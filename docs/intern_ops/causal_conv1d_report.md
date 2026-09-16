# Causal Conv1D 算子接入 vLLM-Plugin-FL 技术报告

**实习生**：刘国新
**负责算子**：Causal Conv1D（Qwen3.6 27B/35B GDN 路径）
**参考实现**：R6（`appleinsky/vllm-plugin-FL` 分支 `qwen36_dense_moe` 的 `csrc/ascend/moe/causal_conv1d/`）、R8（`patch_qwen3_6_gdn.py`）

---

## 1. 背景与目标

Qwen3.6 的 Gated DeltaNet（GDN）结构在 QKV 拼接后接一个因果一维卷积（kernel size=4），用于给状态演化引入短期记忆。该卷积在 Triton 参考实现上于昇腾 NPU 计算效率低，本项目将其替换为 AscendC 原生自定义算子，目标：

- 源码进入 `csrc` 构建系统，可由项目构建流程编译安装；
- 通过 `torch.ops._C_ascend.npu_causal_conv1d_custom` 进入 Qwen 推理路径；
- 保留基线回退开关，用 Profiler 证明实际调用；
- 完成算子级固定用例（Prefill/Decode）与模型级 1K in/1K out 验证，并给出性能数据。

本实现基于 R6 参考分支单算子源码，未合并参考分支的其他改动；工程接入、编译部署、测试与性能验证为本人的独立工作。

## 2. 算子调用链

### 2.1 在模型计算图中的位置

`/workspace/vllm-plugin-FL/vllm_fl/models/qwen3_next.py`：

```
self.conv1d = ColumnParallelLinear(..., kernel_size=4)   # conv 权重 (width, dim)
Prefill: x = rearrange_mixed_qkv(...) → causal_conv1d_fn(x, wt, conv_states, ...)
Decode : x = rearrange_mixed_qkv(...) → causal_conv1d_update(x, wt, conv_states, ...)
```

- conv 输入 = QKV 拼接。`conv_dim = key_dim × 2 + value_dim`；Qwen3.6-27B 下 key=16×128=2048、value=48×128=6144 → **全量 10240，TP4 每卡 2560**，width=4。
- 调用函数 `causal_conv1d_fn/update` 来自 `vllm.model_executor.layers.mamba.ops.causal_conv1d`（Mamba 族共用，即 Triton 基线来源）。

### 2.2 Python 接入与注册

`patch_qwen3_6_gdn.py` 在 `Qwen3NextGatedDeltaNet._forward_core` 路由，把 Triton 版替换为自定义算子。注册依赖 `import vllm_fl._C_ascend`（触发 torch 绑定，30MB .so），随后：

```python
torch.ops._C_ascend.npu_causal_conv1d_custom(...)
```

回退开关：`VLLM_FL_DISABLE_ASCENDC_GDN=1` → 走 Triton 基线路径。

开关双态日志证据（server.log，两种模式各跑一次 1K/1K 验证）：

```
# 启用 AscendC（默认 VLLM_FL_DISABLE_ASCENDC_GDN=0）
[patch_qwen3_6_gdn.py] Patched Qwen3NextGatedDeltaNet ... (AscendC causal_conv1d / fused_gdn_gating / recurrent_gated_delta_rule / gemma_rms_norm)

# 回退 Triton 基线（VLLM_FL_DISABLE_ASCENDC_GDN=1）
[patch_qwen3_6_gdn.py] keep Triton GDN path
```

两组均完成推理，回退路径下算子不进入模型执行路径（Profiler 无 CausalConv1d 记录），模型输出正常，回退开关真实可用。

### 2.3 算子签名与数据布局

```
npu_causal_conv1d_custom(output, x, weight, conv_state, bias_opt,
                         query_start_loc_opt, cache_indices_opt,
                         initial_state_mode_opt, num_accepted_tokens_opt,
                         activation_mode, pad_slot_id, run_mode) -> output
```

- **output 是预分配张量**（第 1 个入参，返回同一对象）；
- 必填：`x`、`weight`、`conv_state`；可选：`bias`、`query_start_loc`、`cache_indices`、`initial_state_mode`、`num_accepted_tokens`；
- 属性：`activation_mode`(0/1)、`pad_slot_id`、`run_mode`(0=prefill/FN，1=decode/UPDATE)；
- dtype：x 仅 bf16/fp16，weight/convStates/bias 与 x 一致；**dim % 16 == 0** 硬性对齐；
- 数据布局：x=`(token数, dim)` 或 `(batch, seqLen, dim)`；weight=`(width, dim)`，width∈[2,4]（`wt[0]`=最老 tap）；convStates=`(num_cache_lines, stateLen, dim)`，stateLen≥width−1，kernel **原地写回**；
- 约束：2D prefill 必须传 `query_start_loc`；`initial_state_mode` 仅 prefill；`num_accepted_tokens` 仅 decode 且 width==4。

## 3. 编译部署

| 步骤 | 命令 |
|---|---|
| 编译（含 .run 打包） | `cd /workspace/vllm-plugin-FL/csrc/ascend && bash build.sh -n "causal_conv1d" -c "ascend910b" --pkg` |
| 安装到独立目录 | `bash build/cann-ops-transformer-custom_linux-aarch64.run --install-path=/workspace/data/lgx/cann_ops --quiet` |
| 持久化 | `echo 'source /workspace/data/lgx/cann_ops/vendors/custom_transformer/bin/set_env.bash' >> ~/.bashrc` |
| 环境修复 | `export PYTHONPATH=/workspace/vllm-plugin-FL:$PYTHONPATH`（editable 安装引导失效时） |

关键点：

- `--pkg` 才生成自解压 `.run`（不带仅出 `.run.json` 清单）；
- 安装到独立目录而非覆盖插件自带 OPP：插件 `_cann_ops_custom` 的 `libcust_opsproto_rt2.0.so` 不含 causal_conv1d 符号，且单算子编译出的 .so 只含本算子，直接覆盖会破坏其余 ~40 个算子注册。`set_env.bash` 用 `${OLD}:${NEW}` 前置拼接 `ASCEND_CUSTOM_OPP_PATH`，与插件目录共存；
- 重新打开 Shell 后 `~/.bashrc` 自动 source，可独立加载，不依赖手工复制 .so；
- 产物：核函数 `build/binary/ascend910b/bin/causal_conv1d` + `causal_conv1d.json` + op_api/op_proto/op_impl。

## 4. 测试与正确性

所有测试固定随机种子、与参考实现比数值，运行于 `vllm-fl-lgx` 容器（`VLLM_PLUGINS=ascend /usr/local/python3.11.14/bin/python3`）。

### 4.1 固定用例（任务书硬性要求）

| 用例 | 配置 | 结果 | 验收点 |
|---|---|---|---|
| Prefill 固定用例 `test_fixed_prefill.py` | x=(8,16) 2D、runMode=0、qsl=[0,8] | out max_abs_diff≈0.0036；state 写回（末尾 width−1 行）**0.0** | ✓ |
| Decode 固定用例 `test_fixed_decode.py` | x=(2,16) 2D、runMode=1 | out max_abs_diff≈0.0015；state 滚动 `[cs[1], x]` **精确 0.0** | ✓ |
| 基线对比 `test_prefill_baseline.py` | vs 框架自带 `causal_conv1d_ref`（Triton） | out 0.0036 **match: True**；state **match: True** | ✓ |

### 4.2 边界与状态缓存（加分项）

| 用例 | 场景 | 结果 |
|---|---|---|
| `test_accept3_varlen.py` | 5 个变长序列 [3,8,5,12,6]，tile 跨序列 | out 0.0038 ✓、state 0.0 ✓（不串） |
| `test_accept4_cacheidx.py` | cacheIndices=[3,0,-1,2,5,99] 乱序 + padding/越界 | out 0.00095 ✓、padding 槽位保持 NaN ✓、state 0.0 ✓ |
| `test_accept5_mtp.py` | spec decode、numAcceptedTokens=[1,2,3]、width=4 | out 0.0032 ✓、state 移位写回 0.0 ✓ |

状态缓存语义（验收 5 点全绿）：

- Prefill 只在序列 tail chunk 写回 state；
- decode 对 convStates **原地写回**，用例须在调用前拷贝初始 state，否则读到更新后状态误判；
- **FN/prefill 初始 state 默认不启用**：不传 `initial_state_mode` 时序列开头历史填 0，必须显式传全 1 才用初始 state（decode 硬编码 hasInit=true）；
- cacheIndices 中 ==padSlotId 或越界的序列整条跳过（不计算不写 state）；
- MTP：`stateTokenOffset = clamp(accepted−1, 0, stateLen−(width−1))`，`WriteBackStateSpec` 移位写回，要求 stateLen≥2+seqLen。

### 4.3 单元测试提交物

`tests/custom_ops_tests/test_causal_conv1d_accuracy.py`：pytest 风格，固定 `torch.manual_seed(0)` + CPU 参考实现，整合验收点 1–5 + 框架 Triton 基线对比，共 6 个 `test_` 函数，服务器全绿。

> 注：任务书提交物路径 `tests/ops/ascend/` 为概括路径，项目实际算子测试目录为 `tests/custom_ops_tests/`。环境限制：服务器 `vllm_fl.utils` 无 `enable_custom_op`（新版才有），本测试不依赖它。

### 4.4 异常用例

| 异常输入 | 处理语义 | 实测结果 |
|---|---|---|
| cacheIndices 越界（如 99） | `ResolveSeqCacheIndex` 判定越界 → 该序列**整条跳过**（不计算、不写 state、输出保持原值） | 越界槽位 out 保持 NaN ✓、state 0.0 ✓ |
| cacheIndices == padSlotId（如 −1，padding 槽位） | 同越界，整条跳过 | padding 槽位 out 保持 NaN ✓ |
| cacheIndices 乱序（[3,0,−1,2,5,99]） | 按槽位映射各自独立处理，互不串扰 | out 0.00095 ✓、state 0.0 ✓ |

与参考实现的差异说明：本算子对非法 cacheIndices 做了**显式跳过**（不计算不写回），不会触发设备端越界写；同类参考实现（如 recurrent_gated_delta_rule）在非法槽位索引下会触发设备端异步异常（错误码 507035），本算子在此更健壮。

## 5. 性能验证

> **归因说明（重要）**：以下模型级「AscendC vs Triton 基线」对比，切换的变量是回退开关 `VLLM_FL_DISABLE_ASCENDC_GDN`。该开关**同时**切换 GDN 路径的 4 个算子（`causal_conv1d` / `fused_gdn_gating` / `recurrent_gated_delta_rule` / `gemma_rms_norm`），因此表中的 TTFT/TPOT 收益是**整条 GDN 路径**的效果，**并非 causal_conv1d 单一算子贡献**。causal_conv1d 算子自身的占比与耗时由 §5.5（Profiler）与 §5.6（Microbenchmark）单独量化。

### 5.1 早期低并发验证（27B, TP4, 1024 in / 1024 out, 3 次重跑）

早期测试确认收益方向：TTFT −43.2%、吞吐 +7.7%、TPOT −6.6%，多次重跑一致。完整原始数据见 `/workspace/results/lgx/20260820_model3x/`。后续性能结论以 §5.3 高并发验收数据为准。

### 5.2 27B 单组高并发预实验（10 次重跑）

27B eager 高并发预实验（`--cases "1024,1024,128" --concurrency 64`），AscendC 与 Triton 各 5 次，剔除 2 个外部干扰样本（NPU 被占 + 冷启动，非最优/最差剔除）后各 4 次取均值：

| 指标 | AscendC 均(×4) | Triton 均(×4) | 提升 |
|---|---|---|---|
| Median TTFT (ms) | 2874.88 | 9584.62 | **−70.0%** |
| Mean TPOT (ms) | 226.85 | 277.58 | −18.3% |
| Output tok/s | 272.44 | 213.76 | **+27.5%** |

剔除离群后两组各自 tight cluster（AscendC TTFT ±1.5%、TPOT ±1.4%；Triton ±1.8%/±1.7%）。TPOT 提升低于 TTFT，符合 §5.6 微基准「decode host 分发受限」结论。完整 10 次原始数据见 `/workspace/results/lgx/`。

> **本节为单组预实验，完整验收以 §5.3 的 4 组 Baseline 为准。**

### 5.3 4 组 Baseline 高并发验收（27B/35B × eager/graph）

mentor 最终验收要求 4 组对比测试，每组 ≥3 次重复，作为算子接入的统计依据。统一高并发配置：`--cases "1024,1024,128" --concurrency 64 --max-num-seqs 64 --max-model-len 8192 --tp 4 --gmem 0.6`；切换 `VLLM_FL_DISABLE_ASCENDC_GDN` 对比 Triton 基线，每组 3 次取均值（剔除外部干扰样本）：

| 组别 | 指标 | Baseline 均值 | AscendC 均值 | 提升 |
|---|---|---|---|---|
| **27B eager** | Throughput (tok/s) | 220.07 | 286.00 | **+30.0%** |
| | Median TTFT (ms) | 9459 | 2949 | −68.8% |
| | Median TPOT (ms) | 278.77 | 217.91 | −21.8% |
| **27B graph** | Throughput (tok/s) | 273.10 | 513.94 | **+88.2%** |
| | Median TTFT (ms) | 9512 | 2802 | −70.5% |
| | Median TPOT (ms) | 224.29 | 120.23 | −46.4% |
| **35B eager** | Throughput (tok/s) | 223.88 | 261.62 | **+16.9%** |
| | Median TTFT (ms) | 8631 | 4734 | −45.2% |
| | Median TPOT (ms) | 274.25 | 237.11 | −13.5% |
| **35B graph** | Throughput (tok/s) | 245.31 | 331.87 | **+35.3%** |
| | Median TTFT (ms) | 8642 | 4526 | −47.7% |
| | Median TPOT (ms) | 249.96 | 186.90 | −25.4% |

结论：

1. **4 组全部正向提升**，吞吐 +16.9%~+88.2%，无一组负增长；
2. **TTFT 改善最显著**：4 组全部 −45% 以上，交互式场景收益最大；
3. **27B dense 收益 > 35B MoE**：conv 占比 0.64% vs 0.25%（见 §5.5），稠密模型中算子占比更大；
4. **graph 模式 TPOT 收益 > eager**：graph −46%/−25% vs eager −22%/−14%，与 §5.6 微基准「decode host 分发受限、graph 捕获抹平开销」一致；
5. **AscendC 三次重复波动 <1%**，远优于 baseline（baseline 35B eager 受外部干扰时 TTFT 曾达 45s，已重跑剔除）。

失败/异常样本（不参与均值，如实记录）：

- 27B graph baseline 09-09 三次全废（OOM/空结果），09-11 重跑 r1/r2/r3；
- 35B eager baseline r1/r3 受干扰（thr 158 / TTFT 45219），09-16 重跑；
- 35B graph asc r1 失败 102/26，09-15 重跑。

### 5.4 35B 低并发验证（TP4, 1024 in / 1024 out, 3 次重跑）

35B MoE 模型低并发测试，6 次数据完全分离：

| 指标 | AscendC 均(×3) | Triton 均(×3) | 提升 |
|---|---|---|---|
| TTFT (ms) | 1186.75 | 1356.98 | **−12.5%** |
| TPOT (ms) | 108.80 | 111.63 | **−2.5%** |
| Output tok/s | 9.11 | 8.86 | +2.7% |

TTFT 增益（12.5%）大于 conv 自身占比（0.25%，见 §5.5），收益来自消除 Triton host 分发/图捕获开销。对比 27B（TTFT −43%），35B 为 MoE、conv 占比被稀释（0.25% vs 0.64%）。35B 高并发验收见 §5.3。

### 5.5 Profiler 证明调用

模型级 profiling 后 4 卡 `op_statistic.csv` 均采集到 `CausalConv1d`（rank-0）：

```
0,CausalConv1d,MIX_AIC,49152,350585.78,5.86,7.132,50.5,0.64
```

Count=49152 = 64 层 × 768 输出 token/层，与 Decode 逐 token 调用吻合（此前误记"48 层"，按 config 为 64 层）；Core Type=MIX_AIC（AI Core 执行）；avg 7.132µs，占整体 0.64%。

35B（Qwen3.6-35B-A3B, TP4 profiling）对比：

```
0,CausalConv1d,MIX_AIC,30720,234801.64,6.92,7.643,45.2,0.248   # r4
0,CausalConv1d,MIX_AIC,30720,239646.22,7.0,7.8,40.46,0.253     # r5
0,CausalConv1d,MIX_AIC,30720,237220.88,6.98,7.722,43,0.25      # r6
```

Count=30720 = 40 层 × 768 输出 token/层（与 config 的 40 层吻合）；avg ≈7.6~7.8µs，**占整体 0.248~0.253%（r4/r5/r6 三次一致）**——约为 27B（0.64%）的 1/3。两层 Count 之比 49152/30720 = 1.6 = 64/40（层数比），每层调用次数一致（均为 768），证明 conv 调用次数随层数线性增长。

**占比与收益的量级差**：35B 上 conv 占比仅 0.248~0.253%（r4/r5/r6 三次一致），却带来 TTFT −12.5% 的收益，说明增益大头来自消除 Triton 参考实现的 host 分发/图捕获/同步开销，而非内核本身（与 5.6 微基准 decode host 分发受限的结论一致）。占比差异（35B 0.25% vs 27B 0.64%）则解释了为什么 35B 收益（−12.5%）仍小于 27B（−43%）。

### 5.6 Microbenchmark（P50/P90, `bench_conv1d.py`, torch.npu.Event 计时）

| 场景 | D | 计时 | p50 (µs) | p90 (µs) |
|---|---|---|---|---|
| PREFILL T=1024 | 2560 | reps=1 | 96.27 | 107.75 |
| DECODE B=1 | 2560 | reps=1 | 86.59 | 98.44 |
| PREFILL T=1024 | 2560 | reps=100 | 34.66 | 37.94 |
| DECODE B=1 | 2560 | reps=100 | 25.42 | 25.99 |
| PREFILL T=1024 | 10240 | reps=100 | 98.78 | 99.06 |
| DECODE B=1 | 10240 | reps=100 | 24.03 | 37.31 |

解读（三层数字互补）：

- **decode 为 host 分发受限**：D 从 2560→10240（4 倍），摊销延迟几乎不变（26.0→27.6µs），单 token 内核小，瓶颈在每调用的 host enqueue ~26µs；
- **prefill 为内核吞吐受限**：维度 4 倍 → 摊销延迟 ~2.8 倍（35.2→98.9µs）；
- **模型 profiler avg 7.13µs** = graph 捕获下纯内核时间，host 分发与同步开销被抹平，故最低。

## 6. 限制与改进方向

- **dtype/形状受限**：仅 bf16/fp16，`dim % 16 == 0`，width∈[2,4]；`num_accepted_tokens` 仅 width==4；
- **状态原地写回**：调用方需注意读取时机（测试已规避）；
- **prefill 初始状态需显式传入**：`initial_state_mode` 缺省不启用，语义与 Triton 基线需对齐说明；
- **decode 增益有限**：瓶颈在 host 分发而非算子本身，graph 捕获后纯内核仅 ~7µs，进一步优化空间小；
- **环境差异**：服务器 `enable_custom_op` 缺失为版本差异，非本算子问题。

## 7. 复现说明

### 7.1 环境信息

| 项 | 值 |
|---|---|
| 硬件 | Ascend 910B3 × 8（测试用 NPU 0–3，TP4） |
| CANN | 9.0.0 |
| Python / torch / torch_npu | 3.11.14 / 2.8.0 / 2.8.0.post2 |
| vLLM / vllm_ascend / vllm-plugin-fl | 0.13.0 / 0.13.0 / 0.1.0（editable） |
| 容器 | `vllm-fl-lgx`（172.16.11.149:3224） |

### 7.2 运行前提

- `export PYTHONPATH=/workspace/vllm-plugin-FL:$PYTHONPATH`；
- `source /workspace/data/lgx/cann_ops/vendors/custom_transformer/bin/set_env.bash`。
- **测试**：脚本在 `/workspace/data/lgx/`（`test_fixed_prefill.py`、`test_fixed_decode.py`、`test_prefill_baseline.py`、`test_accept3_varlen.py`、`test_accept4_cacheidx.py`、`test_accept5_mtp.py`、`test_causal_conv1d_accuracy.py`）；单元测试提交物 `tests/custom_ops_tests/`；基准脚本 `benchmarks/ops/ascend/bench_conv1d.py`。
- **模型级命令**：见 `/workspace/scripts/run_vllm_fl_profile_unified.sh`（AscendC 直接跑；`env VLLM_FL_DISABLE_ASCENDC_GDN=1` 为 Triton 基线）。27B 用 TP4（`--devices 0,1,2,3`），35B 用 TP4（`--devices 0,1,2,3 --model-path /models/Qwen3.6-35B-A3B --gmem 0.6`）。
- **结果归档**：`/workspace/results/lgx/20260818_模型级测试报告`、`20260820_model3x`、`20260820_microbench`、`20260825_accept3to5`、`20260827_35b_model_tp2`。§5.2 高并发 10 次原始 run 目录直接位于 `/workspace/results/lgx/` 下（`atp_qwen3.6-27b_fl_enforce_eager_chunked_lgx_{base,ascendc}_{r1~r5}_tp4_gmem0.6_*`），未再新建子文件夹。

## 8. 提交物清单

| 提交物 | 位置 | 说明 |
|---|---|---|
| 算子实现 | `csrc/ascend/moe/causal_conv1d/`（18 文件） | 完整 AscendC 工程（op_host/op_kernel/torch 适配层） |
| 单元测试 | `tests/custom_ops_tests/test_causal_conv1d_accuracy.py` | 固定种子 + CPU 参考实现，整合验收点 1–5 + Triton 基线对比 |
| 性能脚本 | `benchmarks/ops/ascend/bench_conv1d.py` | 预热/同步/重复，报告 P50/P90 |
| 框架补丁 | `vllm_fl/dispatch/backends/vendor/ascend/patches/patch_qwen3_6_gdn.py` | AscendC 替换 + 回退开关 `VLLM_FL_DISABLE_ASCENDC_GDN` |
| 个人报告 | `docs/intern_ops/`（本文件） | 调用链/编译/测试/性能/限制 |

> 注：任务书提交物路径 `tests/ops/ascend/`、`docs/intern_ops/` 为概括路径；本项目实际算子测试目录为 `tests/custom_ops_tests/`。

## 9. 代码分支与仓库

- **分支**：`add-qwen3_6_ascendc_gdn_ops`（团队共享功能分支）
- **仓库去向**：推个人 Gitee 私有仓库（不直接推官方 `flagos-ai/vllm-plugin-FL`），交作业时提供仓库地址 + 分支名
- **构建产物已从版本控制移除**：`build_out/`、`vllm_fl/_cann_ops_custom/`、`*.so`、`fusion_result.json`、`*.log`、`benchmark_results/` 均不入库（以 `.gitignore` 挡产物）

## 来源与改动范围（声明）

- 算子源码 `csrc/ascend/moe/causal_conv1d/`（18 文件）与 CMake 接入来自 R6 参考分支 `qwen36_dense_moe`，未整体合并参考分支；
- 本人独立完成：编译部署流程（`--pkg` .run 生成、独立安装目录 + OPP 前置）、注册验证、6 组测试脚本与单元测试提交物、microbenchmark 脚本、模型级验证与性能对比；
- 框架补丁 `patch_qwen3_6_gdn.py` 及回退开关 `VLLM_FL_DISABLE_ASCENDC_GDN=1` 来自 R8 参考，未改动。
