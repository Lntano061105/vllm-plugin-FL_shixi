# Fused GDN Gating 算子接入 vLLM-Plugin-FL 技术报告

> 作者：龚昊磊（容器用户名：ghl）
> 项目：vLLM-Plugin-FL 自定义算子接入、编译部署与模型推理验证（6 周实习）
> 负责算子：Fused GDN Gating（`npu_fused_gdn_gating`）
> 目标模型：Qwen3.6 27B / 35B（GDN 线性注意力层）
> 报告日期：2026-09-01（初稿）／2026-09-11（定稿）／2026-09-25（补充模型级 3 次重复对照与 PR 提交状态）｜ 文档位置：`ghl/ghl_report.md`（答辩材料：`ghl/ghl_reply.md`）
> 参考来源：R4（`appleinsky/qwen36_dense_moe` 分支 csrc 实现，commit ab33933）、R8（同分支 Python 接入补丁）——使用情况详见 §4.2，未整体复制实验分支。

---

## 1. 任务背景

本项目面向 vLLM-Plugin-FL 的框架级推理优化，要求复现项目组已有的自定义算子接入工作，理解算子从 csrc 源码、CANN/AscendC 工程、CMake 编译、Python 扩展注册到 vLLM 模型执行路径的完整链路，并在指定模型（Qwen3.6 27B/35B）上完成调用验证。提交以"单算子、最小改动、可独立回退"为原则，基于项目组 main 基线。

本人负责 Fused GDN Gating 算子（27B/35B GDN 路径），固定测试重点为：**固定 Token 数和 Head 数，对比 `g` 与 `beta` 输出**。提交物要求覆盖：源码接入（csrc）、编译部署（可由项目构建流程安装）、Python 调用（torch.ops 注册名/动态库路径/调用结果）、固定测试（算子级固定 Shape + 模型级 1K 输入/1K 输出）、框架接入（进入 Qwen 推理路径 + 回退开关 + Profiler 证据）、性能验证（Microbenchmark 平均时延及 P50/P90；模型侧 TTFT/TPOT/吞吐）、项目汇报（技术报告）。

## 2. 算子定位与调用链

Qwen3.6 的线性注意力层（`Qwen3NextGatedDeltaNet`）前向流程：

```
hidden_states
  ├─ in_proj_qkvz / in_proj_ba          # (num_tokens, hidden) -> qkvz / ba
  ├─ fix_query_key_value_ordering       # 拆出 query/key/value/z/b/a
  ├─ conv1d (causal conv, 含状态缓存)    # AscendC: npu_causal_conv1d_custom
  ├─ fused_gdn_gating  ◄── 本算子       # AscendC: npu_fused_gdn_gating
  │     g   = -exp(A_log) * softplus(a + dt_bias)
  │     beta = sigmoid(b)
  ├─ recurrent / chunk gated delta rule # Decode: npu_recurrent_gated_delta_rule
  │                                     # Prefill: chunk_gated_delta_rule (Triton)
  ├─ RMSNormGated (z 门控) + out_proj
  └─ output
```

自顶向下调用链：

```
Qwen3NextGatedDeltaNet.forward
  → torch.ops.vllm.gdn_attention_core     (vllm_fl/models/qwen3_next.py:477 → 1285)
    → Qwen3NextGatedDeltaNet._forward_core (Triton 基线第 604 行调 fused_gdn_gating)
      → torch.ops._C_ascend.npu_fused_gdn_gating（AscendC 路径，patch 后）
        → fused_gdn_gating_torch_adpt.h    (shape/dtype 校验 + 调 aclnn)
          → aclnnFusedGdnGating            (op_host/op_api：GetWorkspaceSize + Execute)
            → AscendC kernel               (op_kernel/fused_gdn_gating.cpp，AI Core 执行)
```

- torch schema 注册：`csrc/ascend/torch_binding.cpp:2688`。
- 定位结论：该算子位于 `conv1d` 之后、recurrent/chunk gated delta rule 之前，即 GDN 层的门控计算（`g`、`beta`），下游是 linear attention 状态递推。

算子语义（与 vLLM `qwen3_next.fused_gdn_gating` 一致）：

```
x        = a + dt_bias
softplus = (1/beta) * log(1 + exp(beta*x))   若 beta*x <= threshold
           x                                   否则（softplus 退化为恒等）
g        = -exp(A_log) * softplus
beta_out = sigmoid(b)
```

## 3. 算子输入输出、数据类型与数据布局

schema 与 `fused_gdn_gating_torch_adpt.h` 中的约束：

| 参数 | Shape | dtype | 布局 | 说明 |
|---|---|---|---|---|
| A_log | `(num_heads,)` | float32 | ND，1-D | 每头对数衰减，模型参数 |
| a | `(batch, num_heads)` | bf16 / fp16 | ND，2-D | 时间步投影（decay），每头 |
| b | `(batch, num_heads)` | bf16 / fp16 | ND，2-D | 门控投影，dtype 必须与 a 一致 |
| dt_bias | `(num_heads,)` | float32 | ND，1-D | 时间步偏置，dtype 必须与 A_log 一致 |
| g（输出） | `(1, batch, num_heads)` | float32 | ND，3-D | `g = -exp(A_log) * softplus(a + dt_bias)` |
| beta_output（输出） | `(1, batch, num_heads)` | 与 b 同 dtype | ND，3-D | `beta = sigmoid(b)` |

- 约束：`a.size(1) == b.size(1) == A_log.size(0)`；`a`、`b` 同 dtype；`A_log`、`dt_bias` 同 dtype。
- 默认属性：`beta=1.0`、`threshold=20.0`（threshold 用于 softplus 越阈退化，防 exp 溢出）。
- 输出布局与 vLLM Triton 基线一致（`g` 为 fp32、`beta_output` 保持 `b` 的 dtype），可透明替换 GDN 路径中的 Triton `fused_gdn_gating`。
- kernel 内按 tiling key 分派 6 种模板实例（bf16/fp16 输入 × fp32/bf16/fp16 输出），AIV 核执行。

## 4. 工程接入与构建部署

### 4.1 构建部署闭环

```bash
# 1) 编译 torch 扩展（vllm_fl/_C_ascend*.so）
VLLM_VENDOR=ascend python setup.py build_ext --inplace

# 2) 编译并打包 CANN framework 算子（自动拉取 catlass 子模块 → build.sh --pkg → 安装）
cd csrc/ascend && bash build_aclnn.sh ascend910b

# 3) source 部署生成的环境脚本（新 Shell 中验证闭环）
source /workspace/vllm-plugin-FL/vllm_fl/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash

# 4) 验证（连通性 + 算子级固定测试）
python tests/custom_ops_tests/test_fused_gdn_gating.py
pytest tests/ops/ascend/test_fused_gdn_gating.py -m gpu
```

- 产物：自解压包 `csrc/ascend/build/cann-ops-transformer-custom_linux-aarch64.run`（约 21MB）安装到 `vllm_fl/_cann_ops_custom/vendors/custom_transformer/`，与系统 CANN 隔离，**不依赖手工复制 .so**。
- `set_env.bash` 只导出 `ASCEND_CUSTOM_OPP_PATH` 与 `LD_LIBRARY_PATH`（op_api/lib）两个变量；`LD_LIBRARY_PATH` 必须在 Python 进程启动前设置（glibc 只缓存启动时的库搜索路径），"重新打开 Shell → source → 运行"正是部署闭环的关键验证。
- 运行时由 `vllm_fl/__init__.py` 的 `_bootstrap_cann_custom_op_env` 自动自举，默认无需手工 source。

### 4.2 最小改动集合与参考来源

| 类别 | 文件 | 说明 |
|---|---|---|
| 算子本体 | `csrc/ascend/attention/fused_gdn_gating/`（14 文件） | 与参考分支 R4 逐文件 diff **完全一致**（op_host/op_kernel/op_api/torch_adpt） |
| schema 注册 | `csrc/ascend/torch_binding.cpp:51`（include）+ `:2687-2696`（ops.def/ops.impl） | `npu_fused_gdn_gating(...)` |
| 构建清单 | `csrc/ascend/build_aclnn.sh:31`（CUSTOM_OPS 含 fused_gdn_gating） | CANN framework 算子打包 |
| 构建公共 | `csrc/CMakeLists.txt`、`csrc/ascend/CMakeLists.txt` | 扩展分发器 + CANN 算子工程（与他人共用） |
| 环境自举 | `vllm_fl/__init__.py`（`_bootstrap_cann_custom_op_env`） | 免 source 加载（共用） |
| 框架接入 | `vllm_fl/dispatch/backends/vendor/ascend/patch.py` | 注册 patch（共用） |
| 框架接入 | `vllm_fl/dispatch/backends/vendor/ascend/patches/patch_qwen3_6_gdn.py` | 本地裁剪版（577 行 vs R8 907 行），`_forward_core` 挂钩 `npu_fused_gdn_gating`；含默认关闭的 `VLLM_FL_GDN_COUNT_FILE` 计数钩子 |
| 固定测试 | `tests/ops/ascend/test_fused_gdn_gating.py`（25 用例） | 本地新增 |
| 连通性测试 | `tests/custom_ops_tests/test_fused_gdn_gating.py` | 本地新增 |

参考来源说明（如实注明）：
- **R4**（`appleinsky/qwen36_dense_moe` csrc 实现，commit ab33933，2026-07-20）：`csrc/ascend/attention/fused_gdn_gating/` 下 14 个算子源码文件与参考分支逐文件 diff 完全一致。
- **R8**（同分支 Python 接入补丁）：本地 patch 为裁剪版，删除参考分支的实验性路径（PTO megakernel、fused Triton decode kernel、conv1d 权重缓存），仅保留 AscendC 核心接入（causal_conv1d / fused_gdn_gating / recurrent_gated_delta_rule / gemma_rms_norm），符合"单算子、最小改动、可独立回退"原则。

### 4.3 回退开关与调用计数

- `VLLM_FL_DISABLE_ASCENDC_GDN=1`：`_ascendc_ops_available()` 返回 False，保持 vLLM 原生 Triton GDN 路径（开关为 1、`_C_ascend` 不可导入、或 5 个必需 op 缺失时回退）。
- `VLLM_FL_GDN_COUNT_FILE=<path>`（默认关闭，零开销 no-op）：每次真实调用 `npu_fused_gdn_gating` 计数，每 1024 次及进程退出追加 `<pid> <count>` 行，用于与 Profiler 次数交叉核对。

## 5. 测试验证

### 5.1 算子级固定测试（25/25 通过）

`tests/ops/ascend/test_fused_gdn_gating.py`，固定 `NUM_HEADS=32`（27B/35B GDN 层 TP1 头数）、token 数 `[1, 4, 16, 64]`（decode 单步与小段 prefill）、dtype `bf16/fp16`，`rtol=atol=1e-2`：

| 测试类 | 用例 | 数量 |
|---|---|---|
| `TestFusedGdnGatingOutputs` | g/beta 对比 PyTorch 参考（token×dtype 参数化，固定随机种子） | 8 |
| `TestFusedGdnGatingOutputs` | softplus 越阈退化路径（`a=50` 时 `g == -exp(A_log)*x`） | 1 |
| `TestFusedGdnGatingVsTriton` | 对比 vLLM Triton 基线（同上参数化） | 8 |
| `TestFusedGdnGatingValidation` | 8 个 `TORCH_CHECK` 异常分支（维度/dtype/shape 匹配/num_heads） | 8 |

- 连通性测试 `tests/custom_ops_tests/test_fused_gdn_gating.py`：通过（`npu_fused_gdn_gating test passed`）。
- 回退开关验证：默认 `patch_qwen3_6_gdn()` 返回 True 走 AscendC 路径；`VLLM_FL_DISABLE_ASCENDC_GDN=1` 时返回 False 保持 Triton 路径，均有日志可核验。

### 5.2 模型级 1K 输入 / 1K 输出验证（统一脚本）

统一脚本 `/workspace/scripts/run_vllm_fl_profile_unified.sh`（内置 source CANN 环境、FlagGems 白名单、profiler 采集与归档），`--cases "1024,1024,1"`（1 请求 1024 in / 1024 out，eager + chunked）。AscendC/Triton 双路径通过外层 `export VLLM_FL_DISABLE_ASCENDC_GDN=1` 切换（子进程继承）。

**首轮（2026-08-25，各路径 1 次）**

**27B（TP1，gmem 0.9）**：

| 指标 | AscendC（profiler run） | Triton（对照） |
|---|---|---|
| Benchmark duration (s) | 247.96 | 167.83 |
| 生成吞吐 Output tok/s | ~4.3（server.log 引擎稳态值） | 6.10 |
| Mean TTFT (ms) | 未测得* | 922.18 |
| Mean TPOT (ms) | 未测得* | 163.15 |

> *27B AscendC 为 `--bench-profile true` 证据 run：profiler 导出阶段脚本卡住（trace_view.json 15.5GB、CANN 解析 11m43s），vllm bench 结果块未打印，TTFT/TPOT 未测得（总耗时/吞吐从日志提取并如实标注）。

**35B（TP2，gmem 0.7；TP3 结构性不可行，见 §7）**：

| 指标 | AscendC | Triton |
|---|---|---|
| Benchmark duration (s) | 180.18 | 193.43 |
| 生成吞吐 Output tok/s | 5.68 | 5.29 |
| Mean TTFT (ms) | 1058.51 | 1125.14 |
| Mean TPOT (ms) | 175.09 | 187.98 |

**3 次重复对照（2026-09-25，各路径 3 次，均关闭 profiler）**

口径同首轮，仅加 `--bench-profile false --skip-analyse`，规避 27B 上 profiler 导出卡死。12 个 run 串行执行（27B AscendC×3 → 27B Triton×3 → 35B AscendC×3 → 35B Triton×3），全部 rc=0、Successful=1、生成 1024 tokens：

| 分组（n=3） | 吞吐 (tok/s) | TTFT (ms) | TPOT (ms) |
|---|---|---|---|
| 27B AscendC | **6.72 ± 0.24** | 819.69 ± 8.40 | **148.28 ± 5.28** |
| 27B Triton | 6.23 ± 0.07 | 901.67 ± 12.98 | 159.76 ± 1.76 |
| 35B AscendC | **5.60 ± 0.04** | 1050.32 ± 13.84 | **177.82 ± 1.37** |
| 35B Triton | 5.46 ± 0.09 | 1105.74 ± 11.76 | 182.38 ± 3.14 |

AscendC/Triton 比值：27B 吞吐 1.078x、TTFT 0.909x、TPOT 0.928x；35B 吞吐 1.026x、TTFT 0.950x、TPOT 0.975x。各分组的吞吐与 TPOT 区间互不重叠（如 27B AscendC TPOT 142.55~152.96 vs Triton 158.30~161.71），方向结论稳定。逐 run 数值与统计见 `/workspace/results/ghl/20260925_27B35B无profiler对照/result_summary.md`（原始 run 数据与跑批日志已按项目要求于 2026-09-25 删除，驱动脚本保留）。

**Profiler 调用证据（op_statistic.csv，27B AscendC run）**：`FusedGdnGating`（AI_VECTOR_CORE）Count=**49152**（= 48 GDN 层 × 1024 decode 步，精确对应）、Total 249.6ms、device 耗时占比 0.448%；配套 GDN 算子均在列（RecurrentGatedDeltaRule 2.005%、CausalConv1d 0.893%、chunk_gated_delta_rule_fwd_kernel 0.048% 等）。与计数钩子（`VLLM_FL_GDN_COUNT_FILE`）及理论值交叉核对一致。35B 两 run 因磁盘/时间约束为性能对照（`--bench-profile false`），Profiler 证据以 27B 为准。

## 6. 性能分析

### 6.1 算子级 Microbenchmark（W3 交付物）

`/workspace/scripts/ghl/benchmark_fused_gdn_gating.py`：预热 100 次 + 同步 + 测量 500 次，正确性校验先行（与 PyTorch 参考对比，rtol/atol=1e-2），报告 min/mean/P50/P90/P99/max。三后端（AscendC / Triton 基线 / PyTorch 参考）× token 1/4/16/64/256/1024 × bf16/fp16 共 **36 项配置全部通过正确性校验**（2026-08-20 实测）。

代表配置（µs）：

| 配置 | AscendC | Triton 基线 | PyTorch 参考 |
|---|---|---|---|
| tokens=1, bf16 | mean 89.25 / P50 83.96 / P90 85.85 | mean 225.24 / P50 209.73 / P90 218.97 | mean 370.23 |
| tokens=64, bf16 | mean 100.32 / P50 98.84 / P90 94.20 | mean 226.12 | mean 365.75 |
| tokens=1024, bf16 | mean 97.86 / P50 99.24 / P90 103.00 | mean 361.07 / P50 363.97 / P90 358.15 | mean 358.02 |
| tokens=1024, fp16 | mean 95.48 / P50 95.97 / P90 108.44 | mean 370.52 | mean 359.69 |

- AscendC 平均时延约 85-100µs，**不随 token 数增长**（kernel 内 1 次 AIV 调用）；Triton 基线约 195-370µs，随 token 数增长。
- **AscendC/Triton mean 比值 0.26x–0.44x**（AscendC 快 2.3–3.8 倍），tokens=1024 时最快（0.26x-0.27x）。

### 6.2 模型级 vs 算子级性能

| 层 | 27B AscendC | 27B Triton | 35B AscendC | 35B Triton |
|---|---|---|---|---|
| 算子级 mean 时延 | ~85-100µs | ~200-370µs | —（同算子） | — |
| 模型级 1K/1K 生成吞吐（3 次均值，2026-09-25） | **6.72 tok/s** | 6.23 tok/s | **5.60 tok/s** | 5.46 tok/s |
| 模型级 TPOT（3 次均值，2026-09-25） | **148.28 ms** | 159.76 ms | **177.82 ms** | 182.38 ms |

**实测结论**：
- **两条模型上 AscendC 均优于 Triton**：27B 吞吐 +7.8%、TPOT −7.2%、TTFT −9.1%；35B 吞吐 +2.6%、TPOT −2.5%、TTFT −5.0%，与算子级方向（AscendC 快 2.3-3.8 倍）一致；各分组的吞吐与 TPOT 区间互不重叠，方向稳定。
- **首轮（2026-08-25）"27B AscendC 慢于 Triton"已定位为 profiler run 的口径问题**：当次 27B AscendC 为 `--bench-profile true` 取证 run（~4.3 tok/s，profiler 导出卡死导致 TTFT/TPOT 未测得）；2026-09-25 关闭 profiler 后 3 次均值 6.72 tok/s，反超同批 Triton 的 6.23 tok/s。故"用 profiler 产物定位 decode 路径 `npu_recurrent_gated_delta_rule` / state 布局转换开销"的前提不成立，无需再深挖。
- **模型级增益（2.6%~7.8%）远小于算子级增益（2.3-3.8 倍）**：与 Profiler 证据一致——`FusedGdnGating` 仅占 device 耗时 0.448%，GDN 全路径（0.448% + RecurrentGatedDeltaRule 2.005% + CausalConv1d 0.893% + chunk 系列）合计约 3.7%，端到端影响只能到几个百分点量级，该算子不是端到端瓶颈。
- 与 2026-08-25 单次数据互证良好：27B Triton 6.10→6.23 tok/s、TPOT 163.15→159.76 ms；35B AscendC 5.68→5.60、Triton 5.29→5.46。

### 6.3 共享基线重测（27B graph / eager，供对照）

本节记录项目组**共享基线**在本环境的重测结果。注意口径与 §5.2 / §6.2 不同：共享基线为 `--cases "1024,1024,128"`、128 请求、并发 64、TP4、gmem0.6、max-model-len 8192，**不是单请求 1K/1K**，两组数据不可直接比较。

| 基线 | cudagraph 模式 | 成功请求 | Output 吞吐 | TPOT median | 来源 |
|---|---|---|---|---|---|
| 27b_graph（原始记录） | FULL | 65/128 | 266.38 tok/s | 209.45 ms | 2026-08-31~09-01 |
| 27b_eager | —（eager） | 128/128 | 288.20 tok/s | 216.73 ms | 2026-09-10 重测 |
| 35b_eager | —（eager） | 128/128 | 245.88 tok/s | 252.72 ms | 2026-09-10 重测 |
| 35b_graph | PIECEWISE | 128/128 | 322.95 tok/s | 192.32 ms | 2026-09-10 重测 |
| **27b_graph** | **PIECEWISE** | **128/128** | **496.66 tok/s** | **124.07 ms** | **2026-09-25（本次，首次取到有效数据）** |

本次 27B graph 明细：Benchmark duration 263.91 s、Total token throughput 993.33 tok/s（峰值 output 704.00）、TTFT mean/median/P99 = 8231.68 / 2813.94 / 27323.84 ms、TPOT mean/median/P99 = 119.87 / 124.07 / 127.06 ms、ITL mean/median/P99 = 119.75 / 100.49 / 930.84 ms。

- **可用性**：27B graph 在本环境此前累计 10 次失败（FULL 1 + PIECEWISE 9，全部为 `Tried to allocate 650 MiB` 的 NPU OOM）；2026-09-25 按同一口径重测 4 次，**第 3 次成功**并通过严格判据（`Successful>0` 且 `Total generated tokens>10` 且 `duration>60s`），成功 run 的 `server.log` 无任何错误行。其余两次失败**均非 OOM**——一次为 `aicore timeout`（retCode 0x25 / 507014）导致 Worker 进程退出，一次为 EngineCore 连续 4 分钟 `No available shared memory broadcast block found in 60 seconds` 后 fatal error。即该基线**可复现但极不稳定**（累计 13 次尝试成功 1 次），且失败原因不唯一（显存压力 + 设备侧超时 + 引擎挂死），与 §7 第 5 条同一结论。
- **性能**：graph（PIECEWISE）明显优于 eager（27B 496.66 vs 288.20 tok/s、TPOT median 124.07 vs 216.73 ms），符合 graph 模式吸收 decode 调度开销的预期；因 cudagraph 模式不同（原始 FULL vs 本次 PIECEWISE），**与原始 65/128、266.38 tok/s 记录不宜作严格对比**。
- 归档：`/workspace/results/ghl/20260925_27b_graph重测/result_summary.md`（原始 run 数据与跑批日志已按要求删除，run 名与全部数值保留在该摘要中）。

## 7. 限制与遗留问题

1. **提交方式与提交说明（`ghl/git_method.md`）的偏差**：说明要求把 `origin` 指向 `Lntano061105/vllm-plugin-FL_shixi` 后直推；实测当前账号（GitHub `ww1157`）对该仓库无写权限（`Permission ... denied to ww1157`），故按其"如遇权限问题请在群里反馈"处理，并参照仓库既有做法（PR #1/#2/#4/#5 均由 fork 提出）改走 fork 流程，PR #7 目标分支为主仓库 `main`。若项目组要求直推主仓库，需要 collaborator 权限，或把容器公钥加为该仓库 Deploy key。
2. **TP3 对 35B 结构性不可行**：GDN conv 状态维度 8192 无法被 3 整除（vLLM `ensure_divisibility` 断言），35B 取 TP2。
3. **27B AscendC 首轮 profiler run 数据缺失（已闭合）**：该 run 的 profiler 导出阶段卡住（trace_view.json 15.5GB、CANN 解析 11m43s），TTFT/TPOT 未测得；2026-09-25 已用 `--bench-profile false` 补齐 3 次对照（见 §5.2 / §6.2）。
4. **27B/35B 模型级方向"不一致"（已澄清）**：2026-09-25 的 3 次重复数据显示两条模型上 AscendC 均快于 Triton，首轮"27B 慢"系 profiler 口径问题，不再是遗留问题。
5. **27B graph 基线在本环境极不稳定（2026-09-25 重测修订）**：按官方口径（TP4 / gmem0.6 / max-model-len 8192 / 128 请求、并发 64）复测，累计 13 次尝试（FULL 1 + PIECEWISE 12）**仅成功 1 次**。2026-09-25 的 4 次尝试中第 3 次成功取得有效数据：**128/128 全部成功、Output 吞吐 496.66 tok/s、TPOT median 124.07 ms**、duration 263.91 s，`server.log` 无任何错误行；另两次失败**均非 OOM**——一次为 `aicore timeout`（retCode 0x25 / 507014），一次为 EngineCore 连续 4 分钟 `No available shared memory broadcast block found in 60 seconds` 后 fatal error（此前 10 次则全部为 `Tried to allocate 650 MiB` 的 NPU OOM）。即该基线**可复现但不稳定，且失败原因不唯一**（显存压力 + 设备侧超时 + 引擎挂死）。要与原始 FULL 记录严格对比需补同模式重测；要稳定取数需项目组确认配置（capture 尺寸 / max-num-seqs / max-model-len），或按"重试至首次成功并注明尝试次数"记录。归档：`/workspace/results/ghl/20260925_27b_graph重测/`。
6. **flag_gems 5.0.2 pow 编译失败**：模型加载时报 TypeError（cannot convert None to tensor），统一脚本内置 `--flaggems-ops unquantized_fused_moe_method,topk_softmax` 白名单规避。
7. **并发跑批瞬态 aivec 错误**：模型日志无错误、不可复现，疑似跨卡同步粘性错误上报，串行跑批规避。

## 8. 结论

对照任务书要求（工程接入 30% / 正确性与测试 20% / 框架集成与性能 20% / 代码质量文档答辩 30%）：

- ✅ 源码接入：算子进入 `csrc/ascend/attention/fused_gdn_gating/`，含 op_host/op_kernel/op_api/torch_adpt 完整工程与 schema 注册。
- ✅ 编译部署：`build_aclnn.sh` 一键打包安装，重新打开 Shell 后仍可加载，不依赖手工复制 .so。
- ✅ Python 调用：`torch.ops._C_ascend.npu_fused_gdn_gating`，连通性测试通过。
- ✅ 固定测试：算子级 25/25（固定 Shape/dtype/布局，对比 PyTorch 参考与 Triton 基线，含异常用例）；模型级 1K/1K 双路径完成，并补齐各路径 3 次重复对照（2026-09-25）。
- ✅ 框架接入：进入 Qwen 推理路径，`VLLM_FL_DISABLE_ASCENDC_GDN` 回退开关，Profiler 证据（49152 次调用/0.448%）+ 计数钩子交叉核对。
- ✅ 性能验证：Microbenchmark（平均时延 + P50/P90）；模型侧 TTFT/TPOT/吞吐，3 次重复确认 AscendC 在 27B/35B 上均优于 Triton。
- ✅ 项目汇报与提交：技术报告（本文，2026-09-25 更新）+ 答辩材料（`ghl/ghl_reply.md`）；**PR #7 已提交**（`ww1157:ghl/fused_gdn_gating → Lntano061105:main`，提交 `2689811`，19 文件 / +3615 行 / 0 删除），走 fork 流程（原因见 §7 第 1 条）。

**主要结论**：Fused GDN Gating 算子已完成完整接入链路与双路径（AscendC/Triton）验证，并以最小改动集合提交为 PR #7；算子级性能显著优于 Triton 基线（快 2.3-3.8 倍）；模型级经 3 次重复对照确认，27B/35B 上 AscendC 均优于 Triton（吞吐 +7.8% / +2.6%，TPOT −7.2% / −2.5%），首轮观察到的"27B 端到端较慢"已定位为 profiler run 的口径问题。模型级增益远小于算子级增益，与该算子仅占约 0.448% device 耗时的事实一致。

## 附录 A：复现命令速查

```bash
# 环境 source（新 Shell）
source /workspace/vllm-plugin-FL/vllm_fl/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash

# 算子级固定测试（25 项）
pytest tests/ops/ascend/test_fused_gdn_gating.py -m gpu

# 算子级 Microbenchmark
python /workspace/scripts/ghl/benchmark_fused_gdn_gating.py --backends ascendc,triton,torch \
    --tokens 1,4,16,64,256,1024 --dtypes bf16,fp16 --reps 500 --device npu:0

# 模型级 1K/1K（AscendC 默认；Triton 对照加 export VLLM_FL_DISABLE_ASCENDC_GDN=1）
/workspace/scripts/run_vllm_fl_profile_unified.sh \
    --model-path /models/Qwen3.6-27B --model-tag qwen3.6-27b \
    --mode eager --chunked true --devices 4 --tp 1 --gmem 0.9 --max-model-len 4096 \
    --cases "1024,1024,1" --bench-profile false --skip-analyse --run-label <ascendc|triton>_r1
```

## 附录 B：关键文件与产物

| 项 | 路径 |
|---|---|
| 算子源码（与 R4 一致，14 文件） | `csrc/ascend/attention/fused_gdn_gating/` |
| schema 注册 | `csrc/ascend/torch_binding.cpp:2687-2696` |
| 框架接入 patch（提交版：main 版 + 计数钩子） | `vllm_fl/dispatch/backends/vendor/ascend/patches/patch_qwen3_6_gdn.py` |
| 固定测试（25 用例） | `tests/ops/ascend/test_fused_gdn_gating.py` |
| 连通性测试 | `tests/custom_ops_tests/test_fused_gdn_gating.py` |
| 构建脚本 | `csrc/ascend/build_aclnn.sh`（CANN 算子）、`setup.py build_ext --inplace`（torch 扩展） |
| Profiling 统一脚本 / 二次打包 | `/workspace/scripts/run_vllm_fl_profile_unified.sh` / `/workspace/scripts/package_op_statistic.sh` |
| 算子级微基准（个人脚本） | `/workspace/scripts/ghl/benchmark_fused_gdn_gating.py` |
| 微基准结果（2026-08-20） | `/workspace/results/ghl/20260820_算子级微基准/fused_gdn_gating_microbench_20260820.txt` |
| 模型级 1K/1K 结果（2026-08-25，首轮） | `/workspace/results/ghl/20260825_模型级1K1K验证/`（4 个 run + `*_op_statistic.tar.gz`） |
| 模型级无 profiler 重复对照（2026-09-25，各 3 次） | `/workspace/results/ghl/20260925_27B35B无profiler对照/`（`result_summary.md` + 驱动脚本；原始 run 数据与跑批日志已按要求删除） |
| 共享基线重测（27B graph，2026-09-25，128 请求） | `/workspace/results/ghl/20260925_27b_graph重测/result_summary.md`（原始 run 数据与跑批日志已按要求删除） |
| 共享基线重测记录（2026-09-10，eager/graph 4 项） | `ghl/Baseline_results.md` 附录「2026-09-10 重测记录」 |
| 提交分支 / PR | `ww1157/vllm-plugin-FL_shixi:ghl/fused_gdn_gating`（提交 `2689811`）→ PR #7，目标 `Lntano061105/vllm-plugin-FL_shixi:main` |
| 模型权重 | `/models/Qwen3.6-27B`、`/models/Qwen3.6-35B-A3B` |
| 参考分支 | `appleinsky/qwen36_dense_moe`（R4 commit ab33933 / R8） |
| 环境 | Ascend 910B3 × 8、torch 2.8.0 + torch_npu 2.8.0.post2、vllm 0.13.0、CANN 8.5/9.0、flag_gems 5.0.2 |

## 附录 C：交付物清单（2026-09-25 更新）

| 交付物 | 位置 | 状态 |
|---|---|---|
| 技术报告（本文） | `ghl/ghl_report.md` | ✅ 2026-09-25 更新 |
| 答辩材料（提纲/数据/Q&A） | `ghl/ghl_reply.md` | ✅ 完成（2026-09-25 已同步 PR 状态、重复对照与 27B graph 重测结论） |
| 周报汇总（week1~week6） | `ghl/week_report.md` | ✅ 更新 |
| 周报（W39 单周） | `ghl/sixth_report.md` | ✅ 完成 |
| 算子源码（14 文件，与 R4 一致） | `csrc/ascend/attention/fused_gdn_gating/` | ✅ |
| schema 注册 | `csrc/ascend/torch_binding.cpp:2687-2696` | ✅ |
| 构建清单 / 打包脚本 | `csrc/ascend/build_aclnn.sh` | ✅ |
| 框架接入补丁 + 回退开关 + 计数钩子 | `vllm_fl/dispatch/backends/vendor/ascend/patches/patch_qwen3_6_gdn.py` | ✅ |
| 算子级固定测试（25 用例） | `tests/ops/ascend/test_fused_gdn_gating.py` | ✅ 25/25 |
| 连通性测试 | `tests/custom_ops_tests/test_fused_gdn_gating.py` | ✅ 通过 |
| 算子级 Microbenchmark（个人脚本）+ 结果 | `/workspace/scripts/ghl/benchmark_fused_gdn_gating.py`、`/workspace/results/ghl/20260820_算子级微基准/` | ✅ 36 配置通过 |
| 模型级 1K/1K 结果（4 run + 算子统计包） | `/workspace/results/ghl/20260825_模型级1K1K验证/` | ✅ |
| 模型级重复对照（12 run，摘要） | `/workspace/results/ghl/20260925_27B35B无profiler对照/result_summary.md` | ✅ |
| 共享基线重测（27B graph / eager，摘要） | `/workspace/results/ghl/20260925_27b_graph重测/result_summary.md`、`ghl/Baseline_results.md` | ✅ 27B graph 首次取到有效数据（128/128、496.66 tok/s） |
| PR 提交 | PR #7：`ww1157:ghl/fused_gdn_gating → Lntano061105:main`（提交 `2689811`，19 文件 / +3615 行 / 0 删除） | ✅ 已提交（fork 流程，评审中） |
