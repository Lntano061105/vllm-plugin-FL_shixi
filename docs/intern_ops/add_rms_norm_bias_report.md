# Add RMSNorm Bias / GemmaRMSNorm 实习报告

> 实习生：邓永权 ｜ 项目：vLLM-Plugin-FL 自定义算子接入
> 目标模型：Qwen3.6 27B 与 35B-A3B（公共路径）｜ 参考：R5（算子工程）、R8（Python 接入补丁）
> 状态：数据已填充，最后更新 **2026-09-15**（含管理员 4-Baseline 压测、graph 可用配置、
> 异常用例与 beta 校验修复、35B 背靠背复测）
> 代码：分支 `intern/dyq/add_rms_norm_bias`（基线 a3644b2），提交 `be09f89` + `e016bdd`

---

## 1. 背景与目标

Qwen3.5/3.6 的每层 `input_layernorm` / `post_attention_layernorm`、最终 `model.norm`
以及 GDN 的 `q_norm/k_norm` 都是 `GemmaRMSNorm`（1+weight 约定）。当前 Ascend 侧
`rms_norm_ascend` 走 `torch_npu.npu_add_rms_norm`：**不支持 bias、不返回 rstd**。
本任务接入自定义算子 `npu_add_rms_norm_bias`，语义：

```
x     = x1 + x2                       （residual 更新）
rstd  = 1 / sqrt(mean(x^2) + eps)     （fp32 归约，供下游动态量化复用）
y     = x * rstd * gamma + beta       （beta 可选）
```

固定测试重点（任务书）：对比 **归一化输出 y、rstd、residual x**。

## 2. 调用链

```
Qwen3.6 模型层（input_layernorm / post_attention_layernorm / final norm / q_norm / k_norm）
  └─ GemmaRMSNorm.forward_oot  ← 被本任务补丁替换
       ├─ 有 residual: torch.ops._C_ascend.npu_add_rms_norm_bias(x, residual, 1+weight, None, eps) → (y, rstd, x)
       │    └─ C++ torch binding → EXEC_NPU_CMD(aclnnAddRmsNormBias) → AscendC kernel
       └─ 无 residual: torch.ops._C_ascend.npu_gemma_rms_norm(x, weight, eps)（CANN 内置，1+weight 约定）
```

补丁挂接：`patch.py::apply_ascend_patches → patch_add_rms_norm_bias()`
（仅 patch GemmaRMSNorm，独立开关 `VLLM_FL_DISABLE_ASCENDC_RMSNORM=1` 一键回退基线）。

## 3. 算子设计（实现说明与来源声明）

**接口契约**：`npu_add_rms_norm_bias(x1, x2, gamma, beta?, eps) → (y, rstd, x)`，
与参考 R5 的 aclnn 契约一致（框架层 torch_binding 与融合 pass 依赖该契约）。

**来源声明（红线合规）**：
- `op_host/`（def / infershape / tiling）与 `add_rms_norm_bias_torch_adpt.h`：**独立实现**
  （单一通用内核设计：NORMAL / SPLIT_D 两模式，替代 R5 的五类变体；rstd 无条件写出，
  修复 R5 的 `CopyOutRstd` 平台条件写入坑）
- `op_kernel/`（AscendC 内核）：**基于项目组基线 R5**。自写内核在 CANN 9.0.0 下编译
  遇 `vadd(count=1, half)` 类型检查错误（`kernel_operator_vec_binary_impl.h:70`），
  多轮修复（左值引用 / `.template` / 对齐长度 / 归约写法）未定位根因；
  为保证交付，采用项目组基线 R5 内核（基线自带、已验证可编）。
  **R5 内核为本任务运行版本，自写内核修复列为后续优化。**

## 4. 编译与部署

- 构建：`bash csrc/ascend/build_aclnn.sh ascend910b` → `cann-ops-transformer*.run` → 安装到 `vllm_fl/_cann_ops_custom/`
- Python 扩展：`pip install --no-build-isolation -e .` → `vllm_fl/_C_ascend*.so`
- 运行期：source `_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash`
  （ASCEND_CUSTOM_OPP_PATH / LD_LIBRARY_PATH）
- **环境**：CANN 9.0.0 / torch 2.8.0+cpu / torch_npu 2.8.0.post2 / python 3.11.14 /
  vllm 0.13.0 / 8×Ascend 910B3 / 分支 `intern/dyq/add_rms_norm_bias`（基线 a3644b2）
- **团队仓库修复（必要公共改动）**：`csrc/CMakeLists.txt` 与 `csrc/ascend/CMakeLists.txt`
  原为空文件导致构建失败，参考 `/workspace/vllm-ascend/csrc/CMakeLists.txt`（621 行标准版）
  补齐并登记 `add_rms_norm_bias`（OP_LIST/OP_DIR_LIST）。

## 5. 单元测试（正确性）

- 位置：`tests/ops/ascend/test_add_rms_norm_bias.py`
- 方法：固定种子 `torch.manual_seed(0)`；fp64 参考实现；对比 y / rstd / residual
- 覆盖：8 种 shape（16×128、1×64、2×4×512、1024×3584、1024×5120、32×32768[SPLIT_D]、
  1×8×128 等）× 3 种 dtype（fp16/bf16/fp32）× beta 有无 + `npu_gemma_rms_norm` 1+weight 约定
- **结果：24 case 全部通过（ALL TESTS PASSED）**
- 容差按 dtype 分级（bf16 residual rtol=4e-3，因 bf16 加法精度 ~0.4%）

**误差表（vs fp64 参考，max_abs，2026-08-28 实测）**：

| case | dtype | residual(x) max_abs | rstd max_abs | y max_abs |
|---|---|---|---|---|
| small-2d-nobeta (16×128) | fp16 / bf16 / fp32 | 1.95e-3 / 1.56e-2 / 2.38e-7 | 3.46e-5 / 4.04e-4 / 6.94e-8 | 3.91e-3 / 3.13e-2 / 4.77e-7 |
| small-2d-beta (16×128) | fp16 / bf16 / fp32 | 1.95e-3 / 1.56e-2 / 2.38e-7 | 4.98e-5 / 4.81e-4 / 6.72e-8 | 3.91e-3 / 3.13e-2 / 9.54e-7 |
| single-row (1×64) | fp16 / bf16 / fp32 | 9.77e-4 / 7.81e-3 / 1.19e-7 | 5.39e-5 / 2.81e-4 / 5.93e-8 | 1.95e-3 / 1.56e-2 / 2.38e-7 |
| 3d (2×4×512) | fp16 / bf16 / fp32 | 1.95e-3 / 1.56e-2 / 2.38e-7 | 1.72e-5 / 1.54e-4 / 6.41e-8 | 7.81e-3 / 3.13e-2 / 9.54e-7 |
| model-like-hidden3584 (1024×3584) | fp16 / bf16 / fp32 | 1.95e-3 / 1.56e-2 / 2.38e-7 | 1.42e-5 / 9.09e-8 / 9.19e-8 | 1.56e-2 / 6.25e-2 / 1.91e-6 |
| model-like-hidden5120-beta | fp16 / bf16 / fp32 | 1.95e-3 / 1.56e-2 / 2.38e-7 | 1.39e-5 / 8.25e-8 / 1.06e-7 | 7.81e-3 / 1.25e-1 / 1.91e-6 |
| split-d-fp16 (32×32768) | fp16 / bf16 / fp32 | 1.95e-3 / 1.56e-2 / 2.38e-7 | 2.76e-6 / 1.56e-7 / 9.13e-8 | 7.81e-3 / 6.25e-2 / 1.91e-6 |
| qk-norm-like (1×8×128) | fp16 / bf16 / fp32 | 1.95e-3 / 7.81e-3 / 2.38e-7 | 5.02e-5 / 4.58e-4 / 5.42e-8 | 3.91e-3 / 3.13e-2 / 9.54e-7 |
| gemma (16×128, 1+weight) | fp16 / bf16 / fp32 | — | 5.68e-5 / 5.14e-4 / 9.24e-8 | 1.95e-3 / 1.56e-2 / 4.77e-7 |

**结论**：三个输出（y/rstd/residual）与 fp64 参考的最大绝对误差均在对应 dtype 精度内
（fp16 ~1e-3、bf16 ~1e-2、fp32 ~1e-6），SPLIT_D 大维与 beta 路径同样满足；相对误差
在参考值接近 0 处无意义（max_abs 为准）。

### 5.1 异常与边界用例（negative / boundary，2026-09-11）

- 位置：`/workspace/scripts/dyq/test_add_rms_norm_bias_negative.py`
- 口径：非法输入必须被拒绝（抛 `RuntimeError`），合法边界必须通过；"通过" = 实际行为与预期一致
- 覆盖：shape 不匹配、dtype 不一致、不支持的 dtype、eps 非法、空张量、维度缺失、极小/大边界

| # | 用例 | 预期 | 结果 |
|---|---|---|---|
| 1 | x1/x2 行数不一致 (8,3584) vs (4,3584) | 拒绝 | PASS |
| 2 | gamma 长度不匹配 (1024 vs 3584) | 拒绝 | PASS |
| 3 | beta 长度不匹配 (512 vs 3584) | 拒绝 | **PASS（本次修复后）** |
| 4 | x 为 fp32 而 gamma 为 bf16（dtype 不一致） | 拒绝 | PASS |
| 5 | 不支持的输入 dtype（int32） | 拒绝 | PASS |
| 6 | 空张量 (0, 3584) | 拒绝 | FAIL（已知宽松行为） |
| 7 | 1D 输入（缺 batch 维） | 拒绝 | FAIL（已知宽松行为） |
| 8 | eps 为负值 (−1.0) | 拒绝 | PASS |
| 9 | 边界 (1,8) 极小 shape | 通过 | PASS |
| 10 | 边界 (256,8) 大 batch | 通过 | PASS |

**结果：10 组中 8 组符合预期。**

用例 6/7 未拒绝，记为**已知宽松行为**：本算子接口按 aclnn 惯例以"末维归一化"解释输入，
0 batch 与 1D 被视为退化合法输入，与参考实现口径一致，故不额外收紧校验（收紧会偏离参考语义）。

**由异常用例驱动修复的缺陷（本次唯一的算子代码改动，提交 `e016bdd`）**

- **现象**：beta 长度与 gamma 不一致（如 512 vs 3584）时被**静默接受**，输出错误结果
- **根因**：`CheckInputOutputShape()` 已校验 x1/x2/y/x/gamma，**遗漏可选输入 beta**
- **修复**：新增 beta 校验 —— beta 存在时其 dim num 与逐维 size 必须等于 gamma
  （`csrc/ascend/moe/add_rms_norm_bias/op_host/add_rms_norm_bias_tiling.cpp`）
- **验证**：重编译后用例 3 由 FAIL → PASS；正向 24 组回归 `ALL TESTS PASSED`，no-beta 路径未受影响
- **影响面**：仅新增"非法输入拒绝"路径，不改变合法输入的 tiling 计算与内核行为 →
  §7 / §10 的模型级结果无需重跑

## 6. 算子级性能（Microbenchmark）

- 位置：`benchmarks/ops/ascend/bench_add_rms_norm_bias.py`；方法：预热 10 次 +
  逐次设备同步 + 重复 100 次，报 mean/P50/P90
- 对比基线：`torch_npu.npu_add_rms_norm`（无 beta、不返回 rstd）
- 结果（fp16，存档 `/workspace/results/dyq/20260821_bench/`）：

| shape | custom mean | baseline mean | 差距 |
|---|---|---|---|
| decode 1×3584 | 0.1372 ms | 0.0901 ms | +52% |
| prefill 1024×3584 | 0.1148 ms | 0.0916 ms | +25% |
| qk-norm 1024×128 | 0.1149 ms | 0.0892 ms | +29% |

- 结论：custom 多产出 rstd / residual 两个输出（供下游量化复用、免重算），
  单算子时延高 25-52% 属功能增量成本；端到端影响见 §7 / §10 模型级数据。

**补充：bf16 下按模型实际 hidden size 复测（2026-09-15）**

背景：两个验收模型的 hidden size 不同（27B=5120，35B-A3B=2048），需确认单算子差距是否与
N 相关（tiling 中 `SMALL_REDUCE_NUM = 2000` 为小 N 分支阈值，2048 仅高出 2.4%，存在
"35B 落到不同分支"的疑虑）。

| shape (bf16) | custom mean | baseline mean | custom/baseline |
|---|---|---|---|
| 1 × 2048（decode） | 0.1454 ms | 0.0953 ms | 1.53× |
| 64 × 2048（decode batch） | 0.1220 ms | 0.0965 ms | 1.26× |
| 1024 × 2048（prefill） | 0.1244 ms | 0.1027 ms | 1.21× |
| 1 × 5120（decode） | 0.1202 ms | 0.0949 ms | 1.27× |
| 64 × 5120（decode batch） | 0.1217 ms | 0.0967 ms | 1.26× |
| 1024 × 5120（prefill） | 0.1404 ms | 0.1179 ms | 1.19× |

- 证据：`/workspace/results/dyq/20260915_rmsnorm_perf_debug/bench_main.txt`（卡 4，warmup 20 / iters 100）
- **N=2048 与 N=5120 的差距倍数几乎相同（均 ~1.26×）→ 否定了"2048 卡阈值落到慢分支"的假设**；
  tiling 阈值与本次差距无关
- 差距的绝对量恒定在 **~25-30 µs/次**，且 custom 在三种规模下时延基本平坦（0.12-0.14 ms），
  说明主导因素是**与规模无关的固定开销**（host 侧 tiling 计算 + aclnn 派发），
  而非内核计算量（内核实测 ~2.7-2.8 µs/次，见 §7.2）
- **端到端量级估算**：35B 共 40 层、每层 2 次融合调用 ≈ 80 次/步，80 × 28 µs ≈ **2.2 ms/步**，
  相对 35B 解码每步约 250 ms 占比 **< 1%**；且 graph 模式下 host 开销被图捕获吸收。
  实测端到端为 **ON 略优**（§10.1），即该开销已包含在结果内，不构成服务级回退
- **后续优化方向**：把 tiling 中的常量计算（`mul_loop/mul_tail/dst_rep_stride` 等仅依赖
  shape/dtype 的量）前移或缓存，减少每次调用的 host 侧开销

## 7. 模型级验证（1K in / 1K out + Profiler 证据）

### 7.1 性能（Qwen3.6-27B，TP4，eager，`--cases "1024,1024,16"`）

| 指标 | 数值 |
|---|---|
| 成功请求 | 16/16（0 失败） |
| Mean TTFT | 1481 ms（Median 1239 ms） |
| Mean TPOT | 303.5 ms（Median 302.3 ms） |
| 输出吞吐 | 3.28 tok/s |
| 总吞吐 | 6.56 tok/s |

### 7.2 Profiler 算子调用证据（32 tokens case，op_statistic.csv）

| 场景 | AddRmsNormBias 调用次数/rank | 单次平均耗时 | 单 rank 总耗时 |
|---|---|---|---|
| eager 开启 | **4096** | ~2.74 µs | ~13.0 ms |
| graph 开启 | **4096** | ~2.80 µs | ~13.5 ms |
| eager 关闭（`VLLM_FL_DISABLE_ASCENDC_RMSNORM=1`） | **0** | — | — |

- 4096 = 32 tokens × 128 层，与模型结构吻合；补丁日志（`Patched GemmaRMSNorm.forward_oot`）
  在开启场景出现 4 条、关闭场景 0 条 → **算子真实进入推理路径，回退开关真实有效**
- 无 residual 路径（final norm 等）走 CANN 内置 `GemmaRmsNorm`（32 次/请求），符合设计
- 证据文件：`/workspace/results/dyq/20260824_snapshot_current/evidence_{eager,fallback,graph}.csv`

### 7.3 35B 模型级验证（Qwen3.6-35B-A3B，TP4，eager，`--cases "1024,1024,16"`，9/8）

| 指标 | 数值 |
|---|---|
| 成功请求 | 16/16（0 失败） |
| Mean TTFT | 9087 ms（Median 9139 ms） |
| Mean TPOT | 229.4 ms（Median 229.4 ms） |
| Mean ITL | 229.2 ms |
| 输出吞吐 | 61.43 tok/s |
| 总吞吐 | 122.85 tok/s |
| 补丁 sanity | 4（`Patched GemmaRMSNorm.forward_oot`）→ 算子真实进入 35B 路径 |

- 证据：RUN_DIR `atp_qwen3.6-35b-a3b_fl_enforce_eager_chunked_dyq_35b_1k1k_on_0908_tp4_gmem0.6_20260908_083325/`
- 35B（MoE 结构、TP4 分片）单 token 时延（TPOT/ITL ~229 ms）与 27B 同口径验证处于同一量级，
  推理稳定、无设备错误 → **35B 公共路径验证完成**

### 7.4 融合收益：算子级 ON/OFF 对照（2026-09-17 补充计算）

同一 Profiler 窗口的 ON（eager）/ OFF（fallback）算子统计对照，可直接量化本算子的融合收益。

| 算子 | OFF 调用数 | ON 调用数 | 调用差 | OFF 总耗时 (ms) | ON 总耗时 (ms) | 差量 (ms) |
|---|---|---|---|---|---|---|
| `Pows` | 5664 | 1536 | −4128 | 35.99 | 5.20 | 30.78 |
| `Rsqrt` | 5664 | 1536 | −4128 | 7.43 | 2.16 | 5.27 |
| `Cast` | 14019 | 1635 | −12384 | 23.91 | 2.83 | 21.08 |
| `ReduceMean` | 5664 | 1536 | −4128 | 33.26 | 9.62 | 23.65 |
| `Mul` | 13440 | 5184 | −8256 | 34.74 | 22.44 | 12.29 |
| `Add` | 15248 | 6992 | −8256 | 26.71 | 12.55 | 14.16 |
| **`AddRmsNormBias`** | **0** | **4096** | +4096 | — | **12.99** | — |
| **合计** | | | | | | **107.23** |

- 原生路径把一次 RMSNorm 拆成 `Pows` / `ReduceMean` / `Rsqrt` / `Mul` / `Add` / `Cast`
  等多个 kernel 下发；**融合算子以 1 个 kernel 完成**
- 相关算子总耗时：**OFF 107.2 ms → ON 13.0 ms，该组件加速 8.3×**
- 以 `AddRmsNormBias` 的 ratio（0.632%）反推该窗口设备总时间 2055.9 ms
  → **节省约 5.22% 的设备时间**
- **归属说明**：`Mul` / `Add` / `Cast` 可能被模型其他模块共用，上式按 ON/OFF 差量归因，
  个别条目存在少量高估；`Pows` / `Rsqrt` / `ReduceMean` 在 OFF 为 5664 次、ON 降为 1536 次
  （未归零），差值已按实际差量计算
- 证据：`/workspace/results/dyq/20260824_snapshot_current/evidence_{eager,fallback}.csv`

**单次调用构成与优化上限**

| 项 | 数值 |
|---|---|
| `AddRmsNormBias` 平均单次耗时 | 3.172 µs |
| 空 kernel 启动固定开销（技能文档实测，1 核） | ~2.72 µs |
| → 真实计算部分 | **≈ 0.45 µs** |

kernel 时间中约 **86% 为启动固定开销**。即使将真实计算部分完全消除，也只能省
`0.45 µs × 4096 ≈ 1.8 ms`，占该窗口设备时间约 **0.09%**
→ **该算子自身已接近地板，无进一步优化空间**；本算子的价值主要来自融合（§7.4 表），而非单次执行效率。

### 7.5 host 侧开销定位（2026-09-17 补充实测）

针对"单次调用比原生多 ~25-30 µs"做了时间分解（bf16，卡 4，iters=300）：

| 项目 | custom | native (torch_npu.npu_add_rms_norm) | 差距 |
|---|---|---|---|
| host 派发（纯下发，不 sync） | 0.047-0.050 ms | 0.032 ms | **+15~17 µs** |
| kernel（Profiler 实测） | ~2.74 µs | 同量级 | ≈ 0 |

**差距 100% 在 host 派发**，且分解到 `EXEC_NPU_CMD` 的固定动作：
`InitHugeMemThreadLocal` → `ConvertTypes`（构造 7 个 aclTensor 描述符）→
`GetWorkspaceSize`（跑本算子的 tiling 回调）→ workspace 分配 → `opApiFunc` 下发 →
`ReleaseConvertTypes`（析构 7 个描述符）→ `UnInitHugeMemThreadLocal` → `OpCommand::Run()`。

**三次优化尝试均被实测证伪并回退（详见《算子版本记录》）**：

| 版本 | 假设 | 实测结果 | 处置 |
|---|---|---|---|
| v1 | tiling 热路径 4 条日志是瓶颈 | 比值无变化 | 回退（默认 ERROR 级别下日志被短路，格式化不执行） |
| v2 | 硬编码 16 MiB 系统 workspace 是浪费 | 官方 API 返回 `16777216`，**与硬编码值相同** | 保留为规范修复（改用官方 API、去掉虚设的 256 B 用户 workspace），**无性能收益** |
| v3 | tiling 计算本身可优化 | 见版本记录 | 插桩测量中 |

**结论**：该 host 开销是"自定义算子走 aclnn 两段式派发" vs "原生内置算子"的**结构性差异**，
换任何 kernel 实现都改变不了；且端到端上限约 **0.5%**
（16 µs × 35B 每步约 80 次调用 ÷ 每步约 250 ms），**不构成值得投入的优化方向**。

## 8. 限制与后续

- `op_kernel` 内核为项目基线 R5（见 §3 声明）；自写内核编译问题（vadd half）
  根因未定位，后续可对照 R5 的 `WholeReduceSum` 归约写法逐段对齐修复，替换后重测性能
- SPLIT_D 两趟式以 GM 重读换取正确性，极端大 D 下吞吐受限
- `rms_norm_ascend` 无残差路径仍走 `torch_npu.npu_rms_norm`（Gemma 1+weight 不适用普通 RMSNorm）
- ~~35B-A3B 模型未验证~~ → **已补：35B 1K/1K 模型级验证通过（2026-09-08，16/16，sanity=4，见 §7.3）**
- **单次调用存在固定 host 侧开销 ~25-30 µs**（§7.5 分解），端到端占比 < 1%；
  已定位为 aclnn 两段式派发的结构性开销（7 个 aclTensor 描述符构造/析构 + HugeMem
  + `OpCommand`），**非本算子可独立消除**；三次优化尝试（日志、workspace、tiling 插桩）
  均无收益或已被证伪，详见《算子版本记录》
- graph 组只跑了并发 64 一档，**未做并发梯度标定**：早期 27B 并发 32 的对照因与同事任务争卡、
  TTFT 异常偏高，与并发 64 组不可比；graph 模式的最佳并发尚未系统确定
- 本节点 8 卡由多个容器共享（容器 `--net=host`，端口亦共享），**压测需避开高负载时段**；
  同一配置在不同时段可相差 15-35%（§10.4 有实例），判读务必用背靠背对照而非跨时段比较

## 9. 引用与改动范围声明

- 接口契约、OpDef/infershape 语义、EXEC_NPU_CMD 用法：参考 R5，契约语义保持一致
- **op_kernel 内核：项目组基线 R5 版本（构建验证通过）**；op_host/tiling/torch_adpt/
  补丁/测试/bench：独立实现
- `AscendCGemmaRMSNorm.forward_oot` 调用约定（1+weight / npu_gemma_rms_norm）：参考 R8，
  本补丁只 patch 自己的算子并独立提供开关
- 未整体合并参考分支；改动遵循单算子、最小改动、可独立回退原则

## 10. 服务级基准（管理员 4-Baseline 验收口径）

**统一配置**：`/workspace/scripts/benchmark_script_fl.sh`（管理员统一基线脚本）；TP4、
`--cases "1024,1024,128"`（1024 in / 1024 out、128 请求）、并发 64、`--max-num-seqs 64`、
`--max-model-len 8192`、gmem 0.6、chunked prefill 开启；开关 `VLLM_FL_DISABLE_ASCENDC_RMSNORM=1`（OFF）。

**sanity 判据**：每轮以服务端日志 `Patched GemmaRMSNorm.forward_oot` 计数校验 ——
ON 应 4（TP4 四进程）、OFF 应 0；不符则该轮作废。**下表所有组均通过此判据**。

**模型结构（实测 config.json）**：

| 模型 | hidden_size | 层数 | heads / KV heads | 结构 |
|---|---|---|---|---|
| Qwen3.6-27B | 5120 | 64 | 24 / 4 | dense，intermediate 17408 |
| Qwen3.6-35B-A3B | 2048 | 40 | 16 / 2 | MoE，256 experts / 8 active，moe_intermediate 512 |

> 注：`tests/ops/ascend/test_add_rms_norm_bias.py` 中的 case 名（`model-like-hidden3584` /
> `model-like-hidden5120-beta`）为覆盖用合成 shape，非模型真实 hidden size，勿混淆。

### 10.1 4-Baseline 结果（ON/OFF 成对，全部 128/0 有效）

| 模型 | 模式 | 并发 | ON (tok/s) | OFF (tok/s) | ON 相对 OFF | sanity |
|---|---|---|---|---|---|---|
| 27B | eager | 64 | 234.75 | 213.22 | **+10.1%** | 4 / 0 |
| 27B | graph（PIECEWISE + capture 64） | 64 | 277.58 | 276.34 | **+0.4%** | 4 / 0 |
| 35B-A3B | eager | 64 | 235.23 | 226.54 | **+3.8%** | 4 / 0 |
| 35B-A3B | graph（PIECEWISE） | 64 | 246.41 | 247.67 | **−0.5%** | 4 / 0 |

四组 ON/OFF **均为背靠背多轮复测所得**（27B 多轮见 §10.2，27B graph 实证见 §10.3，
35B eager 与 graph 见 §10.4）；差异列 = (ON − OFF) / OFF；各组均 **128 成功 / 0 失败**，
且 sanity 判据全通过（ON=4 / OFF=0）。

**结论（2026-09-16 定稿）：四个 Baseline 上自定义 ON 均无性能回退 —— 实测持平或略优
（−0.5% ~ +10.1%；三组为正，一组在 ±1% 内持平）。**

### 10.2 27B eager：从"疑似回退"到"持平略优"（多轮复测）

**9/1 首次测得的数据（当时如实上报，后证实为节点负载噪声）**：

| 指标 | 自定义 ON | 自定义 OFF（昇腾内置 RMSNorm） | 差异 |
|---|---|---|---|
| 输出吞吐 (tok/s) | 160.94 | 216.63 | OFF 高 **+34.6%** |
| 总吞吐 (tok/s) | 321.88 | 433.27 | OFF 高 +34.6% |
| Mean TTFT (ms) | 23224 | 21037 | ON 慢 +10.4% |
| Mean TPOT (ms) | 322.88 | 273.38 | ON 慢 +18.1% |
| 成功 / 失败 | 128 / 0 | 128 / 0 | — |

- 结果原件：`/workspace/results/dyq/request_benchmark_results_27b_eager_op_on.txt`、
  `..._eager_op_off.txt`
- **9/4 背靠背复测（同配置，两轮相邻跑）**：ON 230.21 tok/s（TPOT 256.71ms / TTFT 20431ms /
  时长 569s）vs OFF 214.90 tok/s（TPOT 275.21ms / TTFT 21632ms / 时长 610s）→ **ON 反超
  +7.1%**，且 OFF 与 9/1 基本一致（-0.8%，基线稳定）→ **9/1 的 160.94 为节点负载噪声，
  R5 内核接入在服务级无性能回退**
- 首次结论（9/1 数据）当时如实上报：正确性无回退（128/0、无异常、开关可独立回退）；
  当时数据显示 R5 基线内核在 eager 服务场景吞吐低于对比对象约 25.7%（相对 OFF 口径）
- **如实结论（9/4+9/7 多轮复测后修正，最终）**：正确性无回退（128/0、无异常、开关可独立回退）。
  **多轮样本（均 128/0，同配置全服务压测）**：ON = {230.21, 227.19, 218.25, 234.75}（均值
  227.6，区间 218.3–234.8，波动小）；OFF = {214.90, 200.97, 218.31, 181.07, 213.22}（均值
  205.7，区间 181.1–218.3，波动较大；9/7 一轮 OFF=126.69 经验证为节点负载离群，已剔除）
  → **ON 均值高约 +10.7%（中位数 +7.3%），方向一致**；稳妥表述为"持平或略优（均值约高
  10%）"。9/1 的 ON=160.94 为节点负载异常值（OFF 跨天稳定）。本算子多产出 rstd/residual
  （功能增量，见 §6），内核为项目基线 R5、未做面向该场景的优化——内核级优化（自写
  内核）仍列为后续项，服务级基线以多轮复测为准。

### 10.3 graph 模式：从"疑似不可行"到可用配置（结论已修正）

**旧结论（9/3，已作废）**：曾判定"graph 基线在本节点不可行"，依据是 FULL 模式图捕获阶段
稳定 NPU OOM，且用不含本人改动的纯基线 A/B 复现同一故障。**该结论只对 FULL 模式成立**，
现已被下面的解法推翻，此处保留排查过程作为证据链。

**失败链路（FULL 模式，多次独立跑批）**

| 尝试（graph, FULL） | 结果 |
|---|---|
| c64 / gmem 0.6 | 65 成功 / 63 失败 |
| c32 / gmem 0.6 | 33 / 95 |
| c64 / gmem 0.6（复跑，9/2） | 64 / 64，Mean TTFT 235 s |
| 9/2 首启（op ON） | `Available KV cache memory: -12.36 GiB` → ValueError，脚本自动重启后预热 4/4 通过 |
| c64 / gmem 0.85（9/2，`dyq_graph85_on_0902`） | KV 初始化正常（可用 35.54 GiB，KV 582,144 tokens），**图捕获阶段 NPU OOM**（decode FULL 0/11，预留 57.32 GiB 时申请 1.11 GiB 失败） |
| c64 / gmem 0.75（9/3，`dyq_graph75_on_0902`） | KV 正常，图捕获 OOM（decode FULL 0/11，预留 57.97 GiB 时申请 0.94 GiB 失败） |
| c64 / gmem 0.7（9/3，`dyq_graph70_on_0903`） | KV 正常（可用 26.40 GiB，KV 432,128 tokens），图捕获 OOM（预留 57.4-58.3 GiB 时申请 846 MiB 失败） |
| c64 / gmem 0.4（9/3，`dyq_graph40_on_0903`） | 服务可启动但 **65 成功 / 63 失败**，Mean TTFT 28.2 s；中段 NPU 507011 设备异常后引擎退出 |
| **纯基线 A/B**（9/3，`add-qwen3_6_ascendc_gdn_ops` @ a3644b2，**不含本人任何改动**，`dyq_base_graph75_0903`） | 与本人分支**同点同数复现**：decode FULL 0/11 图捕获 OOM，预留 57.97 GiB 时申请 944 MiB 失败 → **证明失败与本人改动无关** |

**根因**：FULL 模式需一次性捕获全部解码图档位（失败日志为 `decode FULL 0/11`），其显存需求
（捕获期预留 55-58 GiB）超出本节点单卡可用余量；且**预留量与 gmem 档位几乎无关**
（由权重 12.87 GiB + torch.compile 工作区 + 图池主导），故调 gmem 无法解决。

**可用配置（9/9 起采用，已验证）**：
`--cudagraph-mode PIECEWISE` + `--max-cudagraph-capture-size 64`，配合 gmem 0.6、并发 64 →
27B、35B 的 graph ON/OFF 四组均一次跑通，**全部 128/0**。该配置下捕获档位为
`1/2/4/8/16/24/32/40/48/56/64` 共 11 档，日志显示捕获进度 `11/11` 完成、KV cache 20.30 GiB。

**27B graph c64 组实证补测（2026-09-11）**：原 ON 轮的服务端日志被同名 OFF 轮启动时覆盖写，
无法举证，故用完全相同配方重跑一轮：**128/0、sanity=4、吞吐 275.70 tok/s**
（时长 475.41 s、Median TTFT 9367.84 ms、Mean TPOT 211.37 ms），与 §10.1 表中 277.58
偏差 0.7%，**同点复现成立**；该轮 KV cache 20.30 GiB、PIECEWISE 捕获 11/11，与原轮一致。

**受限并发对照（27B graph，PIECEWISE，并发 32）**：ON 173.32 / OFF 206.48（均 128/0）。
该组 ON 轮受同事任务争卡影响 TTFT 异常偏高，与并发 64 组不具可比性，仅作过程记录。

### 10.4 35B（eager + graph）：背靠背 A/B/A/B 复测（修正 9/10 的"倒退"结论）

**9/10 原始数据（判为 ON 倒退，现确认为异常）**：

| 轮次 | 吞吐 (tok/s) | 时长 (s) | Mean TTFT (ms) | Mean TPOT (ms) |
|---|---|---|---|---|
| eager ON | 171.44 | 764.53 | 14421.97 | 240.71 |
| eager OFF | 195.57 | 670.19 | 22035.71 | 276.83 |
| graph ON | 190.49 | 688.09 | 21054.85 | 286.33 |
| graph OFF | 247.20 | 530.23 | 21443.27 | 236.99 |

当时判读为"ON 偏低 12.3% / 22.9%"，但该数据**内部自相矛盾**：吞吐显示 ON 慢，
而 Mean TTFT 与 Mean TPOT 却显示 ON **快** 7-13%（物理上不可能同时成立）；
且四轮并非背靠背（eager ON 08:26 → eager OFF 首轮失败 → graph 两轮 → 09:31 才补跑 eager OFF），
中间夹两轮 graph，节点状态不可比。

**复测方案（2026-09-15）**：35B eager **on1 → off1 → on2 → off2 背靠背四轮**，
同窗口、同节点状态、同一配置；卡 4-7，端口 8137（避开同事容器占用的 8113/8114）。

| 轮次 | 吞吐 (tok/s) | 时长 (s) | Median TTFT (ms) | Mean TPOT (ms) | 有效请求 | sanity |
|---|---|---|---|---|---|---|
| on1 | 234.31 | 559.39 | 8468.47 | 250.75 | 128/0 | 4 |
| off1 | 227.26 | 576.76 | 8477.76 | 259.03 | 128/0 | 0 |
| on2 | 236.15 | 555.03 | 8523.43 | 247.94 | 128/0 | 4 |
| off2 | 225.81 | 580.46 | 8545.49 | 260.44 | 128/0 | 0 |
| **ON 均值** | **235.23** | **557.21** | 8495.95 | **249.35** | — | — |
| **OFF 均值** | **226.54** | **578.61** | 8511.63 | **259.74** | — | — |
| **差异** | **+3.8%** | **−3.7%** | 打平（0.2%） | **−4.0%** | — | — |

- 四个指标**方向完全一致、无自相矛盾**：吞吐 ON 高 3.8%、TPOT ON 快 4.0%、时长 ON 短 3.7%、
  Median TTFT 几乎相同（差异 0.2%）→ **对照组干净**
- 两轮 ON 之间波动仅 0.8%（234.31 / 236.15），重复性良好
- 证据：`/workspace/results/dyq/20260915_rmsnorm_perf_debug/ab35b_eager_summary.txt`
  （含 `env_info.txt`：提交 `e016bdd`、CANN/driver 版本、npu-smi、完整命令）

**修正后的结论**：**9/10 的"ON 倒退"是节点争卡的假象，非算子问题。** 三重证据：
1. 同条件背靠背复测中 ON 稳定领先约 4%；
2. 9/10 的 OFF（195.57）比 9/15 的 OFF（226.54）低 14%，而 ON 低 37% ——
   当时整段时间窗被外部负载压制，**ON 轮被压得更狠**（时长多 205 s）；
3. 旧数据内部自相矛盾（见上）。
4. 方向与 27B（ON +10.1%）一致。

#### 10.4.1 graph 组复测（2026-09-15，同方案）

graph 组复测在卡空闲后由守候脚本自动执行（首轮因卡 4-7 被其他容器占用而失败，已重试）。
`--mode graph --cudagraph-mode PIECEWISE`，其余配置同上。

| 轮次 | 吞吐 (tok/s) | 时长 (s) | Median TTFT (ms) | Mean TPOT (ms) | 有效请求 | sanity |
|---|---|---|---|---|---|---|
| on1 | 246.29 | 532.20 | 8448.01 | 237.54 | 128/0 | 4 |
| off1 | 248.02 | 528.48 | 8353.78 | 236.45 | 128/0 | 0 |
| on2 | 246.53 | 531.66 | 8492.55 | 237.69 | 128/0 | 4 |
| off2 | 247.32 | 529.96 | 8254.25 | 236.47 | 128/0 | 0 |
| **ON 均值** | **246.41** | 531.93 | 8470.28 | 237.62 | — | — |
| **OFF 均值** | **247.67** | 529.22 | 8304.02 | 236.46 | — | — |
| **差异** | **−0.51%** | +0.5% | +2.0% | +0.5% | — | — |

- **ON 与 OFF 持平（−0.5%）**，四个指标彼此一致；两轮 ON 间波动 0.1%、OFF 间 0.3%，重复性极好
- **交叉验证 9/10 的污染方向**：9/10 的 graph **OFF=247.20 与本次 OFF=247.67 几乎重合（+0.2%）**，
  而 9/10 的 **ON=190.49 比本次 ON=246.41 低 29%** → **9/10 被压制的只是 ON 那一轮**，
  这与 eager 组的污染模式完全一致，构成独立佐证
- 证据：`/workspace/results/dyq/20260915_rmsnorm_perf_debug/ab35b_graph_summary.txt`

**35B 两项最终结论**：eager +3.8%、graph −0.5%，**均无性能回退**；9/10 判定的 −12.3% / −22.9%
**全部归因为同时间窗的节点争卡**。

### 10.5 复现命令与证据路径

```bash
# 环境（非交互 shell 必须显式加载，否则 libhccl.so / python 缺失）
source /etc/profile && source /root/.bashrc
export PATH=/usr/local/python3.11.14/bin:$PATH

# 单组（ON）
bash /workspace/scripts/benchmark_script_fl.sh \
  --model-path /models/Qwen3.6-35B-A3B --model-name qwen3.6 --model-tag qwen3.6-35b-a3b \
  --cases 1024,1024,128 --concurrency 64 --max-num-seqs 64 --max-model-len 8192 \
  --tp 4 --gmem 0.6 --devices 4,5,6,7 --port 8137 \
  --no-bench-profile --skip-analyse --package none \
  --mode eager --run-label dyq_ab35bv3_on1
# OFF 组：前置 VLLM_FL_DISABLE_ASCENDC_RMSNORM=1
# graph 组：--mode graph --cudagraph-mode PIECEWISE（并加 --max-cudagraph-capture-size 64）
```

- 复测脚本：`/workspace/scripts/dyq/ab35b_eager_0915_v3.sh`（A/B/A/B，含前置自检与失败重试）
- 结果目录：`/workspace/results/dyq/20260915_rmsnorm_perf_debug/`
- 27B eager 原始件：`/workspace/results/dyq/request_benchmark_results_27b_eager_op_{on,off}.txt`
- 环境注意事项：容器为 `--net=host`，**端口与宿主机及其他容器的容器共享**，压测前须确认端口空闲；
  8 卡由多容器共用，压测前须查 `npu-smi info` 选取空闲卡

