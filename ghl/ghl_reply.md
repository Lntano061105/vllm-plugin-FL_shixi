# Fused GDN Gating 算子接入 — 答辩材料

> 汇报人：龚昊磊（容器用户名：ghl）
> 项目：vLLM-Plugin-FL 自定义算子接入、编译部署与模型推理验证（6 周实习）
> 负责算子：Fused GDN Gating（`npu_fused_gdn_gating`）｜ 目标模型：Qwen3.6 27B / 35B（GDN 线性注意力层）
> 配套文档：`ghl/ghl_report.md`（技术报告，2026-09-11 定稿）
> 日期：2026-09-11

---

## 0. 答辩速览（开场 30 秒）

Fused GDN Gating 算子已在 vLLM-Plugin-FL 完成**完整接入链路**：`csrc` 源码 → CANN/AscendC 编译打包 → torch 扩展 schema 注册 → Qwen3.6 GDN 推理路径，并带**可回退开关**与 **Profiler 调用证据**。算子级正确性 **25/25 通过**，性能较 vLLM Triton 基线**快 2.3–3.8 倍**且不随 token 数增长；模型级 1K 输入/1K 输出双路径（AscendC/Triton）验证完成。遗留项为 27B 端到端差异定位与 PR 提交（均按项目组安排处理）。

- 建议时长：12–15 分钟汇报 + 3–5 分钟提问。
- 一句话原则：**只用可核验的数据说话**（测试通过数、Profiler 计数、实测时延/吞吐），推测之处明确标注"待定位"。

## 1. 评分点与汇报结构对齐

| 任务书评分点 | 权重 | 对应提纲页 | 核心证据 |
|---|---|---|---|
| 工程接入 | 30% | P3–P6 | csrc 14 文件 + schema 注册 + build_aclnn 打包 + 新 Shell 加载闭环 |
| 正确性与测试 | 20% | P7 | 算子级 25/25（vs PyTorch / vs Triton / 8 个异常分支） |
| 框架集成与性能 | 20% | P8–P11 | patch 挂钩 + 回退开关 + Profiler 49152 次/0.448% + Microbenchmark + 模型级 |
| 代码质量、文档与答辩 | 30% | P2、P12 | 最小改动集合、参考来源如实注明、技术报告 + 本材料 |

## 2. 汇报提纲（逐页要点）

### P1 封面 / 任务概述
- 任务：按任务书完成 1 个自定义算子的"接入—编译—调用—测试—性能—汇报"闭环。
- 本人负责 Fused GDN Gating（Qwen3.6 GDN 层门控），原则"单算子、最小改动、可独立回退"。

### P2 六周计划完成度
- W1 调用链定位 → W2 源码接入 → W3 编译部署 + 算子级固定测试 → W4 框架接入 + 模型级 1K/1K → W5 Microbenchmark + Profiler → W6 技术报告 + 答辩。
- 完成度：工程链路、测试、性能、文档全部完成；PR 提交按项目组要求暂缓（基线/方式待确认）。

### P3 算子定位与调用链
- 展示 GDN 层前向图：`conv1d → fused_gdn_gating(本算子) → recurrent/chunk gated delta rule → RMSNormGated → out_proj`。
- 调用链：`Qwen3NextGatedDeltaNet.forward → torch.ops.vllm.gdn_attention_core → _forward_core → torch.ops._C_ascend.npu_fused_gdn_gating → aclnnFusedGdnGating → AscendC kernel`。
- 一句话定位：算子在 conv1d 之后、linear attention 状态递推之前，负责每头的衰减门 `g` 与写入门 `beta`。

### P4 算子语义与接口
- 公式：`x = a + dt_bias`；`softplus = (1/β)·log(1+exp(βx))`（越阈退化为 `x`）；`g = -exp(A_log)·softplus`；`beta = sigmoid(b)`。
- 接口表（重点讲清 dtype/布局）：
  - 输入 `A_log (H,) fp32`、`a (B,H) bf16/fp16`、`b (B,H) 同 a`、`dt_bias (H,) fp32`；
  - 输出 `g (1,B,H) fp32`、`beta_output (1,B,H) 与 b 同 dtype`；默认 `β=1.0`、`threshold=20.0`。
- 关键点：输出布局与 vLLM Triton 基线**逐位一致**，故可透明替换；`threshold` 防 `exp` 溢出。

### P5 工程接入与最小改动集合
- `csrc/ascend/attention/fused_gdn_gating/`：14 个文件（op_host Tiling/InferShape/OpDef + op_api + op_kernel AscendC + torch_adpt）。
- schema 注册：`csrc/ascend/torch_binding.cpp:2687-2696`；构建清单：`build_aclnn.sh:31`。
- 框架接入：`patches/patch_qwen3_6_gdn.py`（本地裁剪 577 行 vs R8 907 行），仅保留核心接入。
- 参考来源如实注明：算子源码与 **R4**（commit ab33933）逐文件 diff 完全一致；**R8** 补丁为裁剪版，未整体复制实验分支。

### P6 编译部署闭环
- 三步：`setup.py build_ext --inplace`（torch 扩展）→ `cd csrc/ascend && bash build_aclnn.sh ascend910b`（打包 `.run` ≈21MB）→ 安装到 `vllm_fl/_cann_ops_custom/vendors/custom_transformer/`。
- 部署验证要点：`set_env.bash` 只导出 `ASCEND_CUSTOM_OPP_PATH` 与 `LD_LIBRARY_PATH`，必须在 Python 启动前 source；**重新打开 Shell → source → 运行**仍可加载，证明不依赖手工复制 `.so`。
- 运行时 `vllm_fl/__init__.py::_bootstrap_cann_custom_op_env` 自动自举，默认免手工 source。

### P7 正确性验证（算子级 25/25）
- 固定条件：`NUM_HEADS=32`、token `[1,4,16,64]`、dtype `bf16/fp16`、`rtol=atol=1e-2`。
- 三类用例：8 项 vs PyTorch 参考、8 项 vs vLLM Triton 基线、1 项 softplus 越阈退化（`a=50`）、8 项 `TORCH_CHECK` 异常分支（维度/dtype/shape/num_heads）。
- 连通性测试 `tests/custom_ops_tests/test_fused_gdn_gating.py` 通过。

### P8 框架接入与回退开关
- patch 将 `AscendCGatedDeltaNet._forward_core` 挂钩到 `npu_fused_gdn_gating`，进入 Qwen 推理路径。
- 回退：`VLLM_FL_DISABLE_ASCENDC_GDN=1` → `_ascendc_ops_available()` 返回 False，保持原生 Triton 路径（开关为 1 / `_C_ascend` 不可导入 / 5 个必需 op 缺失均回退）。
- 计数钩子：`VLLM_FL_GDN_COUNT_FILE=<path>`（默认关闭、零开销），每 1024 次追加计数，与 Profiler 交叉核对。

### P9 Profiler 证据
- `op_statistic.csv`（27B AscendC run）：`FusedGdnGating`（AI_VECTOR_CORE）**Count=49152 = 48 GDN 层 × 1024 decode 步**，Total 249.6ms，device 占比 **0.448%**；与计数钩子及理论值一致。
- 配套 GDN 算子均在列：`RecurrentGatedDeltaRule` 2.005%、`CausalConv1d` 0.893%、`chunk_gated_delta_rule_fwd_kernel` 0.048%。

### P10 性能：算子级 Microbenchmark
- 36 项配置（3 后端 × token 1/4/16/64/256/1024 × bf16/fp16）全部通过正确性校验。
- AscendC 平均时延 ~85–100µs，**不随 token 数增长**；Triton 基线 ~195–370µs、随 token 增长。
- 比值 **0.26x–0.44x（AscendC 快 2.3–3.8 倍）**，tokens=1024 时最快。

### P11 性能：模型级 1K/1K（eager + chunked）
| 配置 | Run 指标 | AscendC | Triton |
|---|---|---|---|
| 27B TP1 gmem0.9 | 生成吞吐 / TPOT | ~4.3 tok/s* / 未测得* | 6.10 tok/s / 163.15ms |
| 35B TP2 gmem0.7 | 生成吞吐 / TPOT | 5.68 tok/s / 175.09ms | 5.29 tok/s / 187.98ms |

> *27B AscendC 为 profiler run，导出阶段卡住（trace_view.json 15.5GB、解析 11m43s），TTFT/TPOT 未打印，吞吐从引擎日志提取并如实标注。

- 结论：35B 模型级与算子级方向一致（AscendC 略快 ~7%）；27B 模型级慢于 Triton、与算子级方向相反，**待补无 profiler 对照 + 重复数据**；疑似 decode 路径 `recurrent_gated_delta_rule` / state 布局转换开销（待用 `api_statistic.csv`/`trace_view.json` 深挖）。

### P12 限制、遗留与总结
- 遗留：① PR 未提交（基线/方式待项目组确认）；② 27B 模型级差异定位；③ 27B AscendC 无 profiler 对照；④ TP3 对 35B 结构性不可行。
- 总结：链路完整、测试充分、算子级性能达标；下一步聚焦模型级差异定位与提交。

## 3. 关键数据速查（提问时可立即引用）

| 项 | 数值 |
|---|---|
| 算子级测试 | 25/25 通过（8 vs PyTorch + 8 vs Triton + 1 越阈 + 8 异常） |
| 连通性测试 | 通过 |
| 算子级时延 | AscendC ~85–100µs；Triton ~195–370µs；比值 0.26x–0.44x |
| Profiler 计数 | FusedGdnGating 49152 次（=48×1024），占比 0.448% |
| GDN 全路径 device 占比 | FusedGdnGating 0.448% + RecurrentGatedDeltaRule 2.005% + CausalConv1d 0.893% + chunk 0.048% ≈ 3.7% |
| 27B 模型级 | AscendC ~4.3 tok/s（profiler run）vs Triton 6.10 tok/s |
| 35B 模型级 | AscendC 5.68 tok/s / TPOT 175.09ms vs Triton 5.29 tok/s / 187.98ms |
| 算子源码文件数 | 14（与 R4 ab33933 逐文件一致） |
| 本地 patch 行数 | 577 行（R8 907 行，裁剪掉 PTO megakernel 等实验路径） |
| 产物包 | `cann-ops-transformer-custom_linux-aarch64.run` ≈21MB |
| 环境 | 910B3×8、torch 2.8.0 + torch_npu 2.8.0.post2、vllm 0.13.0、CANN 8.5/9.0、flag_gems 5.0.2 |

## 4. 现场演示 / 复现预案（如需实操）

```bash
# 0) 部署环境（新 Shell 内）
source /workspace/vllm-plugin-FL/vllm_fl/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash

# 1) 连通性 + 算子级固定测试（25 项）
python tests/custom_ops_tests/test_fused_gdn_gating.py
pytest tests/ops/ascend/test_fused_gdn_gating.py -m gpu

# 2) 算子级 Microbenchmark（AscendC vs Triton vs PyTorch）
python /workspace/scripts/ghl/benchmark_fused_gdn_gating.py --backends ascendc,triton,torch \
    --tokens 1,4,16,64,256,1024 --dtypes bf16,fp16 --reps 500 --device npu:0

# 3) 模型级 1K/1K（AscendC 默认；Triton 对照前加 export VLLM_FL_DISABLE_ASCENDC_GDN=1）
/workspace/scripts/run_vllm_fl_profile_unified.sh \
    --model-path /models/Qwen3.6-27B --model-tag qwen3.6-27b \
    --mode eager --chunked true --devices 4 --tp 1 --gmem 0.9 --max-model-len 4096 \
    --cases "1024,1024,1" --bench-profile false --skip-analyse --run-label <ascendc|triton>_r1
```

- 备选：若现场不便跑模型，直接展示 `request_benchmark_results.txt` 结果块与 `op_statistic.csv` 片段。
- 兜底：提前确认卡是否空闲（`npu-smi info`），模型级步骤可跳过、以既有数据讲解。

## 5. 预判问答（Q&A）

**Q1：为什么算子级 AscendC 明显快，模型级 27B 反而慢？**
A：两者口径不同。算子级是单算子微基准（预热+同步+重复，测 kernel 时延），AscendC 1 次 AIV 调用、不随 token 增长，确实快 2.3–3.8 倍；模型级是端到端 1K/1K，含调度、跨卡通信与其余 GDN 算子。27B 模型级 AscendC run 是 **profiler run**（带采集开销、且 TTFT/TPOT 未打印），与 `--bench-profile false` 的 Triton 对照不是严格 A/B。更关键的是 profiler 显示 27B 的 GDN 全路径占比仅约 3.7%，单算子收益被端到端摊薄；差异疑似出在 decode 路径 `recurrent_gated_delta_rule` 与 state 布局转换。结论我按"待补无 profiler 对照 + 重复数据"如实记录，没有下确定性结论。

**Q2：为什么 `g` 输出是 fp32，而 `beta_output` 跟 `b` 同 dtype？**
A：`g` 直接参与后续指数/衰减累积，fp32 保精度、避免 bf16/fp16 下衰减量级失真；`beta` 是 sigmoid 结果，本就落在 (0,1)，与 `b` 同 dtype 足够且与 vLLM Triton 基线布局一致，保证可透明替换。

**Q3：`threshold=20.0` 是做什么的？**
A：softplus 在 `βx` 很大时 `exp` 会溢出。设定阈值后，`βx > threshold` 时直接退化为线性 `x`（数学上 softplus 在大值处≈x），既防溢出又保持数值一致；测试里有专门用例（`a=50` 时校验 `g == -exp(A_log)*x`）。

**Q4：怎么证明自定义算子真的进了 Qwen 推理路径，而不是被回退？**
A：三重证据——① patch 生效日志（`Patched Qwen3NextGatedDeltaNet ... fused_gdn_gating ...`）；② Profiler `op_statistic.csv` 中 `FusedGdnGating` 计数 49152 = 48 层 × 1024 decode 步，精确对应；③ 自定义计数钩子 `VLLM_FL_GDN_COUNT_FILE` 与 Profiler 次数交叉一致。

**Q5：回退开关怎么验证有效？**
A：默认 `patch_qwen3_6_gdn()` 返回 True 走 AscendC；设 `VLLM_FL_DISABLE_ASCENDC_GDN=1` 时 `_ascendc_ops_available()` 返回 False、保持 Triton 路径，日志输出 `VLLM_FL_DISABLE_ASCENDC_GDN=1, keep Triton GDN path`，可通过结果与 Profiler 中算子名验证。

**Q6：算子源码与参考分支 R4 逐文件一致，是否算"整体复制"？**
A：任务书要求是"理解链路、按最小改动接入、不得整体复制实验分支"。处理方式是：算子本体（14 文件）与 R4 逐文件一致并在报告中**如实注明来源**；而 Python 接入补丁只做**本地裁剪版**（577 行 vs R8 907 行），删除 PTO megakernel、fused Triton decode kernel、conv1d 权重缓存等实验路径，只保留 AscendC 核心接入，符合"单算子、最小改动、可独立回退"。

**Q7：为什么 35B 不能用 TP3？**
A：GDN 的 conv 状态维度是 8192，8192 % 3 ≠ 0，vLLM `ensure_divisibility` 断言会失败，属结构性不可行，所以 35B 固定用 TP2。

**Q8：模型级为什么只测 1G 输入/1K 输出单请求？**
A：这是任务书规定的模型级验证口径（`--cases "1024,1024,1"`），用于确认算子在真实推理路径下可用并取 TTFT/TPOT/吞吐；并发/多 case 的共享基线对比在 Baseline 工作中单独进行，不与单请求数据混用。

**Q9：怎么证明"新 Shell 后仍能加载"（不依赖手工复制 .so）？**
A：`build_aclnn.sh` 打包成 `.run` 并安装到 `vllm_fl/_cann_ops_custom/vendors/custom_transformer/`；关闭并重开 Shell 后仅 `source set_env.bash`（只设 `ASCEND_CUSTOM_OPP_PATH`/`LD_LIBRARY_PATH`），连通性测试仍通过。关键在 `LD_LIBRARY_PATH` 必须在 Python 启动前生效——glibc 只缓存启动时的库搜索路径。

**Q10：环境上踩过哪些坑？**
A：① flag_gems 5.0.2 的 pow 算子在加载时报 `TypeError: cannot convert None to tensor`，统一脚本用 `--flaggems-ops` 白名单规避；② 27B AscendC profiler 导出卡住（trace_view.json 15.5GB、CANN 解析 11m43s），性能对照改用 `--bench-profile false --skip-analyse`；③ 并发跑批偶发瞬态 aivec 错误（日志无对应错误、不可复现），改串行跑批规避；④ 27B graph 在本环境 OOM（见 Q11）。

**Q11：27B graph 基线为什么没跑出数据？**
A：按共享基线脚本在本环境（TP4/gmem0.6/max-model-len 8192/128 并发）复测，FULL 1 次与 PIECEWISE 9 次全部 OOM（`Tried to allocate 650 MiB`，卡上已分配 56.18 GiB、仅剩 ~25 MiB），其中 PIECEWISE 5 次是 capture 通过后在 warmup 首次推理阶段 OOM。同环境 27b_eager 可 128/128。结论如实标注为"本环境不可复现"，要取数需改 capture 尺寸/max-num-seqs/max-model-len 等配置并声明偏离官方口径。

**Q12：PR 为什么没提交？**
A：最小改动集合与提交说明已整理（与 main 的 merge-base 906fa07），但提交基线与 PR 方式需项目组确认，按指示先不提交，避免在基线未定时产生合并风险。

**Q13：微基准数据可信吗？**
A：脚本先做正确性校验（与 PyTorch 参考对比，rtol/atol=1e-2），再做 100 次预热 + 500 次重复测量并同步，报告 min/mean/P50/P90/P99/max；36 项配置全部通过校验后才进入性能统计，原始结果留存 `fused_gdn_gating_microbench_20260820.txt`。

**Q14：dtype 支持范围？**
A：输入支持 bf16 / fp16（`a`、`b` 同 dtype；`A_log`、`dt_bias` 为 fp32），kernel 内按 tiling key 分派 6 种模板实例（输入 dtype × 输出 dtype 组合），AIV 核执行。

## 6. 风险与兜底

| 风险 | 兜底说法 |
|---|---|
| 现场想跑模型但卡被占用 | 先用 `npu-smi info` 确认；不可用则展示已存结果块与 Profiler 片段讲解 |
| 被追问 27B 端到端差异 | 说明口径差异 + profiler run 限制，给出"待补无 profiler 对照与重复数据"的下一步，不过度推断 |
| 被问 PR 进度 | 说明基线/方式待项目组确认，最小改动集合已就绪 |
| 被问未完成项 | 主动在 P12 列出遗留，避免被动 |
