# 周报汇总（Fused GDN Gating 实习，week1~week4）

> 人员：龚昊磊（容器用户名：ghl）
> 合并自 `week1.md` / `week2.md` / `week3.md`（2026-08-25），原文件已删除；第 4 周报告于 2026-09-01 续写。
> 路径说明：本文档位于仓库 `ghl/` 子目录；文中仓库相对路径相对仓库根 `/workspace/vllm-plugin-FL/`（即加 `../` 前缀）。

---

## 第 1 周（原 week1.md）

人员：龚昊磊（容器用户名：ghl）

本周小结：
本周负责 Fused GDN Gating 算子（27B/35B GDN 路径）的第一阶段工作，目标是定位算子在 Qwen3.6 推理计算图中的位置，理解输入输出数据类型与数据布局，读懂参考实现并确认环境与构建链路，同时完成项目组 Profiling 统一脚本（`/workspace/scripts/` 下 3 个脚本）与手动分步 profiling 流程（`手动profiling测试.md`）的就绪检查与对照核验，为后续模型级验证（统一脚本一键跑 + 手动分步排查双路径）铺路。

已完成工作：
1. 调用链梳理：确认自顶向下调用路径为 Qwen3NextGatedDeltaNet.forward → torch.ops.vllm.gdn_attention_core → _forward_core → torch.ops._C_ascend.npu_fused_gdn_gating（torch schema 注册于 csrc/ascend/torch_binding.cpp:2688）→ aclnnFusedGdnGating（op_api）→ AscendC kernel（op_host 负责 Tiling/InferShape，op_kernel 在 AI Core 执行）；算子位于 conv1d 之后、recurrent/chunk gated delta rule 之前。
2. 算子接口确认：输入 A_log/dt_bias 为 (num_heads,) fp32，a/b 为 (batch, num_heads) bf16/fp16；输出 g 为 (1, batch, num_heads) fp32，beta_output 为 (1, batch, num_heads) 且与 b 同 dtype；语义为 g = -exp(A_log) * softplus(a + dt_bias)，beta = sigmoid(b)，输出布局与 vLLM Triton 基线一致。
3. 环境确认：Ascend 910B3 × 8（本机），CANN 8.5/9.0，torch_npu 2.8.0.post2，vllm 0.13.0；vllm_fl/_C_ascend*.so、libvllm_fl_kernels.so 及 CANN 自定义算子包（vllm_fl/_cann_ops_custom/vendors/custom_transformer/）均已就位。
4. 验证结果：连通性测试 tests/custom_ops_tests/test_fused_gdn_gating.py 通过；算子级固定测试 tests/ops/ascend/test_fused_gdn_gating.py 共 17 项全部通过（固定 head 数 32，token 数 1/4/16/64，bf16/fp16，分别与 PyTorch 参考实现和 vLLM Triton 基线对比，rtol/atol=1e-2，含 softplus 越阈退化路径用例）；回退开关验证通过——默认 patch_qwen3_6_gdn() 返回 True 走 AscendC 路径，设置 VLLM_FL_DISABLE_ASCENDC_GDN=1 时返回 False 保持 Triton 路径，均有日志输出可核验。
5. Profiling 统一脚本就绪检查（本周新增）：确认 `/workspace/scripts/` 下 `run_vllm_ascend_profile_unified.sh`（原生 vllm-ascend）/ `run_vllm_fl_profile_unified.sh`（vllm-plugin-FL，内置 source CANN 环境）/ `package_op_statistic.sh`（二次打包算子统计）3 个脚本可用；`--help` 逐项核对参数（`--cases "I,O,NP"`、`--bench-profile`、`--tp`/`--gmem`/`--devices`/`--flaggems-ops` 等）；确认结果目录规范（`/workspace/results/atp_{model_tag}_fl_{mode}_{chunk}_{label}_tp{tp}_gmem{gmem}_{ts}/`、`latest_run_dir.txt`、`ASCEND_PROFILER_OUTPUT/` 产物清单），并确认 TP 约束：35B 因 GDN conv 状态维度 8192 不能被 3 整除，TP3 结构性不可行（取 TP2）。
6. 手动分步 profiling 流程核验（本周新增）：阅读 `手动profiling测试.md`，确认其与统一脚本内部流程等价（serve 参数/warmup/case 执行/`.current_case_dir` 归档/analyse 为统一脚本各步骤的手动展开），并记录差异点：手动版额外带 `--no-async-scheduling`、结果目录用 `/workspace/new_results/`（非默认 `/workspace/results/`）、`ASCEND_RT_VISIBLE_DEVICES` 先导出 0 后覆盖为 4,5,6,7（最终生效 TP4）、示例模型为 Qwen3-0.6B（小模型快速冒烟）；结论：统一脚本一键失败时可退到手动分步定位故障环节。

下周计划：
在干净状态下复现完整构建部署闭环（cd csrc/ascend && bash build_aclnn.sh ascend910b 打包 → .run 安装 → source set_env.bash → 重新打开 Shell 加载验证）；核对 csrc/ascend/attention/fused_gdn_gating 源码与参考分支 R4/R8 的关系并整理为本人的最小改动提交；补充异常用例（shape/dtype 校验）完善算子级测试；用统一脚本完成模型级 1K 输入、1K 输出验证的前置接入（先以 Qwen3-0.6B 小模型按 `手动profiling测试.md` 手动分步冒烟 + 统一脚本 dry-run，确认 `VLLM_FL_DISABLE_ASCENDC_GDN` 环境变量透传以支持 AscendC/Triton 双路径）。

所遇问题：
暂无

---

## 第 2 周（原 week2.md）

人员：龚昊磊（容器用户名：ghl）

本周小结：
本周按照上周计划继续负责 Fused GDN Gating 算子（27B/35B GDN 路径）的接入工作，完成干净环境下的构建部署闭环复现、参考分支差异核对、异常用例补充，并完成模型级 1K/1K 验证的统一脚本接入（dry-run 与双路径开关确认，脚本就绪待权重）。

已完成工作：
1. 构建部署闭环复现：在干净状态下执行 csrc/ascend/build_aclnn.sh ascend910b，全量重建并打包 CANN framework 算子（含 fused_gdn_gating），生成自解压包 csrc/ascend/build/cann-ops-transformer-custom_linux-aarch64.run（约 21MB）并安装到 vllm_fl/_cann_ops_custom/vendors/custom_transformer/（与系统 CANN 隔离）；重新打开 Shell 后仅需 source set_env.bash 即可加载，连通性测试 tests/custom_ops_tests/test_fused_gdn_gating.py 通过，确认部署不依赖手工复制 .so。
2. 参考分支核对：将本仓库 fused_gdn_gating 接入内容与参考分支 appleinsky/qwen36_dense_moe（R4/R8）逐文件对比——csrc/ascend/attention/fused_gdn_gating/ 下算子源码文件与参考分支完全一致（来源 R4，报告中注明）；框架补丁 patch_qwen3_6_gdn.py 为本地裁剪版，删除了参考分支的实验性路径（PTO megakernel、fused Triton decode kernel、conv1d 权重缓存等），仅保留 AscendC 核心接入（causal_conv1d / fused_gdn_gating / recurrent_gated_delta_rule / gemma_rms_norm），符合"单算子、最小改动、可独立回退"原则；tests/ops/ascend/test_fused_gdn_gating.py 为本地新增。
3. 异常用例补充：在 tests/ops/ascend/test_fused_gdn_gating.py 新增 TestFusedGdnGatingValidation 共 8 个用例，覆盖 Torch Adapter 全部 TORCH_CHECK 校验分支（A_log/dt_bias 维度、a/b 维度、a/b shape 匹配、a/b dtype 一致、A_log/dt_bias dtype 一致、num_heads 匹配）；算子级测试合计 25/25 通过（固定 head 数 32、token 数 1/4/16/64、bf16/fp16，与 PyTorch 参考和 vLLM Triton 基线对比，含越阈路径与异常路径）。
4. 模型级验证统一脚本接入（本周调整）：放弃自研服务化压测脚本，改为接入项目组 Profiling 统一脚本 `run_vllm_fl_profile_unified.sh`（`/workspace/scripts/`）——完成 `--help` 参数核对与脚本实现核对（内置 source CANN 环境、`--flaggems-ops` 白名单、`--cases "1024,1024,1"` 对应任务书 1K 输入/1K 输出要求、`--bench-profile true` 时 profiler 产物 `ASCEND_PROFILER_OUTPUT/op_statistic.csv` 即为任务书要求的"Profiler 证明算子实际调用"证据）；基于实现确认 `VLLM_FL_DISABLE_ASCENDC_GDN=1` 环境变量可透传给脚本内 `vllm serve` 子进程（AscendC/Triton 双路径可直接复用同一脚本）；确认 TP 约束（35B 取 TP2 规避 GDN conv 状态维度 8192 不可被 3 整除）；核验 `手动profiling测试.md` 与统一脚本内部流程等价（见 week1_save.md §5.5），确定故障排查路径：统一脚本一键失败时退到 0.6B 小模型手动分步冒烟（Qwen3-0.6B，`/workspace/new_results/` 结果目录，含 `--no-async-scheduling` 差异项）。注：正式 1K/1K 运行已于 2026-08-25 执行完成（统一脚本，27B/35B 双路径，见 week3.md）；0.6B 冒烟仍为备用排查路径（未执行）。

下周计划：
整理 fused_gdn_gating 单算子最小改动提交（与项目组确认提交基线与 PR 方式）；补齐 27B/35B 模型权重后，用统一脚本执行模型级 1K 输入、1K 输出验证（AscendC/Triton 双路径：`--cases "1024,1024,1" --bench-profile true`），从 request_benchmark_results.txt 取 TTFT/TPOT/吞吐、从 op_statistic.csv 取算子调用证据，并用 package_op_statistic.sh 二次打包；按 W3 目标推进算子级测试交付物。

所遇问题：
Qwen3.6-27B/35B 模型权重不在本机（本地缓存仅有 Qwen3-4B），模型级验证暂时无法执行，需要下载或提供模型路径后按计划推进；其余暂无可复现问题。

---

## 第 3 周（原 week3.md）

人员：龚昊磊（容器用户名：ghl）

本周小结：
本周按上周计划完成三项任务：整理 fused_gdn_gating 单算子最小改动提交（基线确认与改动范围圈定）、在 27B/35B 模型权重就位后执行模型级 1K 输入/1K 输出验证（改用项目组 Profiling 统一脚本，AscendC/Triton 双路径，以 Profiler 产物 `op_statistic.csv` 作为算子实际调用证据）、推进 W3 算子级测试交付物（新增算子级 Microbenchmark，含平均时延与 P50/P90）。

已完成工作：
1. 单算子最小改动整理：确认提交基线为项目组 main 分支（本分支与 main 的 merge-base 为 906fa07）；csrc/ascend/attention/fused_gdn_gating/ 下 14 个算子源码文件与参考分支 R4（appleinsky/qwen36_dense_moe，commit ab33933）逐文件 diff 完全一致（来源已在报告中注明）；框架接入 patch 为本地裁剪版（577 行 vs 参考 R8 的 907 行，删除 PTO megakernel 等实验性路径）；最小改动集合 = 算子源码目录 + torch_binding.cpp 中 npu_fused_gdn_gating 的 schema 注册（含 include）+ build_aclnn.sh 构建清单项 + patch 注册与回退开关（新增默认关闭的 VLLM_FL_GDN_COUNT_FILE 调用计数钩子）+ 本地测试与基准脚本（详见 week3_save.md）。
2. 模型级 1K/1K 验证（统一脚本，已完成，2026-08-25）：27B/35B 双路径 4 个 run 全部完成——27B（TP1，gmem0.9）：AscendC 生成吞吐约 4.3 tok/s（profiler run，TTFT/TPOT 因 profiler 导出卡死未打印）、Triton 6.10 tok/s（TTFT 922ms / TPOT 163ms）；35B（TP2，gmem0.7）：AscendC 5.68 tok/s（TTFT 1059ms / TPOT 175ms）、Triton 5.29 tok/s（TTFT 1125ms / TPOT 188ms）；27B AscendC 的 Profiler 调用证据完整（`op_statistic.csv`：`npu_fused_gdn_gating` 49152 次调用 = 48 GDN 层 × 1024 decode 步，device 耗时占比 0.448%）；TP3 因 GDN conv 状态维度 8192 无法被 3 整除而结构性不可行（35B 取 TP2）。命令与数据明细见 week3_save.md §2。
3. 算子级 Microbenchmark：新增 benchmarks/ops/ascend/benchmark_fused_gdn_gating.py（预热 + 同步 + 重复测试，报告平均时延与 P50/P90，带正确性校验），AscendC vs Triton vs PyTorch 参考三后端 × token 数 1/4/16/64/256/1024 × bf16/fp16 共 36 项配置全部通过正确性检查；AscendC 算子平均时延约 80-90µs，为 Triton 基线（约 195-360µs）的 0.26x-0.44x（即快 2.3-3.8 倍），且不随 token 数增长。
4. 环境问题定位与规避：flag_gems 5.0.2 的 pow 算子 Triton kernel 在模型加载时报 TypeError（cannot convert None to tensor）——统一脚本内置 `--flaggems-ops unquantized_fused_moe_method,topk_softmax` 白名单，加载日志确认 `Enable only the following ops: ['unquantized_fused_moe_method', 'topk_softmax']`，无需手工设置；35B 在 TP3/内存预算过高下分别因 conv 状态维度不可分与 prefill 激活 OOM 失败，最终以 TP2 + gpu-memory-utilization 调低跑通。

下周计划：
按 W4 目标推进：完成最终 PR 提交（按已圈定的最小改动集合提交到 main 基线）；补齐多次重复的稳定性数据；基于统一脚本 profiler 产物（op_statistic.csv / api_statistic.csv / trace_view.json）定位模型级 AscendC 端到端吞吐低于 Triton 的原因（算子级快但模型级慢，疑似 decode 路径 recurrent_gated_delta_rule 或 state 布局转换开销）；完善个人技术报告（../docs/intern_ops/）与答辩材料。

所遇问题：
模型级 1K/1K 验证已于 2026-08-25 完成（27B/35B 双路径 4 个 run，数据见 week3_save.md §2.2/§2.3）：27B AscendC 为 profiler run（生成吞吐 ~4.3 tok/s，TTFT/TPOT 因 profiler 导出卡死未打印，总耗时/吞吐已从日志提取并标注），27B Triton 对照完整（6.10 tok/s、TTFT 922ms、TPOT 163ms）；35B AscendC 5.68 vs Triton 5.29 tok/s（AscendC 略快约 7%，与 8/20 基线方向相反，待重复确认）；27B 上模型级 AscendC 慢于 Triton、与算子级方向相反，疑似 decode 路径 recurrent_gated_delta_rule / state 布局转换开销（待用 profiler 产物深挖）；flag_gems pow 算子编译问题（环境问题，统一脚本内置白名单，已规避）；模型并发跑批期间出现瞬态 aivec 错误（模型日志无错误、不可复现，待观察）。

---

## 第 4 周（2026-08-25 ~ 2026-08-31，续写于 2026-09-01）

人员：龚昊磊（容器用户名：ghl）

本周计划（即第 3 周"下周计划"的 W4 目标）：
1. 完成最终 PR 提交（按已圈定的最小改动集合提交到 main 基线，与项目组确认方式）——**暂缓**：按项目要求先不提交，待项目组确认提交基线与 PR 方式后再执行。
2. 补齐多次重复的稳定性数据（当前各路径各 1 次）——待办（需先补 27B AscendC 无 profiler 对照）。
3. 基于统一脚本 profiler 产物（op_statistic.csv / api_statistic.csv / trace_view.json）定位 27B 模型级 AscendC 慢于 Triton 的原因——待办（疑似 decode 路径 recurrent_gated_delta_rule / state 布局转换开销）。
4. 完善个人技术报告与答辩材料——本周完成技术报告初稿（`ghl/ghl_report.md`）。

本周小结：
本周处于项目第 4 周（8/25~8/31），围绕 W4 计划推进算子收尾工作。核心事项为 PR 提交与技术报告：单算子最小改动集合与提交说明已在第 3 周整理完成（基线 merge-base 906fa07），本周按项目要求**未执行 PR 提交**（提交基线与方式待项目组确认，先不提交）；完成个人技术报告初稿撰写（新建 `ghl/ghl_report.md`，覆盖任务书要求的调用链、工程接入、测试、性能与限制，参考来源 R4/R8 已在文中注明）；并核验当前工作分支与 main 的基线关系（merge-base 906fa07 / main 最新 344e42b），确认最小改动集合与工作树状态未变化，为后续提交做好准备；同时基于项目组共享基线 `Baseline_results.md` 完成本人模型级数据的方向性对比（27B/35B TPOT vs eager/graph 基线，见已完成工作 4）。稳定性数据补跑与 27B 模型级差异定位列入下周计划（本周未执行：27B AscendC 无 profiler 对照需规避 profiler 导出卡死问题，且 PR 基线未定前不重复跑批）。（2026-09-10 补充：完成 Baseline 重测，见已完成工作 5，补上本环境可复现的 eager 数据、35b graph PIECEWISE 数据，并如实记录 27b graph 在 gmem0.6 下 OOM 不可复现。）（2026-09-11 补充：按 `ghl/27b_graph_script.md` 再测 27b_graph PIECEWISE，5 次（首轮 1 + 重试 4）仍全部 OOM、无有效结果，见已完成工作 6。）

已完成工作：
1. 技术报告初稿：新建 `ghl/ghl_report.md`（3-5 页技术报告：任务背景、算子定位与调用链、输入输出 dtype/布局与语义、构建部署闭环、最小改动集合与参考来源（R4 commit ab33933 下 14 个算子文件逐文件一致 / R8 补丁本地裁剪 577 行 vs 907 行）、测试（算子级 25/25 + 连通性 + 模型级 1K/1K 双路径）、性能（算子级 Microbenchmark 平均时延与 P50/P90、模型级 TTFT/TPOT/吞吐）、回退开关与 Profiler 证据（op_statistic.csv 中 FusedGdnGating 49152 次 / device 耗时占比 0.448%）、限制与遗留问题、复现命令）。
2. PR 提交准备（未提交）：确认提交基线 merge-base = 906fa07（origin/main 最新 344e42b），最小改动集合不变（算子源码目录 14 文件 + torch_binding.cpp schema 注册 + build_aclnn.sh 构建清单 + CMake/环境自举公共改动 + patch 与回退开关 + 本地测试与基准脚本，详见 week_save.md §1.3）；按指示**暂不提交**。
3. 环境与仓库跟进：仓库 main 分支新增 Ascend graph backend 相关提交（ab0798f Integrate Ascend NPU graph backend / 163ddfc Dynamo-safe / 8768619 MoE MTP 等，非本人提交）；项目组 8/31 在 27B TP4 上推进 graph 模式共享基线（如 `atp_qwen3.6-27b_fl_graph_chunked_shixi_baseline_tp4_gmem0.6_20260831_152938`，128 并发，非本人算子 run）——记录为环境信息，后续稳定性数据与提交基线需注意与 graph 模式进展对齐。
4. Baseline 对比分析（2026-09-01 新增）：项目组共享基线 `Baseline_results.md`（2026-08-31~09-01，27B/35B × eager/graph，TP4、gmem0.6、128 请求并发 64、`--cases "1024,1024,128"`）就绪后，选取本人第 3 周模型级 1K/1K 数据中的 **27B / 35B TPOT** 与其做方向性对比（本人 run 为单请求、eager+chunked，27B TP1 / 35B TP2，与基线并发/TP 配置不同，非严格 A/B，结论仅供方向参考）：

    | TPOT (ms) | 本人 run（单请求） | 基线 eager（128 并发） | 基线 graph（128 并发） |
    |---|---|---|---|
    | 27B | Triton 163.15 | median 327.89（+101%） | median 209.45（+28%） |
    | 35B | AscendC 175.09 / Triton 187.98 | median 291.42（+55%） | median 188.40（+0.2%） |

    - **35B graph 基线 median TPOT 188.40ms 与本人单请求 Triton 的 187.98ms 几乎一致（+0.2%）**，说明 graph 模式基本吸收了 128 并发的 decode 调度开销；27B graph 209.45ms 较单请求 163.15ms 增约 28%，可能含 TP1→TP4 跨卡通信等配置差异。
    - **eager 基线在 128 并发下 TPOT 相对单请求显著劣化**（27B +101%、35B +55%），反映 eager 逐 step 调度/同步开销。
    - **35B AscendC 单请求 TPOT 175.09ms 优于基线 graph median 188.40ms 约 7%**，与第 3 周"35B AscendC 略快于 Triton"方向一致（配置不同，待补同配置重复 run 验证）。
    - Caveat：基线 graph 两 run 均仅 65/128 成功（63 失败，约 49%），TPOT/吞吐统计口径基于成功请求，且 TTFT P99 达 74.6~76.7s（prefill 排队）；eager run 128/128 成功。TTFT 层面本人单请求 ~0.9~1.1s，基线 128 并发 median 8.5~9.9s（P99 ~75s），主要来自 128×1024 输入的 prefill 排队。基线输出吞吐 164~275 tok/s 为服务级指标，与本人单请求 4.3~6.1 tok/s 场景不同，不作直接比较。

5. Baseline 重测（2026-09-10 新增，续 9/9 启动中止的复测任务）：在本容器环境（910B3 ×8、CANN 9.0.0、驱动 26.0.rc1、vllm 0.13.0+empty、FL 仓库 commit a3644b2 + 工作树改动）按项目组共享基线 `Baseline_results.md` 的 4 个脚本重新测试，得到本人可复现的对照组数据：

    | Baseline | 原始记录（08-31~09-01） | 2026-09-10 重测 |
    |---|---|---|
    | 27b_eager | 128/128；164.00 tok/s；TPOT median 327.89 ms | ✅ 128/128；288.20 tok/s；TPOT median 216.73 ms |
    | 27b_graph | 65/128；266.38 tok/s；TPOT median 209.45 ms | ❌ 无有效结果（FULL 1 次瞬态异常+OOM；PIECEWISE 4 次均 OOM） |
    | 35b_eager | 128/128；185.18 tok/s；TPOT median 291.42 ms | ✅ 128/128；245.88 tok/s；TPOT median 252.72 ms |
    | 35b_graph | 65/128；275.21 tok/s；TPOT median 188.40 ms | ✅ 128/128；322.95 tok/s；TPOT median 192.32 ms（PIECEWISE） |

    - 口径与差异：除 `--port`（8113→8114，8113 被占用）与两个 graph 脚本的 `--cudagraph-mode`（FULL→PIECEWISE，按要求调整）外，其余参数与原始脚本一致；新增严格有效性判据（`Successful>0` 且 `Total generated tokens>10` 且 `duration>60s`），避免瞬态崩溃被客户端误计为成功。
    - **eager 重测值均优于原始记录**（27B 288.20 vs 164.00 tok/s、TPOT median 216.73 vs 327.89 ms；35B 245.88 vs 185.18 tok/s、TPOT median 252.72 vs 291.42 ms），与当日卡负载/调度差异有关，方向性参考。
    - **35b_graph 由 FULL 改为 PIECEWISE 后首次即 128/128 成功**（原始 FULL 仅 65/128），稳定性明显改善；因 cudagraph 模式不同，吞吐不宜与原始 FULL 记录作严格对比（重测 322.95 tok/s、TPOT median 192.32 ms，与原始 median 188.40 ms 接近）。
    - **27b_graph 在本环境不可复现**：FULL 1 次在正式 bench 阶段出现 NPU OOM + vector core exception（507035）级联，客户端把 64 个立即失败请求计为成功（4.69s、1 token），按严格判据判为无效；PIECEWISE 4 次均在 cudagraph capture 阶段 OOM，特征一致（需 650 MiB，已分配 56.18 GiB、仅剩约 23 MiB、57 无碎片）。参考：同环境 27b_eager 可 128/128；27B 每卡 model loading 12.87 GiB、Available KV cache 20.30 GiB。结论：27B graph 在 TP4 / gmem0.6 / 128 并发下 FULL 与 PIECEWISE 均无法完成。
    - 明细（含各 run 完整结果块与失败时间线）见 `ghl/Baseline_results.md` 附录"2026-09-10 重测记录"；环境信息与摘要保留于 `/workspace/results/ghl/20260910_Baseline重测/`（原始 run 数据与跑批日志已按用户要求于 2026-09-10 删除，脚本保留于 `/workspace/scripts/ghl/`）。

6. 27b_graph 重测（2026-09-11 新增）：按 `ghl/27b_graph_script.md` 的脚本（graph + `PIECEWISE`、TP4/gmem0.6/devices 0,1,2,3、端口 8113、`--cases "1024,1024,128"`、并发 64，与 9/10 口径一致）重新测试 27b_graph，**5 次运行（首轮 1 次 + 重试 4 次）全部 OOM，无有效结果**：

    - 5 次均 server 正常启动、PIECEWISE cudagraph capture 19/19 通过（engine init 63.6~67.9s），随后 warmup（默认 `128,128,2,4`）4/4 请求失败、server 崩溃，正式 bench 因端口无服务连接被拒 rc=1。
    - OOM 特征 5 次一致：`Tried to allocate 650.00 MiB (NPU 0; 60.96 GiB total capacity; 56.18 GiB already allocated; 仅剩 23~28 MiB free; 56.23 GiB reserved in total by PyTorch)`。
    - 与 9/10 对比：9/10 的 4 次 PIECEWISE 在 **capture 阶段** OOM；本次 5 次 capture 均通过、改在 **warmup（首次真实推理）阶段** OOM，说明瓶颈是推理激活内存峰值而非 graph 捕获本身。
    - 产物：脚本 `/workspace/scripts/ghl/rerun_27b_graph_20260911.sh`、日志 `rerun_27b_graph_20260911.log`；5 个 run 目录（`..._20260911_003721/_005758/_011447/_013135/_014833`）保留于 `/workspace/results/` 作为 OOM 证据；摘要归档于 `/workspace/results/ghl/20260911_27b_graph重测/`（`env_info.md`、`result_summary.md`）。
    - 结论：27B graph 在本环境（TP4/gmem0.6/8192/128 并发）FULL 与 PIECEWISE 均不可复现（累计 FULL 1 次 + PIECEWISE 9 次失败）；要取数需改配置（capture 尺寸 / max-num-seqs / max-model-len），属偏离官方口径，需与项目组确认。

下周计划：
1. 与项目组确认 PR 提交基线与方式，确认后按最小改动集合完成最终 PR 提交（当前暂缓）。
2. 补齐多次重复稳定性数据：优先 27B AscendC 无 profiler 对照（`--bench-profile false --skip-analyse`，规避 profiler 导出卡死），重复 3 次取均值，确认 27B/35B AscendC vs Triton 方向；另：27B graph 基线在本环境（gmem0.6）取不到数，需与项目组确认可用配置（capture 尺寸 / max-num-seqs / max-model-len）后再补测。
3. 27B 模型级差异定位：用 profiler 产物（api_statistic.csv / trace_view.json）定位 decode 路径开销（recurrent_gated_delta_rule / state 布局转换），先查磁盘余量（GB 级导出文件用完即删）。
4. 技术报告定稿与答辩材料准备（任务书提交方式：技术报告 + 汇报）。

所遇问题：
1. PR 提交暂缓：提交基线与 PR 方式待项目组确认（最小改动集合已整理，merge-base 906fa07；按指示本周不提交）。
2. 27B AscendC profiler run 遗留：profiler 导出阶段卡住（trace_view.json 15.5GB、CANN 解析 11m43s），vllm bench 结果块未打印，TTFT/TPOT 未测得；严格性能对比需补 `--bench-profile false` 对照。
3. 27B/35B 模型级 AscendC vs Triton 方向不一致（27B AscendC ~4.3 慢于 Triton 6.10 tok/s；35B AscendC 5.68 快于 Triton 5.29 tok/s 约 7%），待重复数据确认，疑似 decode 路径 recurrent_gated_delta_rule / state 布局转换开销（待深挖）。
4. 沿用遗留：TP3 对 35B 结构性不可行（GDN conv 状态维度 8192 % 3 = 2）；flag_gems 5.0.2 pow 编译失败（统一脚本白名单已规避）；并发跑批瞬态 aivec 错误（不可复现，待观察）。
5. 27B graph 基线在本环境取不到数（2026-09-11）：按 `27b_graph_script.md` 跑 PIECEWISE 共 5 次（首轮+重试 4）全部 OOM，capture 均通过、warmup 阶段失败（56.18 GiB 已分配、仅剩 ~25 MiB）；累计 FULL 1 次 + PIECEWISE 9 次失败。需与项目组确认可用配置后再补测，否则只能如实标注不可复现。

---

## 第 5 周（2026-09-01 ~ 2026-09-11，续写于 2026-09-11）

人员：龚昊磊（容器用户名：ghl）

本周小结：
本周为项目收尾阶段（覆盖 9/1~9/11），围绕第 4 周"下周计划"中的交付物推进。按项目安排，本周只完成**技术报告定稿**与**答辩材料准备**两项（第 4 周计划中的 PR 提交、27B 稳定性/差异定位数据不作为本周目标）。已完成 `ghl/ghl_report.md` 定稿（报告日期、结论状态、交付物清单更新，并与答辩材料交叉引用）、新建答辩材料 `ghl/ghl_reply.md`（12 页汇报提纲 + 关键数据速查 + 14 条 Q&A 预案 + 评分点对齐 + 演示/兜底预案）。期间另完成 9/9~9/10 共享基线重测的归档与回填（见第 4 周已完成工作 5、6），以及 9/11 的 27b_graph 复测（5 次全 OOM，如实记录）。

已完成工作：
1. 技术报告定稿（`ghl/ghl_report.md`）：
    - 报告日期由"2026-09-01 初稿"更新为"2026-09-01（初稿）／2026-09-11（定稿）"，并标注配套答辩材料路径；
    - §8 结论中"项目汇报"项更新为：技术报告定稿 + 答辩材料完成（PR 提交单独标注为待项目组确认基线/方式）；
    - 新增"附录 C：交付物清单"，逐项列出算子源码/schema 注册/构建脚本/框架补丁/算子级测试/连通性测试/微基准/模型级结果/文档的状态。
2. 答辩材料（新建 `ghl/ghl_reply.md`）：
    - 汇报提纲 12 页：任务概述 → 六周计划完成度 → 算子定位与调用链 → 语义与接口（g/beta、dtype/布局、threshold）→ 工程接入与最小改动集合（R4/R8 来源）→ 编译部署闭环 → 算子级 25/25 正确性 → 框架接入与回退开关/计数钩子 → Profiler 证据（49152 次 / 0.448%）→ 算子级 Microbenchmark → 模型级 1K/1K → 限制与总结；
    - 关键数据速查表（测试数、时延比、Profiler 计数、模型级数值、算子文件数、patch 行数、环境版本）；
    - 与任务书评分点（工程接入 30% / 正确性与测试 20% / 框架集成与性能 20% / 代码质量文档答辩 30%）逐项对齐；
    - 14 条预判问答（Q&A）：算子级快而模型级 27B 慢的口径澄清、dtype/布局设计、softplus threshold、如何证明算子进入推理路径、R4/R8 与"不得整体复制"的边界、TP3 不可行、27B graph OOM、PR 未提交、微基准可信度、dtype 支持范围等；
    - 现场演示/复现命令（连通性测试、算子级测试、微基准、模型级 1K/1K）与风险兜底预案。

下周计划：
项目进入答辩收尾，剩余事项：
1. 按排期完成答辩（材料已就绪：`ghl/ghl_reply.md`）。
2. PR 提交：待项目组确认提交基线与方式后，按已整理的最小改动集合执行（当前暂缓）。
3. （可选）27B 模型级差异定位 + AscendC 无 profiler 对照/重复数据：需先与项目组确认 27B graph 可用配置与资源安排。

所遇问题：
1. PR 提交待项目组确认基线/方式（最小改动集合已就绪，merge-base 906fa07），未提交。
2. 27B graph 基线在本环境取不到数（5 次 PIECEWISE 全部 OOM，累计 FULL 1 + PIECEWISE 9 次失败），需确认可用配置后再补测。
3. 27B 模型级 AscendC 慢于 Triton 的方向性差异仍待定位（疑似 decode 路径 state 布局转换），需无 profiler 对照 + 重复数据。
