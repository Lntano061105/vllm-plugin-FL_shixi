# work_now.md — 工作状态存档（2026-09-10 更新，Baseline 重测任务已完成）

> 容器用户名：ghl（run_rule.md 中的"你的用户名"；结果目录 `/workspace/results/ghl/`、个人脚本 `/workspace/scripts/ghl/`）。
> 路径说明：本文档位于仓库 `ghl/` 子目录；文中仓库相对路径（`csrc/`、`vllm_fl/`、`tests/`、`docs/intern_ops/`、`README.md` 等）相对仓库根 `/workspace/vllm-plugin-FL/`（即加 `../` 前缀）。
> 本文档 2026-09-10 更新（新增 §I 今日存档：Baseline 重测完成，文档已回填、结果已归档）；§H 为 2026-09-09 存档（9/9 启动中止），§0~§6 为 2026-08-25 历史存档。

---

## I. 2026-09-10 存档：Baseline 重测已完成（4 项中 3 项有效，27b_graph 如实记录不可复现）

### I.1 任务与口径（续 9/9 §H）

- 任务：按 `ghl/Baseline_results.md` 4 个脚本重测（TP4、gmem0.6、`--cases "1024,1024,128"`、128 请求并发 64、max-model-len 8192、`--no-bench-profile --skip-analyse --package none`、run-label `shixi_baseline`）。
- 用户确认口径：4 个全跑；端口 8114；文档保留原记录并追加重测章节（按 20260910 命名）；27b_graph 先 FULL 最多 4 次、失败则停下询问。
- 中途用户调整：27b/35b graph 由 FULL 改 **PIECEWISE**（用户给出新命令）；27b_graph 的 4 次 PIECEWISE 仍 OOM 后，用户决定 **不再尝试、如实记录**。

### I.2 结果

| Baseline | Run 目录（公共前缀 `/workspace/results/atp_`，后缀 `_tp4_gmem0.6_`） | 结果 |
|---|---|---|
| 27b_eager | `qwen3.6-27b_fl_enforce_eager_chunked_shixi_baseline_20260910_005225` | ✅ 128/128；454.79s；288.20 tok/s；TTFT med 2885ms；TPOT med 216.73ms |
| 35b_eager | `qwen3.6-35b-a3b_fl_enforce_eager_chunked_shixi_baseline_20260910_011918` | ✅ 128/128；533.06s；245.88 tok/s；TTFT med 4784ms；TPOT med 252.72ms |
| 35b_graph (PIECEWISE) | `qwen3.6-35b-a3b_fl_graph_chunked_shixi_baseline_20260910_025902` | ✅ 128/128；405.86s；322.95 tok/s；TTFT med 4589ms；TPOT med 192.32ms |
| 27b_graph | FULL 1 次 `..._011041`；PIECEWISE 4 次 `..._015126/_020823/_022514/_024210` | ❌ 无有效结果（FULL：OOM + vector core 507035 级联，客户端误计 64/64 成功；PIECEWISE 4 次均 capture 阶段 OOM） |

- 严格有效性判据：`Successful requests>0` 且 `Total generated tokens>10` 且 `Benchmark duration>60s`（防瞬态错误被误计为成功）。
- 27b_graph OOM 证据：`Tried to allocate 650.00 MiB`，`56.18 GiB already allocated`、仅剩约 23 MiB、`56.23 GiB reserved`（4 次一致）；参考同环境 27B 每卡 model loading 12.87 GiB、Available KV cache 20.30 GiB。

### I.3 关键结论

- eager 重测值均优于 8/31~9/1 原始记录（27B 288.20 vs 164.00 tok/s、TPOT med 216.73 vs 327.89ms；35B 245.88 vs 185.18 tok/s、TPOT med 252.72 vs 291.42ms），与当日卡负载/调度差异有关，方向性参考。
- 35b_graph 改 PIECEWISE 后**首次即 128/128**（原 FULL 记录仅 65/128），稳定性明显改善；因 cudagraph 模式不同，吞吐不与原 FULL 记录严格对比。
- 27B graph 在本环境（TP4/gmem0.6/128 并发）FULL 与 PIECEWISE 均 OOM，原 8/31 的 65/128 记录**不可复现**。

### I.4 文档与归档（已完成）

- `ghl/Baseline_results.md`：追加附录「2026-09-10 重测记录」（保留 8/31~9/1 原记录，含 3 项完整结果块 + 27b_graph 失败时间线）。
- `ghl/week_report.md`：本周小结补充一句；已完成工作新增第 5 项（重测对比表与结论）。
- 归档：`/workspace/results/ghl/20260910_Baseline重测/` 仅保留 `env_info.md`、`result_summary.md`；其中 `logs/`（全部跑批日志）与 `runs/`（3 个有效 run 的 run_info.env/run_command.sh/结果原文）已按用户要求于 2026-09-10 删除。
- 跑批脚本：`/workspace/scripts/ghl/rerun_baseline_20260910.sh`（eager + 首次 FULL）、`rerun_graphs_20260910.sh`（严格判据版）、`rerun_graphs_piecewise_20260910.sh`（graph PIECEWISE 重跑）。

### I.5 数据清理（2026-09-10 按要求执行）

- 已删除今日全部测试数据：9 个 run 目录（27b_eager 1、27b_graph 5、35b_eager 1、35b_graph 2，含中止的 `_013911`）、归档 `logs/` 与 `runs/`、`/workspace/scripts/ghl/` 下今日 6 个跑批日志/摘要。
- 删除方式：run_rule 明确路径逐一 `ls` 确认 + 归属校验（只删我方 `shixi_baseline` 目录；`lgx_ascendc_*` 两个他人目录与 `latest_run_dir.txt`（当前指向 lgx 的 run）均未触碰）。
- 保留：跑批脚本（`.sh`）、文档（`Baseline_results.md` / `week_report.md` / `work_now.md`，数值与结论已回填）、归档 `env_info.md` + `result_summary.md`。
- 失败 run 名称仅作为历史标识保留在文档中（其数据已删）。

### I.6 遗留与踩坑

- **本次踩坑**：自写 `wait_port_free` 的 Python heredoc 顶层误用 `return` → `SyntaxError`，致 00:33 首次跑批空转 14 分钟（无结果目录）；已修复重启（失败日志已随本次清理删除）。
- 27B graph 若后续仍需数据：需与项目组确认可用配置（gmem/capture/env），或改用更小 capture/max-num-seqs，均需标注偏离官方口径。

---

## H. 2026-09-09 存档：按 Baseline_results.md 重新测试（9/9 中止，已由 §I 于 2026-09-10 完成）

### H.1 任务背景与已确认口径

- 任务：按 `ghl/Baseline_results.md` 中 4 个脚本（27B/35B × eager/graph，TP4、gmem0.6、`--cases "1024,1024,128"`、128 请求并发 64、max-model-len 8192、`--no-bench-profile --skip-analyse --package none`、run-label `shixi_baseline`）重新测试，结合 `week_report.md` 第 4 周 Baseline 对比分析，完成后**重写两文档数据**。
- 用户已确认两项口径：① 资源安排＝后台轮询等空卡自动串行跑；② 文档更新方式＝`Baseline_results.md` **保留原记录、追加重测章节**，`week_report.md` 同步更新对比数据。
- 入口脚本：`/workspace/scripts/benchmark_script_fl.sh`（项目组统一脚本，vllm serve + vllm bench）。结果目录自动生成 `/workspace/results/atp_{model}_fl_{mode}_{chunk}_{label}_tp4_gmem0.6_{ts}/`。

### H.2 今日执行结果（2026-09-09 实测，数据目录已按用户要求全部删除，数值留档供明日对比参考）

| Baseline | 结果 | 关键数值 |
|---|---|---|
| 27b_eager | ✅ 1 次成功 128/128（run `..._082117`） | duration 466.59s；out 280.91 tok/s；TTFT mean/med 7750/3003ms；TPOT mean/med 218.8/221.6ms、P99 226.0ms |
| 27b_graph | ❌ 全部失败（3 次：端口冲突空跑 / 运行时 OOM 0/128 / server early exit） | OOM 详情：运行时 `NPU out of memory`，54.79GB 已分配、仅剩 83MB free（gmem0.6 下 FULL capture 峰值内存超预算，详见 H.4） |
| 35b_eager | ✅ 1 次成功 128/128（run `..._090715`） | duration 509.16s；out 257.43 tok/s；TTFT med 4718ms；TPOT mean/med 232.4/240.5ms、P99 245.0ms |
| 35b_graph | ⚠️ 2 次均 65/128（run `..._083814`、`..._092517`） | 与 8/31 旧记录同口径（65 成功/63 失败）；`_092517`：duration 198.41s；out 320.81 tok/s；TPOT med 173.25ms |

- 新 eager 数据明显优于 8/31 旧值（27b：280.9 vs 164.0 tok/s、TPOT med 221.6 vs 327.9ms；35b：257.4 vs 185.2 tok/s、TPOT med 240.5 vs 291.4ms），或与当日卡负载/调度差异有关，明日可复跑确认。
- graph 模式 128 请求仅约半数成功（65/63）为复现性现象（8/31、9/1、9/8、9/9 一致），非本次偶发；记录时需如实标注。

### H.3 环境与资源现状（重要）

- **8 卡全天被其他用户跑批占满**（9/8~9/9 存在大量 lgx/dyq/lzh/lx 的 atp_* run），等整组 4 卡空闲常需 1h+；0-3 与 4-7 两组随时可能被抢占。
- **端口 8113 已被其他服务占用**→ 我的 run 统一改用 **8114**（与文档命令唯一差异，已确认空闲范围 8114~8139）。
- 空卡判据（可复用）：某组 4 卡（0-3 或 4-7）HBM 残留 <10GB（实际空闲残留约 3.4GB）且 AICore≈0，**连续 3×60s** 稳定后才可启动，防止误抢正在加载/运行的任务。
- **9/8 参考**：他人对 27b graph FULL shixi_baseline 连跑 9 次仅 3 次有结果（65/63 ×1、128/128 ×2），失败多为启动类瞬时错误（atb so 加载失败等）与运行 OOM → FULL 在此环境极不稳定。

### H.4 已确认踩坑（明日沿用）

| # | 问题 | 解法 |
|---|---|---|
| 1 | `benchmark_script_fl.sh` 结束后 vLLM 子进程退出慢，紧随的下一个 run 因端口未释放**立即失败**（rc=1，生成空 run 目录） | run 之间必须先等待端口可 bind（已写入脚本，最多等 300s） |
| 2 | 27b graph **FULL** gmem0.6 运行时 OOM（vLLM reserved 56.2GB、allocated 54.8GB 后 78MB 都分不出）；9/8 重试 9 次仅 2 次 128/128 | 备选：① 继续 FULL 多试（成功率低）；② 改 `--cudagraph-mode PIECEWISE`（9/9 项目组 dyq/lzh/lx 均在跑 PIECEWISE，gmem 0.6/0.75）——需与项目组确认基线口径后再定 |
| 3 | 跑批中偶发 `aicore/fftsplus aicore error`（周报已知瞬态问题） | 串行跑批规避；单 run 数据异常时重跑该 run |
| 4 | 端口失败/启动失败会留"空 run 目录"（只有脚本初始化文件） | 删除前 `ls` 确认仅含 `analysis.py/run_*.sh/run_info.env/run_summary.txt/client.log/request_benchmark_results.txt(空)/torch_profile` 后按明确路径 `rm -rf` |

### H.5 今日已清理（用户指示"删除今日生成的数据文件"）

- 全部 12 个 run 目录已删（6 个端口失败空目录执行中即删 + 6 个结果目录：082117/083814/090154/090715/092517/100338）；4 个 rerun 运行日志已删；`/workspace/results/latest_run_dir.txt` 已清空；NPU 无本人残留进程。
- **保留**：执行脚本 `/workspace/scripts/ghl/rerun_baseline_20260909.sh`（串行版，含 wait_port；当前仅含 27b_graph/35b_eager/35b_graph 三段，27b_eager 段已注释）与 `/workspace/scripts/ghl/rerun_27b_graph_20260909.sh`（27b graph 专用，最多 4 次重试，成功即停）——明日可直接复用或按需修改。
- `ghl/Baseline_results.md`、`ghl/week_report.md` **均未改动**，保持原状。

### H.6 明日 TODO

1. 确认卡资源与任务范围（是否 4 个 baseline 全跑；建议错开他人高峰）。
2. 恢复 27b_eager 段（如全跑）→ 串行执行 4 baseline（端口 8114）。**27b graph 优先用 FULL 多试；若再 OOM，改用 PIECEWISE 并与项目组确认口径**；eager 模式稳定，预期一次通过。
3. 收集各 run `request_benchmark_results.txt` → `Baseline_results.md` **追加 20260909 重测章节**（保留 8/31~9/1 原记录），`week_report.md` 更新 Baseline 对比数据（graph 的 65/128、OOM/失败次数需如实标注）。
4. 按 run_rule 归档摘要到 `/workspace/results/ghl/20260909_描述/`（含环境信息：commit、CANN、`npu-smi info`、完整命令）。

### H.7 关键路径速查

| 项 | 路径 |
|---|---|
| 文档 | `ghl/Baseline_results.md`、`ghl/week_report.md`（仓库内） |
| 入口脚本 | `/workspace/scripts/benchmark_script_fl.sh`（参数见 H.1；改 `--port 8114`） |
| 复用脚本 | `/workspace/scripts/ghl/rerun_baseline_20260909.sh`、`/workspace/scripts/ghl/rerun_27b_graph_20260909.sh` |
| 模型 | `/models/Qwen3.6-27B`、`/models/Qwen3.6-35B-A3B`（只读） |
| 结果目录 | `/workspace/results/atp_*_shixi_baseline_tp4_gmem0.6_*` |

---

## 历史存档（2026-08-25，任务已完成）

> 背景任务：Fused GDN Gating 算子（Qwen3.6 27B/35B GDN 路径）实习周报 + 项目组 Profiling 统一脚本接入验证。
> 本文档 2026-08-25 版记录 1K/1K 模型级验证完成状态与剩余可选事项（内容已并入周报 week1~4）。

---

## 0. 任务完成度总览

✅ **2026-08-25 已完成全部 TODO**（原 work_now.md 2026-08-24 版 §5 的 1-8 项；第 9 项 0.6B 冒烟为可选，未执行）：

1. ✅ 清理残留（僵尸进程不占资源，端口 8113/8080 已释放）
2. ✅ 提取 27B AscendC 性能数据并补写 `request_benchmark_results.txt`
3. ✅ 核验 Profiler 证据（op_statistic.csv：FusedGdnGating 49152 次 / 0.448%）
4. ✅ 27B Triton 对照（--bench-profile false）
5. ✅ 35B AscendC（--tp 2 --devices 4,5 --gmem 0.7）
6. ✅ 35B Triton 对照
7. ✅ 归档（4 个 run 各生成 `*_op_statistic.tar.gz`）
8. ✅ 周报回填（week1/week2/week3 及 *_save 全部更新，含 week2.md/week3.md）

---

## 1. 环境现状（2026-08-25 实测）

- NPU：910B3 × 8，全卡基本空闲（仅 NPU1 被占 ~28G），本次用 NPU4（27B）/ NPU4,5（35B TP2）
- 软件：torch 2.8.0、torch_npu 2.8.0.post2、vllm 0.13.0、CANN 8.5/9.0、flag_gems 5.0.2
- 模型权重：`/models/Qwen3.6-27B`（64 层/48 GDN 层）、`/models/Qwen3.6-35B-A3B`（40 层/30 GDN 层，67GB/26 safetensors）
- 统一脚本：`/workspace/scripts/run_vllm_fl_profile_unified.sh`、`/workspace/scripts/package_op_statistic.sh`
- **磁盘：已清理至 203G 可用**（删除了 8/13~8/17 旧 run 的 profiler 原始大文件 ~183G，保留轻量结果；昨天证据 run 原始文件 11.5G 保留待后续清理）

---

## 2. 最终数据（统一脚本，1 请求 1024 in / 1024 out，eager + chunked）

| 模型 | 路径 | Run 目录 | 耗时(s) | 生成吞吐(tok/s) | TTFT(ms) | TPOT(ms) |
|---|---|---|---|---|---|---|
| 27B TP1 gmem0.9 | AscendC* | `atp_qwen3.6-27b_..._ascendc_r1_tp1_gmem0.9_20260824_090926` | 247.96 | ~4.3 | N/A* | N/A* |
| 27B TP1 gmem0.9 | Triton | `atp_qwen3.6-27b_..._triton_r1_tp1_gmem0.9_20260825_001525` | 167.83 | 6.10 | 922.18 | 163.15 |
| 35B TP2 gmem0.7 | AscendC | `atp_qwen3.6-35b-a3b_..._ascendc_r1_tp2_gmem0.7_20260825_003035` | 180.18 | 5.68 | 1058.51 | 175.09 |
| 35B TP2 gmem0.7 | Triton | `atp_qwen3.6-35b-a3b_..._triton_r1_tp2_gmem0.7_20260825_005827` | 193.43 | 5.29 | 1125.14 | 187.98 |

*27B AscendC 为 `--bench-profile true` 证据 run：profiler 导出阶段脚本卡住（trace_view.json 15.5GB、CANN 解析 11m43s），vllm bench 结果块未打印——TTFT/TPOT 未测得，总耗时/吞吐从 `request_terminal.log`（tqdm）与 `server.log`（引擎日志 4.0~4.3 tok/s）提取，已如实标注。

**Profiler 证据（op_statistic.csv）**：`FusedGdnGating`（AI_VECTOR_CORE）Count=49152（=1024 tok × 48 GDN 层，精确对应）、Total 249.6ms、Ratio 0.448%；配套 GDN 算子均在列：RecurrentGatedDeltaRule 2.005%、CausalConv1d 0.893%、chunk_gated_delta_rule_fwd_kernel 0.048% 等。`api_statistic.csv`/`step_trace_time.csv` 亦齐（Computing 55.75s / 采集窗口 247.9s）。

**结论要点（如实记录，已写入周报）**：
- 27B 模型级 AscendC（~4.3，profiler run）慢于 Triton（6.10），与 8/20 基线（4.07/4.52 vs 5.32）方向一致；严格对比需补无 profiler 的 27B AscendC 对照
- 35B AscendC（5.68）略快于 Triton（5.29），与 8/20 基线（3.42 vs 4.60）方向相反，待重复确认
- 疑似 27B 开销点：decode 路径 recurrent_gated_delta_rule / state 布局转换（转置、dense KV 重组）

---

## 3. 已确认踩坑（2026-08-25 实测，已回填周报）

| # | 问题 | 原因 | 解法 |
|---|---|---|---|
| 1 | `No available memory for the cache blocks`（exit 1） | 27B bf16 权重 ≈50GiB，TP1 + gmem 0.6 装不下 | `--gmem 0.9 --max-model-len 4096` |
| 2 | `--bench-profile true` 卡住 1h+ | profiler 导出巨大（trace_view.json 15.5GB） | 性能对照一律 `--bench-profile false --skip-analyse`；profiler 证据保留 27B AscendC 一份即可 |
| 3 | 磁盘 99% 满 | GB 级 profiler 原始文件堆积 | 已清理 183G；`msprof_*.json/db`、`torch.python_tracer_func`、`kernel_details.csv` 等用完即删 |
| 4 | TP3 对 35B 结构性不可行 | GDN conv 状态维度 8192 % 3 = 2 | 35B 用 `--tp 2` |
| 5 | 并发跑批瞬态 aivec 错误 | 跨卡同步粘性错误上报 | 串行跑批规避 |

---

## 4. 归档产物

`/workspace/results/ghl/`（20260824/20260825 模型级1K1K验证目录）下 4 个 `*_op_statistic.tar.gz`（27B AscendC 含 op/api/step_trace 3 csv + result；其余含 result）。`latest_run_dir.txt` → ghl 下 35B Triton run。

---

## 5. 剩余可选事项（非阻塞）

1. **0.6B 手动分步冒烟**（`手动profiling测试.md`）：流程验证/答辩素材，未执行（备用）
2. **多次重复稳定性数据**：当前各路径各 1 次，可补跑（27B AscendC 无 profiler 对照尤其值得补）
3. **27B 模型级差异定位**：用 api_statistic.csv / trace_view.json 深挖 decode 路径开销（磁盘注意）
4. **最终 PR 提交**：按 week_save.md（第 3 周）§1.3 最小改动集合提交到 main 基线（与项目组确认方式）
5. **个人技术报告**：`../docs/intern_ops/`，3-5 页 + 答辩材料
6. **清理 8/24 证据 run 原始大文件**（11.5G，ascend_pytorch_profiler_0.db / msprof json / python_tracer_func 等；op_statistic.csv 等轻量结果已归档，可安全删除）

---

## 6. 关键路径速查

| 项 | 路径 |
|---|---|
| 周报（已合并） | `week_report.md`、`week_save.md`（原 week1~3.md / week1~3_save.md 已删除） |
| 统一脚本说明 | `unify_sh.md`、`手动profiling测试.md` |
| 统一脚本 | `/workspace/scripts/run_vllm_fl_profile_unified.sh`、`/workspace/scripts/package_op_statistic.sh` |
| 27B AscendC 证据 run | `/workspace/results/atp_qwen3.6-27b_fl_enforce_eager_chunked_ascendc_r1_tp1_gmem0.9_20260824_090926/` |
| 外层日志（已归档） | `/workspace/results/ghl/20260824_20260825_模型级1K1K验证外层日志/`（run_ascendc/triton/35b 各 .log、gdn_count_*.txt） |
| 算子源码 | `csrc/ascend/attention/fused_gdn_gating/` |
| 框架 patch | `vllm_fl/dispatch/backends/vendor/ascend/patches/patch_qwen3_6_gdn.py` |
| 算子级微基准（个人脚本） | `/workspace/scripts/ghl/benchmark_fused_gdn_gating.py` |
| 模型级补充脚本（个人脚本） | `/workspace/scripts/ghl/benchmark_model_gdn_1k.py` |
