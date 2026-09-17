# Qwen3.6 27B / 35B 四项 Baseline 汇总（PIECEWISE）

- 汇总时间：2026-09-15
- 平台：Ascend 910B3 × 4（TP=4），CANN 9.0.0
- 统一参数：cases 1024,1024,128｜concurrency 64｜max-num-seqs 64｜max-model-len 8192｜gmem 0.6｜chunked prefill 开启｜bench-profile 关闭
- 插件路径：`/workspace/vllm-plugin-FL.bak.20260817`（--fl-repo-path）

## 一、总体结果

| 模型 | 模式 | Output tok/s | Total tok/s | Mean TTFT (ms) | Mean TPOT (ms) | 成功/失败 | 耗时 (s) |
|---|---|---|---|---|---|---|---|
| Qwen3.6-27B | graph (PIECEWISE) | **490.34** | 980.68 | 8441.34 | 121.33 | 128/0 | 267.31 |
| Qwen3.6-35B-A3B | graph (PIECEWISE) | 326.11 | 652.22 | 13842.02 | 181.88 | 128/0 | 401.93 |
| Qwen3.6-27B | eager | 293.21 | 586.41 | 7821.11 | 209.24 | 128/0 | 447.03 |
| Qwen3.6-35B-A3B | eager | 268.38 | 536.76 | 14831.91 | 222.60 | 128/0 | 488.38 |

- 四组全部 128 请求成功、0 失败。
- graph 相比 eager：27B 吞吐 +67.2%（490.34 / 293.21），35B +21.5%（326.11 / 268.38）；TPOT 同步下降 42.0% / 18.3%。

## 二、各组运行目录

| 组 | 容器内 RUN_DIR |
|---|---|
| 27b_graph | `/workspace/results/atp_qwen3.6-27b_fl_graph_chunked_shixi_baseline_pw_tp4_gmem0.6_20260915_115604` |
| 35b_graph | `/workspace/results/atp_qwen3.6-35b-a3b_fl_graph_chunked_shixi_baseline_pw_tp4_gmem0.6_20260911_102049` |
| 27b_eager | `/workspace/results/atp_qwen3.6-27b_fl_enforce_eager_chunked_shixi_baseline_pw_tp4_gmem0.6_20260911_103545` |
| 35b_eager | `/workspace/results/atp_qwen3.6-35b-a3b_fl_enforce_eager_chunked_shixi_baseline_pw_tp4_gmem0.6_20260911_105227` |

## 三、27B graph 补跑说明

- 09-11 首次跑批（devices 0,1,2,3 / port 8113）在 KV cache 分配阶段 OOM，EngineCore 退出，rc=1，未出数。
- 本次改用 devices 4,5,6,7、port 8113（与 09-08 已验证成功组一致），单组串行重跑，12:07:35 完成，rc=0。
- 两次差异仅在卡组；软件栈、模型、参数、插件路径完全相同。设备组 4-7 当前无其他任务占用，结果可复现。

## 四、本地归档清单

| 文件 | 说明 |
|---|---|
| `27b_graph_request_benchmark_results.txt` | 本次补跑原始结果 |
| `35b_graph_request_benchmark_results.txt` | 原始结果 |
| `27b_eager_request_benchmark_results.txt` | 原始结果 |
| `35b_eager_request_benchmark_results.txt` | 原始结果 |
| `*_run_info.env` | 各组运行参数快照（含 DEVICES / PORT / CUDAGRAPH_MODE / FLAGGEMS_OPS） |

## 五、数值可溯源与复核（2026-09-17 复核）

- 复核方式：用 `benchmarks/ops/ascend/analysis.py` 解析上表四个 RUN_DIR 的
  `request_benchmark_results.txt`，输出 `benchmarks/ops/ascend/benchmark_summary_20260915.csv`。
- 复核结果：`output_tok_s` = **490.34 / 326.11 / 293.21 / 268.38**，与一、二节完全一致；
  四行均为 `successful=128`、`failed=0`、`case=i1024_o1024_np128_c64`、`concurrency=64`。
- CSV 每行带 `run`（完整 RUN_DIR 名）与 `time`（压测时刻），可在容器内逐一比对：

| 组 | CSV 中 run（`/workspace/results/` 下） | time |
|---|---|---|
| 27b_graph | `atp_qwen3.6-27b_fl_graph_chunked_shixi_baseline_pw_tp4_gmem0.6_20260915_115604` | 2026-09-15 12:07:25 |
| 35b_graph | `atp_qwen3.6-35b-a3b_fl_graph_chunked_shixi_baseline_pw_tp4_gmem0.6_20260911_102049` | 2026-09-11 10:34:50 |
| 27b_eager | `atp_qwen3.6-27b_fl_enforce_eager_chunked_shixi_baseline_pw_tp4_gmem0.6_20260911_103545` | 2026-09-11 10:51:31 |
| 35b_eager | `atp_qwen3.6-35b-a3b_fl_enforce_eager_chunked_shixi_baseline_pw_tp4_gmem0.6_20260911_105227` | 2026-09-11 11:08:53 |

- 数据源说明：仓库根 `benchmark_results/*.csv` 是 07-18 批次的客户端原始 csv，**没有**模型 / 模式 /
  时间列，与上表四项数值无对应关系，不可作为引用依据；本文件的权威来源是上面四个 RUN_DIR 的
  `request_benchmark_results.txt` 及其聚合产物 `benchmark_summary_20260915.csv`。
- 统计口径：表中为 Mean TTFT / Mean TPOT；Median TPOT 略高（27B graph 121.33 → 125.48 ms），
  引用时需注明口径。
