为便于量化各位算子接入所带来的性能收益，我在此提供4个Baseline供大家对比参考。相关参数及测试结果已整理在飞书文档中，请注意查阅：
https://wcnhom4zaih2.feishu.cn/wiki/PQkPwMUv1iDf9OkteUHcAm6knuf?from=from_copylink

文档中已包含各Baseline的具体运行脚本和配置，请大家按此执行。

每个Baseline均占用4张显卡，单次运行时间约20余分钟。请大家合理协调显卡资源进行测试，避免同时占用8卡，以免影响他人进度。
27b_eager_脚本：
bash benchmark_script_fl.sh \
  --model-path /models/Qwen3.6-27B \
  --model-name qwen3.6 \
  --model-tag qwen3.6-27b \
  --mode eager \
  --cases "1024,1024,128" \
  --concurrency 64 \
  --max-num-seqs 64 \
  --max-model-len 8192 \
  --tp 4 \
  --gmem 0.6 \
  --devices 0,1,2,3 \
  --port 8113 \
  --no-bench-profile \
  --skip-analyse \
  --package none \
  --run-label shixi_baseline


request_benchmark_results_27b_eager:
Run: atp_qwen3.6-27b_fl_enforce_eager_chunked_shixi_baseline_tp4_gmem0.6_20260831_092313
Model: qwen3.6
Model path: /models/Qwen3.6-27B
Mode: eager
CUDAGRAPH_MODE: N/A
Chunked prefill: true
Bench profile: false
Speculative config: 
Cases: 1024,1024,128
Created at: 2026-08-31 09:23:13

========== i1024_o1024_np128_c64 ==========
TIME=2026-08-31 09:47:04
CASE_LOG=/workspace/results/atp_qwen3.6-27b_fl_enforce_eager_chunked_shixi_baseline_tp4_gmem0.6_20260831_092313/torch_profile/i1024_o1024_np128_c64/request_terminal.log
============ Serving Benchmark Result ============
Successful requests:                     128       
Failed requests:                         0         
Maximum request concurrency:             64        
Benchmark duration (s):                  799.21    
Total input tokens:                      131072    
Total generated tokens:                  131072    
Request throughput (req/s):              0.16      
Output token throughput (tok/s):         164.00    
Peak output token throughput (tok/s):    320.00    
Peak concurrent requests:                66.00     
Total token throughput (tok/s):          328.00    
---------------Time to First Token----------------
Mean TTFT (ms):                          21920.85  
Median TTFT (ms):                        9929.60   
P99 TTFT (ms):                           76628.39  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          333.52    
Median TPOT (ms):                        327.89    
P99 TPOT (ms):                           370.00    
---------------Inter-token Latency----------------
Mean ITL (ms):                           333.19    
Median ITL (ms):                         230.00    
P99 ITL (ms):                            3352.56   
==================================================


27b_graph_脚本:
bash benchmark_script_fl.sh \
  --model-path /models/Qwen3.6-27B \
  --model-name qwen3.6 \
  --model-tag qwen3.6-27b \
  --mode graph \
  --cudagraph-mode PIECEWISE \

  --cases "1024,1024,128" \
  --concurrency 64 \
  --max-num-seqs 64 \
  --max-model-len 8192 \
  --tp 4 \
  --gmem 0.6 \
  --devices 0,1,2,3 \
  --port 8113 \
  --no-bench-profile \
  --skip-analyse \
  --package none \
  --run-label shixi_baseline

request_benchmark_results_27b_graph:
Run: atp_qwen3.6-27b_fl_graph_chunked_shixi_baseline_tp4_gmem0.6_20260831_152938
Model: qwen3.6
Model path: /models/Qwen3.6-27B
Mode: graph
CUDAGRAPH_MODE: FULL
Chunked prefill: true
Bench profile: false
Speculative config: 
Cases: 1024,1024,128
Created at: 2026-08-31 15:29:38

========== i1024_o1024_np128_c64 ==========
TIME=2026-08-31 15:41:33
CASE_LOG=/workspace/results/atp_qwen3.6-27b_fl_graph_chunked_shixi_baseline_tp4_gmem0.6_20260831_152938/torch_profile/i1024_o1024_np128_c64/request_terminal.log
============ Serving Benchmark Result ============
Successful requests:                     65        
Failed requests:                         63        
Maximum request concurrency:             64        
Benchmark duration (s):                  239.09    
Total input tokens:                      66560     
Total generated tokens:                  63689     
Request throughput (req/s):              0.27      
Output token throughput (tok/s):         266.38    
Peak output token throughput (tok/s):    448.00    
Peak concurrent requests:                65.00     
Total token throughput (tok/s):          544.77    
---------------Time to First Token----------------
Mean TTFT (ms):                          33936.49  
Median TTFT (ms):                        30123.23  
P99 TTFT (ms):                           76697.71  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          205.53    
Median TPOT (ms):                        209.45    
P99 TPOT (ms):                           237.52    
---------------Inter-token Latency----------------
Mean ITL (ms):                           200.38    
Median ITL (ms):                         159.27    
P99 ITL (ms):                            2682.30   
==================================================

35b_eager_脚本:
bash benchmark_script_fl.sh \
  --model-path /models/Qwen3.6-35B-A3B \
  --model-name qwen3.6 \
  --model-tag qwen3.6-35b-a3b \
  --mode eager \
  --cases "1024,1024,128" \
  --concurrency 64 \
  --max-num-seqs 64 \
  --max-model-len 8192 \
  --tp 4 \
  --gmem 0.6 \
  --devices 0,1,2,3 \
  --port 8113 \
  --no-bench-profile \
  --skip-analyse \
  --package none \
  --run-label shixi_baseline

request_benchmark_results_35b_eager:
Run: atp_qwen3.6-35b-a3b_fl_enforce_eager_chunked_shixi_baseline_tp4_gmem0.6_20260901_034435
Model: qwen3.6
Model path: /models/Qwen3.6-35B-A3B
Mode: eager
CUDAGRAPH_MODE: N/A
Chunked prefill: true
Bench profile: false
Speculative config: 
Cases: 1024,1024,128
Created at: 2026-09-01 03:44:35

========== i1024_o1024_np128_c64 ==========
TIME=2026-09-01 04:06:18
CASE_LOG=/workspace/results/atp_qwen3.6-35b-a3b_fl_enforce_eager_chunked_shixi_baseline_tp4_gmem0.6_20260901_034435/torch_profile/i1024_o1024_np128_c64/request_terminal.log
============ Serving Benchmark Result ============
Successful requests:                     128       
Failed requests:                         0         
Maximum request concurrency:             64        
Benchmark duration (s):                  707.79    
Total input tokens:                      131072    
Total generated tokens:                  131072    
Request throughput (req/s):              0.18      
Output token throughput (tok/s):         185.18    
Peak output token throughput (tok/s):    321.00    
Peak concurrent requests:                66.00     
Total token throughput (tok/s):          370.37    
---------------Time to First Token----------------
Mean TTFT (ms):                          21947.66  
Median TTFT (ms):                        8495.56   
P99 TTFT (ms):                           74534.91  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          296.37    
Median TPOT (ms):                        291.42    
P99 TPOT (ms):                           319.06    
---------------Inter-token Latency----------------
Mean ITL (ms):                           296.08    
Median ITL (ms):                         206.48    
P99 ITL (ms):                            2848.53   
==================================================

35b_graph_脚本：
bash benchmark_script_fl.sh \
  --model-path /models/Qwen3.6-35B-A3B \
  --model-name qwen3.6 \
  --model-tag qwen3.6-35b-a3b \
  --mode graph \
  --cudagraph-mode FULL \
  --cases "1024,1024,128" \
  --concurrency 64 \
  --max-num-seqs 64 \
  --max-model-len 8192 \
  --tp 4 \
  --gmem 0.6 \
  --devices 0,1,2,3 \
  --port 8113 \
  --no-bench-profile \
  --skip-analyse \
  --package none \
  --run-label shixi_baseline

request_benchmark_results_35b_graph:
========== i1024_o1024_np128_c64 ==========
TIME=2026-09-01 02:46:19
CASE_LOG=/workspace/results/atp_qwen3.6-35b-a3b_fl_graph_chunked_shixi_baseline_tp4_gmem0.6_20260901_023452/torch_profile/i1024_o1024_np128_c64/request_terminal.log
============ Serving Benchmark Result ============
Successful requests:                     65        
Failed requests:                         63        
Maximum request concurrency:             64        
Benchmark duration (s):                  222.67    
Total input tokens:                      66560     
Total generated tokens:                  61281     
Request throughput (req/s):              0.29      
Output token throughput (tok/s):         275.21    
Peak output token throughput (tok/s):    495.00    
Peak concurrent requests:                65.00     
Total token throughput (tok/s):          574.13    
---------------Time to First Token----------------
Mean TTFT (ms):                          35391.69  
Median TTFT (ms):                        32938.48  
P99 TTFT (ms):                           74620.85  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          337.57    
Median TPOT (ms):                        188.40    
P99 TPOT (ms):                           4225.96   
---------------Inter-token Latency----------------
Mean ITL (ms):                           182.45    
Median ITL (ms):                         144.97    
P99 ITL (ms):                            2377.90   
==================================================


================================================================================
附：2026-09-10 重测记录（ghl）—— 保留上文 2026-08-31~09-01 原始记录
================================================================================

说明
- 目的：按上文 4 个脚本在本容器环境重新测试，用于与原始记录对比（结合 week_report.md 第 4 周 Baseline 分析）。
- 环境：NPU 910B3 ×8（使用 0,1,2,3）；CANN 9.0.0（/usr/local/Ascend/cann-9.0.0）、驱动 26.0.rc1；
  vllm 0.13.0+empty、torch 2.8.0、torch_npu 2.8.0.post2、flag_gems 5.0.2；
  vllm-plugin-FL 仓库 /workspace/vllm-plugin-FL，commit a3644b2（+ 工作树改动）。
- 与上文命令的差异（仅此两处，其余参数完全一致）：
  ① --port 8113 → 8114（8113 被他人服务占用，8114~8139 空闲）；
  ② 两个 graph 脚本 --cudagraph-mode FULL → PIECEWISE（本环境 FULL 下 27B graph 无法完成，见下）。
  其余参数：TP4、gmem0.6、--cases "1024,1024,128"、并发 64、max-num-seqs 64、
  max-model-len 8192、--no-bench-profile --skip-analyse --package none、run-label shixi_baseline。
- 有效性判据（本次新增，避免瞬态错误被客户端误计为成功）：结果块含 Serving Benchmark Result 且
  Successful requests > 0 且 Total generated tokens > 10 且 Benchmark duration > 60s。

重测结果一览

| Baseline | 原始记录（08-31~09-01） | 2026-09-10 重测 |
|---|---|---|
| 27b_eager | 128/128；164.00 tok/s；TPOT median 327.89 ms | ✅ 128/128；288.20 tok/s；TPOT median 216.73 ms |
| 27b_graph | 65/128；266.38 tok/s；TPOT median 209.45 ms | ❌ 无有效结果（FULL 1 次瞬态异常+OOM；PIECEWISE 4 次均 OOM） |
| 35b_eager | 128/128；185.18 tok/s；TPOT median 291.42 ms | ✅ 128/128；245.88 tok/s；TPOT median 252.72 ms |
| 35b_graph | 65/128；275.21 tok/s；TPOT median 188.40 ms | ✅ 128/128；322.95 tok/s；TPOT median 192.32 ms（PIECEWISE） |

一、27b_eager（128/128 成功）

Run: atp_qwen3.6-27b_fl_enforce_eager_chunked_shixi_baseline_tp4_gmem0.6_20260910_005225
Model: qwen3.6
Model path: /models/Qwen3.6-27B
Mode: eager
CUDAGRAPH_MODE: N/A
Chunked prefill: true
Bench profile: false
Cases: 1024,1024,128
Devices: 0,1,2,3；Port: 8114；run-label: shixi_baseline
Created at: 2026-09-10 00:52:25

========== i1024_o1024_np128_c64 ==========
TIME=2026-09-10 01:08:27
============ Serving Benchmark Result ============
Successful requests:                     128       
Failed requests:                         0         
Maximum request concurrency:             64        
Benchmark duration (s):                  454.79    
Total input tokens:                      131072    
Total generated tokens:                  131072    
Request throughput (req/s):              0.28      
Output token throughput (tok/s):         288.20    
Peak output token throughput (tok/s):    384.00    
Peak concurrent requests:                68.00     
Total token throughput (tok/s):          576.40    
---------------Time to First Token----------------
Mean TTFT (ms):                          8301.68   
Median TTFT (ms):                        2885.42   
P99 TTFT (ms):                           27453.04  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          212.49    
Median TPOT (ms):                        216.73    
P99 TPOT (ms):                           219.22    
---------------Inter-token Latency----------------
Mean ITL (ms):                           212.28    
Median ITL (ms):                         195.96    
P99 ITL (ms):                            927.71    
==================================================

二、35b_eager（128/128 成功）

Run: atp_qwen3.6-35b-a3b_fl_enforce_eager_chunked_shixi_baseline_tp4_gmem0.6_20260910_011918
Model: qwen3.6
Model path: /models/Qwen3.6-35B-A3B
Mode: eager
CUDAGRAPH_MODE: N/A
Chunked prefill: true
Bench profile: false
Cases: 1024,1024,128
Devices: 0,1,2,3；Port: 8114；run-label: shixi_baseline
Created at: 2026-09-10 01:19:18

========== i1024_o1024_np128_c64 ==========
TIME=2026-09-10 01:36:57
============ Serving Benchmark Result ============
Successful requests:                     128       
Failed requests:                         0         
Maximum request concurrency:             64        
Benchmark duration (s):                  533.06    
Total input tokens:                      131072    
Total generated tokens:                  131072    
Request throughput (req/s):              0.24      
Output token throughput (tok/s):         245.88    
Peak output token throughput (tok/s):    358.00    
Peak concurrent requests:                66.00     
Total token throughput (tok/s):          491.77    
---------------Time to First Token----------------
Mean TTFT (ms):                          15289.52  
Median TTFT (ms):                        4783.67   
P99 TTFT (ms):                           50048.16  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          243.83    
Median TPOT (ms):                        252.72    
P99 TPOT (ms):                           255.67    
---------------Inter-token Latency----------------
Mean ITL (ms):                           243.60    
Median ITL (ms):                         212.27    
P99 ITL (ms):                            1580.97   
==================================================

三、35b_graph（PIECEWISE，128/128 成功）

Run: atp_qwen3.6-35b-a3b_fl_graph_chunked_shixi_baseline_tp4_gmem0.6_20260910_025902
Model: qwen3.6
Model path: /models/Qwen3.6-35B-A3B
Mode: graph
CUDAGRAPH_MODE: PIECEWISE（与上文的 FULL 不同）
Chunked prefill: true
Bench profile: false
Cases: 1024,1024,128
Devices: 0,1,2,3；Port: 8114；run-label: shixi_baseline
Created at: 2026-09-10 02:59:02

========== i1024_o1024_np128_c64 ==========
TIME=2026-09-10 03:13:01
============ Serving Benchmark Result ============
Successful requests:                     128       
Failed requests:                         0         
Maximum request concurrency:             64        
Benchmark duration (s):                  405.86    
Total input tokens:                      131072    
Total generated tokens:                  131072    
Request throughput (req/s):              0.32      
Output token throughput (tok/s):         322.95    
Peak output token throughput (tok/s):    448.00    
Peak concurrent requests:                66.00     
Total token throughput (tok/s):          645.89    
---------------Time to First Token----------------
Mean TTFT (ms):                          14407.74  
Median TTFT (ms):                        4588.56   
P99 TTFT (ms):                           47566.86  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          183.25    
Median TPOT (ms):                        192.32    
P99 TPOT (ms):                           194.62    
---------------Inter-token Latency----------------
Mean ITL (ms):                           183.07    
Median ITL (ms):                         152.76    
P99 ITL (ms):                            1493.48   
==================================================

四、27b_graph（未取得有效结果，如实记录）

- FULL 1 次：Run atp_qwen3.6-27b_fl_graph_chunked_shixi_baseline_tp4_gmem0.6_20260910_011041
  server 正常启动、warmup 通过；正式 bench 阶段 2026-09-10 01:16:58 出现 NPU OOM，
  并伴随 vector core exception（错误码 507035）级联报错。客户端把立即失败的 64 个请求计为
  Successful（Benchmark duration 4.69s、Total generated tokens=1、Output token throughput 0.21 tok/s），
  按上述严格判据判为无效。
- PIECEWISE 4 次：
  atp_..._graph_chunked_shixi_baseline_tp4_gmem0.6_20260910_015126（第 1 次）
  atp_..._graph_chunked_shixi_baseline_tp4_gmem0.6_20260910_020823（第 2 次）
  atp_..._graph_chunked_shixi_baseline_tp4_gmem0.6_20260910_022514（第 3 次）
  atp_..._graph_chunked_shixi_baseline_tp4_gmem0.6_20260910_024210（第 4 次）
  4 次均在正式 bench 的 cudagraph capture 阶段 OOM，报错特征一致：
  RuntimeError: NPU out of memory. Tried to allocate 650.00 MiB（60.96 GiB 总容量；
  56.18 GiB already allocated；约 23 MiB free；56.23 GiB reserved in total by PyTorch），无有效结果。
- 参考：同环境 27b_eager 可 128/128 完成；27B 每卡 model loading 12.87 GiB、Available KV cache 20.30 GiB。
- 结论：27B graph 在本环境（TP4 / gmem0.6 / max-model-len 8192 / 128 请求并发 64）
  FULL 与 PIECEWISE 均无法完成；原始 08-31 的 27b_graph 记录（65/128）在本环境不可复现。

备注
- 27B/35B eager 重测值均优于原始记录（27B 288.20 vs 164.00 tok/s、TPOT median 216.73 vs 327.89 ms；
  35B 245.88 vs 185.18 tok/s、TPOT median 252.72 vs 291.42 ms），与当日卡负载/调度差异有关，仅供方向参考。
- 35b_graph 由 FULL 改为 PIECEWISE 后首次即 128/128 成功（原始 FULL 记录为 65/128），
  稳定性明显改善；因 cudagraph 模式不同，吞吐与原始 FULL 记录不宜作严格对比。
- 原始数据说明：本次 raw run 目录（结果原文、run_info.env、服务/客户端日志）已按用户要求于
  2026-09-10 全部删除，上文 run 名仅作历史标识；跑批脚本保留于 /workspace/scripts/ghl/
  （rerun_baseline_20260910.sh、rerun_graphs_20260910.sh、rerun_graphs_piecewise_20260910.sh）。