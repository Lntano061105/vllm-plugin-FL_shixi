# Ascend Op Benchmark Summary

| run | model | mode | case | successful | failed | output_tok_s | total_tok_s | mean_ttft_ms | mean_tpot_ms |
|---|---|---|---|---|---|---|---|---|---|
| atp_qwen3.6-27b_fl_enforce_eager_chunked_shixi_baseline_pw_tp4_gmem0.6_20260911_103545 | qwen3.6 | eager | i1024_o1024_np128_c64 | 128 | 0 | 293.21 | 586.41 | 7821.11 | 209.24 |
| atp_qwen3.6-27b_fl_graph_chunked_shixi_baseline_pw_tp4_gmem0.6_20260915_115604 | qwen3.6 | graph | i1024_o1024_np128_c64 | 128 | 0 | 490.34 | 980.68 | 8441.34 | 121.33 |
| atp_qwen3.6-35b-a3b_fl_enforce_eager_chunked_shixi_baseline_pw_tp4_gmem0.6_20260911_105227 | qwen3.6 | eager | i1024_o1024_np128_c64 | 128 | 0 | 268.38 | 536.76 | 14831.91 | 222.60 |
| atp_qwen3.6-35b-a3b_fl_graph_chunked_shixi_baseline_pw_tp4_gmem0.6_20260911_102049 | qwen3.6 | graph | i1024_o1024_np128_c64 | 128 | 0 | 326.11 | 652.22 | 13842.02 | 181.88 |
