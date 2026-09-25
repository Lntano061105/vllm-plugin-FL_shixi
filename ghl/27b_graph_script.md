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