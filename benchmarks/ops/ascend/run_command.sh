#!/bin/bash
# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Reproducing the four Qwen3.6 baselines on Ascend 910B3 (TP=4).
# Every command below is the exact one used for the archived results under
# benchmark_results/ (09-15 batch). Run them one by one on an idle 4-card group.

FL_REPO_PATH="${FL_REPO_PATH:-/workspace/vllm-plugin-FL}"
BENCH="${BENCH:-/workspace/scripts/benchmark_script_fl.sh}"
COMMON_ARGS="--model-name qwen3.6 --cases 1024,1024,128 --concurrency 64 \
--max-num-seqs 64 --max-model-len 8192 --tp 4 --gmem 0.6 \
--no-bench-profile --skip-analyse --package none --run-label shixi_baseline \
--fl-repo-path ${FL_REPO_PATH}"

# 27B / graph (PIECEWISE) -> Output tok/s 490.34
"${BENCH}" --model-path /models/Qwen3.6-27B --model-tag qwen3.6-27b \
  --mode graph --devices 4,5,6,7 --port 8113 ${COMMON_ARGS}

# 35B-A3B / graph (PIECEWISE) -> Output tok/s 326.11
"${BENCH}" --model-path /models/Qwen3.6-35B-A3B --model-tag qwen3.6-35b-a3b \
  --mode graph --devices 0,1,2,3 --port 8113 ${COMMON_ARGS}

# 27B / eager -> Output tok/s 293.21
"${BENCH}" --model-path /models/Qwen3.6-27B --model-tag qwen3.6-27b \
  --mode eager --devices 0,1,2,3 --port 8113 ${COMMON_ARGS}

# 35B-A3B / eager -> Output tok/s 268.38
"${BENCH}" --model-path /models/Qwen3.6-35B-A3B --model-tag qwen3.6-35b-a3b \
  --mode eager --devices 0,1,2,3 --port 8113 ${COMMON_ARGS}

# Aggregate the four RUN_DIRs into one CSV + Markdown table:
#   python3 benchmarks/ops/ascend/analysis.py --result-root /workspace/results
