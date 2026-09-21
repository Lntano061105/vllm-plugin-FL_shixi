#!/bin/bash
set -euo pipefail
umask 0022

ORIGINAL_ARGS=("$@")

usage() {
  cat <<USAGE
Usage:
  $0 [options]

Core:
  --model-path PATH              Default: /models/Qwen3.6-27B
  --model-name NAME              Default: qwen3.6
  --model-tag TAG                Default: basename(model-path), lowercase
  --mode graph|eager             Default: graph
  --cudagraph-mode FULL|FULL_DECODE_ONLY|PIECEWISE
                                  Graph mode cudagraph setting. Default: FULL
  --chunked true|false           Default: true
  --cases "I,O,NP;I,O,NP"        Default: "32,32,1;32,128,1;128,128,1"
  --concurrency N                Formal bench max concurrency. Default: 1
  --bench-profile true|false     Whether to pass --profile to vllm bench. Default: true
  --profile true|false           Alias of --bench-profile
  --no-bench-profile             Alias of --bench-profile false
  --attention-backend BACKEND|auto
                                  Export VLLM_ATTENTION_BACKEND. Default: TORCH_SDPA
  --run-label LABEL              Extra label in output dir

Serve:
  --port PORT                    Default: 8080
  --tp N                         Default: 4
  --gmem FLOAT                   Default: 0.6
  --max-model-len N              Default: 32768
  --max-num-seqs N               Default: 1
  --speculative-config JSON      Optional vLLM speculative decoding config
  --mtp N                        Shortcut to enable MTP with N speculative tokens
  --devices IDS                  Default: 0,1,2,3
  --fl-repo-path PATH            vllm-plugin-FL source path. Default: /workspace/vllm-plugin-FL

Warmup:
  --warmup "I,O,C,NP"            Default: "128,128,2,4"
  --no-warmup

Output:
  --result-root PATH             Default: /workspace/results
  --package tar.gz|tar|none      Default: tar.gz
  --skip-analyse

Examples:
  $0 --mode graph --cases "1024,2,1;2,1024,1;1024,1024,1" --run-label longcase_np
  $0 --mode eager --chunked false --cases "1024,2,1;2,1024,1;1024,1024,1"
  $0 --mode graph --mtp 1 --cases "128,128,1" --run-label mtp1
USAGE
}

MODEL_PATH="/models/Qwen3.6-27B"
MODEL_NAME="qwen3.6"
MODEL_TAG=""
MODE="graph"
CUDAGRAPH_MODE="FULL"
CHUNKED="true"
CASES="32,32,1;32,128,1;128,128,1"
CASE_CONCURRENCY="1"
BENCH_PROFILE="true"
ATTENTION_BACKEND="TORCH_SDPA"
RUN_LABEL=""
PORT="8080"
TP="4"
GMEM="0.6"
MAX_MODEL_LEN="32768"
MAX_NUM_SEQS="1"
SPECULATIVE_CONFIG=""
NUM_SPEC_TOKENS=""
DEVICES="0,1,2,3"
FL_REPO_PATH="/workspace/vllm-plugin-FL"
WARMUP="128,128,2,4"
RESULT_ROOT="/workspace/results"
PACKAGE_MODE="tar.gz"
SKIP_ANALYSE="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --model-name) MODEL_NAME="$2"; shift 2 ;;
    --model-tag) MODEL_TAG="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --cudagraph-mode) CUDAGRAPH_MODE="$2"; shift 2 ;;
    --chunked) CHUNKED="$2"; shift 2 ;;
    --cases) CASES="$2"; shift 2 ;;
    --concurrency) CASE_CONCURRENCY="$2"; shift 2 ;;
    --bench-profile) BENCH_PROFILE="$2"; shift 2 ;;
    --profile) BENCH_PROFILE="$2"; shift 2 ;;
    --no-bench-profile|--no-profile) BENCH_PROFILE="false"; shift ;;
    --attention-backend) ATTENTION_BACKEND="$2"; shift 2 ;;
    --run-label) RUN_LABEL="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --tp) TP="$2"; shift 2 ;;
    --gmem) GMEM="$2"; shift 2 ;;
    --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
    --max-num-seqs) MAX_NUM_SEQS="$2"; shift 2 ;;
    --speculative-config) SPECULATIVE_CONFIG="$2"; shift 2 ;;
    --mtp) NUM_SPEC_TOKENS="$2"; shift 2 ;;
    --devices) DEVICES="$2"; shift 2 ;;
    --fl-repo-path) FL_REPO_PATH="$2"; shift 2 ;;
    --warmup) WARMUP="$2"; shift 2 ;;
    --no-warmup) WARMUP="none"; shift ;;
    --result-root) RESULT_ROOT="$2"; shift 2 ;;
    --package) PACKAGE_MODE="$2"; shift 2 ;;
    --skip-analyse) SKIP_ANALYSE="true"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] Unknown option: $1"; usage; exit 1 ;;
  esac
done

if [[ "${MODE}" != "graph" && "${MODE}" != "eager" ]]; then
  echo "[ERROR] --mode must be graph or eager"
  exit 1
fi

case "${CUDAGRAPH_MODE}" in
  FULL|FULL_DECODE_ONLY|PIECEWISE) ;;
  *)
    echo "[ERROR] --cudagraph-mode must be FULL, FULL_DECODE_ONLY, or PIECEWISE"
    exit 1
    ;;
esac

if [[ "${CHUNKED}" != "true" && "${CHUNKED}" != "false" ]]; then
  echo "[ERROR] --chunked must be true or false"
  exit 1
fi

if [[ "${BENCH_PROFILE}" != "true" && "${BENCH_PROFILE}" != "false" ]]; then
  echo "[ERROR] --bench-profile/--profile must be true or false"
  exit 1
fi

if [[ "${PACKAGE_MODE}" != "tar.gz" && "${PACKAGE_MODE}" != "tar" && "${PACKAGE_MODE}" != "none" ]]; then
  echo "[ERROR] --package must be tar.gz, tar, or none"
  exit 1
fi

if [[ -n "${NUM_SPEC_TOKENS}" ]]; then
  if [[ -n "${SPECULATIVE_CONFIG}" ]]; then
    echo "[ERROR] --mtp cannot be used together with --speculative-config"
    exit 1
  fi
  if ! [[ "${NUM_SPEC_TOKENS}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] --mtp requires a positive integer"
    exit 1
  fi
  SPECULATIVE_CONFIG="{\"method\":\"mtp\",\"num_speculative_tokens\":${NUM_SPEC_TOKENS},\"model\":\"${MODEL_PATH}\"}"
fi

TS=$(date +%Y%m%d_%H%M%S)
SCRIPT_PATH="$(readlink -f "$0")"
SCRIPT_NAME="$(basename "${SCRIPT_PATH}")"

if [[ -z "${MODEL_TAG}" ]]; then
  MODEL_TAG="$(basename "${MODEL_PATH}" | tr '[:upper:]' '[:lower:]')"
fi

MODE_TAG="${MODE}"
if [[ "${MODE}" == "eager" ]]; then
  MODE_TAG="enforce_eager"
fi

CUDAGRAPH_MODE_DISPLAY="${CUDAGRAPH_MODE}"
if [[ "${MODE}" == "eager" ]]; then
  CUDAGRAPH_MODE_DISPLAY="N/A"
fi

CHUNK_TAG="chunked"
if [[ "${CHUNKED}" == "false" ]]; then
  CHUNK_TAG="nochunk"
fi

RUN_TAG="fl_${MODE_TAG}_${CHUNK_TAG}"
if [[ -n "${RUN_LABEL}" ]]; then
  RUN_TAG="${RUN_TAG}_${RUN_LABEL}"
fi

RUN_NAME="atp_${MODEL_TAG}_${RUN_TAG}_tp${TP}_gmem${GMEM}_${TS}"
RUN_DIR="${RESULT_ROOT}/${RUN_NAME}"
PROFILE_ROOT="${RUN_DIR}/torch_profile"
SERVER_LOG="${RUN_DIR}/server.log"
CLIENT_LOG="${RUN_DIR}/client.log"
REQUEST_BENCH_SUMMARY="${RUN_DIR}/request_benchmark_results.txt"
ANALYSIS_LOG="${RUN_DIR}/analysis_terminal.log"

case "${PACKAGE_MODE}" in
  tar.gz) PACKAGE_PATH="${RESULT_ROOT}/${RUN_NAME}.tar.gz" ;;
  tar) PACKAGE_PATH="${RESULT_ROOT}/${RUN_NAME}.tar" ;;
  none) PACKAGE_PATH="" ;;
esac

TOKENIZER_PATH="${MODEL_PATH}"
HOST="127.0.0.1"
SERVE_HOST="0.0.0.0"
BACKEND="openai-chat"
ENDPOINT="/v1/chat/completions"

mkdir -p "${PROFILE_ROOT}"
echo "${RUN_DIR}" > "${RESULT_ROOT}/latest_run_dir.txt"

{
  printf "%q " "${SCRIPT_PATH}" "${ORIGINAL_ARGS[@]}"
  echo
} > "${RUN_DIR}/run_command.sh"

cat > "${RUN_DIR}/run_info.env" <<INFO
RUN_NAME=${RUN_NAME}
RUN_DIR=${RUN_DIR}
MODEL_PATH=${MODEL_PATH}
SERVED_MODEL_NAME=${MODEL_NAME}
MODEL_TAG=${MODEL_TAG}
BACKEND=${BACKEND}
PLATFORM=ascend
MODE=${MODE}
CUDAGRAPH_MODE=${CUDAGRAPH_MODE_DISPLAY}
CHUNKED=${CHUNKED}
CASES=${CASES}
CASE_CONCURRENCY=${CASE_CONCURRENCY}
BENCH_PROFILE=${BENCH_PROFILE}
ATTENTION_BACKEND=${ATTENTION_BACKEND}
WARMUP=${WARMUP}
TP=${TP}
GPU_MEMORY_UTILIZATION=${GMEM}
MAX_MODEL_LEN=${MAX_MODEL_LEN}
MAX_NUM_SEQS=${MAX_NUM_SEQS}
SPECULATIVE_CONFIG=${SPECULATIVE_CONFIG}
PORT=${PORT}
DEVICES=${DEVICES}
FL_REPO_PATH=${FL_REPO_PATH}
PROFILE_ROOT=${PROFILE_ROOT}
SERVER_LOG=${SERVER_LOG}
CLIENT_LOG=${CLIENT_LOG}
REQUEST_BENCH_SUMMARY=${REQUEST_BENCH_SUMMARY}
PACKAGE_MODE=${PACKAGE_MODE}
PACKAGE_PATH=${PACKAGE_PATH}
SCRIPT_NAME=${SCRIPT_NAME}
SCRIPT_PATH=${SCRIPT_PATH}
INFO

cat > "${RUN_DIR}/run_summary.txt" <<INFO
Run: ${RUN_NAME}
Server log: ${SERVER_LOG}
Client log: ${CLIENT_LOG}
Request benchmark summary: ${REQUEST_BENCH_SUMMARY}
Torch profile root: ${PROFILE_ROOT}
Analysis log: ${ANALYSIS_LOG}
Package: ${PACKAGE_PATH}
Script: ${SCRIPT_NAME}
Command: ${RUN_DIR}/run_command.sh
INFO

cp "${SCRIPT_PATH}" "${RUN_DIR}/${SCRIPT_NAME}"
touch "${RUN_DIR}/analysis.py"

cat > "${REQUEST_BENCH_SUMMARY}" <<INFO
Run: ${RUN_NAME}
Model: ${MODEL_NAME}
Model path: ${MODEL_PATH}
Mode: ${MODE}
CUDAGRAPH_MODE: ${CUDAGRAPH_MODE_DISPLAY}
Chunked prefill: ${CHUNKED}
Bench profile: ${BENCH_PROFILE}
Speculative config: ${SPECULATIVE_CONFIG}
Cases: ${CASES}
Created at: $(date '+%Y-%m-%d %H:%M:%S')
INFO

exec > >(tee -a "${CLIENT_LOG}") 2>&1

export VLLM_PLUGINS=fl
export VLLM_FL_PLATFORM=ascend
export GEMS_VENDOR=ascend
if [[ "${ATTENTION_BACKEND}" != "auto" && "${ATTENTION_BACKEND}" != "none" && -n "${ATTENTION_BACKEND}" ]]; then
  export VLLM_ATTENTION_BACKEND="${ATTENTION_BACKEND}"
else
  unset VLLM_ATTENTION_BACKEND
fi
if [[ -d "${FL_REPO_PATH}" ]]; then
  export PYTHONPATH="${FL_REPO_PATH}:${PYTHONPATH:-}"
fi
export TRITON_ALL_BLOCKS_PARALLEL=1
export TASK_QUEUE_ENABLE=1
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export HCCL_BUFFSIZE=1024
export HCCL_OP_EXPANSION_MODE="AIV"
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export VLLM_FL_FLAGOS_WHITELIST="unquantized_fused_moe_method,topk_softmax"
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:${LD_PRELOAD:-}
export ASCEND_RT_VISIBLE_DEVICES="${DEVICES}"

check_fl_plugin() {
  echo ""
  echo "========== CHECK VLLM FL PLUGIN =========="
  echo "FL_REPO_PATH=${FL_REPO_PATH}"
  echo "VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-<unset>}"
  echo "PYTHONPATH=${PYTHONPATH:-}"

  python - <<'PY'
import importlib
from importlib.metadata import entry_points

module = importlib.import_module("vllm_fl")
module_path = getattr(module, "__file__", None)
plugin_entry_points = []

try:
    all_entry_points = entry_points()
    if hasattr(all_entry_points, "select"):
        candidates = all_entry_points
    else:
        candidates = [ep for eps in all_entry_points.values() for ep in eps]

    for entry_point in candidates:
        value = getattr(entry_point, "value", "")
        group = getattr(entry_point, "group", "")
        if group.startswith("vllm.") and value.startswith("vllm_fl:"):
            plugin_entry_points.append(entry_point)
except Exception as exc:
    print(f"[WARN] Failed to inspect Python entry points: {exc}")

print(f"vllm_fl module: {module_path}")

missing = []
for entry_point in plugin_entry_points:
    attr = entry_point.value.split(":", 1)[1]
    exists = hasattr(module, attr)
    print(
        "entry point "
        f"{entry_point.group}:{entry_point.name} -> {entry_point.value}; "
        f"exists: {exists}"
    )
    if not exists:
        missing.append(attr)

if not plugin_entry_points:
    print("[WARN] No vLLM entry point targeting vllm_fl was found.")

if missing:
    raise SystemExit(
        "[ERROR] vllm_fl is imported, but required vLLM plugin entry point "
        f"function(s) are missing: {sorted(set(missing))}. "
        "Check whether PYTHONPATH points to the correct vllm-plugin-FL source, "
        "or reinstall a vllm-plugin-FL version matching this vLLM build."
    )
PY
}

check_port_available() {
  python - "${PORT}" <<'PY'
import socket
import sys

port = int(sys.argv[1])
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.bind(("0.0.0.0", port))
except OSError as exc:
    raise SystemExit(
        f"[ERROR] Port {port} is already in use before starting vLLM. "
        "Use a different --port or stop the existing server first. "
        f"Bind error: {exc}"
    )
finally:
    sock.close()
PY
}

SERVER_PID=""

parse_warmup() {
  if [[ "${WARMUP}" == "none" ]]; then
    return
  fi

  local clean="${WARMUP//[[:space:]]/}"
  IFS=',' read -r WARMUP_ILEN WARMUP_OLEN WARMUP_CONCURRENCY WARMUP_NUM_PROMPTS extra <<< "${clean}"

  if [[ -z "${WARMUP_ILEN:-}" || -z "${WARMUP_OLEN:-}" || -z "${WARMUP_CONCURRENCY:-}" || -z "${WARMUP_NUM_PROMPTS:-}" || -n "${extra:-}" ]]; then
    echo "[ERROR] --warmup format must be input,output,concurrency,num_prompts"
    exit 1
  fi
}

parse_warmup

stop_server() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "[STOP_SERVER] pid=${SERVER_PID}" | tee -a "${SERVER_LOG}"
    kill "${SERVER_PID}" 2>/dev/null || true
    sleep 5
    if kill -0 "${SERVER_PID}" 2>/dev/null; then
      kill -9 "${SERVER_PID}" 2>/dev/null || true
    fi
  fi
  SERVER_PID=""
}

trap stop_server EXIT

start_server() {
  echo "RUN_DIR=${RUN_DIR}"
  echo "SERVER_LOG=${SERVER_LOG}"
  echo "PROFILE_ROOT=${PROFILE_ROOT}"

  check_fl_plugin
  check_port_available

  local profiler_config
  profiler_config="{\"profiler\":\"torch\",\"torch_profiler_dir\":\"${PROFILE_ROOT}\",\"torch_profiler_with_stack\":false,\"torch_profiler_record_shapes\":false,\"torch_profiler_with_memory\":false,\"torch_profiler_with_flops\":false,\"torch_profiler_use_gzip\":true}"

  SERVE_ARGS=(
    vllm serve "${MODEL_PATH}"
    --served-model-name "${MODEL_NAME}"
    --trust-remote-code
    --tensor-parallel-size "${TP}"
    --host "${SERVE_HOST}"
    --port "${PORT}"
    --gpu-memory-utilization "${GMEM}"
  )

  if [[ "${MODE}" == "graph" ]]; then
    # Ascend uses the eager compiler adapter for graph capture. TorchInductor
    # does not provide an NPU reduction scheduler and fails while lowering
    # operations such as aten.mean.dim. Disable vLLM custom ops because the
    # current FL dispatcher calls ContextVar.get(), which Dynamo cannot trace.
    SERVE_ARGS+=(--compilation-config "{\"cudagraph_mode\": \"${CUDAGRAPH_MODE}\", \"backend\": \"eager\", \"custom_ops\": [\"none\"], \"pass_config\": {\"fuse_norm_quant\": false, \"fuse_act_quant\": false, \"fuse_attn_quant\": false, \"enable_sp\": false, \"fuse_gemm_comms\": false, \"fuse_allreduce_rms\": false}}")
  else
    SERVE_ARGS+=(--enforce-eager)
  fi

  if [[ "${CHUNKED}" == "true" ]]; then
    SERVE_ARGS+=(--enable-chunked-prefill)
  else
    SERVE_ARGS+=(--no-enable-chunked-prefill)
  fi

  SERVE_ARGS+=(
    --no-enable-prefix-caching
    --max-model-len "${MAX_MODEL_LEN}"
    --max-num-seqs "${MAX_NUM_SEQS}"
    --allowed-local-media-path /
    --mm-processor-cache-gb 0
    --profiler-config "${profiler_config}"
  )

  if [[ -n "${SPECULATIVE_CONFIG}" ]]; then
    SERVE_ARGS+=(--speculative-config "${SPECULATIVE_CONFIG}")
  fi

  "${SERVE_ARGS[@]}" > >(tee -a "${SERVER_LOG}") 2>&1 &

  SERVER_PID=$!
  echo "SERVER_PID=${SERVER_PID}" | tee -a "${RUN_DIR}/run_info.env"

  echo "[WAIT_SERVER] http://${HOST}:${PORT}/v1/models"
  for i in $(seq 1 180); do
    if curl -fs "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1; then
      echo "[SERVER_READY]"
      return
    fi

    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      echo "[ERROR] vLLM server exited early. Check ${SERVER_LOG}"
      exit 1
    fi

    sleep 2

    if [[ "${i}" == "180" ]]; then
      echo "[ERROR] server not ready after 360s. Check ${SERVER_LOG}"
      exit 1
    fi
  done
}

warmup_once() {
  if [[ "${WARMUP}" == "none" ]]; then
    return
  fi

  local warmup_log="${RUN_DIR}/warmup_terminal.log"

  {
    echo ""
    echo "========== WARMUP =========="
    echo "TIME=$(date '+%Y-%m-%d %H:%M:%S')"
    echo "[WARMUP] input=${WARMUP_ILEN}, output=${WARMUP_OLEN}, concurrency=${WARMUP_CONCURRENCY}, prompts=${WARMUP_NUM_PROMPTS}"

    vllm bench serve \
      --backend "${BACKEND}" \
      --model "${MODEL_NAME}" \
      --tokenizer "${TOKENIZER_PATH}" \
      --endpoint "${ENDPOINT}" \
      --host "${HOST}" \
      --port "${PORT}" \
      --dataset-name random \
      --random-input-len "${WARMUP_ILEN}" \
      --random-output-len "${WARMUP_OLEN}" \
      --max-concurrency "${WARMUP_CONCURRENCY}" \
      --num-prompts "${WARMUP_NUM_PROMPTS}" \
      --ignore-eos

    echo "[WARMUP_DONE]"
    echo "TIME=$(date '+%Y-%m-%d %H:%M:%S')"
  } 2>&1 | tee -a "${warmup_log}"

  sleep 3
}


append_benchmark_result_summary() {
  local case_name="$1"
  local case_log="$2"

  {
    echo ""
    echo "========== ${case_name} =========="
    echo "TIME=$(date '+%Y-%m-%d %H:%M:%S')"
    echo "CASE_LOG=${case_log}"

    if grep -q "Serving Benchmark Result" "${case_log}"; then
      awk '
        /Serving Benchmark Result/ {capture=1}
        capture {print}
        capture && /^=+$/ {capture=0}
      ' "${case_log}"
    else
      echo "[WARN] Serving Benchmark Result not found in ${case_log}"
    fi
  } >> "${REQUEST_BENCH_SUMMARY}"
}

collect_case_profile_outputs() {
  local case_dir="$1"

  if [[ "${BENCH_PROFILE}" != "true" ]]; then
    return
  fi

  sleep 2

  find "${PROFILE_ROOT}" -mindepth 1 -maxdepth 1 \
    ! -name "$(basename "${case_dir}")" \
    ! -name "i*_o*" \
    ! -name ".current_case_dir" \
    -exec mv -t "${case_dir}" {} + 2>/dev/null || true
}

run_case() {
  local ilen="$1"
  local olen="$2"
  local num_prompts="$3"

  local case_name="i${ilen}_o${olen}_np${num_prompts}_c${CASE_CONCURRENCY}"
  local case_dir="${PROFILE_ROOT}/${case_name}"
  local case_log="${case_dir}/request_terminal.log"

  mkdir -p "${case_dir}"

  {
    echo ""
    echo "========== ${case_name} =========="
    echo "TIME=$(date '+%Y-%m-%d %H:%M:%S')"
    echo "CASE_DIR=${case_dir}"

    echo "[PROFILE_CASE_DIR] ${case_dir}"
    echo "${case_dir}" > "${PROFILE_ROOT}/.current_case_dir"

    echo "[BENCH] input=${ilen}, output=${olen}, concurrency=${CASE_CONCURRENCY}, prompts=${num_prompts}, profile=${BENCH_PROFILE}"

    BENCH_ARGS=(
      vllm bench serve
      --backend "${BACKEND}"
      --model "${MODEL_NAME}"
      --tokenizer "${TOKENIZER_PATH}"
      --endpoint "${ENDPOINT}"
      --host "${HOST}"
      --port "${PORT}"
      --dataset-name random
      --random-input-len "${ilen}"
      --random-output-len "${olen}"
      --max-concurrency "${CASE_CONCURRENCY}"
      --num-prompts "${num_prompts}"
      --ignore-eos
    )

    if [[ "${BENCH_PROFILE}" == "true" ]]; then
      BENCH_ARGS+=(--profile)
    fi

    "${BENCH_ARGS[@]}"

    collect_case_profile_outputs "${case_dir}"

    echo "[PROFILE_DONE]"
    rm -f "${PROFILE_ROOT}/.current_case_dir"

    echo "[DONE] ${case_name}"
    echo "TIME=$(date '+%Y-%m-%d %H:%M:%S')"
  } 2>&1 | tee -a "${case_log}"

  append_benchmark_result_summary "${case_name}" "${case_log}"

  sleep 5
}

run_all_cases() {
  IFS=';' read -ra CASE_SPECS <<< "${CASES}"

  for spec in "${CASE_SPECS[@]}"; do
    spec="${spec//[[:space:]]/}"
    [[ -z "${spec}" ]] && continue

    IFS=',' read -r ilen olen num_prompts extra <<< "${spec}"

    if [[ -z "${ilen:-}" || -z "${olen:-}" || -z "${num_prompts:-}" || -n "${extra:-}" ]]; then
      echo "[ERROR] case format must be input,output,num_prompts. Got: ${spec}"
      exit 1
    fi

    run_case "${ilen}" "${olen}" "${num_prompts}"
  done
}

analyse_profile() {
  if [[ "${SKIP_ANALYSE}" == "true" || "${BENCH_PROFILE}" == "false" ]]; then
    echo "[SKIP_ANALYSE]"
    return
  fi

  {
    echo ""
    echo "========== ANALYSE TORCH NPU PROFILE =========="
    echo "TIME=$(date '+%Y-%m-%d %H:%M:%S')"
    echo "PROFILE_ROOT=${PROFILE_ROOT}"

    python - <<PY
import glob
import os
from torch_npu.profiler.profiler import analyse

profile_root = "${PROFILE_ROOT}"
case_dirs = sorted(glob.glob(os.path.join(profile_root, "i*_o*")))
targets = case_dirs or [profile_root]

for target in targets:
    print(f"[ANALYSE] {target}")
    analyse(target)

print("[ANALYSE_DONE]")
PY

    echo "TIME=$(date '+%Y-%m-%d %H:%M:%S')"
  } 2>&1 | tee -a "${ANALYSIS_LOG}"
}

fix_permissions() {
  echo ""
  echo "========== FIX RESULT PERMISSIONS =========="
  chmod -R a+rX "${RUN_DIR}" || true
}

package_results() {
  if [[ "${PACKAGE_MODE}" == "none" ]]; then
    echo "[SKIP_PACKAGE]"
    return
  fi

  echo ""
  echo "========== PACKAGE SELECTED PROFILER OUTPUTS =========="

  local package_list="${RUN_DIR}/package_filelist.txt"
  : > "${package_list}"

  (
    cd "${RESULT_ROOT}"

    find "${RUN_NAME}/torch_profile" -type f -name "request_terminal.log" >> "${package_list}" 2>/dev/null || true

    if [[ -f "${RUN_NAME}/request_benchmark_results.txt" ]]; then
      echo "${RUN_NAME}/request_benchmark_results.txt" >> "${package_list}"
    fi

    for filename in \
      analysis.db \
      api_statistic.csv \
      ascend_pytorch_profiler_0.db \
      communication.json \
      communication_matrix.json \
      kernel_details.csv \
      op_statistic.csv \
      operator_details.csv \
      trace_view.json
    do
      find "${RUN_NAME}" -type f -path "*/ASCEND_PROFILER_OUTPUT/${filename}" >> "${package_list}" 2>/dev/null || true
    done

    sort -u "${package_list}" -o "${package_list}"

    echo "[PACKAGE_FILE_COUNT] $(wc -l < "${package_list}")"
    cat "${package_list}"

    if [[ ! -s "${package_list}" ]]; then
      echo "[ERROR] No files selected for package."
      exit 1
    fi

    if [[ "${PACKAGE_MODE}" == "tar.gz" ]]; then
      tar -czf "${PACKAGE_PATH}" -T "${package_list}"
    else
      tar -cf "${PACKAGE_PATH}" -T "${package_list}"
    fi
  )

  chmod a+r "${PACKAGE_PATH}" || true
  echo "PACKAGE_PATH=${PACKAGE_PATH}" | tee -a "${RUN_DIR}/run_info.env"
  echo "PACKAGE_PATH=${PACKAGE_PATH}"
}

start_server
warmup_once
run_all_cases

stop_server
trap - EXIT

analyse_profile
fix_permissions
package_results

echo ""
echo "========== ALL DONE =========="
echo "RUN_DIR=${RUN_DIR}"
echo "PACKAGE_PATH=${PACKAGE_PATH}"
