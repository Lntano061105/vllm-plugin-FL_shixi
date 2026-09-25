# 手动profiling测试（0\.6b模型为例）

# qwen0\.6b 基础分步测试命令



本文档把 `scripts/run_vllm_fl_profile_unified.sh` 的核心执行流程拆成最基础的手动命令，便于逐步验证。

示例模型：

```Bash
/models/Qwen3-0.6B
```

服务名：

```Bash
qwen0.6b
```

结果目录：

```Bash
/workspace/new_results/qwen0.6b_basic
```

## 1\. 终端 1：启动 server

```Bash
mkdir -p /workspace/new_results/qwen0.6b_basic/torch_profile

export PYTHONPATH=/workspace/vllm-plugin-FL:$PYTHONPATH
export VLLM_PLUGINS=fl
export VLLM_FL_PLATFORM=ascend
export GEMS_VENDOR=ascend
export VLLM_ATTENTION_BACKEND=TORCH_SDPA
export ASCEND_RT_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_FL_FLAGOS_WHITELIST=unquantized_fused_moe_method,topk_softmax
export FLAGGEMS_ENABLE_OPLIST_PATH=/workspace/new_results/qwen0.6b_basic/flaggems_enable_oplist.txt
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
vllm serve /models/Qwen3-0.6B \
  --served-model-name qwen0.6b \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --host 0.0.0.0 \
  --port 8080 \
  --gpu-memory-utilization 0.6 \
  --compilation-config '{"cudagraph_mode":"FULL","backend":"eager","custom_ops":["all"],"pass_config":{"fuse_norm_quant":false,"fuse_act_quant":false,"fuse_attn_quant":false,"enable_sp":false,"fuse_gemm_comms":false,"fuse_allreduce_rms":false}}' \
  --enable-chunked-prefill \
  --no-enable-prefix-caching \
  --max-model-len 32768 \
  --max-num-seqs 1 \
  --profiler-config '{"profiler":"torch","torch_profiler_dir":"/workspace/new_results/qwen0.6b_basic/torch_profile","torch_profiler_with_stack":true,"torch_profiler_record_shapes":false,"torch_profiler_with_memory":false,"torch_profiler_with_flops":false,"torch_profiler_use_gzip":true}' \
  --no-async-scheduling
```

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=YWU3YzMwZjQ2YmE2MzlmZmVmMjY0NjcwNzQyZmY3NzRfMDAzNzdjNDdjZTBmYTBmNjc5NWY3ZWExMDMyNWE2Y2FfSUQ6NzY3MjM0Nzk5NzE0MTA0NDUwOV8xNzg3NTYxODc4OjE3ODc2NDgyNzhfVjM)

## 2\. 终端 2：检查 server

```Bash
curl http://127.0.0.1:8080/v1/models
```

如果能返回模型列表，说明 server 已经启动。

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=MmI0YTA1ZWRjZTM4YjA0MzFjM2UyZjY5ZTJjMTE0YjZfMjk3OTE0OGZjYjBlMTVmYjY1N2MwNWM2NWM5ZDc5ZjNfSUQ6NzY3MjM0Nzk5ODc3NjY5MTY5MV8xNzg3NTYxODc4OjE3ODc2NDgyNzhfVjM)

## 3\. 终端 2：warmup

```Bash
export PYTHONPATH=/workspace/vllm-plugin-FL:$PYTHONPATH
export VLLM_PLUGINS=fl
export VLLM_FL_PLATFORM=ascend
vllm bench serve \
  --backend openai-chat \
  --model qwen0.6b \
  --tokenizer /models/Qwen3-0.6B \
  --endpoint /v1/chat/completions \
  --host 127.0.0.1 \
  --port 8080 \
  --dataset-name random \
  --random-input-len 128 \
  --random-output-len 128 \
  --max-concurrency 2 \
  --num-prompts 4 \
  --ignore-eos
```

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=M2U1M2VkOWUxM2RmZjMzNTJmYzYyNTA5ODkyYmE5ZDdfODJkNDhkZWFiNmQ5MjI1OGQxYjY2ZDk2YTlkYzNjZTBfSUQ6NzY3MjM0Nzk5OTA0NTA3Nzk2MV8xNzg3NTYxODc4OjE3ODc2NDgyNzhfVjM)

## 4\. 终端 2：case 1

输入 32 tokens，输出 32 tokens，请求数 1，并开启 profile。

```Bash
mkdir -p /workspace/new_results/qwen0.6b_basic/torch_profile/i32_o32_np1_c1
echo /workspace/new_results/qwen0.6b_basic/torch_profile/i32_o32_np1_c1 > /workspace/new_results/qwen0.6b_basic/torch_profile/.current_case_dir
export PYTHONPATH=/workspace/vllm-plugin-FL:$PYTHONPATH
export VLLM_PLUGINS=fl
export VLLM_FL_PLATFORM=ascend
vllm bench serve \
  --backend openai-chat \
  --model qwen0.6b \
  --tokenizer /models/Qwen3-0.6B \
  --endpoint /v1/chat/completions \
  --host 127.0.0.1 \
  --port 8080 \
  --dataset-name random \
  --random-input-len 32 \
  --random-output-len 32 \
  --max-concurrency 1 \
  --num-prompts 1 \
  --ignore-eos \
  --profile
```

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=MmZlZDQ3ZmVjN2Y3NTkzYWJiZDMxODQ2MGFmOGI4ZTJfMDhkZTFjMzk2YTcwYzBkNzMzY2Y0MTQxODNlNjBlYWNfSUQ6NzY3MjM0Nzk5OTYxOTY5NzU5Ml8xNzg3NTYxODc4OjE3ODc2NDgyNzhfVjM)

## 5\. 终端 2：case 2

输入 32 tokens，输出 128 tokens，请求数 1，并开启 profile。

```Bash
mkdir -p /workspace/new_results/qwen0.6b_basic/torch_profile/i32_o128_np1_c1
echo /workspace/new_results/qwen0.6b_basic/torch_profile/i32_o128_np1_c1 > /workspace/new_results/qwen0.6b_basic/torch_profile/.current_case_dir
export PYTHONPATH=/workspace/vllm-plugin-FL:$PYTHONPATH
export VLLM_PLUGINS=fl
export VLLM_FL_PLATFORM=ascend
vllm bench serve \
  --backend openai-chat \
  --model qwen0.6b \
  --tokenizer /models/Qwen3-0.6B \
  --endpoint /v1/chat/completions \
  --host 127.0.0.1 \
  --port 8080 \
  --dataset-name random \
  --random-input-len 32 \
  --random-output-len 128 \
  --max-concurrency 1 \
  --num-prompts 1 \
  --ignore-eos \
  --profile
```

## 6\. 终端 2：case 3

输入 4096 tokens，输出 4096 tokens，请求数 1，并开启 profile。

```Bash
mkdir -p /workspace/new_results/qwen0.6b_basic/torch_profile/i4096_o4096_np1_c1
echo /workspace/new_results/qwen0.6b_basic/torch_profile/i4096_o4096_np1_c1 > /workspace/new_results/qwen0.6b_basic/torch_profile/.current_case_dir
export PYTHONPATH=/workspace/vllm-plugin-FL:$PYTHONPATH
export VLLM_PLUGINS=fl
export VLLM_FL_PLATFORM=ascend
vllm bench serve \
  --backend openai-chat \
  --model qwen0.6b \
  --tokenizer /models/Qwen3-0.6B \
  --endpoint /v1/chat/completions \
  --host 127.0.0.1 \
  --port 8080 \
  --dataset-name random \
  --random-input-len 4096 \
  --random-output-len 4096 \
  --max-concurrency 1 \
  --num-prompts 1 \
  --ignore-eos \
  --profile
```

## 7\. 终端 1：停止 server

在启动 server 的终端按：

```Bash
Ctrl+C
```

## 8\. 可选：分析 profile

```Bash
python - <<'PY'
import glob
from torch_npu.profiler.profiler import analyse

for d in glob.glob("/workspace/new_results/qwen0.6b_basic/torch_profile/i*_o*"):
    print(d)
    analyse(d)
PY
```