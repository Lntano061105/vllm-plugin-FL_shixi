# Profiling统一脚本使用说明

# Profiling 脚本使用说明

本文档说明 `scripts/` 目录下 3 个 Profiling 相关脚本的用途、参数、执行流程和产物结构：

- `scripts/run_vllm_ascend_profile_unified.sh`：使用 `vllm-ascend` 插件启动 vLLM 服务并采集 torch/NPU profiler 数据。

- `scripts/run_vllm_fl_profile_unified.sh`：使用 `vllm-plugin-FL` 启动 vLLM 服务并采集 torch/NPU profiler 数据。

- `scripts/package_op_statistic.sh`：对已完成的 run 目录二次打包，提取算子统计、API 统计和 benchmark 摘要。

## vLLM Ascend Profiling

脚本路径：

```Bash
/workspace/scripts/run_vllm_ascend_profile_unified.sh
```

该脚本会设置 `VLLM_PLUGINS=ascend`，适用于验证原生 `vllm-ascend` 实现的 graph/eager、chunked prefill、MTP 等配置。

### 基本命令

```Bash
/workspace/scripts/run_vllm_ascend_profile_unified.sh \
  --model-path /models/Qwen3.6-27B \
  --model-name qwen3.6 \
  --model-tag qwen3.6-27b \
  --mode graph \
  --cudagraph-mode FULL \
  --chunked true \
  --devices 0,1,2,3 \
  --tp 4 \
  --gmem 0.6 \
  --port 8113 \
  --cases "32,32,1" \
  --bench-profile true \
  --run-label test
```

### eager 模式示例

```Bash
/workspace/scripts/run_vllm_ascend_profile_unified.sh \
  --model-path /models/Qwen3.6-27B \
  --model-name qwen3.6 \
  --model-tag qwen3.6-27b \
  --mode eager \
  --chunked true \
  --devices 0,1,2,3 \
  --tp 4 \
  --gmem 0.6 \
  --cases "32,32,1;32,128,1;128,128,1" \
  --run-label eager
```

### MTP 示例

```Bash
/workspace/scripts/run_vllm_ascend_profile_unified.sh \
  --model-path /models/Qwen3.6-27B \
  --model-name qwen3.6 \
  --model-tag qwen3.6-27b \
  --mode graph \
  --mtp 1 \
  --cases "128,128,1" \
  --run-label mtp1
```

## vLLM FL Profiling

脚本路径：

```Bash
/workspace/scripts/run_vllm_fl_profile_unified.sh
```

该脚本会设置 `VLLM_PLUGINS=fl`、`VLLM_FL_PLATFORM=ascend`、`GEMS_VENDOR=ascend` 等环境变量，适用于验证 `vllm-plugin-FL` 实现。

### 基本命令

```Bash
/workspace/scripts/run_vllm_fl_profile_unified.sh \
  --model-path /models/Qwen3.6-27B \
  --model-name qwen3.6 \
  --model-tag qwen3.6-27b \
  --mode graph \
  --cudagraph-mode FULL \
  --chunked true \
  --devices 0,1,2,3 \
  --tp 4 \
  --gmem 0.6 \
  --port 8080 \
  --cases "32,32,1" \
  --bench-profile true \
  --run-label test
```

### 高并发长上下文示例

```Bash
/workspace/scripts/run_vllm_fl_profile_unified.sh \
  --model-path /models/Qwen3.6-35B-A3B \
  --model-name qwen3.6 \
  --model-tag qwen3.6-35b-a3b \
  --mode eager \
  --chunked true \
  --devices 4,5,6,7 \
  --tp 4 \
  --gmem 0.55 \
  --max-model-len 131072 \
  --max-num-seqs 64 \
  --concurrency 64 \
  --cases "16384,1024,256;65536,1024,256" \
  --bench-profile false \
  --run-label c64_long_context
```

## 通用参数说明

以下参数同时适用于 `run_vllm_ascend_profile_unified.sh` 和 `run_vllm_fl_profile_unified.sh`。

|参数|说明|默认值|
|---|---|---|
|`--model-path PATH`|模型目录路径，同时作为 tokenizer 路径|`/models/Qwen3.6-27B`|
|`--model-name NAME`|vLLM 对外暴露的 served model name|`qwen3.6`|
|`--model-tag TAG`|结果目录中的模型标识；不传时使用模型目录 basename 小写值|自动生成|
|`--mode graph`|eager|执行模式；`graph` 使用 compilation config，`eager` 使用 `--enforce-eager`|
|`--cudagraph-mode FULL`|FULL\_DECODE\_ONLY|graph 模式下的 cudagraph 策略|
|`--chunked true`|false|是否启用 chunked prefill|
|`--cases "I,O,NP;I,O,NP"`|正式测试 case 列表，格式为输入长度、输出长度、请求数|`32,32,1;32,128,1;128,128,1`|
|`--concurrency N`|正式 benchmark 的最大并发数|`1`|
|`--bench-profile true`|false|是否给 `vllm bench serve` 传入 `--profile`|
|`--profile true`|false|`--bench-profile` 的别名|
|`--no-bench-profile`|等价于 `--bench-profile false`|\-|
|`--run-label LABEL`|追加到结果目录名中的自定义标签|空|
|`--port PORT`|vLLM 服务端口；Ascend 脚本默认 `8113`，FL 脚本实际默认 `8080`|见说明|
|`--tp N`|tensor parallel size|`4`|
|`--gmem FLOAT`|`--gpu-memory-utilization`|`0.6`|
|`--max-model-len N`|最大模型长度|`32768`|
|`--max-num-seqs N`|最大序列数|`1`|
|`--speculative-config JSON`|vLLM speculative decoding 配置|空|
|`--mtp N`|MTP 快捷参数，会自动生成 speculative config|空|
|`--devices IDS`|Ascend 可见设备 ID|`0,1,2,3`|
|`--warmup "I,O,C,NP"`|预热 case，格式为输入长度、输出长度、并发、请求数|`128,128,2,4`|
|`--no-warmup`|跳过预热|\-|
|`--result-root PATH`|结果根目录|`/workspace/results`|
|`--package tar.gz`|tar|none\`|
|`--skip-analyse`|跳过 `torch_npu.profiler.profiler.analyse`|false|

## case 和 warmup 格式

正式测试通过 `--cases` 指定，格式为：

```Plain Text
输入长度,输出长度,请求数;输入长度,输出长度,请求数
```

示例：

```Bash
--cases "32,32,1;32,128,1;128,32,1;128,128,1;512,32,1;512,128,1"
```

脚本会为每个 case 生成目录名：

```Plain Text
i{input}_o{output}_np{num_prompts}_c{concurrency}
```

例如：

```Plain Text
i128_o128_np1_c1
```

预热通过 `--warmup` 指定，格式为：

```Plain Text
输入长度,输出长度,并发数,请求数
```

示例：

```Bash
--warmup "128,128,2,4"
```

如不需要预热：

```Bash
--no-warmup
```

## 脚本执行流程

两个 profiling 主脚本的整体流程一致：

1. 解析命令行参数并校验取值。

2. 创建 run 目录、profile 目录和日志文件。

3. 记录本次命令、参数、脚本副本和 run 元信息。

4. 设置 vLLM / Ascend / FL 相关环境变量。

5. 启动 `vllm serve`，并等待 `http://127.0.0.1:{PORT}/v1/models` 可访问。

6. 执行 warmup。

7. 按 `--cases` 逐个执行 `vllm bench serve`。

8. 每个 case 执行前写入 `.current_case_dir`，用于 profiler 输出归档到对应 case 目录。

9. 汇总 benchmark 结果到 `request_benchmark_results.txt`。

10. 停止 vLLM 服务。

11. 调用 `torch_npu.profiler.profiler.analyse` 分析 profiler 数据；Ascend 脚本在 `--skip-analyse` 或 `--bench-profile false` 时跳过，FL 脚本仅在 `--skip-analyse` 时跳过。

12. 修复结果目录权限。

13. 根据 `--package` 打包关键产物。

    

## 结果目录结构

默认结果目录位于 `/workspace/results`。每次运行会生成类似目录：

```Plain Text
/workspace/results/atp_{model_tag}_{platform}_{mode}_{chunk_tag}_{run_label}_tp{tp}_gmem{gmem}_{timestamp}/
```

典型结构如下：

```Plain Text
run_dir/
  run_command.sh
  run_info.env
  run_summary.txt
  server.log
  client.log
  warmup_terminal.log
  request_benchmark_results.txt
  analysis_terminal.log
  package_filelist.txt
  torch_profile/
    i32_o32_np1_c1/
      request_terminal.log
      .../ASCEND_PROFILER_OUTPUT/
        analysis.db
        api_statistic.csv
        ascend_pytorch_profiler_0.db
        communication.json
        communication_matrix.json
        kernel_details.csv
        op_statistic.csv
        operator_details.csv
        trace_view.json
```

关键文件说明：

|文件|说明|
|---|---|
|`run_command.sh`|本次执行命令，便于复现|
|`run_info.env`|本次运行的参数和路径|
|`run_summary.txt`|运行结果摘要|
|`server.log`|vLLM 服务端日志|
|`client.log`|脚本主流程日志|
|`warmup_terminal.log`|warmup 输出|
|`request_benchmark_results.txt`|每个 case 的 benchmark 摘要|
|`analysis_terminal.log`|profiler analyse 输出|
|`torch_profile/*/request_terminal.log`|单个 case 的 benchmark 原始输出|
|`ASCEND_PROFILER_OUTPUT/op_statistic.csv`|device 侧算子统计|
|`ASCEND_PROFILER_OUTPUT/api_statistic.csv`|host/API 侧统计|
|`ASCEND_PROFILER_OUTPUT/trace_view.json`|trace 视图文件|

脚本还会把最新 run 目录写入：

```Plain Text
/workspace/results/latest_run_dir.txt
```

## 打包策略

两个 profiling 主脚本默认在运行结束后打包关键 profiler 产物。

```Bash
--package tar.gz
```

可选值：

- `tar.gz`：生成压缩包，默认值。

- `tar`：生成未压缩 tar 包。

- `none`：不打包，只保留 run 目录。

两个主脚本都会打包以下文件：

- 每个 case 的 `request_terminal.log`

- `ASCEND_PROFILER_OUTPUT/analysis.db`

- `ASCEND_PROFILER_OUTPUT/api_statistic.csv`

- `ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_0.db`

- `ASCEND_PROFILER_OUTPUT/communication.json`

- `ASCEND_PROFILER_OUTPUT/communication_matrix.json`

- `ASCEND_PROFILER_OUTPUT/kernel_details.csv`

- `ASCEND_PROFILER_OUTPUT/op_statistic.csv`

- `ASCEND_PROFILER_OUTPUT/operator_details.csv`

- `ASCEND_PROFILER_OUTPUT/trace_view.json`

此外，Ascend 脚本还会把 `request_benchmark_results.txt` 放入默认压缩包；FL 脚本会在 run 目录中生成该文件，但默认打包清单当前没有包含它。如需要轻量打包 benchmark 摘要和算子统计，可使用下文的 `package_op_statistic.sh`。

## 二次打包算子统计

脚本路径：

```Bash
/workspace/scripts/package_op_statistic.sh
```

该脚本用于对已完成的 run 目录二次打包，仅提取分析算子瓶颈常用的轻量文件：

- `ASCEND_PROFILER_OUTPUT/api_statistic.csv`

- `ASCEND_PROFILER_OUTPUT/op_statistic.csv`

- `ASCEND_PROFILER_OUTPUT/step_trace_time.csv`

- `request_benchmark_results.txt`

使用方式：

```Bash
/workspace/scripts/package_op_statistic.sh /workspace/results/atp_qwen3.6-27b_fl_graph_chunked_baseline_tp4_gmem0.6_20260721_120000
```

输出文件位于 run 目录同级：

```Plain Text
/workspace/results/{RUN_NAME}_op_statistic.tar.gz
```

如果不确定最新 run 目录，可以先查看：

```Bash
cat /workspace/results/latest_run_dir.txt
```

然后执行：

```Bash
/workspace/scripts/package_op_statistic.sh "$(cat /workspace/results/latest_run_dir.txt)"
```

## 常用场景

### 只跑 benchmark，不采集 profiler

```Bash
/workspace/scripts/run_vllm_fl_profile_unified.sh \
  --model-path /models/Qwen3.6-27B \
  --mode graph \
  --bench-profile false \
  --skip-analyse \
  --cases "1024,1024,16" \
  --run-label bench_only
```

### 跳过自动打包

```Bash
/workspace/scripts/run_vllm_fl_profile_unified.sh \
  --model-path /models/Qwen3.6-27B \
  --mode graph \
  --cases "128,128,1" \
  --package none
```

### 对比 graph 和 eager

建议除 `--mode`、`--run-label` 外，其余参数保持一致：



```Bash
/workspace/scripts/run_vllm_fl_profile_unified.sh \
  --model-path /models/Qwen3.6-27B \
  --model-tag qwen3.6-27b \
  --mode graph \
  --chunked true \
  --cases "32,32,1;32,128,1;128,128,1" \
  --run-label graph

/workspace/scripts/run_vllm_fl_profile_unified.sh \
  --model-path /models/Qwen3.6-27B \
  --model-tag qwen3.6-27b \
  --mode eager \
  --chunked true \
  --cases "32,32,1;32,128,1;128,128,1" \
  --run-label eager
```

## 注意事项

- `--mtp` 和 `--speculative-config` 不能同时使用。

- `--mode eager` 下 `--cudagraph-mode` 不生效，脚本会在元信息中记录为 `N/A`。

- `--cases` 中每个 case 必须是 3 段：`input,output,num_prompts`。

- `--warmup` 必须是 4 段：`input,output,concurrency,num_prompts`。

- 如果打包清单没有选中任何文件，打包步骤会报错并退出。

- 如果 vLLM 服务 360 秒内未就绪，脚本会退出并提示查看 `server.log`。

- FL 脚本的 usage 文本中端口默认值写为 `8113`，但脚本实际变量为 `8080`；不希望依赖默认值时建议显式传入 `--port`。

- FL 脚本在 `--bench-profile false` 时仍会尝试执行 analyse；如果只是跑 benchmark，建议同时传入 `--skip-analyse`。

- 多组实验对比时，应固定 `model-path`、`tp`、`gmem`、`max-model-len`、`max-num-seqs`、`chunked`、`cases`、`concurrency` 等参数，只改变待验证变量。





# 附件

\[run\_vllm\_ascend\_profile\_unified\.sh\]

\[run\_vllm\_fl\_profile\_unified\.sh\]

\[package\_op\_statistic\.sh\]