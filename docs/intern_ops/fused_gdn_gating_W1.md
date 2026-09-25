# Fused GDN Gating 第一周工作记录（龚昊磊）

> 项目：vLLM-Plugin-FL 自定义算子接入、编译部署与模型推理验证
> 目标模型：Qwen3.6 27B / 35B（GDN 线性注意力层）
> 负责算子：Fused GDN Gating（`npu_fused_gdn_gating`）
> 参考：R4（csrc 实现）、R8（Python 接入补丁）
> 日期：2026-08-17

## 1. 六周计划

| 周 | 目标 | 提交物 | 状态 |
|---|---|---|---|
| W1 | 定位算子在 GDN 计算图中的位置；理解输入输出 dtype/布局；读懂 R4/R8；确认环境与构建链路 | 调用链笔记（本文件） | 完成 |
| W2 | 算子源码接入 `csrc/ascend`（AscendC/CANN 工程）并接入构建系统 | `csrc/ascend/.../fused_gdn_gating` | 待做 |
| W3 | 编译部署、动态库加载；算子级固定 Shape 测试（固定 Token/Head，对比 g、beta） | `tests/ops/ascend/` + 测试结果 | 部分完成 |
| W4 | 框架接入 Qwen 推理路径、基线回退开关、模型级 1K 输入/1K 输出验证 | `vllm_fl/...` 补丁 + 模型测试记录 | 待做 |
| W5 | Microbenchmark（预热/同步/重复，P50/P90）+ 模型侧 TTFT/TPOT/吞吐；Profiler 证据 | `benchmarks/ops/ascend/` + 性能数据 | 待做 |
| W6 | 技术报告（3–5 页）与答辩 | `docs/intern_ops/` | 待做 |

## 2. 算子在 Qwen GDN 计算图中的位置

Qwen3.6 的线性注意力层（`Qwen3NextGatedDeltaNet`）前向流程：

```
hidden_states
  ├─ in_proj_qkvz / in_proj_ba          # (num_tokens, hidden) -> qkvz / ba
  ├─ fix_query_key_value_ordering       # 拆出 query/key/value/z/b/a
  ├─ conv1d (causal conv, 含状态缓存)    # AscendC: npu_causal_conv1d_custom
  ├─ fused_gdn_gating  ◄── 本算子       # AscendC: npu_fused_gdn_gating
  │     g   = -exp(A_log) * softplus(a + dt_bias)
  │     beta = sigmoid(b)
  ├─ recurrent / chunk gated delta rule # Decode: npu_recurrent_gated_delta_rule
  │                                     # Prefill: chunk_gated_delta_rule (Triton)
  ├─ RMSNormGated (z 门控) + out_proj
  └─ output
```

调用链（自顶向下）：

1. `vllm_fl/models/qwen3_next.py:477` — `Qwen3NextGatedDeltaNet.forward` 调用
   `torch.ops.vllm.gdn_attention_core(mixed_qkv, b, a, core_attn_out, prefix)`。
2. `vllm_fl/models/qwen3_next.py:1285` — `gdn_attention_core` 自定义 op 分派到
   `Qwen3NextGatedDeltaNet._forward_core`（Triton 基线路径调用 `fused_gdn_gating`，第 604 行）。
3. `vllm_fl/dispatch/backends/vendor/ascend/patches/patch_qwen3_6_gdn.py:386` —
   `AscendCGatedDeltaNet._forward_core` 替换为
   `torch.ops._C_ascend.npu_fused_gdn_gating(self.A_log, a, b, self.dt_bias.to(self.A_log.dtype))`。
4. `csrc/ascend/torch_binding.cpp:2688` — torch schema 注册
   `npu_fused_gdn_gating(Tensor A_log, Tensor a, Tensor b, Tensor dt_bias,
   float beta=1.0, float threshold=20.0) -> (Tensor g, Tensor beta_output)`。
5. `csrc/ascend/attention/fused_gdn_gating/fused_gdn_gating_torch_adpt.h` — Torch Adapter
   校验 shape/dtype 后调 `aclnnFusedGdnGating`（CANN 自定义算子 op_api）。
6. `csrc/ascend/attention/fused_gdn_gating/op_host/` — Host 侧 Tiling/InferShape/OpDef。
7. `csrc/ascend/attention/fused_gdn_gating/op_kernel/fused_gdn_gating.cpp` — AscendC kernel
   在 AI Core 上执行（AIV 核，支持 bf16/fp16 输入）。

## 3. 算子输入输出、数据类型与数据布局

schema 与 `fused_gdn_gating_torch_adpt.h` 中的约束：

| 参数 | Shape | dtype | 布局 | 说明 |
|---|---|---|---|---|
| A_log | `(num_heads,)` | float32 | ND，1-D | 对数衰减（每头），模型参数 |
| a | `(batch, num_heads)` | bf16 / fp16 | ND，2-D | 时间步投影（decay），每头 |
| b | `(batch, num_heads)` | bf16 / fp16 | ND，2-D | 门控投影，dtype 必须与 a 一致 |
| dt_bias | `(num_heads,)` | float32 | ND，1-D | 时间步偏置，dtype 必须与 A_log 一致 |
| g | `(1, batch, num_heads)` | float32 | ND，3-D | 输出：`-exp(A_log) * softplus(a + dt_bias)` |
| beta_output | `(1, batch, num_heads)` | 与 b 同 dtype | ND，3-D | 输出：`sigmoid(b)` |

- 约束：`a.size(1) == b.size(1) == A_log.size(0)`；`a`、`b` 同 dtype；`A_log`、`dt_bias` 同 dtype。
- 默认属性：`beta=1.0`、`threshold=20.0`。
- 输出布局与 vLLM Triton 基线一致（`g` 为 fp32、`beta_output` 保持 `b` 的 dtype），
  因此可透明替换 GDN 路径中的 Triton `fused_gdn_gating`。

算子语义（与 vLLM `qwen3_next.fused_gdn_gating` 一致）：

```
x        = a + dt_bias
softplus = (1/beta) * log(1 + exp(beta*x))   若 beta*x <= threshold
           x                                  否则（softplus 退化为恒等）
g        = -exp(A_log) * softplus
beta_out = sigmoid(b)
```

## 4. 环境与构建链路确认

- 硬件：Ascend 910B3 × 3（npu-smi 正常，健康 OK）。
- 软件：Python 3.11.14、torch 2.8.0+cpu + torch_npu 2.8.0.post2、vllm 0.13.0、
  CANN 8.5.0 / 9.0.0（`/usr/local/Ascend/`）。
- 构建产物已就位：`vllm_fl/_C_ascend.cpython-311-aarch64-linux-gnu.so`、
  `vllm_fl/libvllm_fl_kernels.so`；CANN 算子包已安装到
  `vllm_fl/_cann_ops_custom/vendors/custom_transformer/`（含 `bin/set_env.bash`）。
- 构建命令（后续复现用）：
  - torch extension：`VLLM_VENDOR=ascend python setup.py build_ext --inplace`
  - CANN framework 算子包：`cd csrc/ascend && bash build_aclnn.sh ascend910b`
  - 运行前必须：`source vllm_fl/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash`

## 5. 第一周验证结果

### 5.1 连通性测试

`tests/custom_ops_tests/test_fused_gdn_gating.py`：通过
（`npu_fused_gdn_gating test passed`，输出 shape `(1, batch, num_heads)` 校验）。

### 5.2 算子级固定测试

`tests/ops/ascend/test_fused_gdn_gating.py`：17/17 通过（约 12s）。

- 固定 head 数 `NUM_HEADS=32`（Qwen3.6-27B/35B GDN 层 TP=1 时的头数）。
- 固定 token 数 `[1, 4, 16, 64]`（单步 decode 与小段 prefill）。
- dtype：bf16 / fp16。
- 对比对象：
  - PyTorch 参考实现（`torch_reference`，固定随机种子 1234）；
  - vLLM Triton 基线（`vllm.model_executor.models.qwen3_next.fused_gdn_gating`）。
- 断言：`g` 与 `beta_output` 的 shape/dtype 与参考一致，数值 `rtol=atol=1e-2`；
  另含 softplus 越阈退化路径用例（`a=50` 时 `g == -exp(A_log) * x`）。

### 5.3 回退开关

`VLLM_FL_DISABLE_ASCENDC_GDN`（patch_qwen3_6_gdn.py `_ascendc_ops_available`）：

| 设置 | `_ascendc_ops_available()` | `patch_qwen3_6_gdn()` | 日志 |
|---|---|---|---|
| 默认 | True | True | `Patched Qwen3NextGatedDeltaNet ... (AscendC ... fused_gdn_gating ...)` |
| `=1` | False | False | `VLLM_FL_DISABLE_ASCENDC_GDN=1, keep Triton GDN path` |

- 默认时自动 bootstrap CANN 自定义算子环境（`ASCEND_CUSTOM_OPP_PATH` 指向
  `_cann_ops_custom/vendors/custom_transformer`），无需手动 source。
- 开关打开或 `_C_ascend`/算子包不可用时，保持 vLLM 原生 Triton GDN 路径。

## 6. 遗留事项与下一步（W2/W3）

- W1 未做模型级验证（属于 W4：1K 输入 / 1K 输出）。
- W2/W3 需要从干净状态复现一遍完整构建（`build_aclnn.sh` 打包 → `.run` 安装 →
  `set_env.bash` → 加载），确认"重新打开 Shell 后仍能加载"的部署要求。
- 本仓库 `csrc/ascend/attention/fused_gdn_gating/`、CANN 算子包与测试均为仓库既有内容，
  需按任务书要求核对最终提交与基线的关系（参考分支 R4/R8 仅用于理解实现，不得整体复制）。
