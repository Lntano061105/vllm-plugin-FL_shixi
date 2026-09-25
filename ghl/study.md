# Fused GDN Gating 学习笔记：token / head / g / beta 概念

> 对应任务：`work1.md` 第 1 项（龚昊磊，Fused GDN Gating，27B/35B GDN）。
> 路径说明：本文档位于仓库 `ghl/` 子目录；`../` 前缀表示仓库根 `/workspace/vllm-plugin-FL/`。
> 固定测试重点：**固定 Token 数和 Head 数，对比 g 与 beta 输出。**

## 1. 算子在 GDN 层中的位置

Qwen3.6 的 GDN（Gated Delta Net）层中，每个 token 的隐状态先经过
`in_proj_qkvz` 和 `in_proj_ba` 两个投影，得到 `q/k/v/z` 和 `a/b`。
其中 `a`、`b` 的形状都是 `(num_tokens, num_v_heads)`，即每个 token、每个 head 一个标量，
用于控制递推（`qwen3_next.py:450-457`）。

`fused_gdn_gating` 把 `a`、`b` 和两个逐 head 的学习参数 `A_log`、`dt_bias` 合成
`g`、`beta` 两个输出（`patch_qwen3_6_gdn.py:426-428`）：

```python
g, beta = torch.ops._C_ascend.npu_fused_gdn_gating(self.A_log, a, b, self.dt_bias)
```

## 2. token 怎么理解

- "序列里的第几个位置"。算子内部把它当作 `batch` 维（kernel 里叫 `numBatches_`，每行一个 token）。
- 对外 `a`/`b` 形状为 `(T, H)`，输出为 `(1, T, H)` —— 前导 `1` 是给 recurrent 内核用的 batch 维；
  vLLM 把多序列压平成一维 token 流，所以 batch 恒为 1。
- "固定 Token 数" 即测试中的 `TOKEN_COUNTS = [1, 4, 16, 64]`
  （`tests/ops/ascend/test_fused_gdn_gating.py:43`）：
  - `1`：decode 每步只处理 1 个 token；
  - `4/16/64`：prefill 的 chunk 大小。
  - 覆盖真实推理的两种调用模式（decode 与 prefill）。

## 3. head 怎么理解

- GDN 的递推状态分 head：每个 head `h` 有自己的状态 `S_h`（形状 `[Dk, Dv]`）
  及自带参数 `A_log[h]`、`dt_bias[h]`。
- `a[t, h]`、`b[t, h]` 表示"第 t 个 token 在第 h 个 head 上的门控值"；
  输出 `g`、`beta` 同样是每个 `(token, head)` 一个值。
- Qwen3.6 27B/35B 的 GDN 层在 TP=1 下为 **32 个 head**（测试中 `NUM_HEADS = 32`），
  即"固定 Head 数"。

## 4. g 和 beta 怎么理解

两者是 Gated Delta Rule 递推的两个门控。递推式：

```
S_t = exp(g_t) · S_{t-1} + beta_t · k_t · v_t^T      (S 为 [Dk, Dv] 的状态)
```

### g —— 状态的衰减（decay）

- 公式：`g = -exp(A_log) · softplus(a + dt_bias)`（内核注释见
  `vllm_fl/_cann_ops_custom/.../fused_gdn_gating/fused_gdn_gating.h:12`）。
- g 为负数，`exp(g) ∈ (0,1)` 是遗忘因子：上一时刻状态保留多少。g 越负 → 遗忘越快。
- `A_log`：每 head 学习到的对数衰减率（`nn.Parameter`，形状 `num_v_heads // tp_size`）。
- `a + dt_bias`：输入相关的"时间步"（类似 Mamba 的 Δt），`softplus` 保证其为正。

### beta —— 写入门 / delta rule 的学习率

- 公式：`beta = sigmoid(b) ∈ (0,1)`。
- delta rule 更新为 `S += beta · k·v^T`：
  - beta ≈ 1：新键值对完整写入状态；
  - beta ≈ 0：只衰减不写入。
- 这是 "Gated"（门控）的由来。

两个输出直接喂给 `npu_recurrent_gated_delta_rule`（decode 路径）或 chunk 内核（prefill 路径）
驱动状态递推。

## 5. 与上游 Triton 基线的一致性

上游 Triton 基线是同一套公式（`qwen3_next.py:1362-1395`）：

```python
g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
beta_output = b.sigmoid()
```

固定测试即把 AscendC 算子的 `g`、`beta` 与 Triton 基线及纯 PyTorch 参考实现逐元素对比
（rtol/atol 取 1e-2）。

## 6. 一句话总结

| 概念 | 含义 |
| ---- | ---- |
| token | 输入序列中的位置，对应 `a`/`b` 的行（测试固定 1/4/16/64） |
| head  | 状态通道，对应 `a`/`b` 的列（测试固定 32） |
| g     | 每步状态的衰减量（log 尺度，负数；`exp(g)` 为遗忘因子） |
| beta  | 每步把新 k·v 写进状态的强度（sigmoid 后 0~1） |

## 参考文件

- 任务文档：`work1.md`（第 5 节第 1 行）
- Torch 适配层：`csrc/ascend/attention/fused_gdn_gating/fused_gdn_gating_torch_adpt.h`
- AscendC 内核：`csrc/ascend/attention/fused_gdn_gating/op_kernel/`
- Python 接入补丁（R8）：`vllm_fl/dispatch/backends/vendor/ascend/patches/patch_qwen3_6_gdn.py`
- 上游模型：`vllm/model_executor/models/qwen3_next.py`（`fused_gdn_gating`）
- 算子级固定测试：`tests/ops/ascend/test_fused_gdn_gating.py`

## 7. 需要学习的知识点

任务横跨三层：模型层（GDN 是什么）→ 框架层（vLLM 怎么调）→ 硬件层（AscendC 算子怎么写），缺哪层补哪层。

### 第 1 层：模型层（理解 token / head / g / beta 的出处）

- token 概念：tokenizer（BPE/SentencePiece）、词元、embedding、hidden_size、序列长度
- Transformer 基础：QKV 投影、注意力公式、多头注意力（head 从哪来）
- 线性注意力 / SSM：递推状态 S、Mamba 的 selectivity、delta rule（`S += beta·k·vᵀ`）、门控、recurrent（逐 token）与 chunk（逐块）两种计算范式
- GDN 具体结构：`a/b` 来自 `in_proj_ba`，`A_log/dt_bias` 是每 head 学习参数（见本文档第 1-4 节）

### 第 2 层：框架层（理解算子怎么被调用）

- vLLM 推理流程：prefill/decode 两种模式、KV cache、attention metadata、请求怎么变成 token 流
- 张量基本功：shape/stride/dtype/contiguous、`(1, T, H)` 这类形状为什么这么排
- 自定义算子注册链路：`torch.ops` → C++ extension（`TORCH_LIBRARY`）→ aclnn → 内核；以及 patch/回退开关机制

### 第 3 层：硬件层（理解算子内核怎么写）

- 昇腾 NPU 体系结构：AI Core、达芬奇架构、存储层级 GM→L1/L2→UB
- AscendC 编程：矢量计算、tiling（数据切分）、数据搬运、流水并行
- CANN 工程：op_host（infershape/tiling）+ op_kernel 分离、算子部署（即 `csrc/ascend/attention/fused_gdn_gating/` 的目录结构）
- 数值计算：fp16/bf16 精度问题、softplus/sigmoid 的数值稳定（内核中 threshold=20 即防 exp 溢出）

## 8. 书籍与资源

| 层级 | 资源 | 说明 |
| ---- | ---- | ---- |
| 模型层 | 《动手学深度学习》（李沐等，d2l.ai 在线免费） | token、embedding、注意力、Transformer 都有配套代码，最实用 |
| 模型层 | 李宏毅《机器学习》课程（B站） | 概念讲得最直观，先建立直觉再上代码 |
| 模型层 | 《深度学习》（花书） | 只作数学/优化词典式查阅，不要从头读，对 Transformer 讲得少 |
| 模型层 | 论文：Attention Is All You Need → Mamba → Mamba-2 → DeltaNet → Gated Delta Networks → Qwen3-Next 技术报告 | 按此顺序读，GDN 是这几篇的收束 |
| 框架层 | vLLM 官方文档 + 源码（`/vllm-workspace/vllm`） | 跟着 `qwen3_next.py` 的调用链走一遍就是最好的教材 |
| 框架层 | fla 库（GitHub: fla-org/flash-linear-attention） | GDN 算子的 Triton 参考实现，注释里有完整递推公式 |
| 硬件层 | 《昇腾AI处理器架构与编程——深入理解CANN技术原理及应用》（华为，机械工业出版社） | 达芬奇架构与 CANN 全局观 |
| 硬件层 | CANN 官方文档（昇腾社区 hiascend.com，AscendC 编程指南） | 以官方文档为准，比书更新快，AscendC 部分直接看它 |
| 硬件层 | 《C++ Primer（第5版）中文版》 | C++ 基础（读内核源码会用到） |
| 硬件层 | 《大规模并行处理器编程实战》（Kirk & Hwu，PMPP） | 讲 CUDA，但 tiling/存储层级/向量化概念完全可迁移到 AscendC，强烈建议 |
| 硬件层 | 《CUDA C编程：基础与实践》（樊哲勇，可选） | 想快速上手类 CUDA 写法时再读 |

## 9. 建议学习路线（按任务时间线）

1. **概念期（本周）**：读 Gated Delta Networks 论文 + d2l 的注意力章节；把 `g = -exp(A_log)·softplus(a+dt_bias)`、`beta = sigmoid(b)` 用 PyTorch 手推一遍
2. **接入期（下周）**：在 vLLM 源码里跟一遍 GDN 层调用链，对照 `patch_qwen3_6_gdn.py` 看 AscendC 替换点；跑通 `tests/ops/ascend/test_fused_gdn_gating.py`
3. **算子期（再往后）**：AscendC 官方教程入门，对照 `csrc/ascend/attention/fused_gdn_gating/` 的 op_host/op_kernel 逐文件读，配合 PMPP 补并行思维

> 注意：花书（2016 年）对 Transformer 和如今的大模型栈覆盖很少，别当主教材；昇腾相关的书籍少且滞后，官方文档优先。
