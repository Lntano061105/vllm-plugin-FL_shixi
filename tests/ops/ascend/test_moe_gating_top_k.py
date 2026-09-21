#!/usr/bin/env python3
"""MoE Gating Top-K 正确性测试。

与参考实现（PyTorch 原生 Top-K + Softmax）对比专家选择、权重和归一化。
路径动态发现，不写死绝对路径。
"""
import glob
import os

import numpy as np
import torch

# 1. 动态加载算子
import vllm_fl

_VLLM_FL_DIR = os.path.dirname(os.path.abspath(vllm_fl.__file__))
_SO_FILES = glob.glob(os.path.join(_VLLM_FL_DIR, "_C_ascend*.so"))
if not _SO_FILES:
    raise RuntimeError(f"_C_ascend*.so not found under {_VLLM_FL_DIR}")
torch.ops.load_library(_SO_FILES[0])
print(f"Loaded op library: {_SO_FILES[0]}")


def reference_topk(x: torch.Tensor, k: int, renorm: int = 0,
                   norm_type: int = 0, routed_scaling_factor: float = 1.0):
    """PyTorch reference for the operator's semantics.

    norm_type=0: softmax; norm_type=1: sigmoid.
    renorm=1: renormalize the top-k weights to sum to 1.
    """
    if norm_type == 0:
        scores = torch.softmax(x.float(), dim=-1)
    elif norm_type == 1:
        scores = torch.sigmoid(x.float())
    else:
        raise ValueError(f"unsupported norm_type={norm_type}")

    topk_weights, topk_ids = torch.topk(scores, k, dim=-1, sorted=False)

    if renorm:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    if routed_scaling_factor != 1.0:
        topk_weights = topk_weights * routed_scaling_factor

    return topk_weights, topk_ids


def run_case(num_tokens, num_experts, k, renorm=0, norm_type=0,
             routed_scaling_factor=1.0, dtype=torch.float16):
    """One correctness case: run op and compare against reference."""
    torch.manual_seed(42)

    x = torch.randn(num_tokens, num_experts, dtype=torch.float32).npu()

    # 调用算子
    y, expert_idx, out = torch.ops._C_ascend.moe_gating_top_k(
        x, k, 1, 1, 0, renorm, norm_type, True,
        routed_scaling_factor, 1e-20,
    )

    # Reference（在 CPU 上用 float32 算）
    ref_weights, ref_ids = reference_topk(
        x.cpu(), k, renorm=renorm, norm_type=norm_type,
        routed_scaling_factor=routed_scaling_factor,
    )

    # ---- 1. 专家索引必须完全一致（排序无关）----
    sorted_ref_ids = torch.sort(ref_ids, dim=-1).values
    sorted_op_ids = torch.sort(expert_idx.cpu().long(), dim=-1).values
    assert torch.equal(sorted_op_ids, sorted_ref_ids), (
        f"expert_idx mismatch:\n"
        f"  op  = {sorted_op_ids[:3]}\n"
        f"  ref = {sorted_ref_ids[:3]}"
    )

    # ---- 2. 权重按索引对齐后对比 ----
    # 由于算子返回的 topk_ids 顺序可能与 torch.topk 不同，需要按 id 匹配
    op_weights = y.cpu().float()
    op_ids = expert_idx.cpu().long()

    aligned_ref_weights = torch.empty_like(op_weights)
    for i in range(num_tokens):
        for j in range(k):
            eid = op_ids[i, j].item()
            # 在 reference 里找这个 expert 的权重
            mask = ref_ids[i] == eid
            assert mask.any(), f"token {i}: op expert {eid} not in reference"
            aligned_ref_weights[i, j] = ref_weights[i][mask].max()

    torch.testing.assert_close(
        op_weights, aligned_ref_weights, rtol=2e-3, atol=2e-3,
    )

    # ---- 3. renorm=1 时，权重和必须为 1 ----
    if renorm:
        weight_sum = op_weights.sum(dim=-1)
        torch.testing.assert_close(
            weight_sum, torch.ones_like(weight_sum),
            rtol=1e-3, atol=1e-3,
        )

    print(f"  [PASS] tokens={num_tokens}, experts={num_experts}, k={k}, "
          f"renorm={renorm}, norm_type={norm_type}, dtype={dtype}")


def main():
    print("=" * 70)
    print("MoE Gating Top-K 正确性测试（与 reference 数值对比）")
    print("=" * 70)

    # 覆盖实际模型 shape：Qwen3-35B 有 256 专家，k=2
    cases = [
        # (num_tokens, num_experts, k, renorm, norm_type)
        (4,    8,   2, 0, 0),   # 极小用例
        (4,    8,   2, 1, 0),   # renorm=1
        (128,  64,  2, 0, 0),
        (256,  128, 2, 1, 0),
        (512,  256, 2, 0, 0),
        (1024, 256, 2, 1, 0),   # Qwen3-35B 典型 shape
        (2048, 256, 2, 0, 0),   # 更大 token 数
        (2048, 256, 4, 0, 0),   # k=4
    ]

    print("\n开始测试...")
    for num_tokens, num_experts, k, renorm, norm_type in cases:
        run_case(num_tokens, num_experts, k, renorm=renorm,
                 norm_type=norm_type)

    print("\n" + "=" * 70)
    print("✅ 全部用例通过")
    print("=" * 70)


if __name__ == "__main__":
    main()
