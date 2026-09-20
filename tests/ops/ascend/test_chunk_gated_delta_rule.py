"""tests/ops/ascend/test_chunk_gated_delta_rule.py
完整版 Chunk Gated Delta Rule 算子（npu_chunk_gated_delta_rule）单元测试

要求（任务书 §3-R4 / §4-R2 / §6 评价）：
- 可独立运行：python3 test_chunk_gated_delta_rule.py  （也兼容 pytest 收集）
- 固定随机种子：torch.manual_seed(42)
- 包含参考实现：ref_gated_delta_rule（fp32 逐 token 递推，数学逐字对齐 R9）
- 覆盖四重点：变长序列 / 初始 State / 最终 State / 实际长度输入
- 异常用例：非法 dtype / 非法 shape 应报错（任务书"异常用例"）

判定标准：余弦相似度 cos > 0.998（官方容差，bf16 state 累积误差 √num_chunks×2⁻⁸）
运行环境：昇腾 NPU + torch_npu（需先 source set_env.bash）
"""
import torch
import torch_npu
import vllm_fl._C_ascend  # noqa: F401  确保算子注册

TORCH_OP = "torch.ops._C_ascend.npu_chunk_gated_delta_rule"
COS_THRESHOLD = 0.998


# ---------------------------------------------------------------------------
# 参考实现（fp32，逐 token 递推）
# ---------------------------------------------------------------------------
def ref_gated_delta_rule(q, k, v, g, beta, initial_state, seqlens, scale):
    """逐 token 递推的 fp32 参考实现。

    GQA：ratio = Nv // Nk，q/k 按 ratio repeat_interleave 到每个 value head。
    State S 布局 (B, Nv, Dv, Dk)，先衰减再更新：
        S[b] = S[b] * exp(g[t]) + w ⊗ k_t
        out[t] = scale * S[b] @ q_t
    """
    T, Nk, Dk = q.shape
    Nv, Dv = v.shape[1], v.shape[2]
    ratio = Nv // Nk
    B = initial_state.shape[0]
    S = initial_state.to(torch.float32).clone()
    out = torch.zeros(T, Nv, Dv, dtype=torch.float32, device=q.device)
    start = 0
    for b in range(B):
        L = int(seqlens[b])
        for t in range(start, start + L):
            k_t = k[t].repeat_interleave(ratio, dim=0)
            q_t = q[t].repeat_interleave(ratio, dim=0)
            u = torch.einsum("nvk,nk->nv", S[b], k_t)
            w = beta[t].unsqueeze(-1) * (v[t] - u)
            if g is not None:
                S[b] = S[b] * torch.exp(g[t]).view(Nv, 1, 1)
            S[b] = S[b] + torch.einsum("nv,nk->nvk", w, k_t)
            out[t] = scale * torch.einsum("nvk,nk->nv", S[b], q_t)
        start += L
    return out, S


def cosine(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    if a.norm() == 0 and b.norm() == 0:
        return 1.0
    if a.norm() == 0 or b.norm() == 0:
        return 0.0
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


# ---------------------------------------------------------------------------
# 用例生成（固定随机种子）
# ---------------------------------------------------------------------------
def make_inputs(Nk, Nv, Dk, Dv, seqlens, with_g=True, nonzero_init=True):
    """生成固定种子的输入张量（TND 布局，与算子契约一致）。"""
    torch.manual_seed(42)
    dev = "npu"
    B = len(seqlens)
    T = sum(seqlens)
    q = torch.randn(T, Nk, Dk, dtype=torch.bfloat16, device=dev)
    k = torch.randn(T, Nk, Dk, dtype=torch.bfloat16, device=dev)
    q = q / q.norm(dim=-1, keepdim=True)   # L2 归一化（与调用侧 l2norm_fwd 一致）
    k = k / k.norm(dim=-1, keepdim=True)
    v = torch.randn(T, Nv, Dv, dtype=torch.bfloat16, device=dev) * 0.1
    beta = torch.rand(T, Nv, dtype=torch.bfloat16, device=dev).sigmoid()
    g = (-torch.nn.functional.softplus(torch.randn(T, Nv, device=dev), beta=1.0)).float() if with_g else None
    if nonzero_init:
        initial_state = torch.randn(B, Nv, Dv, Dk, dtype=torch.bfloat16, device=dev) * 0.01
    else:
        initial_state = torch.zeros(B, Nv, Dv, Dk, dtype=torch.bfloat16, device=dev)
    actual_seq_lengths = torch.tensor(seqlens, dtype=torch.int32, device=dev)
    scale = Dk ** -0.5
    return q, k, v, beta, g, initial_state, actual_seq_lengths, scale


def run_case(name, Nk, Nv, Dk, Dv, seqlens, with_g=True, nonzero_init=True):
    """单个正确性用例：AscendC 输出 vs fp32 参考，双输出（out/final_state）都对拍。"""
    q, k, v, beta, g, initial_state, actual_seq_lengths, scale = make_inputs(
        Nk, Nv, Dk, Dv, seqlens, with_g, nonzero_init)
    out, final_state = torch.ops._C_ascend.npu_chunk_gated_delta_rule(
        q, k, v, beta, initial_state, actual_seq_lengths, g=g, scale_value=scale)
    torch.npu.synchronize()
    ref_out, ref_state = ref_gated_delta_rule(
        q.float(), k.float(), v.float(), g, beta.float(), initial_state, seqlens, scale)
    cos_out = cosine(out.cpu(), ref_out.cpu())
    cos_state = cosine(final_state.cpu(), ref_state.cpu())
    ok = cos_out > COS_THRESHOLD and cos_state > COS_THRESHOLD
    print(f"[{name}] T={sum(seqlens)} seqlens={seqlens} Nk={Nk} Nv={Nv} Dk={Dk} Dv={Dv}")
    print(f"  out cos={cos_out:.6f}  final_state cos={cos_state:.6f}  {'PASS' if ok else 'FAIL'}")
    assert ok, f"[{name}] 数值对拍未过容差 cos > {COS_THRESHOLD}"
    return True


# ---------------------------------------------------------------------------
# 异常用例（任务书"异常用例"要求）
# ---------------------------------------------------------------------------
def test_wrong_dtype():
    """异常用例 1：query 传 FP32（契约要求 BF16），应抛出异常。"""
    q, k, v, beta, g, initial_state, actual_seq_lengths, scale = make_inputs(
        2, 4, 64, 64, seqlens=[64])
    raised = False
    try:
        torch.ops._C_ascend.npu_chunk_gated_delta_rule(
            q.float(), k, v, beta, initial_state, actual_seq_lengths,
            g=g, scale_value=scale)
    except Exception as e:
        raised = True
        print(f"[异常dtype] 正确抛出异常: {type(e).__name__}: {str(e)[:80]}")
    assert raised, "[异常dtype] 期望抛出异常但未抛"
    return True


def test_wrong_shape():
    """异常用例 2：actual_seq_lengths 的 B 维与 initial_state 不匹配，应抛出异常。"""
    q, k, v, beta, g, initial_state, actual_seq_lengths, scale = make_inputs(
        2, 4, 64, 64, seqlens=[64])
    # initial_state 为 (1,Nv,Dv,Dk)，这里故意传 2 个长度的 actual_seq_lengths → B 维冲突
    bad_lens = torch.tensor([32, 32], dtype=torch.int32, device="npu")
    raised = False
    try:
        torch.ops._C_ascend.npu_chunk_gated_delta_rule(
            q, k, v, beta, initial_state, bad_lens, g=g, scale_value=scale)
    except Exception as e:
        raised = True
        print(f"[异常shape] 正确抛出异常: {type(e).__name__}: {str(e)[:80]}")
    assert raised, "[异常shape] 期望抛出异常但未抛"
    return True


# ---------------------------------------------------------------------------
# 四重点固定用例（任务书 §5）
# ---------------------------------------------------------------------------
def test_fixed_cases():
    run_case("等长+初始State",  Nk=2, Nv=4, Dk=64, Dv=64, seqlens=[128],           with_g=True,  nonzero_init=True)
    run_case("变长[7,65]",      Nk=2, Nv=4, Dk=64, Dv=64, seqlens=[7, 65],          with_g=True,  nonzero_init=True)
    run_case("chunk边界[63,64,65]", Nk=2, Nv=4, Dk=64, Dv=64, seqlens=[63, 64, 65], with_g=True,  nonzero_init=True)
    run_case("L=1实际长度",     Nk=2, Nv=4, Dk=64, Dv=64, seqlens=[1],              with_g=True,  nonzero_init=True)
    run_case("零初始State",     Nk=2, Nv=4, Dk=64, Dv=64, seqlens=[128],           with_g=True,  nonzero_init=False)
    run_case("无g衰减",         Nk=2, Nv=4, Dk=64, Dv=64, seqlens=[64],            with_g=False, nonzero_init=True)
    return True


def main():
    print("=" * 56)
    print("Chunk Gated Delta Rule 单元测试  (参考实现: fp32 逐 token 递推)")
    print(f"算子: {TORCH_OP}   容差: cos > {COS_THRESHOLD}")
    print("=" * 56)
    test_fixed_cases()
    test_wrong_dtype()
    test_wrong_shape()
    print("=" * 56)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
