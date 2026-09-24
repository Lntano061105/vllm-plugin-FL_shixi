
"""tests/ops/ascend/test_chunk_gated_delta_rule.py
完整版 Chunk Gated Delta Rule 算子（npu_chunk_gated_delta_rule）单元测试

要求（任务书 §3-R4 / §4-R2 / §6 评价）：
- 可独立运行：python3 test_chunk_gated_delta_rule.py  （也兼容 pytest 收集）
- 固定随机种子：torch.manual_seed(42)
- 包含参考实现：ref_gated_delta_rule（fp32 逐 token 递推，数学与 kernel 一致）
- 覆盖四重点：变长序列 / 初始 State / 最终 State / 实际长度输入
- 异常用例：非法 dtype / 非法 shape 应报错

判定标准（三判据，缺一不可）：
  cos      > 0.998                     余弦相似度（官方容差）
  rel_l2   < 5e-2                      相对 L2 误差 ‖a-b‖₂ / ‖b‖₂（官方容差）
  max_err  <= ATOL + RTOL · max|b|     逐元素最大误差上界（numpy.allclose 约定）
  说明：仅用 cos 无法发现「结果整体放大 / 偏移」这类错误（整体 ×10 时 cos 仍≈1）；
        rel_l2 对整体缩放敏感，但对少量元素的局部大偏差不敏感，
        故再补一项逐元素最大误差上界（绝对 + 相对混合），三者互为补充。

参考实现的递推顺序（与 kernel 一致，见 csrc/.../op_kernel/*/chunk_gated_delta_rule_stage2.h：
stage2 先 CalGCumExp 对 state 做 exp(g) 衰减，再用衰减后的 state 算 v'/attn_inter）：
    S ← S · exp(g_t)                       ← 1) 先衰减（原地）
    u  = S @ k_t                           ← 2) 用衰减后的 state
    w  = beta_t · (v_t − u)                ← 3) 更新量
    S ← S + w ⊗ k_t                        ← 4) 原地累加
    out_t = scale · (S @ q_t)              ← 5) 读出（用第 4 步更新后的 S）
（修正前顺序为「先算 u/w、再衰减」，与本 kernel 不一致；见 ref_wrong_order_for_contrast。）

运行环境：
  - 纯 CPU（无卡，用于参考实现与判据自检）：
        cd <仓库根> && python3 tests/ops/ascend/test_chunk_gated_delta_rule.py --cpu-only
  - 昇腾 NPU（完整用例，需算子已编译并加载 CANN 算子包环境；解释器需已安装 torch_npu）：
        cd <仓库根>
        source vllm_fl/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash
        python3 tests/ops/ascend/test_chunk_gated_delta_rule.py
"""
import sys

import torch

# ---------------------------------------------------------------------------
# 判据阈值
COS_THRESHOLD = 0.998      # 余弦相似度（官方容差）
REL_L2_THRESHOLD = 5e-2    # 相对 L2 误差 ‖a-b‖₂/‖b‖₂（官方容差）
# 逐元素最大误差上界按 numpy.allclose 约定：|a-b| <= ATOL + RTOL * max|b|
# （纯绝对阈值会随输出量级变化而误判，故采用绝对 + 相对混合上界）
ATOL = 1e-2                # 绝对误差下限（覆盖 b 接近 0 的元素）
RTOL = 5e-2                # 相对误差系数（覆盖 b 量级较大的元素）

TORCH_OP = "torch.ops._C_ascend.npu_chunk_gated_delta_rule"
CPU_ONLY = "--cpu-only" in sys.argv

# ---------------------------------------------------------------------------
# 环境探测：NPU 相关依赖做成可选，保证无卡时也能跑 CPU 用例
NPU_AVAILABLE = False
NPU_IMPORT_ERROR = ""
try:
    import torch_npu  # noqa: F401

    import vllm_fl._C_ascend  # noqa: F401  确保算子注册

    NPU_AVAILABLE = True
except Exception as _e:  # pragma: no cover - 环境相关
    NPU_IMPORT_ERROR = f"{type(_e).__name__}: {_e}"


def require_npu(name):
    """NPU 用例前置检查；不可用时明确跳过（不伪装通过）。"""
    if not NPU_AVAILABLE:
        print(f"[SKIP] {name}：当前无可用 NPU 算子环境（{NPU_IMPORT_ERROR}）")
        return False
    return True


# ---------------------------------------------------------------------------
# 判据：cos + 相对 L2 + 最大绝对误差
def metrics(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    if a.norm() == 0 and b.norm() == 0:
        cos = 1.0
    elif a.norm() == 0 or b.norm() == 0:
        cos = 0.0
    else:
        cos = torch.nn.functional.cosine_similarity(
            a.unsqueeze(0), b.unsqueeze(0)).item()
    rel_l2 = (torch.norm(a - b) / torch.norm(b).clamp_min(1e-12)).item()
    max_err = (a - b).abs().max().item()
    # 逐元素最大误差上界：绝对 + 相对（相对项按参考张量的量级 max|b| 计算）
    err_bound = ATOL + RTOL * b.abs().max().item()
    norm_ratio = (torch.norm(a) / torch.norm(b).clamp_min(1e-12)).item()
    return {"cos": cos, "rel_l2": rel_l2, "max_err": max_err,
            "err_bound": err_bound, "norm_ratio": norm_ratio}


def judge(a, b, name="", verbose=True):
    """三判据判定（cos / 相对 L2 / 逐元素最大误差上界）；返回 (ok, metrics)。"""
    m = metrics(a, b)
    ok = (m["cos"] > COS_THRESHOLD
          and m["rel_l2"] < REL_L2_THRESHOLD
          and m["max_err"] <= m["err_bound"])
    if verbose:
        print(f"  {name}: cos={m['cos']:.6f} (> {COS_THRESHOLD})"
              f"  rel_l2={m['rel_l2']:.3e} (< {REL_L2_THRESHOLD})"
              f"  max_err={m['max_err']:.3e} (<= {m['err_bound']:.3e})"
              f"  ‖a‖/‖b‖={m['norm_ratio']:.4f}  -> {'PASS' if ok else 'FAIL'}")
    return ok, m


# ---------------------------------------------------------------------------
# 参考实现（fp32，逐 token 递推）—— 顺序与 kernel 一致：先衰减，再算更新量
def ref_gated_delta_rule(q, k, v, g, beta, initial_state, seqlens, scale):
    """逐 token 递推的 fp32 参考实现。

    GQA：ratio = Nv // Nk，q/k 按 ratio repeat_interleave 到每个 value head。
    State S 布局 (B, Nv, Dv, Dk)：
        S = S · exp(g_t)                     # 1) 先衰减（kernel: CalGCumExp）
        u = S @ k_t                          # 2) 用衰减后的 state
        w = beta_t · (v_t − u)               # 3) 更新量
        S = S + w ⊗ k_t                      # 4) 再累加
        out_t = scale · (S @ q_t)
    """
    T, Nk, Dk = q.shape
    Nv, Dv = v.shape[1], v.shape[2]
    ratio = Nv // Nk
    B = initial_state.shape[0]
    # 参考实现语义为 fp32：内部统一 dtype，避免调用方传入 fp64 等导致运算不匹配
    q = q.to(torch.float32)
    k = k.to(torch.float32)
    v = v.to(torch.float32)
    beta = beta.to(torch.float32)
    if g is not None:
        g = g.to(torch.float32)
    S = initial_state.to(torch.float32).clone()
    out = torch.zeros(T, Nv, Dv, dtype=torch.float32, device=q.device)
    start = 0
    for b in range(B):
        L = int(seqlens[b])
        for t in range(start, start + L):
            k_t = k[t].repeat_interleave(ratio, dim=0)
            q_t = q[t].repeat_interleave(ratio, dim=0)
            if g is not None:
                # 1) 先衰减
                S[b] = S[b] * torch.exp(g[t]).view(Nv, 1, 1)
            # 2) 用衰减后的 state 计算 u
            u = torch.einsum("nvk,nk->nv", S[b], k_t)
            # 3) 更新量
            w = beta[t].unsqueeze(-1) * (v[t] - u)
            # 4) 累加
            S[b] = S[b] + torch.einsum("nv,nk->nvk", w, k_t)
            out[t] = scale * torch.einsum("nvk,nk->nv", S[b], q_t)
        start += L
    return out, S


def ref_wrong_order_for_contrast(q, k, v, g, beta, initial_state, seqlens, scale):
    """修正前的错误顺序（先算 u/w 再衰减），仅用于 CPU 对照，说明差异来源。"""
    T, Nk, Dk = q.shape
    Nv, Dv = v.shape[1], v.shape[2]
    ratio = Nv // Nk
    B = initial_state.shape[0]
    # 参考实现语义为 fp32：内部统一 dtype，避免调用方传入 fp64 等导致运算不匹配
    q = q.to(torch.float32)
    k = k.to(torch.float32)
    v = v.to(torch.float32)
    beta = beta.to(torch.float32)
    if g is not None:
        g = g.to(torch.float32)
    S = initial_state.to(torch.float32).clone()
    out = torch.zeros(T, Nv, Dv, dtype=torch.float32, device=q.device)
    start = 0
    for b in range(B):
        L = int(seqlens[b])
        for t in range(start, start + L):
            k_t = k[t].repeat_interleave(ratio, dim=0)
            q_t = q[t].repeat_interleave(ratio, dim=0)
            u = torch.einsum("nvk,nk->nv", S[b], k_t)      # ← 用了未衰减的 state
            w = beta[t].unsqueeze(-1) * (v[t] - u)
            if g is not None:
                S[b] = S[b] * torch.exp(g[t]).view(Nv, 1, 1)
            S[b] = S[b] + torch.einsum("nv,nk->nvk", w, k_t)
            out[t] = scale * torch.einsum("nvk,nk->nv", S[b], q_t)
        start += L
    return out, S


# ===========================================================================
# CPU 用例（无需 NPU，可随时运行）
# ===========================================================================
def test_ref_single_token_cpu():
    """CPU 单 token 锚点：验证参考实现的递推顺序。

    构造（Dk=Dv=64 → scale = 64^-0.5 = 0.125；Nk=Nv=1）：
        q = k = one-hot e0；beta = 1；v[0] = 0.25；
        initial_state[0,0,0,0] = 0.25；g = -20（exp(-20) ≈ 2.06e-9，近似全衰减）

    正确顺序（先衰减）：
        S ← 0.25·e^-20 ≈ 0；w = 1·(0.25 − ≈0) ≈ 0.25；S ≈ 0.25
        out = 0.125 × 0.25 = 0.03125
    错误顺序（先算 u 再衰减）：
        u = S·k = 0.25；w = 0.25 − 0.25 = 0；S ≈ 0
        out ≈ 0
    """
    Dk = Dv = 64
    scale = Dk ** -0.5
    T = 1
    q = torch.zeros(T, 1, Dk, dtype=torch.float64)
    k = torch.zeros(T, 1, Dk, dtype=torch.float64)
    q[0, 0, 0] = 1.0
    k[0, 0, 0] = 1.0
    v = torch.zeros(T, 1, Dv, dtype=torch.float64)
    v[0, 0, 0] = 0.25
    beta = torch.ones(T, 1, dtype=torch.float64)
    g = torch.full((T, 1), -20.0, dtype=torch.float64)
    S0 = torch.zeros(1, 1, Dv, Dk, dtype=torch.float64)
    S0[0, 0, 0, 0] = 0.25
    seqlens = [1]

    out_ok, state_ok = ref_gated_delta_rule(q, k, v, g, beta, S0, seqlens, scale)
    out_bad, state_bad = ref_wrong_order_for_contrast(q, k, v, g, beta, S0, seqlens, scale)

    print(f"[CPU 单token锚点] 正确顺序 out={out_ok[0, 0, 0].item():.8f}"
          f"  错误顺序 out={out_bad[0, 0, 0].item():.8e}")
    print(f"[CPU 单token锚点] 正确顺序 final_state={state_ok[0, 0, 0, 0].item():.8f}"
          f"  错误顺序 final_state={state_bad[0, 0, 0, 0].item():.8e}")

    expected = 0.03125
    assert abs(out_ok[0, 0, 0].item() - expected) < 1e-6, \
        f"参考实现 out 应为 {expected}，实际 {out_ok[0, 0, 0].item()}"
    assert abs(state_ok[0, 0, 0, 0].item() - 0.25) < 1e-6, \
        f"参考实现 final_state 应为 0.25，实际 {state_ok[0, 0, 0, 0].item()}"
    assert abs(out_bad[0, 0, 0].item()) < 1e-6, \
        "错误顺序应给 ≈0，用于说明该用例能区分两种递推顺序"
    print("[CPU 单token锚点] PASS（正确顺序 = 0.03125，错误顺序 ≈ 0，可区分）")
    return True


def test_metric_rejects_scaled_output():
    """判据自检：整体缩放与局部尖峰都必须被判 FAIL（cos 单独做不到）。

    - 整体 ×10：cos 仍≈1，但 rel_l2 = 9.0、max_err 远超上界 -> FAIL
    - 单点尖峰 +0.5：rel_l2 仅 ~1e-2（< 5e-2）察觉不到，但 max_err 超上界 -> FAIL
      （说明新增的逐元素最大误差项确实补上了 rel_l2 的盲区）
    """
    torch.manual_seed(0)
    ref = torch.randn(64, 4, 8, dtype=torch.float64)

    ok_same, _ = judge(ref, ref.clone(), "identical", verbose=False)
    assert ok_same, "完全相同的张量应判定通过"

    scaled = ref * 10.0
    ok_scaled, m_scaled = judge(scaled, ref, "scaled-x10", verbose=False)
    print(f"[判据自检] 整体×10：cos={m_scaled['cos']:.6f}（仍≈1，说明 cos 挡不住）"
          f"  rel_l2={m_scaled['rel_l2']:.4f}  max_err={m_scaled['max_err']:.4f}"
          f" / 上界{m_scaled['err_bound']:.4f}"
          f"  -> {'PASS(不合格判据)' if ok_scaled else 'FAIL(预期)'}")
    assert not ok_scaled, "整体放大 10 倍仍判定通过 → 判据不足"
    assert m_scaled["cos"] > 0.999, "预期 cos 对整体缩放不敏感"

    spike = ref.clone()
    spike.view(-1)[0] += 0.5
    ok_spike, m_spike = judge(spike, ref, "spike", verbose=False)
    print(f"[判据自检] 单点尖峰+0.5：cos={m_spike['cos']:.6f}"
          f"  rel_l2={m_spike['rel_l2']:.4f}（< {REL_L2_THRESHOLD}，rel_l2 挡不住）"
          f"  max_err={m_spike['max_err']:.4f} / 上界{m_spike['err_bound']:.4f}"
          f"  -> {'PASS(不合格判据)' if ok_spike else 'FAIL(预期)'}")
    assert not ok_spike, "单点最大误差超界仍判定通过 → 判据不足"
    assert m_spike["rel_l2"] < REL_L2_THRESHOLD, "预期 rel_l2 对单点尖峰不敏感"

    print("[判据自检] PASS（整体缩放由 rel_l2 拦住，局部尖峰由 max_err 拦住）")
    return True



def test_ref_shapes_cpu():
    """CPU 冒烟：参考实现在变长 + 多 batch 下的形状与有限性检查。"""
    torch.manual_seed(42)
    Nk, Nv, Dk, Dv = 2, 4, 64, 64
    seqlens = [7, 65]
    T = sum(seqlens)
    q = torch.randn(T, Nk, Dk, dtype=torch.float64)
    k = torch.randn(T, Nk, Dk, dtype=torch.float64)
    q = q / q.norm(dim=-1, keepdim=True)
    k = k / k.norm(dim=-1, keepdim=True)
    v = torch.randn(T, Nv, Dv, dtype=torch.float64) * 0.1
    beta = torch.rand(T, Nv, dtype=torch.float64).sigmoid()
    g = -torch.nn.functional.softplus(torch.randn(T, Nv, dtype=torch.float64))
    S0 = torch.randn(2, Nv, Dv, Dk, dtype=torch.float64) * 0.01
    out, state = ref_gated_delta_rule(q, k, v, g, beta, S0, seqlens, Dk ** -0.5)
    assert out.shape == (T, Nv, Dv), out.shape
    assert state.shape == (2, Nv, Dv, Dk), state.shape
    assert torch.isfinite(out).all() and torch.isfinite(state).all()
    print(f"[CPU 冒烟] PASS  out{tuple(out.shape)}  final_state{tuple(state.shape)}")
    return True


# ===========================================================================
# NPU 用例
# ===========================================================================
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
    g = (-torch.nn.functional.softplus(
        torch.randn(T, Nv, device=dev), beta=1.0)).float() if with_g else None
    if nonzero_init:
        initial_state = torch.randn(B, Nv, Dv, Dk, dtype=torch.bfloat16, device=dev) * 0.01
    else:
        initial_state = torch.zeros(B, Nv, Dv, Dk, dtype=torch.bfloat16, device=dev)
    actual_seq_lengths = torch.tensor(seqlens, dtype=torch.int32, device=dev)
    scale = Dk ** -0.5
    return q, k, v, beta, g, initial_state, actual_seq_lengths, scale


def run_case(name, Nk, Nv, Dk, Dv, seqlens, with_g=True, nonzero_init=True):
    """单个正确性用例：AscendC 输出 vs fp32 参考，双输出（out/final_state）三判据对拍。"""
    if not require_npu(name):
        return None
    q, k, v, beta, g, initial_state, actual_seq_lengths, scale = make_inputs(
        Nk, Nv, Dk, Dv, seqlens, with_g, nonzero_init)
    out, final_state = torch.ops._C_ascend.npu_chunk_gated_delta_rule(
        q, k, v, beta, initial_state, actual_seq_lengths, g=g, scale_value=scale)
    torch.npu.synchronize()
    ref_out, ref_state = ref_gated_delta_rule(
        q.float(), k.float(), v.float(), g, beta.float(), initial_state, seqlens, scale)

    print(f"[{name}] T={sum(seqlens)} seqlens={seqlens} Nk={Nk} Nv={Nv} Dk={Dk} Dv={Dv}")
    ok_out, _ = judge(out.cpu(), ref_out.cpu(), "out")
    ok_state, _ = judge(final_state.cpu(), ref_state.cpu(), "final_state")
    ok = ok_out and ok_state
    assert ok, (f"[{name}] 双输出三判据未全过：cos > {COS_THRESHOLD}, "
                f"rel_l2 < {REL_L2_THRESHOLD}, max_err <= ATOL + RTOL*max|b|（ATOL={ATOL}, RTOL={RTOL}）")
    return True


def test_wrong_dtype():
    """异常用例 1：query 传 FP32（契约要求 BF16），应抛出异常。"""
    if not require_npu("异常dtype"):
        return None
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
    if not require_npu("异常shape"):
        return None
    q, k, v, beta, g, initial_state, actual_seq_lengths, scale = make_inputs(
        2, 4, 64, 64, seqlens=[64])
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


def test_fixed_cases():
    """四重点固定用例（任务书 §5）。"""
    if not require_npu("固定用例"):
        return None
    run_case("等长+初始State",      Nk=2, Nv=4, Dk=64, Dv=64, seqlens=[128],           with_g=True,  nonzero_init=True)
    run_case("变长[7,65]",          Nk=2, Nv=4, Dk=64, Dv=64, seqlens=[7, 65],          with_g=True,  nonzero_init=True)
    run_case("chunk边界[63,64,65]", Nk=2, Nv=4, Dk=64, Dv=64, seqlens=[63, 64, 65],    with_g=True,  nonzero_init=True)
    run_case("L=1实际长度",         Nk=2, Nv=4, Dk=64, Dv=64, seqlens=[1],              with_g=True,  nonzero_init=True)
    run_case("零初始State",         Nk=2, Nv=4, Dk=64, Dv=64, seqlens=[128],           with_g=True,  nonzero_init=False)
    run_case("无g衰减",             Nk=2, Nv=4, Dk=64, Dv=64, seqlens=[64],            with_g=False, nonzero_init=True)
    return True


def main():
    print("=" * 72)
    print("Chunk Gated Delta Rule 单元测试")
    print(f"算子: {TORCH_OP}")
    print(f"判据: cos > {COS_THRESHOLD}  且  rel_l2 < {REL_L2_THRESHOLD}"
          f"  且  max_err <= ATOL + RTOL*max|b|（ATOL={ATOL}, RTOL={RTOL}）")
    print("=" * 72)

    print("\n--- CPU 用例（无需 NPU）---")
    test_ref_single_token_cpu()
    test_metric_rejects_scaled_output()
    test_ref_shapes_cpu()

    if CPU_ONLY:
        print("\n[cpu-only] 已跳过 NPU 用例")
        print("=" * 72)
        print("CPU TESTS PASSED")
        return

    print("\n--- NPU 用例 ---")
    if not NPU_AVAILABLE:
        print(f"[SKIP] NPU 环境不可用（{NPU_IMPORT_ERROR}）")
        print("=" * 72)
        print("CPU TESTS PASSED / NPU TESTS SKIPPED（请在有卡环境重跑以完成实测）")
        return

    test_fixed_cases()
    test_wrong_dtype()
    test_wrong_shape()
    print("=" * 72)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()

