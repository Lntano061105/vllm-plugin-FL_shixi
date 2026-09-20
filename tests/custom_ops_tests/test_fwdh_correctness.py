"""fwd_h 算子正确性对拍 v2（官方 cpu_reference + final_state 对比）
覆盖：h_state 序列 / v_new / final_state，断言 cosine >= 0.99
"""
import torch
import torch_npu  # noqa: F401  引入 npu 设备接口

import vllm_fl._C_ascend  # noqa: F401  确保算子注册

CHUNK_SIZE = 64

def npu_chunk_gdr_fwd_h(k, w, u, g, initial_state=None, chunk_size=64):
    return torch.ops._C_ascend.chunk_gated_delta_rule_fwd_h(
        k, w, u, g=g, initial_state=initial_state,
        output_final_state=True, chunk_size=chunk_size, save_new_value=True,
    )

def cpu_reference(k, w, u, g, initial_state=None, chunk_size=64):
    """CPU fp32 reference（官方原版）"""
    k, w, u, g = k.float(), w.float(), u.float(), g.float()
    B, Hg, T, K = k.shape
    HV, V = u.shape[1], u.shape[3]
    NT = T // chunk_size
    h = initial_state.float().clone() if initial_state is not None else torch.zeros(B, HV, K, V)
    h_chunks = [h.clone()]
    v_new = torch.zeros_like(u)
    for c in range(NT):
        t0 = c * chunk_size
        W_chunk = w[:, :, t0:t0+chunk_size, :]
        ws = torch.einsum("bhik,bhkv->bhiv", W_chunk, h)
        g_chunk = g[:, :, t0:t0+chunk_size]
        v_update = torch.zeros(B, HV, chunk_size, V)
        for i in range(chunk_size):
            gi_cum = g_chunk[:, :, -1] - g_chunk[:, :, i]
            vn = u[:, :, t0+i, :] - ws[:, :, i, :]
            v_new[:, :, t0+i, :] = vn
            v_update[:, :, i, :] = gi_cum.unsqueeze(-1).exp() * vn
        K_chunk = k[:, :, t0:t0+chunk_size, :]
        h_work = torch.einsum("bhik,bhiv->bhkv", K_chunk, v_update)
        h = h * g_chunk[:, :, -1:].unsqueeze(-1).exp() + h_work
        h_chunks.append(h.clone())
    return h_chunks, v_new

def cosine(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    if a.norm() == 0 and b.norm() == 0:
        return 1.0
    if a.norm() == 0 or b.norm() == 0:
        return 0.0
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

def run_case(B, Hg, HV, T, K, V):
    torch.manual_seed(42)
    DTYPE = torch.float16
    k = torch.randn(B, Hg, T, K, dtype=DTYPE) * 0.1
    w = torch.randn(B, Hg, T, K, dtype=DTYPE) * 0.1
    u = torch.randn(B, HV, T, V, dtype=DTYPE) * 0.1
    g = (-torch.rand(B, HV, T) * 0.1).float()
    init = torch.randn(B, HV, K, V, dtype=DTYPE) * 0.01
    dev = "npu"
    h_ref, vn_ref = cpu_reference(k, w, u, g, init, CHUNK_SIZE)
    h_out, vn_out, final_state = npu_chunk_gdr_fwd_h(
        k.to(dev), w.to(dev), u.to(dev), g.to(dev), init.to(dev), CHUNK_SIZE)
    torch.npu.synchronize()
    h_npu, vn_npu = h_out.cpu().float(), vn_out.cpu().float()
    fs_npu = final_state.cpu().float()
    NT = T // CHUNK_SIZE
    print(f"--- case B={B} Hg={Hg} HV={HV} T={T} K={K} V={V} ---")
    print("h_out:", tuple(h_out.shape), "v_new:", tuple(vn_out.shape), "final:", tuple(final_state.shape))
    ok = True
    for c in range(min(NT + 1, h_npu.shape[2])):
        cos = cosine(h_npu[0, :, c], h_ref[c])
        print(f"  h_state[{c}] cos = {cos:.6f}")
        if cos < 0.99:
            ok = False
    for c in range(NT):
        t0, t1 = c * CHUNK_SIZE, (c+1) * CHUNK_SIZE
        cos = cosine(vn_npu[:, :, t0:t1], vn_ref[:, :, t0:t1])
        print(f"  v_new[{c}] cos = {cos:.6f}")
        if cos < 0.99:
            ok = False
    cos_fs = cosine(fs_npu, h_ref[NT])
    print(f"  final_state cos = {cos_fs:.6f}  (目标: h_ref[NT])")
    if cos_fs < 0.99:
        ok = False
    print("  PASS" if ok else "  FAIL")
    return ok

if __name__ == "__main__":
    all_ok = True
    all_ok &= run_case(1, 1, 1, 128, 128, 128)
    all_ok &= run_case(1, 2, 2, 128, 128, 128)
    print("=" * 30)
    print("全部通过" if all_ok else "存在失败")
