"""变长序列验证 v2（全用例传 cu_seqlens，绕开 flag_gems i_t bug）"""
import torch
import torch_npu
import vllm_fl._C_ascend
from vllm_fl.dispatch.backends.vendor.ascend.impl.fla.chunk import chunk_gated_delta_rule as cgdr

def cosine(a, b):
    a, b = a.cpu().flatten().double(), b.cpu().flatten().double()
    if a.norm() == 0 and b.norm() == 0:
        return 1.0
    if a.norm() == 0 or b.norm() == 0:
        return 0.0
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

def main():
    dev = "npu"
    # ===== 用例 A：官方手算解析用例（带 cu_seqlens）=====
    print("=== 用例 A：手算解析验证（scale=1/sqrt(K) 语义）===")
    q = torch.tensor([[[[1.0, 0.0]]]], dtype=torch.bfloat16).to(dev)
    k = torch.tensor([[[[1.0, 0.0]]]], dtype=torch.bfloat16).to(dev)
    v = torch.tensor([[[[10.0, 20.0, 30.0]]]], dtype=torch.bfloat16).to(dev)
    g = torch.zeros(1, 1, 1, dtype=torch.float32, device=dev)
    beta = torch.ones(1, 1, 1, dtype=torch.bfloat16, device=dev)
    init = torch.tensor([[[[1.0, 2.0], [4.0, 8.0], [16.0, 32.0]]]], dtype=torch.bfloat16, device=dev)
    cu1 = torch.tensor([0, 1], dtype=torch.long, device=dev)
    o, fs = cgdr(q, k, v, g, beta, initial_state=init, output_final_state=True, cu_seqlens=cu1)
    torch.npu.synchronize()
    expected_o = torch.tensor([[[[10.0, 20.0, 30.0]]]], dtype=torch.float32) / (2.0**0.5)
    expected_fs = torch.tensor([[[[10.0, 2.0], [20.0, 8.0], [30.0, 32.0]]]], dtype=torch.float32)
    ca = cosine(o, expected_o); cb = cosine(fs, expected_fs)
    print(f"  o cos={ca:.6f}  final_state cos={cb:.6f}  (期望≈1.0)")
    ok_a = ca > 0.999 and cb > 0.999
    print("  用例A:", "PASS" if ok_a else "FAIL")

    # ===== 用例 B：变长序列 L=[5,12]，T=17 =====
    print("=== 用例 B：变长序列 L=[5,12] cu_seqlens=[0,5,17] ===")
    torch.manual_seed(7)
    H, Hv, K, V = 4, 8, 64, 64
    L1, L2 = 5, 12
    T = L1 + L2
    q = torch.randn(1, T, H, K, dtype=torch.bfloat16, device=dev)
    k = torch.randn(1, T, H, K, dtype=torch.bfloat16, device=dev)
    v = torch.randn(1, T, Hv, V, dtype=torch.bfloat16, device=dev)
    g = (-torch.nn.functional.softplus(torch.randn(1, T, Hv, device=dev), beta=1.0)).float()
    beta = torch.rand(1, T, Hv, dtype=torch.bfloat16, device=dev).sigmoid()
    init = torch.randn(2, Hv, K, V, dtype=torch.bfloat16, device=dev) * 0.01
    cu = torch.tensor([0, L1, T], dtype=torch.long, device=dev)
    o_v, fs_v = cgdr(q, k, v, g, beta, initial_state=init, output_final_state=True, cu_seqlens=cu)
    torch.npu.synchronize()
    print("  o:", tuple(o_v.shape), "final_state:", tuple(fs_v.shape))
    o1, fs1 = cgdr(q[:, :L1], k[:, :L1], v[:, :L1], g[:, :L1], beta[:, :L1],
                   initial_state=init[0:1], output_final_state=True,
                   cu_seqlens=torch.tensor([0, L1], dtype=torch.long, device=dev))
    o2, fs2 = cgdr(q[:, L1:], k[:, L1:], v[:, L1:], g[:, L1:], beta[:, L1:],
                   initial_state=init[1:2], output_final_state=True,
                   cu_seqlens=torch.tensor([0, L2], dtype=torch.long, device=dev))
    torch.npu.synchronize()
    c1 = cosine(o_v[:, :L1], o1); c2 = cosine(o_v[:, L1:], o2)
    c3 = cosine(fs_v[0], fs1[0]); c4 = cosine(fs_v[1], fs2[0])
    print(f"  seq0(L=5)  o cos={c1:.6f}  fs cos={c3:.6f}")
    print(f"  seq1(L=12) o cos={c2:.6f}  fs cos={c4:.6f}")
    ok_b = all(x > 0.99 for x in [c1, c2, c3, c4])
    print("  用例B:", "PASS" if ok_b else "FAIL")

    print("=" * 30)
    print("变长验证全部通过" if (ok_a and ok_b) else "存在失败")

if __name__ == "__main__":
    main()
