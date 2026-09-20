"""fwd_h 算子 Microbenchmark：预热 + 同步 + 重复，报 mean/P50/P90"""
import time
import torch
import torch_npu
import vllm_fl._C_ascend

CHUNK = 64

def bench_once(k, w, u, g, init):
    torch.ops._C_ascend.chunk_gated_delta_rule_fwd_h(
        k, w, u, g=g, initial_state=init,
        output_final_state=True, chunk_size=CHUNK, save_new_value=True)

def main():
    torch.manual_seed(42)
    B, Hg, Hv, T, K, V = 1, 4, 8, 1024, 128, 128   # 接近模型 prefill 规模
    dev = "npu"
    k = (torch.randn(B, Hg, T, K, dtype=torch.bfloat16, device=dev) * 0.1)
    w = (torch.randn(B, Hg, T, K, dtype=torch.bfloat16, device=dev) * 0.1)
    u = (torch.randn(B, Hv, T, V, dtype=torch.bfloat16, device=dev) * 0.1)
    g = (-torch.rand(B, Hv, T) * 0.1).float().to(dev)
    init = (torch.randn(B, Hv, K, V, dtype=torch.bfloat16, device=dev) * 0.01)

    WARMUP, REPEAT = 20, 100
    for _ in range(WARMUP):
        bench_once(k, w, u, g, init)
    torch.npu.synchronize()

    times = []
    for _ in range(REPEAT):
        t0 = time.perf_counter()
        bench_once(k, w, u, g, init)
        torch.npu.synchronize()
        times.append((time.perf_counter() - t0) * 1000)  # ms
    times.sort()
    mean = sum(times) / len(times)
    p50 = times[len(times) // 2]
    p90 = times[int(len(times) * 0.90)]
    print(f"shape: B={B} Hg={Hg} Hv={Hv} T={T} K={K} V={V} chunk={CHUNK}")
    print(f"WARMUP={WARMUP} REPEAT={REPEAT}")
    print(f"Mean: {mean:.3f} ms | P50: {p50:.3f} ms | P90: {p90:.3f} ms")
    print(f"min={times[0]:.3f} ms | max={times[-1]:.3f} ms")

if __name__ == "__main__":
    main()
