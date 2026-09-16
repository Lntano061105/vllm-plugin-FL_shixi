import argparse
import torch, torch_npu, vllm_fl._C_ascend

def bench(call, warmup, iters, reps):
    for _ in range(warmup):
        call()
    torch.npu.synchronize()
    us = []
    for _ in range(iters):
        s = torch.npu.Event(enable_timing=True)
        e = torch.npu.Event(enable_timing=True)
        s.record()
        for _ in range(reps):
            call()
        e.record()
        torch.npu.synchronize()
        us.append(s.elapsed_time(e) * 1000.0 / reps)
    t = torch.tensor(us)
    p = torch.quantile(t, torch.tensor([0.5, 0.9])).tolist()
    print(f"min={t.min().item():.2f} p50={p[0]:.2f} p90={p[1]:.2f} max={t.max().item():.2f} avg={t.mean().item():.2f} us  (n={iters} x reps={reps})")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dim', type=int, default=3584, help='GDN conv 输入维度 (TP4 时每卡=hidden/4)')
    ap.add_argument('--width', type=int, default=4)
    ap.add_argument('--tokens', type=int, default=1024, help='prefill 序列长度')
    ap.add_argument('--batch', type=int, default=1, help='decode 批大小')
    ap.add_argument('--mode', choices=['prefill', 'decode', 'both'], default='both')
    ap.add_argument('--warmup', type=int, default=100)
    ap.add_argument('--iters', type=int, default=1000)
    ap.add_argument('--reps', type=int, default=1, help='每个计时窗口内连发次数(摊销launch开销, 1=逐次)')
    ap.add_argument('--device', type=int, default=0)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    torch.npu.set_device(a.device)
    q = torch.bfloat16
    D, W = a.dim, a.width
    wt = torch.randn(W, D, dtype=q, device='npu') * 0.3
    bias = torch.randn(D, dtype=q, device='npu') * 0.1

    def do(mode, n, tag):
        if mode == 'prefill':
            x = torch.randn(n, D, dtype=q, device='npu')
            cs = torch.zeros(1, W - 1, D, dtype=q, device='npu')
            qsl = torch.tensor([0, n], dtype=torch.int32, device='npu')
            out = torch.empty(n, D, dtype=q, device='npu')
            call = lambda: torch.ops._C_ascend.npu_causal_conv1d_custom(
                out, x, wt, cs, bias, qsl, None, None, None, 0, -1, 0)
        else:
            x = torch.randn(n, D, dtype=q, device='npu')
            cs = torch.zeros(n, W - 1, D, dtype=q, device='npu')
            out = torch.empty(n, D, dtype=q, device='npu')
            call = lambda: torch.ops._C_ascend.npu_causal_conv1d_custom(
                out, x, wt, cs, bias, None, None, None, None, 0, -1, 1)
        print(f"{tag}: mode={mode} D={D} W={W} n={n} | ", end='')
        bench(call, a.warmup, a.iters, a.reps)

    if a.mode in ('prefill', 'both'):
        do('prefill', a.tokens, 'PREFILL')
    if a.mode in ('decode', 'both'):
        do('decode', a.batch, 'DECODE')

if __name__ == '__main__':
    main()
