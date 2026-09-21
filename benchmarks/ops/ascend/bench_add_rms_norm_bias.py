# Copyright (c) 2026 Intern deliverable (Add RMSNorm Bias / GemmaRMSNorm).
# SPDX-License-Identifier: Apache-2.0
#
# Microbenchmark for npu_add_rms_norm_bias vs the torch_npu baseline
# (torch_npu.npu_add_rms_norm, which does not support beta and does not
# return rstd). Methodology follows the task book: warmup + per-iteration
# device sync + repeated runs; report mean / P50 / P90 / min / max latency.
#
# Usage:
#   python bench_add_rms_norm_bias.py
#   python bench_add_rms_norm_bias.py --iters 200 --warmup 20 --dtype bfloat16
#
# NOTE: run inside the container with the custom-op package built and
# vllm_fl._C_ascend importable.

import argparse
import statistics
import time

import torch

try:
    import torch_npu  # noqa: F401
except ImportError as e:  # pragma: no cover
    raise SystemExit(f"torch_npu required: {e}")

import vllm_fl._C_ascend  # noqa: F401  (registers torch.ops._C_ascend.*)

DEVICE = "npu:0"


def _bench(fn, warmup, iters):
    """Time fn() with a device sync per iteration; return per-iter latencies (ms)."""
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()

    lats = []
    for _ in range(iters):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.npu.synchronize()
        lats.append((time.perf_counter() - t0) * 1e3)
    return lats


def _summarize(name, lats, numel):
    lats = sorted(lats)
    mean = statistics.fmean(lats)
    p50 = lats[len(lats) // 2]
    p90 = lats[int(len(lats) * 0.90)]
    gbps = numel * 2 / (mean * 1e-3) / 1e9  # fp16/bf16 2 bytes/elem (rough, 3 tensors in+2 out)
    print(f"{name:<28} mean={mean:8.4f}ms  P50={p50:8.4f}ms  P90={p90:8.4f}ms  "
          f"min={min(lats):8.4f}ms  max={max(lats):8.4f}ms")
    return dict(name=name, mean=mean, p50=p50, p90=p90)


def main():
    parser = argparse.ArgumentParser(description="AddRmsNormBias microbenchmark")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--shapes", nargs="*", default=None,
                        help="e.g. '1,3584' '1024,3584' '1024,128' (default: decode/prefill/qk-norm)")
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    shapes = args.shapes or ["1,3584", "1024,3584", "1024,128"]

    print(f"device: {DEVICE}  dtype: {args.dtype}  warmup: {args.warmup}  iters: {args.iters}\n")

    for spec in shapes:
        rows, cols = (int(v) for v in spec.split(","))
        torch.manual_seed(0)
        x1 = torch.randn(rows, cols, device=DEVICE, dtype=dtype)
        x2 = torch.randn(rows, cols, device=DEVICE, dtype=dtype)
        gamma = (torch.randn(cols, device=DEVICE, dtype=dtype) + 0.5)
        eps = 1e-6
        numel = rows * cols

        print(f"--- shape ({rows}, {cols}) dtype={args.dtype} ---")

        def custom():
            return torch.ops._C_ascend.npu_add_rms_norm_bias(x1, x2, gamma, None, eps)

        def baseline():
            # torch_npu baseline: no beta support, rstd discarded
            return torch_npu.npu_add_rms_norm(x1, x2, gamma, eps)

        # correctness sanity (custom vs baseline y)
        y_c, rstd_c, x_c = custom()
        y_b, _, x_b = baseline()
        torch.npu.synchronize()
        assert torch.allclose(y_c, y_b, rtol=2e-2, atol=2e-2), f"y mismatch at {spec}"
        assert torch.allclose(x_c, x_b, rtol=1e-3, atol=1e-3), f"x mismatch at {spec}"

        _summarize(f"custom   ({rows},{cols})", _bench(custom, args.warmup, args.iters), numel)
        _summarize(f"baseline ({rows},{cols})", _bench(baseline, args.warmup, args.iters), numel)
        print()


if __name__ == "__main__":
    main()
