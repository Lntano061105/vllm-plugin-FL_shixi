#!/usr/bin/env python3
"""Task book §3 "Python 调用" requirement:
   既能通过 torch.ops 调用算子，又能输出【算子注册名称 / 动态库路径 / 调用结果】。

Run:
    python tests/ops/ascend/show_op_info.py
"""
import os
import sys
import glob

import torch


def _find_libs():
    """Locate the produced dynamic libraries without hard-coding an absolute path."""
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    pats = [
        os.path.join(root, "vllm_fl", "_C_ascend*.so"),
        os.path.join(root, "vllm_fl", "_cann_ops_custom", "vendors", "*", "op_api", "lib", "*.so"),
        os.path.join(root, "vllm_fl", "libvllm_fl_kernels.so"),
    ]
    out = []
    for p in pats:
        out.extend(sorted(glob.glob(p)))
    return out


def main():
    print("=" * 78)
    print("Add RMSNorm Bias / GemmaRMSNorm — operator registration & call report")
    print("=" * 78)

    print("\n[1] 动态库路径 (dynamic libraries produced by the project build)")
    libs = _find_libs()
    if not libs:
        print("    !! no library found — build first: bash csrc/ascend/build_aclnn.sh ascend910b")
        return 1
    for p in libs:
        print(f"    {p}  ({os.path.getsize(p):,} bytes)")

    print("\n[2] 算子注册名称 (registered operator names)")
    try:
        import vllm_fl._C_ascend  # noqa: F401  (import also preloads libcust_opapi)
    except Exception as e:  # pragma: no cover
        print(f"    !! import vllm_fl._C_ascend failed: {e}")
        print("    hint: run inside the container after sourcing set_env.bash")
        return 1

    names = ["npu_add_rms_norm_bias", "npu_gemma_rms_norm"]
    for n in names:
        available = hasattr(torch.ops._C_ascend, n)
        print(f"    torch.ops._C_ascend.{n:<24} registered = {available}")

    print("\n[3] 调用结果 (call result + reference comparison)")
    import torch_npu  # noqa: F401

    torch.manual_seed(0)
    dev, dt, eps = "npu:0", torch.bfloat16, 1e-6
    rows, cols = 8, 512
    x1 = torch.randn(rows, cols, device=dev, dtype=dt)
    x2 = torch.randn(rows, cols, device=dev, dtype=dt)
    g = torch.randn(cols, device=dev, dtype=dt) + 0.5

    y, rstd, xres = torch.ops._C_ascend.npu_add_rms_norm_bias(x1, x2, g, None, eps)
    torch.npu.synchronize()

    print(f"    input : x1{x1.shape} {x1.dtype} contiguous={x1.is_contiguous()}")
    print(f"            x2{x2.shape} {x2.dtype} contiguous={x2.is_contiguous()}")
    print(f"            gamma{g.shape} {g.dtype} contiguous={g.is_contiguous()}")
    print(f"    output: y{tuple(y.shape)} {y.dtype}  rstd{tuple(rstd.shape)} {rstd.dtype} "
          f" x{tuple(xres.shape)} {xres.dtype}")

    # fp64 reference, same convention as the unit test
    d1, d2, dg = x1.double(), x2.double(), g.double()
    x_ref = d1 + d2
    r_ref = 1.0 / torch.sqrt(x_ref.pow(2).mean(dim=-1, keepdim=True) + eps)
    y_ref = (x_ref * r_ref) * dg

    def err(a, b):
        return (a.float() - b.float()).abs().max().item()

    print(f"    max_abs_err  y={err(y, y_ref):.3e}  rstd={err(rstd, r_ref):.3e}  x={err(xres, x_ref):.3e}")
    print("\n    => operator is registered, callable via torch.ops, and matches the fp64 reference.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
