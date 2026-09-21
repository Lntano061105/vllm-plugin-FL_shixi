# Copyright (c) 2026 Intern deliverable (Add RMSNorm Bias / GemmaRMSNorm).
# SPDX-License-Identifier: Apache-2.0
#
# Unit tests for the custom AddRmsNormBias / GemmaRMSNorm operators.
#
# Fixed test focus (task book): compare normalized output y, rstd and
# residual x against an fp64 reference implementation, with a fixed random
# seed, on fixed shapes covering:
#   - dtype: fp16 / bf16 / fp32
#   - beta: absent (None) and present (bias path)
#   - shapes: small (q/k norm like), model-like hidden sizes, and the
#     SPLIT_D path (num_col larger than the UB budget)
#
# Usage:
#   python test_add_rms_norm_bias.py            # run all cases
#   python test_add_rms_norm_bias.py --shapes 16,128 1024,5120
#   pytest test_add_rms_norm_bias.py            # pytest-compatible
#
# NOTE: requires an Ascend environment (torch_npu) and the built custom-op
# package (bash csrc/ascend/build_aclnn.sh ascend910b), i.e. run inside the
# container.

import argparse
import os
import sys

import torch

# ---------------------------------------------------------------------------
# environment bootstrap
# ---------------------------------------------------------------------------
def _enable_custom_op() -> bool:
    """Point ASCEND_CUSTOM_OPP_PATH / LD_LIBRARY_PATH at the packaged custom
    op installed under vllm_fl/_cann_ops_custom (same idea as the repo's
    vllm_fl.utils.enable_custom_op, kept local so the test is standalone)."""
    from pathlib import Path

    try:
        import vllm_fl._C_ascend  # noqa: F401  (importing also preloads opapi via plugin)
        custom_ops_root = Path(vllm_fl.__file__).parent / "_cann_ops_custom"
    except Exception:
        custom_ops_root = None

    if custom_ops_root is not None and custom_ops_root.is_dir():
        vendor_dir = custom_ops_root / "vendors" / "custom_transformer"
        lib_dir = vendor_dir / "op_api" / "lib"
        opp = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")
        if str(vendor_dir) not in opp:
            os.environ["ASCEND_CUSTOM_OPP_PATH"] = str(vendor_dir) + (":" + opp if opp else "")
        ld = os.environ.get("LD_LIBRARY_PATH", "")
        if str(lib_dir) not in ld:
            os.environ["LD_LIBRARY_PATH"] = str(lib_dir) + (":" + ld if ld else "")
        import ctypes
        try:
            ctypes.CDLL(str(lib_dir / "libcust_opapi.so"), mode=ctypes.RTLD_LOCAL)
        except OSError:
            pass
        return True
    return False


def _require_device() -> None:
    if not _enable_custom_op():
        print(
            "ERROR: vllm_fl/_cann_ops_custom is not installed.\n"
            "Build the CANN framework operators first, e.g.:\n"
            "  bash csrc/ascend/build_aclnn.sh ascend910b",
            file=sys.stderr,
        )
        sys.exit(1)
    try:
        import torch_npu  # noqa: F401
    except ImportError as e:
        print(f"ERROR: torch_npu is not available: {e}", file=sys.stderr)
        sys.exit(1)
    try:
        import vllm_fl._C_ascend  # noqa: F401,F811
    except ImportError as e:
        print(f"ERROR: vllm_fl._C_ascend is not importable: {e}", file=sys.stderr)
        sys.exit(1)


DEVICE = "npu:0"

# ---------------------------------------------------------------------------
# fp64 reference implementations
# ---------------------------------------------------------------------------
def ref_add_rms_norm_bias(x1, x2, gamma, eps, beta=None):
    """Reference for npu_add_rms_norm_bias in fp64.

    x     = x1 + x2
    rstd  = 1 / sqrt(mean(x^2) + eps)
    y     = x * rstd * gamma (+ beta)
    """
    x1 = x1.double()
    x2 = x2.double()
    gamma = gamma.double()
    x = x1 + x2
    rstd = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    y = x * rstd * gamma
    if beta is not None:
        y = y + beta.double()
    return y, rstd, x


def ref_gemma_rms_norm(x, gamma, eps):
    """Reference for npu_gemma_rms_norm (Gemma 1+weight convention)."""
    x = x.double()
    gamma = gamma.double()
    rstd = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    y = x * (1.0 + gamma) * rstd
    return y, rstd


# ---------------------------------------------------------------------------
# per-case check
# ---------------------------------------------------------------------------
def _assert_close(name, got, want, rtol, atol):
    max_abs = (got.double() - want.double()).abs().max().item()
    rel = (got.double() - want.double()).abs().div(want.double().abs() + 1e-12).max().item()
    print(f"  [{name}] max_abs={max_abs:.3e} max_rel={rel:.3e}")
    if not torch.allclose(got, want, rtol=rtol, atol=atol, equal_nan=False):
        max_abs = (got.double() - want.double()).abs().max().item()
        rel = (got.double() - want.double()).abs().div(want.double().abs() + 1e-12).max().item()
        raise AssertionError(
            f"{name}: max_abs={max_abs:.3e} max_rel={rel:.3e} "
            f"(rtol={rtol}, atol={atol})")


def run_case(x1, x2, gamma, eps, beta, dtype, rtol_y, atol_y, rtol_r, atol_r, tag):
    """Run one fixed case and verify y / rstd / x against the fp64 reference."""
    # layout: assert the contract described at the top of this file
    x1 = x1.to(device=DEVICE, dtype=dtype)
    x2 = x2.to(device=DEVICE, dtype=dtype)
    gamma = gamma.to(device=DEVICE, dtype=dtype)
    if beta is not None:
        beta = beta.to(device=DEVICE, dtype=dtype)
    assert x1.is_contiguous() and x2.is_contiguous() and gamma.is_contiguous(), "inputs must be contiguous"
    # The op reduces over the LAST dim of x1/x2 (gamma has 1 dim). Cases in this file use
    # 2-D (M, N) model-like shapes and 1-D q/k-norm-like shapes, so both are legitimate.
    assert x1.dim() == x2.dim() and gamma.dim() == 1, "x1/x2 must share a rank; gamma must be 1-D"
    assert x1.shape[-1] == gamma.shape[0] if x1.dim() >= 1 else True, "gamma must match the last dim of x"
    print(f"[layout] x={tuple(x1.shape)}/{x1.dtype}/stride={tuple(x1.stride())} "
          f"gamma={tuple(gamma.shape)}/{gamma.dtype} beta={'yes' if beta is not None else 'none'}")

    x2 = x2.to(device=DEVICE, dtype=dtype)
    gamma = gamma.to(device=DEVICE, dtype=dtype)
    beta = None if beta is None else beta.to(device=DEVICE, dtype=dtype)

    y, rstd, x_out = torch.ops._C_ascend.npu_add_rms_norm_bias(
        x1, x2, gamma, beta, float(eps))

    # --- shapes / dtypes ---------------------------------------------------
    assert y.shape == x1.shape, f"{tag}: y shape {y.shape} != {x1.shape}"
    assert x_out.shape == x1.shape, f"{tag}: x shape {x_out.shape} != {x1.shape}"
    assert y.dtype == dtype, f"{tag}: y dtype {y.dtype} != {dtype}"
    assert x_out.dtype == dtype, f"{tag}: x dtype {x_out.dtype} != {dtype}"
    assert rstd.dtype == torch.float32, f"{tag}: rstd dtype {rstd.dtype} != fp32"

    want_rstd_shape = list(x1.shape[:-gamma.dim()]) + [1] * gamma.dim()
    assert list(rstd.shape) == want_rstd_shape, (
        f"{tag}: rstd shape {list(rstd.shape)} != {want_rstd_shape}")

    # --- numerics vs fp64 reference ----------------------------------------
    y_ref, rstd_ref, x_ref = ref_add_rms_norm_bias(x1.cpu(), x2.cpu(), gamma.cpu(), eps, beta.cpu() if beta is not None else None)

    # residual: x == x1 + x2 (same dtype arithmetic; tolerance per dtype:
    # bf16/fp16 addition carries ~0.4%/0.1% relative error at |x|~3)
    TOL_X = {torch.float16: (2e-3, 2e-3), torch.bfloat16: (4e-3, 4e-3), torch.float32: (1e-3, 1e-3)}
    rtol_x, atol_x = TOL_X[dtype]
    _assert_close(f"{tag}/residual(x)", x_out.cpu().double(), x_ref, rtol=rtol_x, atol=atol_x)
    # rstd: fp32 output vs fp64 reference (fp32 accumulation error dominated)
    _assert_close(f"{tag}/rstd", rstd.cpu().double(), rstd_ref, rtol=rtol_r, atol=atol_r)
    # normalized output (cast back to the input dtype by the op)
    _assert_close(f"{tag}/y", y.cpu(), y_ref.to(dtype), rtol=rtol_y, atol=atol_y)


# ---------------------------------------------------------------------------
# fixed test cases
# ---------------------------------------------------------------------------
def _cases():
    torch.manual_seed(0)
    cases = []
    # (shape, eps, use_beta, tag)
    specs = [
        ((16, 128), 1e-6, False, "small-2d-nobeta"),
        ((16, 128), 1e-6, True, "small-2d-beta"),
        ((1, 64), 1e-6, False, "single-row"),
        ((2, 4, 512), 1e-6, True, "3d"),
        ((1024, 3584), 1e-6, False, "model-like-hidden3584"),
        ((1024, 5120), 1e-5, True, "model-like-hidden5120-beta"),
        ((32, 32768), 1e-6, False, "split-d-fp16"),   # D > UB budget -> SPLIT_D
        ((1, 8, 128), 1e-6, False, "qk-norm-like"),    # q_norm/k_norm path
    ]
    for (shape, eps, use_beta, tag) in specs:
        x1 = torch.randn(*shape)
        x2 = torch.randn(*shape)
        gamma = torch.randn(shape[-1]) + 0.5  # positive-ish, like a learned scale
        beta = torch.randn(shape[-1]) if use_beta else None
        cases.append((x1, x2, gamma, eps, beta, tag))
    return cases


TOL = {
    torch.float16: dict(rtol_y=2e-2, atol_y=2e-2, rtol_r=1e-3, atol_r=1e-5),
    torch.bfloat16: dict(rtol_y=2e-2, atol_y=2e-2, rtol_r=1e-3, atol_r=1e-5),
    torch.float32: dict(rtol_y=1e-4, atol_y=1e-5, rtol_r=1e-4, atol_r=1e-6),
}


def test_add_rms_norm_bias_all():
    """Run every (case, dtype) combination; fixed seed => reproducible."""
    torch.manual_seed(0)
    for (x1, x2, gamma, eps, beta, tag) in _cases():
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            tol = TOL[dtype]
            run_case(
                x1, x2, gamma, eps, beta, dtype,
                tol["rtol_y"], tol["atol_y"], tol["rtol_r"], tol["atol_r"],
                f"{tag}/{dtype}")


def test_gemma_rms_norm():
    """npu_gemma_rms_norm (no-residual path) uses the 1+weight convention."""
    torch.manual_seed(0)
    x = torch.randn(16, 128)
    gamma = torch.randn(128) * 0.1  # Gemma offset weights are small
    eps = 1e-6
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        y, rstd = torch.ops._C_ascend.npu_gemma_rms_norm(
            x.to(device=DEVICE, dtype=dtype),
            gamma.to(device=DEVICE, dtype=dtype),
            eps)
        y_ref, rstd_ref = ref_gemma_rms_norm(x, gamma, eps)
        tol = TOL[dtype]
        _assert_close(f"gemma/y/{dtype}", y.cpu(), y_ref.to(dtype), tol["rtol_y"], tol["atol_y"])
        _assert_close(f"gemma/rstd/{dtype}", rstd.cpu().double(), rstd_ref, tol["rtol_r"], tol["atol_r"])
        assert rstd.dtype == torch.float32


def _main():
    parser = argparse.ArgumentParser(description="AddRmsNormBias unit tests")
    parser.add_argument("--shapes", nargs="*", default=None,
                        help="override shapes as 'rows,cols' (e.g. '16,128' '1024,5120')")
    args = parser.parse_args()

    _require_device()
    print(f"device: {DEVICE}")
    print(f"op registered: npu_add_rms_norm_bias={hasattr(torch.ops._C_ascend, 'npu_add_rms_norm_bias')}, "
          f"npu_gemma_rms_norm={hasattr(torch.ops._C_ascend, 'npu_gemma_rms_norm')}")

    if args.shapes is not None:
        # user-provided shapes: build ad-hoc cases
        torch.manual_seed(0)
        for spec in args.shapes:
            rows, cols = (int(v) for v in spec.split(","))
            x1 = torch.randn(rows, cols)
            x2 = torch.randn(rows, cols)
            gamma = torch.randn(cols) + 0.5
            for dtype in (torch.float16, torch.bfloat16, torch.float32):
                tol = TOL[dtype]
                run_case(x1, x2, gamma, 1e-6, None, dtype,
                         tol["rtol_y"], tol["atol_y"], tol["rtol_r"], tol["atol_r"],
                         f"{rows}x{cols}/{dtype}")
        print("all user-provided cases passed")
        return

    test_gemma_rms_norm()
    test_add_rms_norm_bias_all()
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    _main()
