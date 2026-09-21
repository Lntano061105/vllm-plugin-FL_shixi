# Copyright (c) 2026 BAAI. All rights reserved.

"""
Ascend normalization operator implementations.
"""

from __future__ import annotations

from typing import Optional, Union

import torch


def _custom_rmsnorm_enabled() -> bool:
    """Whether the custom fused AddRmsNormBias operator may be used.

    Delegates to the framework patch module so that the disable switch and the
    dependency checks (CANN custom-op run package, _C_ascend extension, required
    op names) stay identical on both entry points. Returning True while the CANN
    run package is missing would make the call fail at runtime.
    """
    try:
        from ..patches.patch_add_rms_norm_bias import _custom_ops_available

        return _custom_ops_available()
    except Exception:
        return False


def rms_norm_ascend(
    obj,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """
    RMS normalization using Ascend NPU.

    Args:
        obj: The calling obj (e.g., RMSNorm layer)
        x: Input tensor
        residual: Optional residual tensor to add before normalization

    Returns:
        Normalized tensor, or tuple of (normalized, residual) if residual is provided
    """
    import torch_npu

    weight = obj.weight
    epsilon = obj.variance_epsilon

    if _custom_rmsnorm_enabled() and residual is not None:
        # Custom fused path: add + RMSNorm in one operator, and the residual is
        # returned from the operator instead of a separate add.
        # NOTE: the no-residual path intentionally keeps the torch_npu baseline,
        # because npu_gemma_rms_norm applies the Gemma 1+weight convention which
        # does not hold for the plain RMSNorm wrapper.
        try:
            y, _rstd, residual_out = torch.ops._C_ascend.npu_add_rms_norm_bias(
                x, residual, weight, None, epsilon
            )
            return y, residual_out
        except Exception:
            pass  # fall through to the baseline on any runtime failure

    # Baseline (torch_npu): no beta support, rstd discarded.
    if residual is not None:
        x, _, residual = torch_npu.npu_add_rms_norm(x, residual, weight, epsilon)
        return x, residual

    x, _ = torch_npu.npu_rms_norm(x, weight, epsilon)
    return x
