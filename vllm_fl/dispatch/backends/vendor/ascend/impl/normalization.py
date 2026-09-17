# Copyright (c) 2026 BAAI. All rights reserved.

"""
Ascend normalization operator implementations.
"""

from __future__ import annotations

import os
from typing import Optional, Union

import torch


def _custom_rmsnorm_enabled() -> bool:
    """Whether the custom AddRmsNormBias operator may be used.

    Returns False when explicitly disabled via the environment switch, when the
    compiled extension is unavailable, or when the operator is not registered.
    """
    if os.environ.get("VLLM_FL_DISABLE_ASCENDC_RMSNORM", "0") == "1":
        return False
    try:
        import vllm_fl._C_ascend  # noqa: F401
        return hasattr(torch.ops._C_ascend, "npu_add_rms_norm_bias")
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
