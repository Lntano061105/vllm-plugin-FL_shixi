# Copyright (c) 2026 Intern deliverable (Add RMSNorm Bias / GemmaRMSNorm).
# SPDX-License-Identifier: Apache-2.0
#
# Framework patch: wire the custom AddRmsNormBias / GemmaRMSNorm operators into
# the Qwen3.5/Qwen3.6 inference path.
#
# Design (independent of the reference R8 patch, which patches the whole GDN
# op set together):
#   * only patches GemmaRMSNorm (the common path: input_layernorm /
#     post_attention_layernorm / final norm / q_norm / k_norm) and the
#     dispatch-managed rms_norm backend (impl/normalization.py);
#   * enable/disable switch:  VLLM_FL_DISABLE_ASCENDC_RMSNORM=1  keeps the
#     original torch_npu / vLLM-native path;
#   * automatic fallback: if vllm_fl._C_ascend or the required ops are
#     missing at runtime, the patch is skipped (upstream behavior kept).
#
# How to enable (one-line hook, idempotent):
#   in vllm_fl/dispatch/backends/vendor/ascend/patch.py::apply_ascend_patches,
#   add:
#       patch_add_rms_norm_bias()
#   alongside the other patch_*() calls.

import logging
import os

import torch

logger = logging.getLogger(__name__)

# =1 -> keep the original (torch_npu / vLLM-native) implementation.
_ENV_SWITCH = "VLLM_FL_DISABLE_ASCENDC_RMSNORM"

_REQUIRED_OPS = (
    "npu_add_rms_norm_bias",
    "npu_gemma_rms_norm",
)


def _custom_ops_available() -> bool:
    """Custom ops usable? Controlled by the switch + runtime availability."""
    if os.environ.get(_ENV_SWITCH, "0") == "1":
        logger.info("%s=1, keep the torch_npu / vLLM-native RMSNorm path", _ENV_SWITCH)
        return False
    try:
        import vllm_fl._C_ascend  # noqa: F401
    except Exception as e:
        logger.warning("vllm_fl._C_ascend unavailable (%s); RMSNorm patch skipped", e)
        return False
    missing = [name for name in _REQUIRED_OPS if not hasattr(torch.ops._C_ascend, name)]
    if missing:
        logger.warning("torch.ops._C_ascend missing %s; RMSNorm patch skipped", missing)
        return False
    return True


class AscendCGemmaRMSNorm:
    """GemmaRMSNorm forward_oot backed by the custom operators.

    Semantics (Gemma 1+weight convention):
      * residual is not None (fused add+rmsnorm path):
            y, rstd, residual_out = npu_add_rms_norm_bias(x, residual, 1+weight, None, eps)
            -> returns (y, residual_out)
      * residual is None (plain rmsnorm path):
            y, rstd = npu_gemma_rms_norm(x, weight, eps)   # op applies 1+weight internally
    """

    def forward_oot(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is not None:
            y, _, residual_out = torch.ops._C_ascend.npu_add_rms_norm_bias(
                x, residual, 1.0 + self.weight, None, self.variance_epsilon
            )
            return y, residual_out
        y, _ = torch.ops._C_ascend.npu_gemma_rms_norm(
            x, self.weight, self.variance_epsilon
        )
        return y


def patch_add_rms_norm_bias() -> bool:
    """Apply the AddRmsNormBias framework patch.

    Returns True when the custom ops were wired in; otherwise the original
    (torch_npu / vLLM-native) implementation is kept.
    """
    if not _custom_ops_available():
        return False
    try:
        from vllm.model_executor.layers.layernorm import GemmaRMSNorm

        GemmaRMSNorm.forward_oot = AscendCGemmaRMSNorm.forward_oot
        logger.info(
            "Patched GemmaRMSNorm.forward_oot with AscendC "
            "npu_add_rms_norm_bias / npu_gemma_rms_norm "
            "(disable with %s=1)", _ENV_SWITCH,
        )
        return True
    except Exception as e:
        logger.warning("Failed to patch GemmaRMSNorm for Ascend: %s", e)
        return False
