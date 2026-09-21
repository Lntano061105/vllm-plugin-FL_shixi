# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Patch Qwen3 MoE routing to use the AscendC MoE Gating Top-K operator.
#
# The vllm_fl plugin defines FusedMoEFL (a subclass of vLLM's FusedMoE)
# with its own select_experts method.  Patching FusedMoE alone has no
# effect because the subclass does not dispatch to the parent method.
# This patch therefore targets FusedMoEFL when available, and falls back
# to FusedMoE otherwise.

import glob
import logging
import os
from functools import wraps

import torch

logger = logging.getLogger(__name__)


def _load_ascend_op_library():
    """Load the _C_ascend torch extension from the vllm_fl package."""
    import vllm_fl

    vllm_fl_dir = os.path.dirname(os.path.abspath(vllm_fl.__file__))
    so_files = glob.glob(os.path.join(vllm_fl_dir, "_C_ascend*.so"))
    if not so_files:
        raise RuntimeError(
            f"_C_ascend*.so not found under {vllm_fl_dir}; "
            "did the vllm_fl package build correctly?"
        )
    torch.ops.load_library(so_files[0])
    logger.info("[MoePatch] Loaded Ascend op library: %s", so_files[0])



def _resolve_target_cls():
    """Return the FusedMoE subclass actually used by the model.

    The vllm_fl plugin registers FusedMoEFL and the model instantiates
    that subclass, so its own select_experts wins over the parent's.
    """
    try:
        from vllm_fl.ops.fused_moe.layer import FusedMoEFL

        logger.info("[MoePatch] Using FusedMoEFL from vllm_fl")
        return FusedMoEFL
    except ImportError:
        from vllm.model_executor.layers.fused_moe.layer import FusedMoE

        logger.info("[MoePatch] FusedMoEFL not found, using FusedMoE")
        return FusedMoE


def apply_patch():
    """Patch the FusedMoE subclass used by the model with AscendC Top-K."""
    try:
        _load_ascend_op_library()
        target_cls = _resolve_target_cls()

        original_select_experts = target_cls.select_experts

        @wraps(original_select_experts)
        def patched_select_experts(self, hidden_states, router_logits):
            # Only substitute the plain fused_topk path. Grouped topk,
            # EPLB, bias-corrected routing and custom routing functions
            # keep using the original implementation.
            can_use_ascendc = (
                not self.enable_eplb
                and not self.use_grouped_topk
                and self.e_score_correction_bias is None
                and self.custom_routing_function is None
            )
            if not can_use_ascendc:
                return original_select_experts(self, hidden_states, router_logits)

            try:
                topk_weights, topk_ids, _ = torch.ops._C_ascend.moe_gating_top_k(
                    router_logits,
                    k=self.top_k,
                    k_group=1,
                    group_count=1,
                    group_select_mode=0,
                    renorm=int(self.renormalize),
                    norm_type=0,  # 0: softmax, 1: sigmoid
                    out_flag=False,
                    routed_scaling_factor=self.routed_scaling_factor,
                    eps=1e-20,
                )
                indices_type = self.quant_method.topk_indices_dtype
                if indices_type is not None and topk_ids.dtype != indices_type:
                    topk_ids = topk_ids.to(dtype=indices_type)

                logger.debug(
                    "[MoePatch] moe_gating_top_k used: weights=%s ids=%s",
                    tuple(topk_weights.shape),
                    tuple(topk_ids.shape),
                )
                return topk_weights, topk_ids, None
            except Exception as e:
                logger.warning(
                    "[MoePatch] moe_gating_top_k failed (%s); "
                    "falling back to original select_experts",
                    e,
                )
                return original_select_experts(self, hidden_states, router_logits)

        target_cls.select_experts = patched_select_experts
        logger.info(
            "[MoePatch] Patched %s.select_experts with AscendC moe_gating_top_k",
            target_cls.__name__,
        )
        return True
    except Exception as e:
        logger.error("[MoePatch] Failed to apply patch: %s", e)
        return False
