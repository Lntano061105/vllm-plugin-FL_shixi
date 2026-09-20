# Copyright (c) 2026 Intern deliverable (Add RMSNorm Bias / GemmaRMSNorm).
# SPDX-License-Identifier: Apache-2.0
#
# Framework patch: wire the custom AddRmsNormBias / GemmaRMSNorm operators into
# the Qwen3.5/Qwen3.6 inference path.
#
# Design (independent of the reference R8 patch, which patches the whole GDN
# op set together):
#   * only patches GemmaRMSNorm (the common path: input_layernorm /
#     post_attention_layernorm / final norm) and the dispatch-managed rms_norm
#     backend (impl/normalization.py);
#   * enable/disable switch:  VLLM_FL_DISABLE_ASCENDC_RMSNORM=1 restores the
#     true baseline (torch_npu / vLLM-native), including when another patch has
#     already replaced GemmaRMSNorm.forward_oot;
#   * automatic fallback: the patch is skipped - and the baseline restored -
#     when the CANN custom-op package or the _C_ascend bindings are unavailable.
#
# How to enable (one-line hook, idempotent):
#   in vllm_fl/dispatch/backends/vendor/ascend/patch.py::apply_ascend_patches,
#   add:
#       patch_add_rms_norm_bias()
#   alongside the other patch_*() calls.

import ctypes
import logging
import os

import torch

logger = logging.getLogger(__name__)

# =1 -> restore the original (torch_npu / vLLM-native) implementation.
_ENV_SWITCH = "VLLM_FL_DISABLE_ASCENDC_RMSNORM"

_REQUIRED_OPS = (
    "npu_add_rms_norm_bias",
    "npu_gemma_rms_norm",
)

# Directory name of the packaged CANN custom-op vendor produced by
# csrc/ascend/build_aclnn.sh and installed under
# vllm_fl/_cann_ops_custom/vendors/. The _C_ascend bindings can be importable
# while this package is missing (e.g. the extension was built but the run
# package was never installed), in which case calling the op fails at runtime.
_CUSTOM_OPP_MARKER = "custom_transformer"

# Modules whose GemmaRMSNorm.forward_oot override may be removed when this
# switch is off. Restricting the removal to plugin patches guarantees that a
# future vLLM-native forward_oot is never deleted by accident.
_PATCH_MODULE_PREFIXES = ("vllm_fl.", "vllm_ascend.")


def _bootstrap_custom_op_env() -> bool:
    """Make the packaged CANN custom-op package discoverable at runtime.

    Same idea as the GDN patch / vllm-ascend ``bootstrap_custom_op_env``:
    prepend the packaged ``_cann_ops_custom/vendors/custom_transformer`` dir to
    ``ASCEND_CUSTOM_OPP_PATH`` so the server does not have to be launched from a
    shell that already sourced ``set_env.bash``. The OPP path is scanned lazily
    by the AscendCL runtime at the first custom-op call, so setting it before
    any op invocation is sufficient; the variable is also inherited by the
    spawned worker processes.

    ``libcust_opapi.so`` is additionally preloaded by absolute path: the aclnn
    adapter resolves the custom symbols with ``dlopen("libcust_opapi.so")`` by
    bare name, which only searches the *startup-time* ``LD_LIBRARY_PATH``
    (glibc caches it) - but finds the already-loaded library by SONAME after a
    preload. ``RTLD_LOCAL`` is used on purpose: ``RTLD_GLOBAL`` leads to a
    double-free at process teardown.

    Returns True when the package is present and usable.
    """
    try:
        import vllm_fl._C_ascend as _ext  # noqa: F401
    except Exception as e:
        logger.warning("Failed to import vllm_fl._C_ascend: %s", e)
        return False

    vendor_dir = os.path.join(
        os.path.dirname(_ext.__file__), "_cann_ops_custom", "vendors",
        _CUSTOM_OPP_MARKER,
    )
    if not os.path.isdir(vendor_dir):
        logger.warning("CANN custom op package not found at %s", vendor_dir)
        return False

    opp_path = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")
    if vendor_dir not in opp_path:
        os.environ["ASCEND_CUSTOM_OPP_PATH"] = (
            vendor_dir + (":" + opp_path if opp_path else "")
        )

    lib_dir = os.path.join(vendor_dir, "op_api", "lib")
    lib_path = os.path.join(lib_dir, "libcust_opapi.so")
    if not os.path.isfile(lib_path):
        logger.warning("CANN custom op library not found at %s", lib_path)
        return False

    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if lib_dir not in ld_path:
        os.environ["LD_LIBRARY_PATH"] = lib_dir + (":" + ld_path if ld_path else "")
    try:
        ctypes.CDLL(lib_path, mode=ctypes.RTLD_LOCAL)
    except OSError as e:
        logger.warning("Failed to preload libcust_opapi.so: %s", e)
        return False
    return True


def _custom_ops_available() -> bool:
    """Custom ops usable? Controlled by the switch + runtime availability.

    Three conditions must hold, and all three are checked here (the extension
    and the registered op names alone are not sufficient - the CANN run package
    must also be present, otherwise the call fails at runtime):

      1. the switch is not set;
      2. the packaged CANN custom-op package exists and is on the OPP path;
      3. ``vllm_fl._C_ascend`` imports and registers the required op names.
    """
    if os.environ.get(_ENV_SWITCH, "0") == "1":
        logger.info("%s=1, keep the torch_npu / vLLM-native RMSNorm path", _ENV_SWITCH)
        return False

    # The CANN run package must be present. This check is deliberately *not*
    # skipped when ASCEND_CUSTOM_OPP_PATH already contains the marker: the
    # variable may have been exported by a shell profile (or set by another
    # patch) while the package itself was never installed, in which case the
    # extension imports and the op names register, but the call fails at
    # runtime. _bootstrap_custom_op_env() validates the directory and the
    # library and is idempotent, so it is always run.
    if not _bootstrap_custom_op_env():
        logger.warning(
            "CANN custom op environment is not set and auto-bootstrap "
            "failed; keep the torch_npu / vLLM-native RMSNorm path"
        )
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


def _restore_baseline_forward_oot() -> None:
    """Restore the true baseline for ``GemmaRMSNorm.forward_oot``.

    ``VLLM_FL_DISABLE_ASCENDC_RMSNORM=1`` must disable the AscendC RMSNorm path
    as a whole, not merely decline to install this patch. Another patch in the
    same family (``patch_qwen3_6_gdn``) also assigns
    ``GemmaRMSNorm.forward_oot`` and its implementation calls
    ``npu_add_rms_norm_bias`` too, so simply returning early would leave the
    AscendC path active and the "OFF" baseline would not be a baseline.

    Removing the class-level override makes attribute lookup fall through to
    ``CustomOp.forward_oot`` -> ``forward_native`` -> ``GemmaRMSNorm.forward_static``,
    which is exactly the pre-patch behaviour. Only overrides installed by a
    plugin patch are removed, so a vLLM-native ``forward_oot`` (should upstream
    add one) is preserved.
    """
    try:
        from vllm.model_executor.layers.layernorm import GemmaRMSNorm
    except Exception as e:  # pragma: no cover - vLLM always has this module
        logger.warning("Cannot import GemmaRMSNorm for baseline restore: %s", e)
        return

    current = GemmaRMSNorm.__dict__.get("forward_oot")
    if current is None:
        return  # nothing installed -> already the baseline
    module = getattr(current, "__module__", "") or ""
    if not module.startswith(_PATCH_MODULE_PREFIXES):
        logger.info(
            "GemmaRMSNorm.forward_oot is defined by %s (not a plugin patch); "
            "leaving it untouched", module,
        )
        return
    del GemmaRMSNorm.forward_oot
    logger.info(
        "Restored the baseline GemmaRMSNorm.forward_oot (removed the override "
        "from %s); AscendC RMSNorm is disabled", module,
    )


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

    Returns True when the custom ops were wired in; otherwise the baseline
    (torch_npu / vLLM-native) implementation is left in place *and* any
    AscendC override installed by an earlier patch is removed, so the switch
    is effective on its own.
    """
    if not _custom_ops_available():
        _restore_baseline_forward_oot()
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
        _restore_baseline_forward_oot()
        return False
