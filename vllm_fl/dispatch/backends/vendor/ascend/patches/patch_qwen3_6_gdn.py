#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# Copyright (c) 2026 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# mypy: ignore-errors

"""AscendC fused-op patch for Qwen3.5/Qwen3.6 GatedDeltaNet (GDN) layers.

Ports the vllm-ascend ``AscendGatedDeltaNetAttention`` integration
(``vllm_ascend/ops/gdn.py`` and ``vllm_ascend/ops/layernorm.py``) to the
FL plugin on vLLM 0.13:

* ``npu_causal_conv1d_custom`` replaces the Triton ``causal_conv1d_fn`` /
  ``causal_conv1d_update`` calls inside ``Qwen3NextGatedDeltaNet._forward_core``.
* ``npu_fused_gdn_gating`` replaces the Triton ``fused_gdn_gating``.
* ``npu_recurrent_gated_delta_rule`` replaces ``fused_recurrent_gated_delta_rule``
  on the (speculative-)decode paths.  The chunked prefill path keeps the
  existing Triton ``chunk_gated_delta_rule`` (already patched to the Ascend
  implementation by ``patch_fla_ops``).
* ``npu_gemma_rms_norm`` / ``npu_add_rms_norm_bias`` back
  ``GemmaRMSNorm.forward_oot``.

Layout notes (must stay consistent with the kernels):

* conv_state: vLLM 0.13 allocates the GDN conv cache as
  ``(state_len, conv_dim)`` per slot, which is exactly what the AscendC
  kernel expects, so the cache is passed through *without* the transpose
  used by the Triton path.
* ssm_state: the AscendC ``recurrent_gated_delta_rule`` kernel expects the
  state in ``(Hv, Dv, Dk)`` layout (see
  ``csrc/ascend/attention/recurrent_gated_delta_rule``), while vLLM 0.13
  allocates ``(Hv, Dk, Dv)``.  ``get_state_shape`` is therefore patched to
  swap the last two dims, and the chunked-prefill path transposes the
  initial/final state at the boundary.
* ``actual_seq_lengths`` of ``npu_recurrent_gated_delta_rule`` follows the
  cu_seqlens convention ``[0, len_1, ..., len_B]`` (batch = numel - 1).
* mamba KV-cache: upstream stores conv/ssm states interleaved inside one
  (padded) page per block; the AscendC kernels address the state caches
  assuming dense per-state tensors, so ``_reshape_kv_cache_tensors`` is
  wrapped to regroup the state views into dense per-state tensors over the
  same raw storage (transparent to the Triton fallback path, which uses
  explicit strides).

The patch bootstraps the CANN custom-op environment automatically
(``ASCEND_CUSTOM_OPP_PATH`` pointing at the packaged
``_cann_ops_custom/vendors/custom_transformer``) and is only skipped when
the ``_C_ascend`` bindings or the op package are unavailable; otherwise the
existing Triton path is kept. Set ``VLLM_FL_DISABLE_ASCENDC_GDN=1`` to
force the Triton path.
"""

import atexit
import logging
import math
import os

import torch
from vllm.attention.backends.abstract import AttentionMetadata
from vllm.attention.backends.utils import PAD_SLOT_ID
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator
from vllm.model_executor.models.qwen3_next import Qwen3NextGatedDeltaNet
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.kv_cache_interface import MambaSpec

import vllm.model_executor.models.qwen3_next as _qwen3_next_lib

from ..impl.fla.l2norm import l2norm_fwd

logger = logging.getLogger(__name__)

_CUSTOM_OPP_MARKER = "custom_transformer"
_REQUIRED_OPS = (
    "npu_causal_conv1d_custom",
    "npu_fused_gdn_gating",
    "npu_recurrent_gated_delta_rule",
    "npu_gemma_rms_norm",
    "npu_add_rms_norm_bias",
)


def _install_gdn_gating_counter() -> None:
    """Env-gated invocation counter for ``npu_fused_gdn_gating``.

    When ``VLLM_FL_GDN_COUNT_FILE`` points at a writable path, every real
    call of the AscendC gating op is tallied and appended to that file as
    ``<pid> <count>`` lines (flushed every 1024 calls and at exit).  Used to
    prove the op actually executes inside the model inference path (task
    requirement), including multi-process / multi-card TP setups where a
    wrapper installed in the driver process would not see the calls.  The
    wrap is a no-op unless the env var is set, so the normal patch behavior
    is unchanged.
    """
    count_file = os.environ.get("VLLM_FL_GDN_COUNT_FILE", "")
    if not count_file:
        return
    orig = torch.ops._C_ascend.npu_fused_gdn_gating
    state = {"count": 0}

    def _flush() -> None:
        if state["count"]:
            try:
                with open(count_file, "a") as fh:
                    fh.write(f"{os.getpid()} {state['count']}\n")
            except OSError:  # e.g. disk full during shutdown; never fail a call
                pass
            state["count"] = 0

    def _counting_op(*args, **kwargs):
        state["count"] += 1
        if state["count"] % 1024 == 0:
            _flush()
        return orig(*args, **kwargs)

    torch.ops._C_ascend.npu_fused_gdn_gating = _counting_op
    atexit.register(_flush)
    logger.info("npu_fused_gdn_gating invocation counter -> %s", count_file)


def _soc_is_ascend950():
    """判断当前设备是否为 ascend950（唯一支持 FP32 state 的平台）。

    查询失败时保守返回 False（按 ascend910b 处理，仅允许 bf16 state）。
    """
    names = []
    try:
        names.append(str(torch.npu.get_device_name(0)))
    except Exception:
        pass
    if not names:
        try:
            import torch_npu  # noqa: F401

            names.append(str(torch_npu.npu.get_device_name(0)))
        except Exception:
            pass
    for name in names:
        if "950" in name.lower():
            return True
    return False


def _ascendc_chunk_gdn_supported(query, key, value, beta, initial_state, g):
    """判断当前输入能否走 AscendC 完整版 chunk GDR 算子；返回 (是否可用, 原因)。

    判据与算子侧原型定义 / tiling 的校验保持一致，任一条件不满足即回退 Triton 基线：
      1) 算子已注册（未编译或 .so 未加载时 hasattr 为 False）
      2) dtype 契约：query / key / value / beta 为 bf16；g（可选）为 fp32
      3) 平台 x state dtype：ascend910b 仅支持 bf16 state，FP32 state 需 ascend950
    """
    try:
        if not hasattr(torch.ops._C_ascend, "npu_chunk_gated_delta_rule"):
            return False, "operator npu_chunk_gated_delta_rule not registered"
    except Exception as exc:  # 扩展未加载等
        return False, "op registration check failed: %s" % exc

    if query.dtype != torch.bfloat16 or key.dtype != torch.bfloat16:
        return False, "query/key dtype must be bf16, got %s/%s" % (query.dtype, key.dtype)
    if value.dtype != torch.bfloat16 or beta.dtype != torch.bfloat16:
        return False, "value/beta dtype must be bf16, got %s/%s" % (value.dtype, beta.dtype)
    if g is not None and g.dtype != torch.float32:
        return False, "g dtype must be fp32, got %s" % g.dtype

    state_dtype = initial_state.dtype
    if state_dtype not in (torch.bfloat16, torch.float32):
        return False, "initial_state dtype unsupported: %s" % state_dtype
    if state_dtype == torch.float32 and not _soc_is_ascend950():
        return False, "fp32 state requires ascend950 (ascend910b supports bf16 only)"

    return True, ""



def _bootstrap_custom_op_env() -> bool:
    """Make the packaged CANN custom-op package discoverable at runtime.

    Same idea as vllm-ascend's ``bootstrap_custom_op_env``: prepend the
    packaged ``_cann_ops_custom/vendors/custom_transformer`` dir to
    ``ASCEND_CUSTOM_OPP_PATH`` so users do not have to source
    ``set_env.bash`` before launching the server. The OPP path is scanned
    lazily by the AscendCL runtime at the first custom-op call, so setting
    it here (before any op invocation) is sufficient; the variable is also
    inherited by spawned worker processes.

    Additionally preload ``libcust_opapi.so`` by absolute path: the aclnn
    adapter resolves custom symbols via ``dlopen("libcust_opapi.so")`` by
    bare name, which only searches the *startup-time* ``LD_LIBRARY_PATH``
    (glibc caches it) — but finds the already-loaded library by SONAME
    after a preload. ``RTLD_LOCAL`` is used on purpose: ``RTLD_GLOBAL``
    leads to a double-free at process teardown.
    """
    try:
        import vllm_fl._C_ascend as _ext  # noqa: F401
    except Exception as e:
        logger.warning("Failed to import vllm_fl._C_ascend: %s", e)
        return False
    vendor_dir = os.path.join(
        os.path.dirname(_ext.__file__), "_cann_ops_custom", "vendors", _CUSTOM_OPP_MARKER
    )
    if not os.path.isdir(vendor_dir):
        logger.warning("CANN custom op package not found at %s", vendor_dir)
        return False
    opp_path = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")
    if vendor_dir not in opp_path:
        os.environ["ASCEND_CUSTOM_OPP_PATH"] = (
            vendor_dir + (":" + opp_path if opp_path else "")
        )
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    lib_dir = os.path.join(vendor_dir, "op_api", "lib")
    if lib_dir not in ld_path:
        os.environ["LD_LIBRARY_PATH"] = lib_dir + (":" + ld_path if ld_path else "")
        try:
            import ctypes

            ctypes.CDLL(
                os.path.join(lib_dir, "libcust_opapi.so"), mode=ctypes.RTLD_LOCAL
            )
        except OSError as e:
            logger.warning("Failed to preload libcust_opapi.so: %s", e)
            return False
    return True


def _ascendc_ops_available() -> bool:
    """Check the CANN custom-op env and the ``_C_ascend`` torch bindings."""
    if os.environ.get("VLLM_FL_DISABLE_ASCENDC_GDN", "0") == "1":
        logger.info("VLLM_FL_DISABLE_ASCENDC_GDN=1, keep Triton GDN path")
        return False
    if _CUSTOM_OPP_MARKER not in os.environ.get("ASCEND_CUSTOM_OPP_PATH", ""):
        if not _bootstrap_custom_op_env():
            logger.warning(
                "CANN custom op environment is not set and auto-bootstrap "
                "failed; keep Triton GDN path"
            )
            return False
    try:
        import vllm_fl._C_ascend  # noqa: F401
    except Exception as e:
        logger.warning("Failed to import vllm_fl._C_ascend: %s; keep Triton GDN path", e)
        return False
    missing = [name for name in _REQUIRED_OPS if not hasattr(torch.ops._C_ascend, name)]
    if missing:
        logger.warning("torch.ops._C_ascend missing ops %s; keep Triton GDN path", missing)
        return False
    return True


def _build_actual_seq_lengths(
    query_start_loc: torch.Tensor,
    num_sequences: int,
) -> torch.Tensor:
    """Build ``[0, len_1, ..., len_B]`` cu-seqlens style actual_seq_lengths."""
    actual_seq_lengths = torch.empty_like(query_start_loc[: num_sequences + 1])
    actual_seq_lengths[:1].copy_(query_start_loc[:1])
    torch.sub(
        query_start_loc[1 : num_sequences + 1],
        query_start_loc[:num_sequences],
        out=actual_seq_lengths[1:],
    )
    return actual_seq_lengths


def _patch_mamba_cache_dense_layout() -> None:
    """Rebuild mamba KV-cache views as dense per-state tensors.

    Upstream vLLM lays out the conv/ssm states of a mamba block interleaved
    inside one (padded) page, so the per-state views have a first-dim stride
    larger than the dense block size. The AscendC kernels address the state
    cache assuming dense per-state tensors (verified: in-place state updates
    land at wrong offsets with the paged views), while the Triton kernels
    take explicit strides and work with either layout. Wrap
    ``ModelRunnerFL._reshape_kv_cache_tensors`` so that, after the original
    reshape, every MambaSpec layer's state views are rebuilt as dense,
    grouped per-state views over the same raw storage. This is semantically
    transparent for all other consumers (they index the views by block id).
    """
    from vllm.utils.torch_utils import get_dtype_size

    from vllm_fl.worker.model_runner import ModelRunnerFL

    orig_reshape = ModelRunnerFL._reshape_kv_cache_tensors

    def _reshape_kv_cache_tensors_dense_mamba(
        self,
        kv_cache_config,
        kv_cache_raw_tensors,
        kernel_block_sizes,
    ):
        kv_caches = orig_reshape(
            self, kv_cache_config, kv_cache_raw_tensors, kernel_block_sizes
        )
        for group in self._kv_cache_spec_attn_group_iterator():
            kv_cache_spec = group.kv_cache_spec
            if not isinstance(kv_cache_spec, MambaSpec):
                continue
            if group.kv_cache_group_id == len(kernel_block_sizes):
                continue
            for layer_name in group.layer_names:
                if layer_name in self.runner_only_attn_layers:
                    continue
                raw_tensor = kv_cache_raw_tensors[layer_name]
                num_blocks = raw_tensor.numel() // kv_cache_spec.page_size_bytes
                state_tensors = []
                storage_offset_bytes = 0
                raw_u8 = raw_tensor.view(torch.uint8)
                for shape, dtype in zip(kv_cache_spec.shapes, kv_cache_spec.dtypes):
                    dtype_size = get_dtype_size(dtype)
                    num_bytes = num_blocks * math.prod(shape) * dtype_size
                    tensor = (
                        raw_u8[storage_offset_bytes : storage_offset_bytes + num_bytes]
                        .view(dtype)
                        .view(num_blocks, *shape)
                    )
                    state_tensors.append(tensor)
                    storage_offset_bytes += num_bytes
                kv_caches[layer_name] = state_tensors
        return kv_caches

    ModelRunnerFL._reshape_kv_cache_tensors = _reshape_kv_cache_tensors_dense_mamba
    logger.info("Patched mamba KV-cache views to dense per-state layout for AscendC GDN ops")


class AscendCGatedDeltaNet(Qwen3NextGatedDeltaNet):
    """GDN layer backed by the AscendC fused kernels (eager mode)."""

    def get_state_shape(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        conv_state_shape, temporal_state_shape = (
            MambaStateShapeCalculator.gated_delta_net_state_shape(
                self.tp_size,
                self.num_k_heads,
                self.num_v_heads,
                self.head_k_dim,
                self.head_v_dim,
                self.conv_kernel_size,
                self.num_spec,
            )
        )
        # The AscendC recurrent_gated_delta_rule kernel keeps the ssm state
        # in (Hv, Dv, Dk) layout; vLLM 0.13 uses (Hv, Dk, Dv).
        num_v_heads, head_k_dim, head_v_dim = temporal_state_shape
        return conv_state_shape, (num_v_heads, head_v_dim, head_k_dim)

    def _forward_core(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
    ):
        """
        Core attention computation (called by custom op).
        """
        forward_context = get_forward_context()
        attn_metadata: AttentionMetadata = forward_context.attn_metadata

        if attn_metadata is None:
            # V1 profile run
            return

        assert isinstance(attn_metadata, dict)
        attn_metadata = attn_metadata[self.prefix]
        assert isinstance(attn_metadata, GDNAttentionMetadata)
        has_initial_state = attn_metadata.has_initial_state
        spec_query_start_loc = attn_metadata.spec_query_start_loc
        non_spec_query_start_loc = attn_metadata.non_spec_query_start_loc
        spec_sequence_masks = attn_metadata.spec_sequence_masks
        spec_token_indx = attn_metadata.spec_token_indx
        non_spec_token_indx = attn_metadata.non_spec_token_indx
        spec_state_indices_tensor = attn_metadata.spec_state_indices_tensor  # noqa: E501
        non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor  # noqa: E501
        self_kv_cache = self.kv_cache[forward_context.virtual_engine]
        # conv cache is already (slot, state_len, conv_dim): pass through.
        conv_state = self_kv_cache[0]
        ssm_state = self_kv_cache[1]
        num_actual_tokens = attn_metadata.num_actual_tokens
        num_accepted_tokens = attn_metadata.num_accepted_tokens

        mixed_qkv = mixed_qkv[:num_actual_tokens]
        b = b[:num_actual_tokens]
        a = a[:num_actual_tokens]

        # 1. Convolution sequence transformation
        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )
        # The AscendC kernel expects the conv weight as (width, dim).
        conv_weights_t = conv_weights.transpose(0, 1)
        activation_mode = 1 if self.activation else 0

        if spec_sequence_masks is not None:
            if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
                mixed_qkv_spec = mixed_qkv
                mixed_qkv_non_spec = None
            else:
                mixed_qkv_spec = mixed_qkv.index_select(0, spec_token_indx)
                mixed_qkv_non_spec = mixed_qkv.index_select(0, non_spec_token_indx)
        else:
            mixed_qkv_spec = None
            mixed_qkv_non_spec = mixed_qkv

        # 1.1: Process the multi-query part
        if spec_sequence_masks is not None:
            spec_num_rows = spec_query_start_loc.size(0) - 1
            mixed_qkv_spec_out = torch.empty_like(mixed_qkv_spec)
            torch.ops._C_ascend.npu_causal_conv1d_custom(
                mixed_qkv_spec_out,
                mixed_qkv_spec,
                conv_weights_t,
                conv_state,
                self.conv1d.bias,
                spec_query_start_loc,
                spec_state_indices_tensor[:spec_num_rows],
                None,  # initial_state_mode
                num_accepted_tokens,
                activation_mode,
                PAD_SLOT_ID,
                1,  # run_mode: decode/speculative update
            )
            mixed_qkv_spec = mixed_qkv_spec_out

        # 1.2: Process the remaining part
        if attn_metadata.num_prefills > 0:
            non_spec_num_rows = non_spec_query_start_loc.size(0) - 1
            mixed_qkv_non_spec_out = torch.empty_like(mixed_qkv_non_spec)
            torch.ops._C_ascend.npu_causal_conv1d_custom(
                mixed_qkv_non_spec_out,
                mixed_qkv_non_spec,
                conv_weights_t,
                conv_state,
                self.conv1d.bias,
                non_spec_query_start_loc,
                non_spec_state_indices_tensor[:non_spec_num_rows],
                has_initial_state,  # initial_state_mode
                None,  # num_accepted_tokens
                activation_mode,
                PAD_SLOT_ID,
                0,  # run_mode: varlen prefill
            )
            mixed_qkv_non_spec = mixed_qkv_non_spec_out
        elif attn_metadata.num_decodes > 0:
            mixed_qkv_non_spec_out = torch.empty_like(mixed_qkv_non_spec)
            torch.ops._C_ascend.npu_causal_conv1d_custom(
                mixed_qkv_non_spec_out,
                mixed_qkv_non_spec,
                conv_weights_t,
                conv_state,
                self.conv1d.bias,
                non_spec_query_start_loc,
                non_spec_state_indices_tensor[: attn_metadata.num_decodes],
                None,  # initial_state_mode
                None,  # num_accepted_tokens
                activation_mode,
                PAD_SLOT_ID,
                1,  # run_mode: decode update
            )
            mixed_qkv_non_spec = mixed_qkv_non_spec_out
        else:
            mixed_qkv_non_spec = None

        query_spec, key_spec, value_spec = self.rearrange_mixed_qkv(mixed_qkv_spec)
        query_non_spec, key_non_spec, value_non_spec = self.rearrange_mixed_qkv(
            mixed_qkv_non_spec
        )

        # 2. Recurrent attention
        g, beta = torch.ops._C_ascend.npu_fused_gdn_gating(
            self.A_log, a, b, self.dt_bias.to(self.A_log.dtype)
        )

        if spec_sequence_masks is not None:
            if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
                g_spec = g
                beta_spec = beta
                g_non_spec = None
                beta_non_spec = None
            else:
                g_spec = g.index_select(1, spec_token_indx)
                beta_spec = beta.index_select(1, spec_token_indx)
                g_non_spec = g.index_select(1, non_spec_token_indx)
                beta_non_spec = beta.index_select(1, non_spec_token_indx)
        else:
            g_spec = None
            beta_spec = None
            g_non_spec = g
            beta_non_spec = beta

        # 2.1: Process the multi-query part
        if spec_sequence_masks is not None:
            actual_seq_lengths = _build_actual_seq_lengths(
                spec_query_start_loc, attn_metadata.num_spec_decodes
            )
            query_spec = l2norm_fwd(query_spec)
            key_spec = l2norm_fwd(key_spec)
            # The AscendC kernel does not apply the q/k L2 norm in-kernel,
            # and writes the updated state back in place.
            core_attn_out_spec = torch.ops._C_ascend.npu_recurrent_gated_delta_rule(
                query=query_spec.squeeze(0),
                key=key_spec.squeeze(0),
                value=value_spec.squeeze(0),
                g=g_spec.squeeze(0),
                beta=beta_spec.squeeze(0),
                state=ssm_state,
                scale=key_spec.shape[-1] ** -0.5,
                actual_seq_lengths=actual_seq_lengths,
                ssm_state_indices=spec_state_indices_tensor.flatten(),
                num_accepted_tokens=num_accepted_tokens.to(torch.int32),
            ).unsqueeze(0)
        else:
            core_attn_out_spec, last_recurrent_state = None, None

        # 2.2: Process the remaining part
        if attn_metadata.num_prefills > 0:
            # 开关控制（任务书 §3-R5：保留基线回退开关，参考官方 PR #12607）：
            #   VLLM_FL_USE_ACLNN_CHUNK_GDN=1（默认）→ AscendC npu_chunk_gated_delta_rule
            #   VLLM_FL_USE_ACLNN_CHUNK_GDN=0 或 VLLM_FL_DISABLE_ASCENDC_GDN=1 → 回退 Triton 基线
            # 两个开关统一用宽松比较：仅显式 "1" 视为开启，其余取值一律回退 Triton。
            # 原写法 int() 会在 VLLM_FL_USE_ACLNN_CHUNK_GDN="" 或 "abc" 时抛
            # ValueError，直接在模型前向路径中断整个推理。
            _use_env = os.environ.get("VLLM_FL_USE_ACLNN_CHUNK_GDN", "1")
            use_aclnn = _use_env.strip() == "1"
            if not use_aclnn and _use_env.strip() != "0":
                logger.info(
                    "VLLM_FL_USE_ACLNN_CHUNK_GDN=%r 非预期取值，按回退处理（仅 '1' 开启）",
                    _use_env,
                )
            if os.environ.get("VLLM_FL_DISABLE_ASCENDC_GDN", "0").strip() == "1":
                use_aclnn = False
            if use_aclnn:
                # 可用性判断（PR 评审意见）：算子未注册、dtype 或平台不支持时回退 Triton 基线
                _asc_ok, _asc_why = _ascendc_chunk_gdn_supported(
                    query_non_spec, key_non_spec, value_non_spec, beta_non_spec,
                    ssm_state[non_spec_state_indices_tensor], g_non_spec,
                )
                if not _asc_ok:
                    logger.info(
                        "AscendC chunk GDN unavailable (%s), fallback to Triton GDN path",
                        _asc_why,
                    )
                    use_aclnn = False
            if use_aclnn:
                # AscendC npu_chunk_gated_delta_rule: TND 布局 + 原生 (Dv,Dk) state + 逐长度
                actual_seq_lengths = (
                    non_spec_query_start_loc[1:] - non_spec_query_start_loc[:-1]
                ).to(torch.int32)
                # AscendC 算子内部不做 q/k L2 归一化，需外部先做
                q_asc = l2norm_fwd(query_non_spec)
                k_asc = l2norm_fwd(key_non_spec)
                # 原生 (Dv,Dk) 布局，不转置；clone 后清零无初始 state 的序列
                initial_state = ssm_state[non_spec_state_indices_tensor].clone()
                initial_state[~has_initial_state, ...] = 0
                (
                    core_attn_out_non_spec,
                    last_recurrent_state,
                ) = torch.ops._C_ascend.npu_chunk_gated_delta_rule(
                    q_asc.squeeze(0),
                    k_asc.squeeze(0),
                    value_non_spec.squeeze(0),
                    beta_non_spec.squeeze(0),
                    initial_state,
                    actual_seq_lengths,
                    g=g_non_spec.squeeze(0),
                    scale_value=key_non_spec.shape[-1] ** -0.5,
                )
                core_attn_out_non_spec = core_attn_out_non_spec.unsqueeze(0)
                # Init cache（AscendC 输出原生 (Dv,Dk) 布局，不转置）
                ssm_state[non_spec_state_indices_tensor] = last_recurrent_state.to(ssm_state.dtype)
            else:
                # Triton 回退（基线 chunk_gated_delta_rule，FLA (Hv,Dk,Dv) 布局，边界转置）
                initial_state = (
                    ssm_state[non_spec_state_indices_tensor].transpose(-1, -2).contiguous()
                )
                initial_state[~has_initial_state, ...] = 0
                (
                    core_attn_out_non_spec,
                    last_recurrent_state,
                ) = _qwen3_next_lib.chunk_gated_delta_rule(
                    q=query_non_spec,
                    k=key_non_spec,
                    v=value_non_spec,
                    g=g_non_spec,
                    beta=beta_non_spec,
                    initial_state=initial_state,
                    output_final_state=True,
                    cu_seqlens=non_spec_query_start_loc,
                    head_first=False,
                    use_qk_l2norm_in_kernel=True,
                )
                # Init cache（Triton 输出 FLA (Hv,Dk,Dv)，写回需转置）
                ssm_state[non_spec_state_indices_tensor] = (
                    last_recurrent_state.transpose(-1, -2).contiguous().to(ssm_state.dtype)
                )
        elif attn_metadata.num_decodes > 0:
            actual_seq_lengths = _build_actual_seq_lengths(
                non_spec_query_start_loc, attn_metadata.num_decodes
            )
            query_non_spec = l2norm_fwd(query_non_spec)
            key_non_spec = l2norm_fwd(key_non_spec)
            core_attn_out_non_spec = torch.ops._C_ascend.npu_recurrent_gated_delta_rule(
                query=query_non_spec.squeeze(0),
                key=key_non_spec.squeeze(0),
                value=value_non_spec.squeeze(0),
                g=g_non_spec.squeeze(0),
                beta=beta_non_spec.squeeze(0),
                state=ssm_state,
                scale=key_non_spec.shape[-1] ** -0.5,
                actual_seq_lengths=actual_seq_lengths,
                ssm_state_indices=non_spec_state_indices_tensor[
                    : attn_metadata.num_decodes
                ],
            ).unsqueeze(0)
        else:
            core_attn_out_non_spec, last_recurrent_state = None, None

        # 3. Merge core attention output
        if spec_sequence_masks is not None and core_attn_out_non_spec is not None:
            merged_out = torch.empty(
                (1, num_actual_tokens, *core_attn_out_spec.shape[2:]),
                dtype=core_attn_out_non_spec.dtype,
                device=core_attn_out_non_spec.device,
            )
            merged_out.index_copy_(1, spec_token_indx, core_attn_out_spec)
            merged_out.index_copy_(1, non_spec_token_indx, core_attn_out_non_spec)
            core_attn_out[:num_actual_tokens] = merged_out.squeeze(0)
        elif spec_sequence_masks is not None:
            core_attn_out[:num_actual_tokens] = core_attn_out_spec.squeeze(0)
        else:
            core_attn_out[:num_actual_tokens] = core_attn_out_non_spec.squeeze(0)


class AscendCGemmaRMSNorm(GemmaRMSNorm):
    """GemmaRMSNorm backed by the AscendC ``npu_gemma_rms_norm`` kernel."""

    def forward_oot(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is not None:
            x, _, residual = torch.ops._C_ascend.npu_add_rms_norm_bias(
                x, residual, 1.0 + self.weight, None, self.variance_epsilon
            )
            return x, residual

        # npu_gemma_rms_norm implements the Gemma (1 + weight) convention
        # internally, so the raw weight is passed (same as vllm-ascend).
        x, _ = torch.ops._C_ascend.npu_gemma_rms_norm(
            x, self.weight, self.variance_epsilon
        )
        return x


def patch_qwen3_6_gdn() -> bool:
    """Apply the AscendC GDN patch for Qwen3.5/Qwen3.6.

    Returns True when the AscendC kernels were wired in; otherwise the
    upstream/Triton implementations are kept.
    """
    if not _ascendc_ops_available():
        return False

    _install_gdn_gating_counter()
    Qwen3NextGatedDeltaNet.get_state_shape = AscendCGatedDeltaNet.get_state_shape
    Qwen3NextGatedDeltaNet._forward_core = AscendCGatedDeltaNet._forward_core
    _patch_mamba_cache_dense_layout()
    GemmaRMSNorm.forward_oot = AscendCGemmaRMSNorm.forward_oot
    logger.info(
        "Patched Qwen3NextGatedDeltaNet and GemmaRMSNorm for Ascend "
        "(AscendC causal_conv1d / fused_gdn_gating / recurrent_gated_delta_rule "
        "/ gemma_rms_norm)"
    )
    return True
