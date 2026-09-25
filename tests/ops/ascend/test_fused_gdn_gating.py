# Copyright (c) 2026 BAAI. All rights reserved.

"""Op-level fixed tests for the AscendC ``npu_fused_gdn_gating`` kernel.

Task: Fused GDN Gating (27B/35B GDN).  Fixed-test focus: fixed token count
and head count, comparing the ``g`` and ``beta`` outputs against the vLLM
Triton baseline and a plain PyTorch reference.

Reference semantics (vLLM ``fused_gdn_gating``, qwen3_next.py):

    x        = a + dt_bias
    softplus = (1/beta) * log(1 + exp(beta*x))          if beta*x <= threshold
               x                                        otherwise
    g        = -exp(A_log) * softplus
    beta_out = sigmoid(b)

Output layout: ``g`` is float32 ``(1, batch, num_heads)`` and ``beta_out``
keeps ``b``'s dtype ``(1, batch, num_heads)`` -- both matching the Triton
baseline, so the AscendC op can replace it transparently in the GDN path.
"""

import os

# vLLM registers both the 'fl' and 'ascend' platform plugins on this box;
# pin to 'fl' before importing any vllm module.
os.environ.setdefault("VLLM_PLUGINS", "fl")

import pytest
import torch
import torch_npu  # noqa: F401

import vllm_fl._C_ascend  # noqa: F401  (registers torch.ops._C_ascend)

pytestmark = pytest.mark.gpu

RTOL = 1e-2
ATOL = 1e-2

# Fixed head count used by the Qwen3.6-27B/35B GDN layers (TP=1).
NUM_HEADS = 32

# Fixed token counts: single decode step and small prefill chunks.
TOKEN_COUNTS = [1, 4, 16, 64]

DTYPES = [torch.bfloat16, torch.float16]


def torch_reference(A_log, a, b, dt_bias, beta=1.0, threshold=20.0):
    """Plain PyTorch reference of the gating formula."""
    x = a.float() + dt_bias.float()
    softplus_x = torch.where(
        beta * x <= threshold,
        (1.0 / beta) * torch.log1p(torch.exp(beta * x)),
        x,
    )
    g = (-torch.exp(A_log.float()) * softplus_x).unsqueeze(0)
    beta_out = torch.sigmoid(b.float()).unsqueeze(0).to(b.dtype)
    return g.float(), beta_out


def triton_reference(A_log, a, b, dt_bias):
    """vLLM Triton baseline (kept as the framework fallback path)."""
    from vllm.model_executor.models.qwen3_next import fused_gdn_gating

    return fused_gdn_gating(A_log, a, b, dt_bias)


@pytest.fixture(scope="module")
def gdn_inputs(device):
    """Deterministic A_log/dt_bias/a/b tensors on the device."""
    torch.manual_seed(1234)
    A_log = torch.randn(NUM_HEADS, dtype=torch.float32, device=device)
    dt_bias = torch.randn(NUM_HEADS, dtype=torch.float32, device=device)
    return A_log, dt_bias


class TestFusedGdnGatingOutputs:
    """Compare g / beta outputs against the PyTorch reference."""

    @pytest.mark.parametrize("num_tokens", TOKEN_COUNTS)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_g_and_beta_match_reference(self, gdn_inputs, device, num_tokens, dtype):
        A_log, dt_bias = gdn_inputs
        a = torch.randn(num_tokens, NUM_HEADS, dtype=dtype, device=device)
        b = torch.randn(num_tokens, NUM_HEADS, dtype=dtype, device=device)

        g, beta = torch.ops._C_ascend.npu_fused_gdn_gating(A_log, a, b, dt_bias)
        g_ref, beta_ref = torch_reference(A_log, a, b, dt_bias)

        assert g.shape == (1, num_tokens, NUM_HEADS), g.shape
        assert beta.shape == (1, num_tokens, NUM_HEADS), beta.shape
        assert g.dtype == torch.float32
        assert beta.dtype == dtype
        assert torch.allclose(g.cpu(), g_ref.cpu(), rtol=RTOL, atol=ATOL), (
            f"g mismatch at num_tokens={num_tokens} dtype={dtype}"
        )
        assert torch.allclose(beta.cpu().float(), beta_ref.cpu().float(),
                              rtol=RTOL, atol=ATOL), (
            f"beta mismatch at num_tokens={num_tokens} dtype={dtype}"
        )

    def test_softplus_threshold_path(self, gdn_inputs, device):
        """When beta*x > threshold, softplus degenerates to identity."""
        A_log, dt_bias = gdn_inputs
        a = torch.full((4, NUM_HEADS), 50.0, dtype=torch.bfloat16, device=device)
        b = torch.zeros(4, NUM_HEADS, dtype=torch.bfloat16, device=device)

        g, beta = torch.ops._C_ascend.npu_fused_gdn_gating(
            A_log, a, b, dt_bias, 1.0, 20.0
        )
        g_ref, beta_ref = torch_reference(A_log, a, b, dt_bias, 1.0, 20.0)
        # x = 50 + dt_bias > 20, so g == -exp(A_log) * x
        g_manual = -torch.exp(A_log.float()) * (a.float() + dt_bias.float())
        assert torch.allclose(g.cpu(), g_manual.unsqueeze(0).cpu(),
                              rtol=RTOL, atol=ATOL)
        assert torch.allclose(g.cpu(), g_ref.cpu(), rtol=RTOL, atol=ATOL)
        assert torch.allclose(beta.cpu().float(), beta_ref.cpu().float(),
                              rtol=RTOL, atol=ATOL)


class TestFusedGdnGatingVsTriton:
    """Compare the AscendC op against the vLLM Triton baseline."""

    @pytest.mark.parametrize("num_tokens", TOKEN_COUNTS)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_matches_triton_baseline(self, gdn_inputs, device, num_tokens, dtype):
        A_log, dt_bias = gdn_inputs
        a = torch.randn(num_tokens, NUM_HEADS, dtype=dtype, device=device)
        b = torch.randn(num_tokens, NUM_HEADS, dtype=dtype, device=device)

        g, beta = torch.ops._C_ascend.npu_fused_gdn_gating(A_log, a, b, dt_bias)
        g_tri, beta_tri = triton_reference(A_log, a, b, dt_bias)

        assert torch.allclose(g.cpu(), g_tri.cpu(), rtol=RTOL, atol=ATOL), (
            f"g vs Triton mismatch at num_tokens={num_tokens} dtype={dtype}"
        )
        assert torch.allclose(beta.cpu().float(), beta_tri.cpu().float(),
                              rtol=RTOL, atol=ATOL), (
            f"beta vs Triton mismatch at num_tokens={num_tokens} dtype={dtype}"
        )


class TestFusedGdnGatingValidation:
    """Shape/dtype validation enforced by the Torch adapter
    (``csrc/ascend/attention/fused_gdn_gating/fused_gdn_gating_torch_adpt.h``).
    """

    @pytest.fixture(autouse=True)
    def valid_inputs(self, device):
        """Reference-valid inputs that each invalid case mutates."""
        self.A_log = torch.randn(NUM_HEADS, dtype=torch.float32, device=device)
        self.a = torch.randn(4, NUM_HEADS, dtype=torch.bfloat16, device=device)
        self.b = torch.randn(4, NUM_HEADS, dtype=torch.bfloat16, device=device)
        self.dt_bias = torch.randn(NUM_HEADS, dtype=torch.float32, device=device)

    def _run(self, **kw):
        return torch.ops._C_ascend.npu_fused_gdn_gating(
            kw.get("A_log", self.A_log),
            kw.get("a", self.a),
            kw.get("b", self.b),
            kw.get("dt_bias", self.dt_bias),
        )

    def test_a_log_must_be_1d(self):
        with pytest.raises(RuntimeError, match="A_log should be 1-D"):
            self._run(A_log=self.A_log.unsqueeze(0))

    def test_dt_bias_must_be_1d(self):
        with pytest.raises(RuntimeError, match="dt_bias should be 1-D"):
            self._run(dt_bias=self.dt_bias.unsqueeze(0))

    def test_a_must_be_2d(self):
        with pytest.raises(RuntimeError, match="a should be 2-D"):
            self._run(a=self.a.squeeze(0).unsqueeze(-1))

    def test_b_must_be_2d(self):
        with pytest.raises(RuntimeError, match="b should be 2-D"):
            self._run(b=self.b.squeeze(0).unsqueeze(-1))

    def test_a_b_shape_mismatch(self):
        with pytest.raises(RuntimeError, match="a and b must have the same shape"):
            self._run(b=self.b[:, : NUM_HEADS // 2])

    def test_a_b_dtype_mismatch(self):
        with pytest.raises(RuntimeError, match="a and b must have the same dtype"):
            self._run(b=self.b.to(torch.float16))

    def test_a_log_dt_bias_dtype_mismatch(self):
        with pytest.raises(RuntimeError, match="A_log and dt_bias must have the same dtype"):
            self._run(dt_bias=self.dt_bias.to(torch.bfloat16))

    def test_num_heads_mismatch(self):
        with pytest.raises(RuntimeError, match="num_heads"):
            self._run(A_log=self.A_log[: NUM_HEADS // 2])
