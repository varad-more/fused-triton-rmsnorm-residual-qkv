"""Correctness tests for RMSNorm + residual + QKV projection.

Currently tests the PyTorch baseline against itself (manual reference).
The fused Triton kernel will be plugged in later.
"""

import pytest
import torch

from triton_fused_rmsnorm_qkv.baseline import rmsnorm_residual_qkv

# ---------------------------------------------------------------------------
# Reference implementation (manual, for cross-checking the baseline)
# ---------------------------------------------------------------------------


def _reference_rmsnorm_residual_qkv(x, residual, rms_weight, qkv_weight, eps):
    """Minimal manual reference — no code sharing with baseline.py."""
    hidden = x.float() + residual.float()
    var = hidden.pow(2).mean(dim=-1, keepdim=True)
    normed = hidden * torch.rsqrt(var + eps)
    normed = normed * rms_weight.float()
    normed = normed.to(x.dtype)
    qkv = normed @ qkv_weight.t()
    q, k, v = qkv.chunk(3, dim=-1)
    return q, k, v


# ---------------------------------------------------------------------------
# Shape / dtype grid
# ---------------------------------------------------------------------------

SHAPES = [
    (1, 128, 512),
    (2, 256, 1024),
    (4, 512, 2048),
    (1, 2048, 4096),
]

DTYPES = [torch.float16, torch.bfloat16]


@pytest.fixture(params=SHAPES, ids=lambda s: f"B{s[0]}_S{s[1]}_H{s[2]}")
def shape(request):
    return request.param


@pytest.fixture(params=DTYPES, ids=lambda d: str(d).split(".")[-1])
def dtype(request):
    return request.param


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBaselineCorrectness:
    """Verify baseline against a hand-written reference."""

    def test_matches_manual_reference(self, shape, dtype):
        B, S, H = shape
        torch.manual_seed(42)

        x = torch.randn(B, S, H, dtype=dtype)
        residual = torch.randn(B, S, H, dtype=dtype)
        rms_weight = torch.randn(H, dtype=dtype)
        qkv_weight = torch.randn(3 * H, H, dtype=dtype)
        eps = 1e-5

        q, k, v = rmsnorm_residual_qkv(x, residual, rms_weight, qkv_weight, eps)
        q_ref, k_ref, v_ref = _reference_rmsnorm_residual_qkv(
            x, residual, rms_weight, qkv_weight, eps
        )

        atol = 1e-2 if dtype == torch.float16 else 2e-2
        rtol = 1e-2

        torch.testing.assert_close(q, q_ref, atol=atol, rtol=rtol)
        torch.testing.assert_close(k, k_ref, atol=atol, rtol=rtol)
        torch.testing.assert_close(v, v_ref, atol=atol, rtol=rtol)

    def test_output_shapes(self, shape, dtype):
        B, S, H = shape
        torch.manual_seed(0)

        x = torch.randn(B, S, H, dtype=dtype)
        residual = torch.randn(B, S, H, dtype=dtype)
        rms_weight = torch.randn(H, dtype=dtype)
        qkv_weight = torch.randn(3 * H, H, dtype=dtype)

        q, k, v = rmsnorm_residual_qkv(x, residual, rms_weight, qkv_weight)

        assert q.shape == (B, S, H)
        assert k.shape == (B, S, H)
        assert v.shape == (B, S, H)
        assert q.dtype == dtype

    def test_output_dtype_preserved(self, dtype):
        B, S, H = 1, 32, 256
        torch.manual_seed(0)

        x = torch.randn(B, S, H, dtype=dtype)
        residual = torch.randn(B, S, H, dtype=dtype)
        rms_weight = torch.randn(H, dtype=dtype)
        qkv_weight = torch.randn(3 * H, H, dtype=dtype)

        q, k, v = rmsnorm_residual_qkv(x, residual, rms_weight, qkv_weight)

        assert q.dtype == dtype
        assert k.dtype == dtype
        assert v.dtype == dtype


class TestEdgeCases:
    """Sanity checks for degenerate inputs."""

    def test_zero_residual(self):
        B, S, H = 2, 64, 256
        torch.manual_seed(0)

        x = torch.randn(B, S, H, dtype=torch.float16)
        residual = torch.zeros(B, S, H, dtype=torch.float16)
        rms_weight = torch.ones(H, dtype=torch.float16)
        qkv_weight = torch.randn(3 * H, H, dtype=torch.float16)

        q, k, v = rmsnorm_residual_qkv(x, residual, rms_weight, qkv_weight)

        assert not torch.isnan(q).any()
        assert not torch.isnan(k).any()
        assert not torch.isnan(v).any()

    def test_unit_weight(self):
        B, S, H = 1, 16, 128
        torch.manual_seed(0)

        x = torch.randn(B, S, H, dtype=torch.float32)
        residual = torch.zeros(B, S, H, dtype=torch.float32)
        rms_weight = torch.ones(H, dtype=torch.float32)
        qkv_weight = torch.eye(H, dtype=torch.float32).repeat(3, 1)  # (3H, H)

        q, k, v = rmsnorm_residual_qkv(x, residual, rms_weight, qkv_weight)

        # With identity projection and unit rms_weight, Q == K == V == normed(x)
        torch.testing.assert_close(q, k, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(k, v, atol=1e-5, rtol=1e-5)
