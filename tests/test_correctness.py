"""Correctness tests for RMSNorm + residual + QKV projection.

Tests the PyTorch baseline against a manual reference, and (when CUDA is
available) tests the fused Triton kernel against the baseline.
"""

import pytest
import torch

from triton_fused_rmsnorm_qkv.baseline import rmsnorm_residual_qkv

_has_cuda = torch.cuda.is_available()
if _has_cuda:
    from triton_fused_rmsnorm_qkv.kernel import fused_rmsnorm_residual_qkv

# ---------------------------------------------------------------------------
# Reference implementation (manual, for cross-checking the baseline)
# ---------------------------------------------------------------------------


def _reference_rmsnorm_residual_qkv(x, residual, rms_weight, qkv_weight, eps):
    """Minimal manual reference — no code sharing with baseline.py."""
    hidden = (x + residual).float()
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

        # Matmul accumulation error scales with hidden dim; use generous tolerances
        # for half-precision CPU numerics (no tensor cores → different rounding).
        atol = 2e-1 if dtype == torch.bfloat16 else 5e-2
        rtol = 5e-2

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


# ---------------------------------------------------------------------------
# Fused Triton kernel tests (CUDA only)
# ---------------------------------------------------------------------------

CUDA_SHAPES = [
    (1, 128, 512),
    (2, 256, 1024),
    (4, 512, 2048),
    (1, 2048, 4096),
]


@pytest.mark.skipif(not _has_cuda, reason="CUDA not available")
class TestFusedKernelCorrectness:
    """Verify fused Triton kernel matches the PyTorch baseline."""

    @pytest.fixture(params=CUDA_SHAPES, ids=lambda s: f"B{s[0]}_S{s[1]}_H{s[2]}")
    def shape(self, request):
        return request.param

    @pytest.fixture(params=[torch.float16, torch.bfloat16], ids=lambda d: str(d).split(".")[-1])
    def dtype(self, request):
        return request.param

    def test_matches_baseline(self, shape, dtype):
        B, S, H = shape
        torch.manual_seed(42)
        device = "cuda"

        x = torch.randn(B, S, H, dtype=dtype, device=device)
        residual = torch.randn(B, S, H, dtype=dtype, device=device)
        rms_weight = torch.randn(H, dtype=dtype, device=device)
        qkv_weight = torch.randn(3 * H, H, dtype=dtype, device=device)
        eps = 1e-5

        q_base, k_base, v_base = rmsnorm_residual_qkv(
            x, residual, rms_weight, qkv_weight, eps
        )
        q_fused, k_fused, v_fused = fused_rmsnorm_residual_qkv(
            x, residual, rms_weight, qkv_weight, eps
        )

        # Accumulation error scales with sqrt(H): our kernel uses fp32 accumulation
        # while the PyTorch baseline uses cuBLAS tensor cores (native bf16/fp16).
        # Both are valid half-precision matmul implementations; tolerance reflects
        # expected divergence between the two paths, not numerical bugs.
        scale = (H / 512) ** 0.5
        base_atol = 2e-1 if dtype == torch.bfloat16 else 5e-2
        atol = base_atol * scale
        rtol = 1e-1

        torch.testing.assert_close(q_fused, q_base, atol=atol, rtol=rtol)
        torch.testing.assert_close(k_fused, k_base, atol=atol, rtol=rtol)
        torch.testing.assert_close(v_fused, v_base, atol=atol, rtol=rtol)

    def test_output_shapes(self, shape, dtype):
        B, S, H = shape
        torch.manual_seed(0)
        device = "cuda"

        x = torch.randn(B, S, H, dtype=dtype, device=device)
        residual = torch.randn(B, S, H, dtype=dtype, device=device)
        rms_weight = torch.randn(H, dtype=dtype, device=device)
        qkv_weight = torch.randn(3 * H, H, dtype=dtype, device=device)

        q, k, v = fused_rmsnorm_residual_qkv(x, residual, rms_weight, qkv_weight)

        assert q.shape == (B, S, H)
        assert k.shape == (B, S, H)
        assert v.shape == (B, S, H)
        assert q.dtype == dtype

    def test_no_nans(self, shape, dtype):
        B, S, H = shape
        torch.manual_seed(0)
        device = "cuda"

        x = torch.randn(B, S, H, dtype=dtype, device=device)
        residual = torch.zeros(B, S, H, dtype=dtype, device=device)
        rms_weight = torch.ones(H, dtype=dtype, device=device)
        qkv_weight = torch.randn(3 * H, H, dtype=dtype, device=device)

        q, k, v = fused_rmsnorm_residual_qkv(x, residual, rms_weight, qkv_weight)

        assert not torch.isnan(q).any()
        assert not torch.isnan(k).any()
        assert not torch.isnan(v).any()
