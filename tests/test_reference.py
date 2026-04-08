import torch

from fused_triton_rmsnorm_residual_qkv import (
    fused_rmsnorm_residual_qkv,
    reference_rmsnorm_residual_qkv,
)


def _run_case(device: str, dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    batch, seq, hidden = 2, 3, 128
    head_dim = 64

    x = torch.randn(batch, seq, hidden, device=device, dtype=dtype)
    residual = torch.randn(batch, seq, hidden, device=device, dtype=dtype)
    rms_weight = torch.randn(hidden, device=device, dtype=dtype)
    qkv_weight = torch.randn(head_dim * 3, hidden, device=device, dtype=dtype)
    qkv_bias = torch.randn(head_dim * 3, device=device, dtype=dtype)

    got = fused_rmsnorm_residual_qkv(x, residual, rms_weight, qkv_weight, qkv_bias)
    ref = reference_rmsnorm_residual_qkv(x, residual, rms_weight, qkv_weight, qkv_bias)

    atol = 1e-4 if dtype in (torch.float16, torch.bfloat16) else 1e-5
    rtol = 1e-3 if dtype in (torch.float16, torch.bfloat16) else 1e-5

    torch.testing.assert_close(got.residual_out, ref.residual_out, atol=atol, rtol=rtol)
    torch.testing.assert_close(got.normed, ref.normed, atol=atol, rtol=rtol)
    torch.testing.assert_close(got.q, ref.q, atol=atol, rtol=rtol)
    torch.testing.assert_close(got.k, ref.k, atol=atol, rtol=rtol)
    torch.testing.assert_close(got.v, ref.v, atol=atol, rtol=rtol)


def test_cpu_reference_path_matches() -> None:
    _run_case("cpu", torch.float32)


def test_cuda_path_matches_when_available() -> None:
    if not torch.cuda.is_available():
        return
    _run_case("cuda", torch.float16)
