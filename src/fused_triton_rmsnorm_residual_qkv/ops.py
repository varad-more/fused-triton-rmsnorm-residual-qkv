from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None
    tl = None


@dataclass
class FusedQKVOutputs:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    residual_out: torch.Tensor
    normed: torch.Tensor


def _next_power_of_2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _reference_residual_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    residual_out = x + residual
    rms = residual_out.pow(2).mean(dim=-1, keepdim=True)
    normed = residual_out * torch.rsqrt(rms + eps)
    normed = normed * weight
    return normed, residual_out


if triton is not None:

    @triton.jit
    def _residual_rmsnorm_fwd_kernel(
        x_ptr,
        residual_ptr,
        weight_ptr,
        residual_out_ptr,
        norm_out_ptr,
        n_cols,
        eps,
        stride_x_row,
        stride_residual_row,
        stride_residual_out_row,
        stride_norm_out_row,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols

        x = tl.load(x_ptr + row * stride_x_row + offsets, mask=mask, other=0.0).to(tl.float32)
        residual = tl.load(
            residual_ptr + row * stride_residual_row + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        residual_out = x + residual
        mean_square = tl.sum(residual_out * residual_out, axis=0) / n_cols
        inv_rms = tl.rsqrt(mean_square + eps)
        norm_out = residual_out * inv_rms * weight

        tl.store(
            residual_out_ptr + row * stride_residual_out_row + offsets,
            residual_out,
            mask=mask,
        )
        tl.store(norm_out_ptr + row * stride_norm_out_row + offsets, norm_out, mask=mask)


def residual_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if x.shape != residual.shape:
        raise ValueError(f"x and residual must match, got {x.shape} vs {residual.shape}")
    if x.shape[-1] != weight.numel():
        raise ValueError(
            f"weight size must match hidden dim, got hidden={x.shape[-1]} weight={weight.numel()}"
        )

    if (
        triton is None
        or not x.is_cuda
        or not residual.is_cuda
        or not weight.is_cuda
    ):
        return _reference_residual_rmsnorm(x, residual, weight, eps)

    if not x.is_contiguous() or not residual.is_contiguous() or not weight.is_contiguous():
        x = x.contiguous()
        residual = residual.contiguous()
        weight = weight.contiguous()

    hidden = x.shape[-1]
    flat_x = x.reshape(-1, hidden)
    flat_residual = residual.reshape(-1, hidden)

    residual_out = torch.empty_like(flat_x)
    norm_out = torch.empty_like(flat_x)

    block_size = min(max(128, _next_power_of_2(hidden)), 65536)
    grid = (flat_x.shape[0],)

    _residual_rmsnorm_fwd_kernel[grid](
        flat_x,
        flat_residual,
        weight,
        residual_out,
        norm_out,
        hidden,
        eps,
        flat_x.stride(0),
        flat_residual.stride(0),
        residual_out.stride(0),
        norm_out.stride(0),
        BLOCK_SIZE=block_size,
        num_warps=4 if block_size <= 1024 else 8,
    )

    return norm_out.reshape_as(x), residual_out.reshape_as(x)


def reference_rmsnorm_residual_qkv(
    x: torch.Tensor,
    residual: torch.Tensor,
    rms_weight: torch.Tensor,
    qkv_weight: torch.Tensor,
    qkv_bias: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> FusedQKVOutputs:
    normed, residual_out = _reference_residual_rmsnorm(x, residual, rms_weight, eps)
    qkv = F.linear(normed, qkv_weight, qkv_bias)
    q, k, v = torch.chunk(qkv, 3, dim=-1)
    return FusedQKVOutputs(q=q, k=k, v=v, residual_out=residual_out, normed=normed)


def fused_rmsnorm_residual_qkv(
    x: torch.Tensor,
    residual: torch.Tensor,
    rms_weight: torch.Tensor,
    qkv_weight: torch.Tensor,
    qkv_bias: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> FusedQKVOutputs:
    if qkv_weight.ndim != 2:
        raise ValueError(f"qkv_weight must be rank-2, got shape {qkv_weight.shape}")
    if qkv_weight.shape[0] % 3 != 0:
        raise ValueError("qkv_weight output dimension must be divisible by 3")
    if qkv_weight.shape[1] != x.shape[-1]:
        raise ValueError(
            f"qkv_weight input dim must match hidden dim, got {qkv_weight.shape[1]} vs {x.shape[-1]}"
        )

    normed, residual_out = residual_rmsnorm(x, residual, rms_weight, eps=eps)
    qkv = F.linear(normed, qkv_weight, qkv_bias)
    q, k, v = torch.chunk(qkv, 3, dim=-1)
    return FusedQKVOutputs(q=q, k=k, v=v, residual_out=residual_out, normed=normed)
