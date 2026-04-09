"""Fused Triton kernel: residual add + RMSNorm + packed QKV projection.

Single GPU launch that eliminates two HBM round-trips compared to the
unfused PyTorch baseline. Each program instance processes one row (one token)
of the (B*S, H) input.

Strategy:
  - BLOCK_H is the next power-of-2 >= H, so the full hidden dim fits in
    one register tile.
  - Phase 1: Load x and residual, compute hidden = x + residual, RMSNorm
    in fp32 (variance, rsqrt, gain), keep normed vector in registers.
  - Phase 2: For each of the 3 output chunks (Q, K, V), tile over output
    columns in groups of BLOCK_H. For each tile, do a 2D weight load
    (BLOCK_H output cols × BLOCK_H inner dim), broadcast-multiply with
    the normed vector, reduce, and store.

This avoids materializing the intermediate normalized tensor to HBM.
"""

from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_rmsnorm_residual_qkv_kernel(
    X_ptr,
    Residual_ptr,
    RMSWeight_ptr,
    QKVWeight_ptr,   # (3*H, H) row-major
    Out_ptr,         # (M, 3*H)
    M,
    H: tl.constexpr,
    stride_x_row,
    stride_r_row,
    stride_w_row,
    stride_o_row,
    eps,
    BLOCK_H: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offs = tl.arange(0, BLOCK_H)
    mask = col_offs < H

    # -- Phase 1: residual add + RMSNorm --
    x = tl.load(X_ptr + row_idx * stride_x_row + col_offs, mask=mask, other=0.0)
    r = tl.load(Residual_ptr + row_idx * stride_r_row + col_offs, mask=mask, other=0.0)
    hidden = x + r

    hidden_f32 = hidden.to(tl.float32)
    var = tl.sum(hidden_f32 * hidden_f32, axis=0) / H
    rstd = 1.0 / tl.sqrt(var + eps)

    w = tl.load(RMSWeight_ptr + col_offs, mask=mask, other=0.0)
    normed = (hidden_f32 * rstd * w.to(tl.float32)).to(x.dtype)  # (BLOCK_H,)

    # -- Phase 2: QKV projection --
    # normed is fully in registers (BLOCK_H >= H).
    # Output = normed @ QKVWeight^T, split into 3 chunks of H.
    for c in tl.static_range(3):
        base_row = c * H
        for j_start in range(0, H, BLOCK_H):
            j_offs = tl.arange(0, BLOCK_H) + j_start
            j_mask = j_offs < H

            acc = tl.zeros([BLOCK_H], dtype=tl.float32)
            for k_start in range(0, H, BLOCK_H):
                k_offs = tl.arange(0, BLOCK_H) + k_start
                k_mask = k_offs < H

                # 2D load: weight_tile[j, k] = QKVWeight[base_row + j_offs[j], k_offs[k]]
                w_ptrs = (QKVWeight_ptr
                          + (base_row + j_offs[:, None]) * stride_w_row
                          + k_offs[None, :])
                w_tile = tl.load(w_ptrs, mask=j_mask[:, None] & k_mask[None, :], other=0.0)

                n_tile = tl.where(k_mask, normed, 0.0)
                acc += tl.sum(w_tile * n_tile[None, :], axis=1)

            out_col = base_row + j_offs
            tl.store(
                Out_ptr + row_idx * stride_o_row + out_col,
                acc.to(x.dtype),
                mask=j_mask,
            )


def fused_rmsnorm_residual_qkv(
    x: torch.Tensor,
    residual: torch.Tensor,
    rms_weight: torch.Tensor,
    qkv_weight: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused residual add + RMSNorm + QKV projection.

    Drop-in replacement for baseline.rmsnorm_residual_qkv.
    """
    assert x.is_cuda, "Fused kernel requires CUDA tensors"
    B, S, H = x.shape
    M = B * S

    x_flat = x.reshape(M, H)
    r_flat = residual.reshape(M, H)
    out = torch.empty(M, 3 * H, dtype=x.dtype, device=x.device)

    BLOCK_H = triton.next_power_of_2(H)
    grid = (M,)

    _fused_rmsnorm_residual_qkv_kernel[grid](
        x_flat, r_flat, rms_weight, qkv_weight, out,
        M, H,
        x_flat.stride(0),
        r_flat.stride(0),
        qkv_weight.stride(0),
        out.stride(0),
        eps,
        BLOCK_H=BLOCK_H,
    )

    out = out.reshape(B, S, 3 * H)
    q, k, v = out.chunk(3, dim=-1)
    return q, k, v
