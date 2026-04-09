"""Fused Triton kernel: residual add + RMSNorm + packed QKV projection.

Single GPU launch that eliminates two HBM round-trips compared to the
unfused PyTorch baseline. Each program instance handles a (BLOCK_M × BLOCK_N)
output tile and uses tl.dot (tensor cores) for the QKV projection.

Grid: (ceil(M / BLOCK_M), ceil(3*H / BLOCK_N))

Strategy:
  - Phase 1: Each program loads its (BLOCK_M, H) slice of hidden = x + r in
    BLOCK_K-column chunks, accumulates sum-of-squares, computes rstd per row.
  - Phase 2: Re-reads the same hidden slice (expected L1-resident for small
    BLOCK_M) per K-tile, applies RMSNorm gain, then tl.dot with the
    (BLOCK_K, BLOCK_N) weight tile to accumulate into the output.

HBM traffic analysis (per SM):
  - x, r:      2 × BLOCK_M × H  (read twice; stays in L1 between passes)
  - W:         BLOCK_N × H  (each program reads a unique row-stripe of W)
  - rms_weight: H per program (read once per K-tile, BLOCK_K elems at a time)
  - output:    BLOCK_M × BLOCK_N (written once)

Aggregated, W is read once total (different programs read disjoint row-stripes),
and x/r re-reads are L1-cache hits for BLOCK_M × H ≤ L1 capacity (~128 KB).

This avoids materializing the intermediate normalized tensor to HBM.
"""

from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl

# Tile sizes — chosen so BLOCK_M*BLOCK_K*2 bytes ≤ L1 (128 KB on A10G):
#   16 × 64 × 2 = 2 KB (fp16) — plenty of headroom for double-buffering.
# BLOCK_M and BLOCK_N must be multiples of 16 for tensor-core alignment.
_BLOCK_M = 16
_BLOCK_N = 128
_BLOCK_K = 64


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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    IS_BF16: tl.constexpr,   # True → bfloat16 output, False → float16
):
    pid_m = tl.program_id(0)   # row tile
    pid_n = tl.program_id(1)   # output-column tile

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    m_offs = m_start + tl.arange(0, BLOCK_M)   # (BLOCK_M,)
    n_offs = n_start + tl.arange(0, BLOCK_N)   # (BLOCK_N,)
    m_mask = m_offs < M
    n_mask = n_offs < 3 * H

    # ── Phase 1: residual add + RMSNorm (compute rstd) ──────────────────────
    # Load hidden = x + r in BLOCK_K columns at a time, accumulate variance.
    var = tl.zeros([BLOCK_M], dtype=tl.float32)
    for k_start in range(0, H, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)   # (BLOCK_K,)
        k_mask = k_offs < H

        x_tile = tl.load(
            X_ptr + m_offs[:, None] * stride_x_row + k_offs[None, :],
            mask=m_mask[:, None] & k_mask[None, :], other=0.0,
        )  # (BLOCK_M, BLOCK_K)
        r_tile = tl.load(
            Residual_ptr + m_offs[:, None] * stride_r_row + k_offs[None, :],
            mask=m_mask[:, None] & k_mask[None, :], other=0.0,
        )  # (BLOCK_M, BLOCK_K)

        h_tile_f32 = (x_tile + r_tile).to(tl.float32)  # (BLOCK_M, BLOCK_K)
        var += tl.sum(h_tile_f32 * h_tile_f32, axis=1)   # (BLOCK_M,)

    rstd = 1.0 / tl.sqrt(var / H + eps)   # (BLOCK_M,) fp32

    # ── Phase 2: QKV projection using tl.dot ────────────────────────────────
    # For each K-tile:
    #   normed[m, k] = hidden[m, k] * rstd[m] * rms_weight[k]
    #   acc[m, n]   += normed[m, :] @ W[n, :]^T
    # W is (3*H, H) row-major; we load W^T tile as (BLOCK_K, BLOCK_N).
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k_start in range(0, H, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < H

        # Re-read hidden (BLOCK_M × BLOCK_K ≤ L1, so these are cache hits).
        x_tile = tl.load(
            X_ptr + m_offs[:, None] * stride_x_row + k_offs[None, :],
            mask=m_mask[:, None] & k_mask[None, :], other=0.0,
        )
        r_tile = tl.load(
            Residual_ptr + m_offs[:, None] * stride_r_row + k_offs[None, :],
            mask=m_mask[:, None] & k_mask[None, :], other=0.0,
        )
        h_tile_f32 = (x_tile + r_tile).to(tl.float32)

        w_rms = tl.load(RMSWeight_ptr + k_offs, mask=k_mask, other=0.0).to(tl.float32)
        # normed in original dtype for tensor-core compatibility
        normed_tile = (h_tile_f32 * rstd[:, None] * w_rms[None, :]).to(x_tile.dtype)
        # (BLOCK_M, BLOCK_K) in fp16/bf16

        # Weight tile W^T: shape (BLOCK_K, BLOCK_N)
        # W[n, k] lives at QKVWeight_ptr + n*stride_w_row + k
        # We want w_t[k_idx, n_idx] = W[n_offs[n_idx], k_offs[k_idx]]
        w_t = tl.load(
            QKVWeight_ptr + n_offs[None, :] * stride_w_row + k_offs[:, None],
            mask=k_mask[:, None] & n_mask[None, :], other=0.0,
        )  # (BLOCK_K, BLOCK_N) in fp16/bf16

        # Tensor-core matmul: (BLOCK_M, BLOCK_K) @ (BLOCK_K, BLOCK_N) → fp32 acc
        acc = tl.dot(normed_tile, w_t, acc, out_dtype=tl.float32)

    # ── Store output ─────────────────────────────────────────────────────────
    out_cast = tl.bfloat16 if IS_BF16 else tl.float16
    out_ptrs = Out_ptr + m_offs[:, None] * stride_o_row + n_offs[None, :]
    tl.store(out_ptrs, acc.to(out_cast), mask=m_mask[:, None] & n_mask[None, :])


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

    grid = (triton.cdiv(M, _BLOCK_M), triton.cdiv(3 * H, _BLOCK_N))

    _fused_rmsnorm_residual_qkv_kernel[grid](
        x_flat, r_flat, rms_weight, qkv_weight, out,
        M, H,
        x_flat.stride(0),
        r_flat.stride(0),
        qkv_weight.stride(0),
        out.stride(0),
        eps,
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_BLOCK_N,
        BLOCK_K=_BLOCK_K,
        IS_BF16=(x.dtype == torch.bfloat16),
    )

    out = out.reshape(B, S, 3 * H)
    q, k, v = out.chunk(3, dim=-1)
    return q, k, v
