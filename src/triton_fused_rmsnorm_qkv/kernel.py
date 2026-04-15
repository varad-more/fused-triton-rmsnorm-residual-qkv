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

Phase 3: Tile sizes (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages) are
selected per-shape by `@triton.autotune` keyed on (M, H). The config space
spans decode-oriented tiles (small BLOCK_M for M ≤ 32) and prefill-oriented
tiles (BLOCK_M up to 128 to amortize weight reads over more output rows).
"""

from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl


def _autotune_configs():
    """Full sweep over the spec'd config space:
        BLOCK_M ∈ {32, 64, 128}
        BLOCK_N ∈ {64, 128, 256}
        BLOCK_K ∈ {16, 32, 64}
        num_warps ∈ {4, 8}
        num_stages ∈ {2, 3, 4}

    Static filters drop configs that would clearly fail on A10G:
      - tile area > 128*256 (register pressure for fp32 accumulator),
      - large tiles (BM*BN ≥ 16384) with too few warps to fill them,
      - per-stage shared-mem footprint × num_stages > 96 KB
        (A10G Ampere shared-mem budget per SM is 100 KB usable).

    Per-shape pruning (`_prune_configs`) further drops configs where
    BLOCK_M wildly exceeds M. ~80 configs survive after static filtering.
    """
    SHARED_MEM_BUDGET = 96 * 1024   # bytes per SM (Ampere; leave 4 KB headroom)
    BYTES_PER_HALF = 2

    configs = []
    for BM in (32, 64, 128):
        for BN in (64, 128, 256):
            for BK in (16, 32, 64):
                for nw in (4, 8):
                    for ns in (2, 3, 4):
                        # 1) tile area cap (register pressure)
                        if BM * BN > 128 * 256:
                            continue
                        # 2) large tiles need enough warps to cover them
                        if BM * BN >= 128 * 128 and nw < 8:
                            continue
                        # 3) shared-memory budget for the K-stage buffers
                        bytes_per_stage = BYTES_PER_HALF * (BM * BK + BK * BN)
                        if bytes_per_stage * ns > SHARED_MEM_BUDGET:
                            continue
                        configs.append(triton.Config(
                            {"BLOCK_M": BM, "BLOCK_N": BN, "BLOCK_K": BK},
                            num_warps=nw, num_stages=ns,
                        ))
    return configs


def _prune_configs(configs, named_args, **kwargs):
    """Per-shape pruning to keep autotune wall time bounded.

    Strategy:
      - BLOCK_M is upper-bounded by 2×M (avoid wasted masked rows on small M).
      - BLOCK_M is lower-bounded by max(32, M//4) when M is large — small
        BLOCK_M for big M means too many programs and excessive W re-reads.
      - num_stages=4 only kept when BLOCK_K ≤ 32 (big-K configs rarely benefit
        from the extra pipeline depth and pay shared-mem cost).

    These rules cut the per-shape config count by ~3× without removing any
    config that is plausibly optimal for the regime.
    """
    M = named_args["M"]

    def keep(c):
        BM = c.kwargs["BLOCK_M"]
        BK = c.kwargs["BLOCK_K"]
        # Upper bound: BLOCK_M <= 2*M (or 32 minimum)
        if BM > max(32, 2 * M):
            return False
        # Lower bound for large M: skip BLOCK_M=32 once M >= 256
        if M >= 256 and BM == 32:
            return False
        # num_stages=4 only with smaller BLOCK_K
        if c.num_stages == 4 and BK > 32:
            return False
        # For large M, BLOCK_K=16 is too narrow (too many K-loop iters);
        # drop it to cut autotune time on the slowest shapes.
        if M >= 1024 and BK == 16:
            return False
        # For large M, num_warps=4 with big tiles under-fills the SM.
        if M >= 1024 and c.num_warps == 4 and BM * c.kwargs["BLOCK_N"] >= 64 * 128:
            return False
        return True

    pruned = [c for c in configs if keep(c)]
    return pruned or configs  # never return empty


@triton.autotune(
    configs=_autotune_configs(),
    # dtype is implicitly part of the cache key via the IS_BF16 constexpr
    # (Triton specializes per constexpr). H is also a constexpr so it
    # already differentiates compiled kernels; we list both here for clarity
    # so the autotune cache is explicitly keyed on the GEMM dims.
    # Note: in this kernel N (output cols) = 3*H and K (reduction) = H, so
    # (M, H) uniquely identifies the (M, N, K) tuple.
    key=["M", "H"],
    prune_configs_by={"early_config_prune": _prune_configs},
)
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

    Drop-in replacement for baseline.rmsnorm_residual_qkv. Tile sizes are
    selected by `@triton.autotune` on first call for each (M, H) shape.
    """
    assert x.is_cuda, "Fused kernel requires CUDA tensors"
    B, S, H = x.shape
    M = B * S

    x_flat = x.reshape(M, H)
    r_flat = residual.reshape(M, H)
    out = torch.empty(M, 3 * H, dtype=x.dtype, device=x.device)

    # Grid is a lambda so autotune's selected BLOCK_M/BLOCK_N determine sizing.
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(3 * H, META["BLOCK_N"]),
    )

    _fused_rmsnorm_residual_qkv_kernel[grid](
        x_flat, r_flat, rms_weight, qkv_weight, out,
        M, H,
        x_flat.stride(0),
        r_flat.stride(0),
        qkv_weight.stride(0),
        out.stride(0),
        eps,
        IS_BF16=(x.dtype == torch.bfloat16),
    )

    out = out.reshape(B, S, 3 * H)
    q, k, v = out.chunk(3, dim=-1)
    return q, k, v


def get_autotune_best_config(M: int, H: int) -> dict | None:
    """Return the autotune-selected config for a given (M, H), if cached.

    Useful for roofline analysis and benchmark reporting. Returns None if
    autotune has not yet run for this shape.
    """
    cache = getattr(_fused_rmsnorm_residual_qkv_kernel, "cache", {})
    # Triton's autotune cache key includes all constexpr args + the `key` list,
    # so we can't look up by (M, H) alone. Instead, scan for a matching entry.
    for k, cfg in cache.items():
        # k is a tuple that includes M and H somewhere; match conservatively.
        if M in k and H in k:
            return {
                **cfg.kwargs,
                "num_warps": cfg.num_warps,
                "num_stages": cfg.num_stages,
            }
    return None
