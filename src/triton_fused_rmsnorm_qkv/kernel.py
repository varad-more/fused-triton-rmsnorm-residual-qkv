"""Fused Triton kernel: residual add + RMSNorm + packed QKV projection.

This module provides two Triton kernels and an adaptive dispatch function
that together fuse the residual_add -> RMSNorm -> QKV_linear pipeline into
a single GPU launch, eliminating two HBM round-trips.

Kernels:
    1. _fused_rmsnorm_residual_qkv_kernel (2D grid, decode path)
       Grid: (ceil(M/BLOCK_M), ceil(3H/BLOCK_N))
       Used when M <= 32 to fill SMs via output-column parallelism.

    2. _fused_rmsnorm_residual_qkv_persistent_kernel (1D grid, prefill path)
       Grid: (ceil(M/BLOCK_M),)
       Used when M > 32 to avoid redundant hidden-tile reads across N-tiles.

Public API:
    fused_rmsnorm_residual_qkv(x, residual, rms_weight, qkv_weight, eps)
        -> (Q, K, V)
    get_autotune_best_config(M, H) -> dict | None

Algorithm (both kernels):
    Phase 1 -- Variance accumulation:
        For each BLOCK_K column chunk of hidden = x + r, accumulate
        sum-of-squares in fp32. Compute rstd = 1/sqrt(var/H + eps).

    Phase 2 -- Fused normalize + matmul:
        For each K-tile: re-read hidden (L1 cache hits), apply RMSNorm
        gain, then tl.dot with weight tile -> fp32 accumulator.

HBM traffic (theoretical minimum, per launch):
    Reads:  (2*M*H + H + 3*H*H) * elem_bytes   # x, r, rms_weight, W
    Writes: 3*M*H * elem_bytes                  # packed Q||K||V
    The intermediate normalized tensor never touches HBM.

Tile sizes are selected per-shape by @triton.autotune keyed on (M, H).
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
    # Each program computes one (BLOCK_M, BLOCK_N) output tile.
    # pid_m selects the row tile, pid_n selects the output-column tile.
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    m_offs = m_start + tl.arange(0, BLOCK_M)   # (BLOCK_M,) row indices
    n_offs = n_start + tl.arange(0, BLOCK_N)   # (BLOCK_N,) output-col indices
    m_mask = m_offs < M       # guard against partial row tiles
    n_mask = n_offs < 3 * H   # guard against partial column tiles

    # ── Phase 1: Residual add + RMSNorm variance ────────────────────────────
    # Accumulate sum-of-squares of hidden = x + r in fp32 to avoid overflow.
    # This pass reads (BLOCK_M, H) of x and r -- the data will remain in L1
    # for reuse in Phase 2 since BLOCK_M * H * 2 bytes fits within L1.
    var = tl.zeros([BLOCK_M], dtype=tl.float32)
    for k_start in range(0, H, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < H

        x_tile = tl.load(
            X_ptr + m_offs[:, None] * stride_x_row + k_offs[None, :],
            mask=m_mask[:, None] & k_mask[None, :], other=0.0,
        )  # (BLOCK_M, BLOCK_K)
        r_tile = tl.load(
            Residual_ptr + m_offs[:, None] * stride_r_row + k_offs[None, :],
            mask=m_mask[:, None] & k_mask[None, :], other=0.0,
        )  # (BLOCK_M, BLOCK_K)

        # Fused residual add + cast to fp32 for numerically stable variance
        h_tile_f32 = (x_tile + r_tile).to(tl.float32)
        var += tl.sum(h_tile_f32 * h_tile_f32, axis=1)   # (BLOCK_M,)

    # RMSNorm inverse standard deviation: rstd_i = 1 / sqrt(var_i / H + eps)
    rstd = 1.0 / tl.sqrt(var / H + eps)   # (BLOCK_M,) in fp32

    # ── Phase 2: Fused normalize + QKV matmul via tl.dot ────────────────────
    # For each K-tile, we compute:
    #   normed[m, k] = hidden[m, k] * rstd[m] * rms_weight[k]
    #   acc[m, n]   += normed[m, :] @ W[n, :]^T
    #
    # The re-read of x and r hits L1 cache (same addresses as Phase 1).
    # W is (3*H, H) row-major; we load it transposed as (BLOCK_K, BLOCK_N).
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k_start in range(0, H, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < H

        # Re-read hidden tile (L1 cache hits from Phase 1)
        x_tile = tl.load(
            X_ptr + m_offs[:, None] * stride_x_row + k_offs[None, :],
            mask=m_mask[:, None] & k_mask[None, :], other=0.0,
        )
        r_tile = tl.load(
            Residual_ptr + m_offs[:, None] * stride_r_row + k_offs[None, :],
            mask=m_mask[:, None] & k_mask[None, :], other=0.0,
        )
        h_tile_f32 = (x_tile + r_tile).to(tl.float32)

        # Apply RMSNorm: normed = hidden * rstd * gain, then cast back to
        # input dtype (fp16/bf16) for tensor-core compatibility with tl.dot.
        w_rms = tl.load(RMSWeight_ptr + k_offs, mask=k_mask, other=0.0).to(tl.float32)
        normed_tile = (h_tile_f32 * rstd[:, None] * w_rms[None, :]).to(x_tile.dtype)
        # normed_tile: (BLOCK_M, BLOCK_K) in fp16/bf16

        # Load transposed weight tile: W_T[k, n] = W[n, k]
        # Each pid_n reads a disjoint stripe of W, so W is read exactly once
        # across the full grid.
        w_t = tl.load(
            QKVWeight_ptr + n_offs[None, :] * stride_w_row + k_offs[:, None],
            mask=k_mask[:, None] & n_mask[None, :], other=0.0,
        )  # (BLOCK_K, BLOCK_N) in fp16/bf16

        # Tensor-core matmul with fp32 accumulation to prevent precision loss
        acc = tl.dot(normed_tile, w_t, acc, out_dtype=tl.float32)

    # ── Store output tile ───────────────────────────────────────────────────
    # Cast fp32 accumulator back to input dtype before writing to HBM
    out_cast = tl.bfloat16 if IS_BF16 else tl.float16
    out_ptrs = Out_ptr + m_offs[:, None] * stride_o_row + n_offs[None, :]
    tl.store(out_ptrs, acc.to(out_cast), mask=m_mask[:, None] & n_mask[None, :])


# ---------------------------------------------------------------------------
# Persistent-in-N kernel (Phase 4)
#
# Grid: (ceil(M / BLOCK_M), 1) — each program handles ALL 3H output columns
# for its BLOCK_M row tile. Hidden (x + r) is read from HBM once in Phase 1,
# then reused from L1/L2 cache across all N-tiles in Phase 2. Weight W is
# read exactly once across the full grid (each N-tile reads a disjoint stripe).
# ---------------------------------------------------------------------------


def _autotune_configs_persistent():
    """Config space for the persistent-in-N kernel.

    BLOCK_M ∈ {16, 32, 64, 128}  (includes 16 for transition regime)
    BLOCK_N ∈ {64, 128, 256}
    BLOCK_K ∈ {32, 64}
    num_warps ∈ {4, 8}
    num_stages ∈ {2, 3}
    """
    SHARED_MEM_BUDGET = 96 * 1024
    BYTES_PER_HALF = 2

    configs = []
    for BM in (16, 32, 64, 128):
        for BN in (64, 128, 256):
            for BK in (32, 64):
                for nw in (4, 8):
                    for ns in (2, 3):
                        # Register pressure: fp32 accumulator tiles
                        if BM * BN > 128 * 256:
                            continue
                        # Large tiles need enough warps
                        if BM * BN >= 128 * 128 and nw < 8:
                            continue
                        # Shared-memory for K-stage pipeline buffers
                        bytes_per_stage = BYTES_PER_HALF * (BM * BK + BK * BN)
                        if bytes_per_stage * ns > SHARED_MEM_BUDGET:
                            continue
                        configs.append(triton.Config(
                            {"BLOCK_M": BM, "BLOCK_N": BN, "BLOCK_K": BK},
                            num_warps=nw, num_stages=ns,
                        ))
    return configs


def _prune_configs_persistent(configs, named_args, **kwargs):
    """Per-shape pruning for persistent-in-N kernel."""
    M = named_args["M"]

    def keep(c):
        BM = c.kwargs["BLOCK_M"]
        BK = c.kwargs["BLOCK_K"]
        # Upper bound: don't waste rows
        if BM > max(16, 2 * M):
            return False
        # Lower bound for large M: small BLOCK_M means too many programs
        if M >= 256 and BM < 32:
            return False
        if M >= 512 and BM == 32:
            return False
        # num_stages=3 only with smaller BLOCK_K
        if c.num_stages == 3 and BK > 32:
            return False
        # For large M, num_warps=4 with big tiles under-fills SM
        if M >= 1024 and c.num_warps == 4 and BM * c.kwargs["BLOCK_N"] >= 64 * 128:
            return False
        return True

    pruned = [c for c in configs if keep(c)]
    return pruned or configs


@triton.autotune(
    configs=_autotune_configs_persistent(),
    key=["M", "H"],
    prune_configs_by={"early_config_prune": _prune_configs_persistent},
)
@triton.jit
def _fused_rmsnorm_residual_qkv_persistent_kernel(
    X_ptr,
    Residual_ptr,
    RMSWeight_ptr,
    QKVWeight_ptr,   # (3*H, H) row-major
    Out_ptr,         # (M, 3*H)
    M,
    H: tl.constexpr,
    THREE_H: tl.constexpr,  # = 3 * H, passed as constexpr for the N-loop bound
    stride_x_row,
    stride_r_row,
    stride_w_row,
    stride_o_row,
    eps,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    # 1D grid: each program handles ALL output columns for its row tile.
    # This eliminates the redundant hidden-tile reads that plague the 2D
    # kernel at large M (where ceil(3H/BLOCK_N) sibling programs each
    # independently re-read the same rows -- up to 19x read amplification).
    pid_m = tl.program_id(0)

    m_start = pid_m * BLOCK_M
    m_offs = m_start + tl.arange(0, BLOCK_M)
    m_mask = m_offs < M

    # ── Phase 1: Residual add + RMSNorm variance ──────────────────────────
    # Identical to the 2D kernel: accumulate sum-of-squares in fp32.
    # The hidden data loaded here stays in L1/L2 for reuse across all
    # N-tiles in Phase 2.
    var = tl.zeros([BLOCK_M], dtype=tl.float32)
    for k_start in range(0, H, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < H

        x_tile = tl.load(
            X_ptr + m_offs[:, None] * stride_x_row + k_offs[None, :],
            mask=m_mask[:, None] & k_mask[None, :], other=0.0,
        )
        r_tile = tl.load(
            Residual_ptr + m_offs[:, None] * stride_r_row + k_offs[None, :],
            mask=m_mask[:, None] & k_mask[None, :], other=0.0,
        )

        h_tile_f32 = (x_tile + r_tile).to(tl.float32)
        var += tl.sum(h_tile_f32 * h_tile_f32, axis=1)

    rstd = 1.0 / tl.sqrt(var / H + eps)

    # ── Phase 2: Persistent N-loop -- QKV projection ─────────────────────
    # Outer loop iterates over output-column tiles (N dimension = 3H).
    # Inner loop reduces over the K dimension (hidden dim H).
    #
    # Key insight: hidden (x+r) is re-read once per N-tile, but since
    # Phase 1 already loaded it into L1/L2, these are cache hits. The
    # weight matrix W is still read exactly once total (each N-tile
    # reads a disjoint BLOCK_N-wide column stripe).
    for n_start in range(0, THREE_H, BLOCK_N):
        n_offs = n_start + tl.arange(0, BLOCK_N)
        n_mask = n_offs < THREE_H

        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

        for k_start in range(0, H, BLOCK_K):
            k_offs = k_start + tl.arange(0, BLOCK_K)
            k_mask = k_offs < H

            # Re-read hidden (L1/L2 cache hits from Phase 1)
            x_tile = tl.load(
                X_ptr + m_offs[:, None] * stride_x_row + k_offs[None, :],
                mask=m_mask[:, None] & k_mask[None, :], other=0.0,
            )
            r_tile = tl.load(
                Residual_ptr + m_offs[:, None] * stride_r_row + k_offs[None, :],
                mask=m_mask[:, None] & k_mask[None, :], other=0.0,
            )
            h_tile_f32 = (x_tile + r_tile).to(tl.float32)

            # RMSNorm: normalize + apply learnable gain
            w_rms = tl.load(RMSWeight_ptr + k_offs, mask=k_mask, other=0.0).to(tl.float32)
            normed_tile = (h_tile_f32 * rstd[:, None] * w_rms[None, :]).to(x_tile.dtype)

            # Load transposed weight tile W_T[k, n] = W[n, k]
            w_t = tl.load(
                QKVWeight_ptr + n_offs[None, :] * stride_w_row + k_offs[:, None],
                mask=k_mask[:, None] & n_mask[None, :], other=0.0,
            )

            # Tensor-core matmul with fp32 accumulation
            acc = tl.dot(normed_tile, w_t, acc, out_dtype=tl.float32)

        # Store this N-tile's output, cast back to input dtype
        out_cast = tl.bfloat16 if IS_BF16 else tl.float16
        out_ptrs = Out_ptr + m_offs[:, None] * stride_o_row + n_offs[None, :]
        tl.store(out_ptrs, acc.to(out_cast), mask=m_mask[:, None] & n_mask[None, :])


# ---------------------------------------------------------------------------
# Dispatch threshold: below this M, use 2D grid (decode); above, persistent.
#
# Rationale: At M <= 32, the persistent kernel launches only 1 program
# (ceil(32/BLOCK_M) = 1 for BLOCK_M >= 32), leaving 79 of 80 SMs idle.
# The 2D grid adds pid_n parallelism (ceil(3H/BLOCK_N) ~= 192 programs)
# to fill SMs. Above M = 32, enough row tiles exist to keep SMs busy,
# and the persistent kernel's better data reuse wins.
# ---------------------------------------------------------------------------
_PERSISTENT_THRESHOLD = 32


def fused_rmsnorm_residual_qkv(
    x: torch.Tensor,
    residual: torch.Tensor,
    rms_weight: torch.Tensor,
    qkv_weight: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused residual add + RMSNorm + QKV projection.

    Drop-in replacement for baseline.rmsnorm_residual_qkv. Dispatches to:
      - 2D grid kernel for decode (M ≤ 32): fills SMs via pid_n parallelism
      - Persistent-in-N kernel for prefill (M > 32): reads hidden once, loops N
    Tile sizes selected by `@triton.autotune` per (M, H) shape.
    """
    assert x.is_cuda, "Fused kernel requires CUDA tensors"
    B, S, H = x.shape
    M = B * S

    x_flat = x.reshape(M, H)
    r_flat = residual.reshape(M, H)
    out = torch.empty(M, 3 * H, dtype=x.dtype, device=x.device)

    if M <= _PERSISTENT_THRESHOLD:
        # Decode path: 2D grid fills SMs
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
    else:
        # Prefill path: persistent-in-N, 1D grid
        grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)
        _fused_rmsnorm_residual_qkv_persistent_kernel[grid](
            x_flat, r_flat, rms_weight, qkv_weight, out,
            M, H, 3 * H,
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

    Checks the persistent kernel cache first (prefill path), then falls
    back to the 2D kernel cache (decode path).
    """
    for kernel in (
        _fused_rmsnorm_residual_qkv_persistent_kernel,
        _fused_rmsnorm_residual_qkv_kernel,
    ):
        cache = getattr(kernel, "cache", {})
        for k, cfg in cache.items():
            if M in k and H in k:
                return {
                    **cfg.kwargs,
                    "num_warps": cfg.num_warps,
                    "num_stages": cfg.num_stages,
                }
    return None
