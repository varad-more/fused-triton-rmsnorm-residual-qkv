"""Fused Triton kernel: residual add + RMSNorm + packed projection.

This module provides two Triton kernels and an adaptive dispatch function
that together fuse the residual_add -> RMSNorm -> QKV_linear pipeline into
a single GPU launch, eliminating two HBM round-trips.

Since Phase 6, the kernels accept an arbitrary output width `N_OUT` as a
constexpr, so callers can use un-padded GQA weights `(H_q + 2*H_kv, H)`
instead of zero-padded `(3*H, H)`. They also gate the residual add on a
`HAS_RESIDUAL` constexpr to skip the extra HBM loads when no residual is
supplied (Llama-3 places the residual *after* attention, so the decode
block doesn't have one at the fused-norm-QKV point).

Kernels:
    1. _fused_rmsnorm_residual_qkv_kernel (2D grid, decode path)
       Grid: (ceil(M/BLOCK_M), ceil(N_OUT/BLOCK_N))
       Used when M <= 32 to fill SMs via output-column parallelism.

    2. _fused_rmsnorm_residual_qkv_persistent_kernel (1D grid, prefill path)
       Grid: (ceil(M/BLOCK_M),)
       Used when M > 32 to avoid redundant hidden-tile reads across N-tiles.

Public API:
    fused_rmsnorm_residual_matmul(x, residual, rms_weight, weight, eps)
        -> (M, N_OUT) raw output. `residual` may be None; `weight` may have
        any N_OUT. Low-level entrypoint used by the GQA integration.

    fused_rmsnorm_residual_qkv(x, residual, rms_weight, qkv_weight, eps)
        -> (Q, K, V) triple. Thin wrapper for the classic MHA packed layout
        `qkv_weight: (3*H, H)`; kept for API compatibility.

    get_autotune_best_config(M, H) -> dict | None

Algorithm (both kernels):
    Phase 1 -- Variance accumulation:
        For each BLOCK_K column chunk of hidden = (x + r) or just x,
        accumulate sum-of-squares in fp32. Compute rstd = 1/sqrt(var/H + eps).

    Phase 2 -- Fused normalize + matmul:
        For each K-tile: re-read hidden (L1 cache hits), apply RMSNorm
        gain, then tl.dot with weight tile -> fp32 accumulator.

HBM traffic (theoretical minimum, per launch):
    Reads:  (M*H [+ M*H if residual] + H + N_OUT*H) * elem_bytes
    Writes: M*N_OUT * elem_bytes
    The intermediate normalized tensor never touches HBM.

Tile sizes are selected per-shape by @triton.autotune keyed on (M, H, N_OUT).
"""

from __future__ import annotations

from typing import Optional, Tuple

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
                        if BM * BN > 128 * 256:
                            continue
                        if BM * BN >= 128 * 128 and nw < 8:
                            continue
                        bytes_per_stage = BYTES_PER_HALF * (BM * BK + BK * BN)
                        if bytes_per_stage * ns > SHARED_MEM_BUDGET:
                            continue
                        configs.append(triton.Config(
                            {"BLOCK_M": BM, "BLOCK_N": BN, "BLOCK_K": BK},
                            num_warps=nw, num_stages=ns,
                        ))
    return configs


def _prune_configs(configs, named_args, **kwargs):
    M = named_args["M"]

    def keep(c):
        BM = c.kwargs["BLOCK_M"]
        BK = c.kwargs["BLOCK_K"]
        if BM > max(32, 2 * M):
            return False
        if M >= 256 and BM == 32:
            return False
        if c.num_stages == 4 and BK > 32:
            return False
        if M >= 1024 and BK == 16:
            return False
        if M >= 1024 and c.num_warps == 4 and BM * c.kwargs["BLOCK_N"] >= 64 * 128:
            return False
        return True

    pruned = [c for c in configs if keep(c)]
    return pruned or configs


@triton.autotune(
    configs=_autotune_configs(),
    # Key includes N_OUT so native-GQA (N_OUT=H+2*H_kv) and MHA (N_OUT=3*H)
    # shapes get independent autotune cache entries.
    key=["M", "H", "N_OUT"],
    prune_configs_by={"early_config_prune": _prune_configs},
)
@triton.jit
def _fused_rmsnorm_residual_qkv_kernel(
    X_ptr,
    Residual_ptr,
    RMSWeight_ptr,
    QKVWeight_ptr,   # (N_OUT, H) row-major
    Out_ptr,         # (M, N_OUT)
    M,
    H: tl.constexpr,
    N_OUT: tl.constexpr,   # output width (3*H for MHA, H+2*H_kv for GQA)
    stride_x_row,
    stride_r_row,
    stride_w_row,
    stride_o_row,
    eps,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    IS_BF16: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,   # skip residual loads when False
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    m_offs = m_start + tl.arange(0, BLOCK_M)
    n_offs = n_start + tl.arange(0, BLOCK_N)
    m_mask = m_offs < M
    n_mask = n_offs < N_OUT

    # ── Phase 1: (optional residual add) + RMSNorm variance ───────────────
    var = tl.zeros([BLOCK_M], dtype=tl.float32)
    for k_start in range(0, H, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < H

        x_tile = tl.load(
            X_ptr + m_offs[:, None] * stride_x_row + k_offs[None, :],
            mask=m_mask[:, None] & k_mask[None, :], other=0.0,
        )
        if HAS_RESIDUAL:
            r_tile = tl.load(
                Residual_ptr + m_offs[:, None] * stride_r_row + k_offs[None, :],
                mask=m_mask[:, None] & k_mask[None, :], other=0.0,
            )
            h_tile_f32 = (x_tile + r_tile).to(tl.float32)
        else:
            h_tile_f32 = x_tile.to(tl.float32)
        var += tl.sum(h_tile_f32 * h_tile_f32, axis=1)

    rstd = 1.0 / tl.sqrt(var / H + eps)

    # ── Phase 2: Fused normalize + matmul via tl.dot ───────────────────────
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k_start in range(0, H, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < H

        x_tile = tl.load(
            X_ptr + m_offs[:, None] * stride_x_row + k_offs[None, :],
            mask=m_mask[:, None] & k_mask[None, :], other=0.0,
        )
        if HAS_RESIDUAL:
            r_tile = tl.load(
                Residual_ptr + m_offs[:, None] * stride_r_row + k_offs[None, :],
                mask=m_mask[:, None] & k_mask[None, :], other=0.0,
            )
            h_tile_f32 = (x_tile + r_tile).to(tl.float32)
        else:
            h_tile_f32 = x_tile.to(tl.float32)

        w_rms = tl.load(RMSWeight_ptr + k_offs, mask=k_mask, other=0.0).to(tl.float32)
        normed_tile = (h_tile_f32 * rstd[:, None] * w_rms[None, :]).to(x_tile.dtype)

        w_t = tl.load(
            QKVWeight_ptr + n_offs[None, :] * stride_w_row + k_offs[:, None],
            mask=k_mask[:, None] & n_mask[None, :], other=0.0,
        )

        acc = tl.dot(normed_tile, w_t, acc, out_dtype=tl.float32)

    out_cast = tl.bfloat16 if IS_BF16 else tl.float16
    out_ptrs = Out_ptr + m_offs[:, None] * stride_o_row + n_offs[None, :]
    tl.store(out_ptrs, acc.to(out_cast), mask=m_mask[:, None] & n_mask[None, :])


def _autotune_configs_persistent():
    SHARED_MEM_BUDGET = 96 * 1024
    BYTES_PER_HALF = 2

    configs = []
    for BM in (16, 32, 64, 128):
        for BN in (64, 128, 256):
            for BK in (32, 64):
                for nw in (4, 8):
                    for ns in (2, 3):
                        if BM * BN > 128 * 256:
                            continue
                        if BM * BN >= 128 * 128 and nw < 8:
                            continue
                        bytes_per_stage = BYTES_PER_HALF * (BM * BK + BK * BN)
                        if bytes_per_stage * ns > SHARED_MEM_BUDGET:
                            continue
                        configs.append(triton.Config(
                            {"BLOCK_M": BM, "BLOCK_N": BN, "BLOCK_K": BK},
                            num_warps=nw, num_stages=ns,
                        ))
    return configs


def _prune_configs_persistent(configs, named_args, **kwargs):
    M = named_args["M"]

    def keep(c):
        BM = c.kwargs["BLOCK_M"]
        BK = c.kwargs["BLOCK_K"]
        if BM > max(16, 2 * M):
            return False
        if M >= 256 and BM < 32:
            return False
        if M >= 512 and BM == 32:
            return False
        if c.num_stages == 3 and BK > 32:
            return False
        if M >= 1024 and c.num_warps == 4 and BM * c.kwargs["BLOCK_N"] >= 64 * 128:
            return False
        return True

    pruned = [c for c in configs if keep(c)]
    return pruned or configs


@triton.autotune(
    configs=_autotune_configs_persistent(),
    key=["M", "H", "N_OUT"],
    prune_configs_by={"early_config_prune": _prune_configs_persistent},
)
@triton.jit
def _fused_rmsnorm_residual_qkv_persistent_kernel(
    X_ptr,
    Residual_ptr,
    RMSWeight_ptr,
    QKVWeight_ptr,   # (N_OUT, H) row-major
    Out_ptr,         # (M, N_OUT)
    M,
    H: tl.constexpr,
    N_OUT: tl.constexpr,   # output width; replaces legacy THREE_H
    stride_x_row,
    stride_r_row,
    stride_w_row,
    stride_o_row,
    eps,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    IS_BF16: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
):
    pid_m = tl.program_id(0)

    m_start = pid_m * BLOCK_M
    m_offs = m_start + tl.arange(0, BLOCK_M)
    m_mask = m_offs < M

    # ── Phase 1: Variance ──────────────────────────────────────────────────
    var = tl.zeros([BLOCK_M], dtype=tl.float32)
    for k_start in range(0, H, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < H

        x_tile = tl.load(
            X_ptr + m_offs[:, None] * stride_x_row + k_offs[None, :],
            mask=m_mask[:, None] & k_mask[None, :], other=0.0,
        )
        if HAS_RESIDUAL:
            r_tile = tl.load(
                Residual_ptr + m_offs[:, None] * stride_r_row + k_offs[None, :],
                mask=m_mask[:, None] & k_mask[None, :], other=0.0,
            )
            h_tile_f32 = (x_tile + r_tile).to(tl.float32)
        else:
            h_tile_f32 = x_tile.to(tl.float32)
        var += tl.sum(h_tile_f32 * h_tile_f32, axis=1)

    rstd = 1.0 / tl.sqrt(var / H + eps)

    # ── Phase 2: Persistent N-loop ─────────────────────────────────────────
    for n_start in range(0, N_OUT, BLOCK_N):
        n_offs = n_start + tl.arange(0, BLOCK_N)
        n_mask = n_offs < N_OUT

        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

        for k_start in range(0, H, BLOCK_K):
            k_offs = k_start + tl.arange(0, BLOCK_K)
            k_mask = k_offs < H

            x_tile = tl.load(
                X_ptr + m_offs[:, None] * stride_x_row + k_offs[None, :],
                mask=m_mask[:, None] & k_mask[None, :], other=0.0,
            )
            if HAS_RESIDUAL:
                r_tile = tl.load(
                    Residual_ptr + m_offs[:, None] * stride_r_row + k_offs[None, :],
                    mask=m_mask[:, None] & k_mask[None, :], other=0.0,
                )
                h_tile_f32 = (x_tile + r_tile).to(tl.float32)
            else:
                h_tile_f32 = x_tile.to(tl.float32)

            w_rms = tl.load(RMSWeight_ptr + k_offs, mask=k_mask, other=0.0).to(tl.float32)
            normed_tile = (h_tile_f32 * rstd[:, None] * w_rms[None, :]).to(x_tile.dtype)

            w_t = tl.load(
                QKVWeight_ptr + n_offs[None, :] * stride_w_row + k_offs[:, None],
                mask=k_mask[:, None] & n_mask[None, :], other=0.0,
            )

            acc = tl.dot(normed_tile, w_t, acc, out_dtype=tl.float32)

        out_cast = tl.bfloat16 if IS_BF16 else tl.float16
        out_ptrs = Out_ptr + m_offs[:, None] * stride_o_row + n_offs[None, :]
        tl.store(out_ptrs, acc.to(out_cast), mask=m_mask[:, None] & n_mask[None, :])


# ---------------------------------------------------------------------------
# Dispatch + public API
# ---------------------------------------------------------------------------
_PERSISTENT_THRESHOLD = 32


def fused_rmsnorm_residual_matmul(
    x: torch.Tensor,
    residual: Optional[torch.Tensor],
    rms_weight: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Fused (optional residual) + RMSNorm + arbitrary-width matmul.

    Low-level entrypoint for Phase 6. `weight.shape[0]` (= N_OUT) can be
    anything — in particular `H + 2*H_kv` for native GQA, avoiding the
    zero-padded `(3*H, H)` packing used in Phase 5.

    Args:
        x: (B, S, H) activations in fp16 or bf16.
        residual: (B, S, H) residual stream, or None to skip the add.
        rms_weight: (H,) RMSNorm gain.
        weight: (N_OUT, H) row-major projection matrix.
        eps: RMSNorm epsilon.

    Returns:
        (B, S, N_OUT) tensor in the same dtype as x.
    """
    assert x.is_cuda, "Fused kernel requires CUDA tensors"
    B, S, H = x.shape
    N_OUT = weight.shape[0]
    assert weight.shape[1] == H, f"weight must be (N_OUT, H); got {weight.shape} for H={H}"
    M = B * S

    x_flat = x.reshape(M, H)
    out = torch.empty(M, N_OUT, dtype=x.dtype, device=x.device)

    has_residual = residual is not None
    if has_residual:
        r_flat = residual.reshape(M, H)
        r_ptr = r_flat
        r_stride = r_flat.stride(0)
    else:
        # Unused inside kernel (HAS_RESIDUAL=False guards all loads), but we
        # still need a valid pointer + stride to satisfy the JIT signature.
        r_ptr = x_flat
        r_stride = x_flat.stride(0)

    if M <= _PERSISTENT_THRESHOLD:
        grid = lambda META: (
            triton.cdiv(M, META["BLOCK_M"]),
            triton.cdiv(N_OUT, META["BLOCK_N"]),
        )
        _fused_rmsnorm_residual_qkv_kernel[grid](
            x_flat, r_ptr, rms_weight, weight, out,
            M, H, N_OUT,
            x_flat.stride(0),
            r_stride,
            weight.stride(0),
            out.stride(0),
            eps,
            IS_BF16=(x.dtype == torch.bfloat16),
            HAS_RESIDUAL=has_residual,
        )
    else:
        grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)
        _fused_rmsnorm_residual_qkv_persistent_kernel[grid](
            x_flat, r_ptr, rms_weight, weight, out,
            M, H, N_OUT,
            x_flat.stride(0),
            r_stride,
            weight.stride(0),
            out.stride(0),
            eps,
            IS_BF16=(x.dtype == torch.bfloat16),
            HAS_RESIDUAL=has_residual,
        )

    return out.reshape(B, S, N_OUT)


def fused_rmsnorm_residual_qkv(
    x: torch.Tensor,
    residual: Optional[torch.Tensor],
    rms_weight: torch.Tensor,
    qkv_weight: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused residual + RMSNorm + packed QKV projection — MHA convenience wrapper.

    Thin wrapper around `fused_rmsnorm_residual_matmul` for the classic
    `qkv_weight: (3*H, H)` layout. Returns the (Q, K, V) triple.

    For GQA (Llama-3, Mistral, Qwen2), call `fused_rmsnorm_residual_matmul`
    directly with a `(H + 2*H_kv, H)` weight and split the result.
    """
    H = x.shape[-1]
    assert qkv_weight.shape == (3 * H, H), (
        f"qkv_weight must be (3*H, H); got {qkv_weight.shape}. "
        "For GQA, call fused_rmsnorm_residual_matmul directly."
    )
    out = fused_rmsnorm_residual_matmul(x, residual, rms_weight, qkv_weight, eps)
    q, k, v = out.chunk(3, dim=-1)
    return q, k, v


# ---------------------------------------------------------------------------
# Autotune-off ablation entrypoint (Week 4 ablation)
# ---------------------------------------------------------------------------


def fused_rmsnorm_residual_matmul_fixed(
    x: torch.Tensor,
    residual: Optional[torch.Tensor],
    rms_weight: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Low-level matmul entrypoint with autotune disabled.

    Same contract as `fused_rmsnorm_residual_matmul` but calls the
    underlying JITFunction with a hardcoded Week-2 tile config. Used by
    the Week 4 ablation to quantify the speedup contribution of autotune.
    """
    assert x.is_cuda
    B, S, H = x.shape
    N_OUT = weight.shape[0]
    assert weight.shape[1] == H
    M = B * S

    x_flat = x.reshape(M, H)
    out = torch.empty(M, N_OUT, dtype=x.dtype, device=x.device)

    has_residual = residual is not None
    if has_residual:
        r_flat = residual.reshape(M, H)
        r_ptr = r_flat
        r_stride = r_flat.stride(0)
    else:
        r_ptr = x_flat
        r_stride = x_flat.stride(0)

    # Naive Week-2 tile; shrink BLOCK_M on decode to avoid masking waste.
    BLOCK_M = 32 if M <= 32 else 64
    BLOCK_N = 128
    BLOCK_K = 32
    num_warps = 4
    num_stages = 2

    if M <= _PERSISTENT_THRESHOLD:
        kernel_fn = _fused_rmsnorm_residual_qkv_kernel.fn
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N_OUT, BLOCK_N))
        kernel_fn[grid](
            x_flat, r_ptr, rms_weight, weight, out,
            M, H, N_OUT,
            x_flat.stride(0),
            r_stride,
            weight.stride(0),
            out.stride(0),
            eps,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            IS_BF16=(x.dtype == torch.bfloat16),
            HAS_RESIDUAL=has_residual,
            num_warps=num_warps, num_stages=num_stages,
        )
    else:
        kernel_fn = _fused_rmsnorm_residual_qkv_persistent_kernel.fn
        grid = (triton.cdiv(M, BLOCK_M),)
        kernel_fn[grid](
            x_flat, r_ptr, rms_weight, weight, out,
            M, H, N_OUT,
            x_flat.stride(0),
            r_stride,
            weight.stride(0),
            out.stride(0),
            eps,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            IS_BF16=(x.dtype == torch.bfloat16),
            HAS_RESIDUAL=has_residual,
            num_warps=num_warps, num_stages=num_stages,
        )

    return out.reshape(B, S, N_OUT)


def fused_rmsnorm_residual_qkv_fixed(
    x: torch.Tensor,
    residual: Optional[torch.Tensor],
    rms_weight: torch.Tensor,
    qkv_weight: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Same interface as `fused_rmsnorm_residual_qkv` but with autotune off."""
    H = x.shape[-1]
    assert qkv_weight.shape == (3 * H, H)
    out = fused_rmsnorm_residual_matmul_fixed(x, residual, rms_weight, qkv_weight, eps)
    q, k, v = out.chunk(3, dim=-1)
    return q, k, v


def get_autotune_best_config(M: int, H: int) -> dict | None:
    """Return the autotune-selected config for a given (M, H), if cached."""
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
