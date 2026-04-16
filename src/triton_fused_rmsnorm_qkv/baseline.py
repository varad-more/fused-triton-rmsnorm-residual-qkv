"""PyTorch baseline: RMSNorm + residual add + packed QKV projection.

All ops are standard PyTorch -- no fusion, no Triton. This serves as the
correctness reference and performance floor for the fused kernel.

The operation computed is:
    hidden = x + residual
    normed = RMSNorm(hidden, weight=rms_weight, eps=eps)
    Q, K, V = split(normed @ qkv_weight^T, 3, dim=-1)

Performance characteristics (unfused):
    - 3 separate CUDA kernel launches (add, norm, matmul)
    - 2 intermediate HBM round-trips (hidden, normed are materialized)
    - cuBLAS handles the matmul via highly optimized GEMM routines
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def rmsnorm_residual_qkv(
    x: torch.Tensor,
    residual: torch.Tensor,
    rms_weight: torch.Tensor,
    qkv_weight: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unfused baseline: residual add -> RMSNorm -> packed QKV linear -> split.

    Args:
        x: Input activations, shape (B, S, H). fp16 or bf16.
        residual: Residual stream, shape (B, S, H). Same dtype as x.
        rms_weight: RMSNorm learnable gain parameter, shape (H,).
        qkv_weight: Packed QKV projection matrix, shape (3*H, H).
            Rows [0:H] = W_Q, [H:2H] = W_K, [2H:3H] = W_V.
        eps: RMSNorm epsilon for numerical stability. Default 1e-6.

    Returns:
        Tuple of (Q, K, V), each of shape (B, S, H) in the input dtype.
    """
    # Step 1: Residual connection -- adds the skip-connection stream
    # to the current layer's input. This is the first op in each
    # transformer decoder block (post-norm architecture).
    hidden = x + residual

    # Step 2: RMSNorm -- unlike LayerNorm, RMSNorm does not subtract the mean.
    # This saves one reduction and is empirically equivalent for LLMs.
    #
    # Formula: normed = hidden * rsqrt(mean(hidden^2) + eps) * weight
    #
    # We upcast to fp32 for the variance computation to avoid overflow
    # in the sum-of-squares (fp16 max is 65504; squaring a hidden value
    # of 256 already exceeds this).
    inp_dtype = hidden.dtype
    hidden_f32 = hidden.to(torch.float32)
    variance = hidden_f32.pow(2).mean(dim=-1, keepdim=True)
    normed = hidden_f32 * torch.rsqrt(variance + eps)
    normed = (normed * rms_weight.to(torch.float32)).to(inp_dtype)

    # Step 3: Packed QKV projection -- a single matmul computes Q, K, V
    # simultaneously. F.linear computes normed @ qkv_weight^T, producing
    # shape (B, S, 3*H). We then split along the last dimension.
    qkv = F.linear(normed, qkv_weight)
    q, k, v = qkv.chunk(3, dim=-1)

    return q, k, v
