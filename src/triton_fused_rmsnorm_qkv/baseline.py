"""PyTorch baseline: RMSNorm + residual add + packed QKV projection.

All ops are standard PyTorch — no fusion, no Triton. This serves as the
correctness reference and performance floor for the fused kernel.
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
    """Unfused baseline: residual add → RMSNorm → packed QKV linear → split.

    Args:
        x: Input activations, shape (B, S, H).
        residual: Residual stream, shape (B, S, H).
        rms_weight: RMSNorm gain, shape (H,).
        qkv_weight: Packed QKV projection, shape (3*H, H).
        eps: RMSNorm epsilon.

    Returns:
        (Q, K, V) each of shape (B, S, H).
    """
    # 1. residual add
    hidden = x + residual

    # 2. RMSNorm (cast to fp32 for numeric stability, then back)
    inp_dtype = hidden.dtype
    hidden_f32 = hidden.to(torch.float32)
    variance = hidden_f32.pow(2).mean(dim=-1, keepdim=True)
    normed = hidden_f32 * torch.rsqrt(variance + eps)
    normed = (normed * rms_weight.to(torch.float32)).to(inp_dtype)

    # 3. packed QKV projection + 3-way split
    qkv = F.linear(normed, qkv_weight)
    q, k, v = qkv.chunk(3, dim=-1)

    return q, k, v
