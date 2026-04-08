from __future__ import annotations

import argparse
import time

import torch

from fused_triton_rmsnorm_residual_qkv import (
    fused_rmsnorm_residual_qkv,
    reference_rmsnorm_residual_qkv,
)


def bench(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.perf_counter()
    return (end - start) * 1000.0 / iters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seq", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    if args.hidden % args.heads != 0:
        raise SystemExit("hidden must be divisible by heads")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    head_dim = args.hidden // args.heads

    x = torch.randn(args.batch, args.seq, args.hidden, device=device, dtype=dtype)
    residual = torch.randn_like(x)
    rms_weight = torch.randn(args.hidden, device=device, dtype=dtype)
    qkv_weight = torch.randn(head_dim * args.heads * 3, args.hidden, device=device, dtype=dtype)
    qkv_bias = torch.randn(head_dim * args.heads * 3, device=device, dtype=dtype)

    fused_ms = bench(
        lambda: fused_rmsnorm_residual_qkv(x, residual, rms_weight, qkv_weight, qkv_bias),
        args.warmup,
        args.iters,
    )
    ref_ms = bench(
        lambda: reference_rmsnorm_residual_qkv(x, residual, rms_weight, qkv_weight, qkv_bias),
        args.warmup,
        args.iters,
    )

    print(f"device={device} dtype={dtype}")
    print(f"fused_like_path_ms={fused_ms:.3f}")
    print(f"reference_path_ms={ref_ms:.3f}")
    print(f"speedup={ref_ms / fused_ms:.3f}x")


if __name__ == "__main__":
    main()
