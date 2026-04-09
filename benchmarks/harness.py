"""Benchmark harness for RMSNorm + residual + QKV projection.

Uses torch.utils.benchmark.Timer with proper CUDA sync, warmup,
and median-of-100 timing. Outputs a pandas DataFrame to CSV.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from itertools import product

import pandas as pd
import torch
from torch.utils.benchmark import Timer

from triton_fused_rmsnorm_qkv.baseline import rmsnorm_residual_qkv

_has_cuda = torch.cuda.is_available()
if _has_cuda:
    from triton_fused_rmsnorm_qkv.kernel import fused_rmsnorm_residual_qkv


# ---------------------------------------------------------------------------
# Model configurations
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelConfig:
    name: str
    hidden: int
    n_heads: int
    head_dim: int


MODEL_CONFIGS = [
    ModelConfig("Llama-3-8B", hidden=4096, n_heads=32, head_dim=128),
    ModelConfig("Mistral-7B", hidden=4096, n_heads=32, head_dim=128),
    ModelConfig("Qwen2-7B", hidden=3584, n_heads=28, head_dim=128),
]

BATCH_SIZES = [1, 4, 16]
SEQ_LENS = [128, 512, 2048]

# Decode step: single new token per request, M = B (≤ BLOCK_M → W read once).
DECODE_BATCH_SIZES = [1, 4, 8, 16]
DECODE_SEQ_LEN = 1


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def _make_inputs(
    batch: int, seqlen: int, hidden: int, dtype: torch.dtype, device: str
):
    x = torch.randn(batch, seqlen, hidden, dtype=dtype, device=device)
    residual = torch.randn_like(x)
    rms_weight = torch.randn(hidden, dtype=dtype, device=device)
    qkv_weight = torch.randn(3 * hidden, hidden, dtype=dtype, device=device)
    return x, residual, rms_weight, qkv_weight


def bench_baseline(
    batch: int,
    seqlen: int,
    model: ModelConfig,
    dtype: torch.dtype,
    device: str,
    num_repeats: int = 100,
    num_warmup: int = 10,
) -> dict:
    """Benchmark the PyTorch baseline and return a result dict."""
    x, residual, rms_weight, qkv_weight = _make_inputs(
        batch, seqlen, model.hidden, dtype, device
    )

    timer = Timer(
        stmt="rmsnorm_residual_qkv(x, residual, rms_weight, qkv_weight)",
        globals={
            "rmsnorm_residual_qkv": rmsnorm_residual_qkv,
            "x": x,
            "residual": residual,
            "rms_weight": rms_weight,
            "qkv_weight": qkv_weight,
        },
        label=f"baseline/{model.name}",
        sub_label=f"B={batch},S={seqlen}",
        description="PyTorch baseline",
        num_threads=1,
    )

    # warmup
    for _ in range(num_warmup):
        rmsnorm_residual_qkv(x, residual, rms_weight, qkv_weight)
    if device == "cuda":
        torch.cuda.synchronize()

    measurement = timer.blocked_autorange(min_run_time=0.5)

    # Compute memory bandwidth estimate
    # Reads: x + residual + rms_weight + qkv_weight
    # Writes: Q + K + V (intermediate normed is not counted as final output)
    elem_bytes = x.element_size()
    tokens = batch * seqlen
    H = model.hidden
    read_bytes = tokens * H * 2 * elem_bytes + H * elem_bytes + 3 * H * H * elem_bytes
    write_bytes = tokens * 3 * H * elem_bytes
    total_bytes = read_bytes + write_bytes
    bandwidth_gbps = total_bytes / measurement.median / 1e9

    return {
        "model": model.name,
        "batch": batch,
        "seqlen": seqlen,
        "hidden": model.hidden,
        "dtype": str(dtype).split(".")[-1],
        "device": device,
        "impl": "baseline",
        "median_ms": measurement.median * 1e3,
        "iqr_ms": measurement.iqr * 1e3,
        "bandwidth_gbps": bandwidth_gbps,
    }


def bench_fused(
    batch: int,
    seqlen: int,
    model: ModelConfig,
    dtype: torch.dtype,
    device: str,
    num_warmup: int = 10,
) -> dict:
    """Benchmark the fused Triton kernel and return a result dict."""
    assert device == "cuda", "Fused kernel requires CUDA"
    x, residual, rms_weight, qkv_weight = _make_inputs(
        batch, seqlen, model.hidden, dtype, device
    )

    timer = Timer(
        stmt="fused_rmsnorm_residual_qkv(x, residual, rms_weight, qkv_weight)",
        globals={
            "fused_rmsnorm_residual_qkv": fused_rmsnorm_residual_qkv,
            "x": x,
            "residual": residual,
            "rms_weight": rms_weight,
            "qkv_weight": qkv_weight,
        },
        label=f"fused/{model.name}",
        sub_label=f"B={batch},S={seqlen}",
        description="Triton fused kernel",
        num_threads=1,
    )

    for _ in range(num_warmup):
        fused_rmsnorm_residual_qkv(x, residual, rms_weight, qkv_weight)
    torch.cuda.synchronize()

    measurement = timer.blocked_autorange(min_run_time=0.5)

    elem_bytes = x.element_size()
    tokens = batch * seqlen
    H = model.hidden
    read_bytes = tokens * H * 2 * elem_bytes + H * elem_bytes + 3 * H * H * elem_bytes
    write_bytes = tokens * 3 * H * elem_bytes
    total_bytes = read_bytes + write_bytes
    bandwidth_gbps = total_bytes / measurement.median / 1e9

    return {
        "model": model.name,
        "batch": batch,
        "seqlen": seqlen,
        "hidden": model.hidden,
        "dtype": str(dtype).split(".")[-1],
        "device": device,
        "impl": "fused",
        "median_ms": measurement.median * 1e3,
        "iqr_ms": measurement.iqr * 1e3,
        "bandwidth_gbps": bandwidth_gbps,
    }


def run_benchmark_grid(device: str | None = None) -> pd.DataFrame:
    """Run the full benchmark grid and return a DataFrame."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dtype = torch.float16 if device == "cuda" else torch.float32

    results = []
    grid = list(product(MODEL_CONFIGS, BATCH_SIZES, SEQ_LENS))
    total = len(grid)

    impls = ["baseline"] + (["fused"] if device == "cuda" else [])

    for impl in impls:
        print(f"\n{'─' * 60}")
        print(f"  {impl.upper()}")
        print(f"{'─' * 60}")
        for i, (model, batch, seqlen) in enumerate(grid, 1):
            print(f"[{i}/{total}] {model.name}  B={batch} S={seqlen} … ", end="", flush=True)
            try:
                if impl == "baseline":
                    result = bench_baseline(batch, seqlen, model, dtype, device)
                else:
                    result = bench_fused(batch, seqlen, model, dtype, device)
                results.append(result)
                print(f"{result['median_ms']:.3f} ms  ({result['bandwidth_gbps']:.1f} GB/s)")
            except torch.cuda.OutOfMemoryError:
                print("OOM — skipped")
            except Exception as e:
                print(f"ERROR: {e}")

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def run_decode_benchmark(device: str = "cuda") -> pd.DataFrame:
    """Decode-step benchmark: B in DECODE_BATCH_SIZES, S=1 (M = B ≤ BLOCK_M)."""
    dtype = torch.float16
    results = []
    grid = list(product(MODEL_CONFIGS, DECODE_BATCH_SIZES))
    total = len(grid)

    impls = ["baseline", "fused"]
    for impl in impls:
        print(f"\n{'─' * 60}")
        print(f"  DECODE — {impl.upper()}  (S=1)")
        print(f"{'─' * 60}")
        for i, (model, batch) in enumerate(grid, 1):
            print(f"[{i}/{total}] {model.name}  B={batch} S=1 … ", end="", flush=True)
            try:
                if impl == "baseline":
                    result = bench_baseline(batch, DECODE_SEQ_LEN, model, dtype, device)
                else:
                    result = bench_fused(batch, DECODE_SEQ_LEN, model, dtype, device)
                results.append(result)
                print(f"{result['median_ms'] * 1e3:.1f} μs  ({result['bandwidth_gbps']:.1f} GB/s)")
            except Exception as e:
                print(f"ERROR: {e}")

    return pd.DataFrame(results)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
    print()

    df = run_benchmark_grid(device)

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / "baseline.csv"
    df.to_csv(out_path, index=False)

    print(f"\n{'=' * 72}")
    print(f"Results saved to {out_path}")
    print(f"{'=' * 72}\n")
    print(df.to_string(index=False))

    # Speedup summary
    if "fused" in df["impl"].values and "baseline" in df["impl"].values:
        base = df[df["impl"] == "baseline"].set_index(["model", "batch", "seqlen"])
        fused = df[df["impl"] == "fused"].set_index(["model", "batch", "seqlen"])
        speedup = base["median_ms"] / fused["median_ms"]
        print(f"\n{'=' * 72}")
        print("Speedup (baseline / fused):")
        print(speedup.to_string())
        print(f"Mean speedup: {speedup.mean():.2f}x")

    if device == "cuda":
        print(f"\n{'=' * 72}")
        print("DECODE BENCHMARK (S=1 — target regime for this kernel)")
        print(f"{'=' * 72}")
        df_decode = run_decode_benchmark(device)
        out_path_decode = results_dir / "decode.csv"
        df_decode.to_csv(out_path_decode, index=False)
        print(f"\nDecode results saved to {out_path_decode}")
        if "fused" in df_decode["impl"].values and "baseline" in df_decode["impl"].values:
            bd = df_decode[df_decode["impl"] == "baseline"].set_index(["model", "batch", "seqlen"])
            fd = df_decode[df_decode["impl"] == "fused"].set_index(["model", "batch", "seqlen"])
            sp = bd["median_ms"] / fd["median_ms"]
            print("Decode speedup (baseline / fused):")
            print(sp.to_string())
            print(f"Mean decode speedup: {sp.mean():.2f}x")


if __name__ == "__main__":
    main()
