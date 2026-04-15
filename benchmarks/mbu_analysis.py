"""Memory Bandwidth Utilization (MBU) analysis for the fused kernel.

For each (model, batch, seqlen, dtype) shape:
  1. Computes theoretical HBM bytes moved (inputs + weights + output).
  2. Measures wall time using the project harness (torch.utils.benchmark).
  3. Reports achieved bandwidth = bytes / time.
  4. Reports MBU = achieved / 600 GB/s (A10G peak HBM bandwidth).

Outputs:
  - benchmarks/results/mbu.csv (full table)
  - stdout: formatted summary table

The "theoretical bytes" assume each tensor is read or written exactly once from
HBM. Re-reads that hit L1/L2 are not counted, so achieved BW > 100% of HBM BW
is possible and indicates effective cache reuse. MBU < 100% indicates either
HBM-bound traffic ratios less than ideal, or compute-bound regions.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from itertools import product
from pathlib import Path

import pandas as pd
import torch

from triton_fused_rmsnorm_qkv.baseline import rmsnorm_residual_qkv

if torch.cuda.is_available():
    from triton_fused_rmsnorm_qkv.kernel import (
        _fused_rmsnorm_residual_qkv_kernel,
        fused_rmsnorm_residual_qkv,
    )

from harness import (
    BATCH_SIZES,
    DECODE_BATCH_SIZES,
    DECODE_SEQ_LEN,
    MODEL_CONFIGS,
    SEQ_LENS,
    ModelConfig,
    bench_baseline,
    bench_fused,
)


# A10G HBM2 peak bandwidth (NVIDIA spec sheet): 600 GB/s.
A10G_PEAK_BW_GBPS = 600.0
# A10G fp16/bf16 tensor-core peak (dense): 125 TFLOPs/s.
# Reported for context — when arithmetic intensity exceeds peak_FLOPs/peak_BW
# (~208 FLOPs/byte on A10G), the kernel is compute-bound and MBU will be low
# by definition. MFU (FLOPs utilization) is the meaningful metric in that regime.
A10G_PEAK_TFLOPS_FP16 = 125.0


# ---------------------------------------------------------------------------
# Theoretical traffic
# ---------------------------------------------------------------------------


def theoretical_bytes(M: int, H: int, elem_bytes: int) -> dict:
    """Bytes of HBM traffic if every tensor is read/written exactly once.

    Reads:
      - x:           M * H elements
      - residual:    M * H elements
      - rms_weight:  H elements
      - qkv_weight:  3*H * H elements
    Writes:
      - out:         M * 3*H elements
    """
    read_elems = 2 * M * H + H + 3 * H * H
    write_elems = 3 * M * H
    return {
        "read_bytes": read_elems * elem_bytes,
        "write_bytes": write_elems * elem_bytes,
        "total_bytes": (read_elems + write_elems) * elem_bytes,
    }


# ---------------------------------------------------------------------------
# Autotune lookup
# ---------------------------------------------------------------------------


def lookup_autotune_choice(M: int, H: int, dtype: torch.dtype) -> dict | None:
    """Find the autotune-selected config for (M, H) in the kernel cache.

    Triton's autotune cache key is a tuple containing the values of the
    ``key=[...]`` args in declaration order, plus dtype-specialization tags.
    Our kernel uses ``key=["M", "H"]`` so the prefix of the cache key is
    ``(M, H, ...)``.
    """
    cache = getattr(_fused_rmsnorm_residual_qkv_kernel, "cache", {})
    for k, cfg in cache.items():
        if isinstance(k, tuple) and len(k) >= 2 and k[0] == M and k[1] == H:
            return {
                "BLOCK_M": cfg.kwargs["BLOCK_M"],
                "BLOCK_N": cfg.kwargs["BLOCK_N"],
                "BLOCK_K": cfg.kwargs["BLOCK_K"],
                "num_warps": cfg.num_warps,
                "num_stages": cfg.num_stages,
            }
    return None


# ---------------------------------------------------------------------------
# Per-shape MBU run
# ---------------------------------------------------------------------------


def analyze_shape(
    model: ModelConfig,
    batch: int,
    seqlen: int,
    dtype: torch.dtype,
    impl: str = "fused",
    device: str = "cuda",
) -> dict:
    """Benchmark one shape and compute MBU."""
    if impl == "fused":
        bench = bench_fused(batch, seqlen, model, dtype, device)
    elif impl == "baseline":
        bench = bench_baseline(batch, seqlen, model, dtype, device)
    else:
        raise ValueError(f"unknown impl: {impl}")

    M = batch * seqlen
    H = model.hidden
    elem_bytes = torch.tensor([], dtype=dtype).element_size()
    traffic = theoretical_bytes(M, H, elem_bytes)

    time_s = bench["median_ms"] / 1e3
    achieved_bw_gbps = traffic["total_bytes"] / time_s / 1e9
    mbu_pct = achieved_bw_gbps / A10G_PEAK_BW_GBPS * 100.0

    # Compute-side: 2 * M * (3H) * H FMA-pair FLOPs for the QKV matmul.
    # (RMSNorm + residual add are negligible — O(M*H) vs O(M*3H*H).)
    flops = 2.0 * M * (3 * H) * H
    achieved_tflops = flops / time_s / 1e12
    mfu_pct = achieved_tflops / A10G_PEAK_TFLOPS_FP16 * 100.0
    arith_intensity = flops / traffic["total_bytes"]   # FLOPs per byte

    autotune_cfg = lookup_autotune_choice(M, H, dtype) if impl == "fused" else None

    return {
        "model": model.name,
        "batch": batch,
        "seqlen": seqlen,
        "M": M,
        "hidden": H,
        "dtype": str(dtype).split(".")[-1],
        "impl": impl,
        "time_us": bench["median_ms"] * 1e3,
        "iqr_us": bench["iqr_ms"] * 1e3,
        "read_bytes": traffic["read_bytes"],
        "write_bytes": traffic["write_bytes"],
        "total_bytes": traffic["total_bytes"],
        "achieved_bw_gbps": achieved_bw_gbps,
        "peak_bw_gbps": A10G_PEAK_BW_GBPS,
        "mbu_pct": mbu_pct,
        "achieved_tflops": achieved_tflops,
        "peak_tflops": A10G_PEAK_TFLOPS_FP16,
        "mfu_pct": mfu_pct,
        "arith_intensity_flops_per_byte": arith_intensity,
        "autotune": autotune_cfg,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_mbu_grid(
    dtypes: list[torch.dtype],
    include_baseline: bool = False,
    decode_only: bool = False,
    prefill_only: bool = False,
) -> pd.DataFrame:
    assert torch.cuda.is_available(), "MBU analysis requires CUDA"

    rows = []

    if not decode_only:
        prefill_grid = list(product(MODEL_CONFIGS, BATCH_SIZES, SEQ_LENS))
        print(f"\n{'═' * 72}")
        print(f"  PREFILL GRID — {len(prefill_grid) * len(dtypes)} shapes")
        print(f"{'═' * 72}")
        for dtype in dtypes:
            for i, (model, batch, seqlen) in enumerate(prefill_grid, 1):
                print(
                    f"[{i:2d}/{len(prefill_grid)}] {dtype.__str__().split('.')[-1]:8s} "
                    f"{model.name:11s} B={batch:2d} S={seqlen:4d} … ",
                    end="", flush=True,
                )
                try:
                    row = analyze_shape(model, batch, seqlen, dtype, impl="fused")
                    rows.append(row)
                    print(
                        f"{row['time_us']:8.1f} μs   "
                        f"{row['achieved_bw_gbps']:6.1f} GB/s   "
                        f"MBU={row['mbu_pct']:5.1f}%"
                    )
                except torch.cuda.OutOfMemoryError:
                    print("OOM")
                except Exception as e:
                    print(f"ERROR: {e}")
                if include_baseline:
                    try:
                        bl = analyze_shape(model, batch, seqlen, dtype, impl="baseline")
                        rows.append(bl)
                    except Exception as e:
                        print(f"  baseline ERROR: {e}")

    if not prefill_only:
        decode_grid = list(product(MODEL_CONFIGS, DECODE_BATCH_SIZES))
        print(f"\n{'═' * 72}")
        print(f"  DECODE GRID (S={DECODE_SEQ_LEN}) — {len(decode_grid) * len(dtypes)} shapes")
        print(f"{'═' * 72}")
        for dtype in dtypes:
            for i, (model, batch) in enumerate(decode_grid, 1):
                print(
                    f"[{i:2d}/{len(decode_grid)}] {dtype.__str__().split('.')[-1]:8s} "
                    f"{model.name:11s} B={batch:2d} S={DECODE_SEQ_LEN:4d} … ",
                    end="", flush=True,
                )
                try:
                    row = analyze_shape(model, batch, DECODE_SEQ_LEN, dtype, impl="fused")
                    rows.append(row)
                    print(
                        f"{row['time_us']:8.1f} μs   "
                        f"{row['achieved_bw_gbps']:6.1f} GB/s   "
                        f"MBU={row['mbu_pct']:5.1f}%"
                    )
                except Exception as e:
                    print(f"ERROR: {e}")
                if include_baseline:
                    try:
                        bl = analyze_shape(model, batch, DECODE_SEQ_LEN, dtype, impl="baseline")
                        rows.append(bl)
                    except Exception as e:
                        print(f"  baseline ERROR: {e}")

    return pd.DataFrame(rows)


def format_summary(df: pd.DataFrame) -> str:
    """Render the spec'd table: shape, dtype, time_us, achieved_bw_gbps, peak_bw_gbps, MBU_pct.

    Adds MFU and arithmetic-intensity columns so compute-bound shapes (where
    MBU is low by construction) can be identified at a glance.
    """
    fused = df[df["impl"] == "fused"].copy()
    fused["shape"] = fused.apply(
        lambda r: f"{r['model']} B={r['batch']:>2} S={r['seqlen']:>4}", axis=1
    )
    show = fused[
        [
            "shape", "dtype", "time_us",
            "achieved_bw_gbps", "peak_bw_gbps", "mbu_pct",
            "achieved_tflops", "mfu_pct", "arith_intensity_flops_per_byte",
        ]
    ].copy()
    show["time_us"] = show["time_us"].round(1)
    show["achieved_bw_gbps"] = show["achieved_bw_gbps"].round(1)
    show["peak_bw_gbps"] = show["peak_bw_gbps"].astype(int)
    show["mbu_pct"] = show["mbu_pct"].round(1)
    show["achieved_tflops"] = show["achieved_tflops"].round(1)
    show["mfu_pct"] = show["mfu_pct"].round(1)
    show["arith_intensity_flops_per_byte"] = show["arith_intensity_flops_per_byte"].round(0)
    show = show.rename(columns={"arith_intensity_flops_per_byte": "ai_FLOPs/B"})
    return show.to_string(index=False)


def main():
    parser = argparse.ArgumentParser(description="MBU analysis for fused kernel")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "both"], default="fp16")
    parser.add_argument("--decode-only", action="store_true")
    parser.add_argument("--prefill-only", action="store_true")
    parser.add_argument("--with-baseline", action="store_true")
    parser.add_argument(
        "--out", type=Path,
        default=Path(__file__).parent / "results" / "mbu.csv",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("MBU analysis requires CUDA")

    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Peak HBM bandwidth assumed: {A10G_PEAK_BW_GBPS:.0f} GB/s")

    dtypes = {
        "fp16": [torch.float16],
        "bf16": [torch.bfloat16],
        "both": [torch.float16, torch.bfloat16],
    }[args.dtype]

    df = run_mbu_grid(
        dtypes=dtypes,
        include_baseline=args.with_baseline,
        decode_only=args.decode_only,
        prefill_only=args.prefill_only,
    )

    args.out.parent.mkdir(exist_ok=True, parents=True)
    # autotune dict isn't a flat column — flatten before save
    df_save = df.copy()
    if "autotune" in df_save.columns:
        df_save["autotune"] = df_save["autotune"].apply(
            lambda d: ",".join(f"{k}={v}" for k, v in d.items()) if d else ""
        )
    df_save.to_csv(args.out, index=False)

    print(f"\n{'═' * 72}")
    print(f"  Summary (fused only)")
    print(f"{'═' * 72}")
    print(format_summary(df))

    print(f"\n{'═' * 72}")
    print(f"  Autotune choices (per shape)")
    print(f"{'═' * 72}")
    seen = set()
    fused = df[df["impl"] == "fused"]
    for _, r in fused.iterrows():
        key = (r["M"], r["hidden"], r["dtype"])
        if key in seen:
            continue
        seen.add(key)
        cfg = r["autotune"]
        cfg_s = ",".join(f"{k}={v}" for k, v in cfg.items()) if cfg else "(missing)"
        print(f"  M={r['M']:>5}  H={r['hidden']:>4}  {r['dtype']:8s}  →  {cfg_s}")

    summary = fused.groupby("model")["mbu_pct"].agg(["min", "median", "max"])
    print(f"\n{'═' * 72}")
    print("  MBU summary by model (fused)")
    print(f"{'═' * 72}")
    print(summary.round(1).to_string())

    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
