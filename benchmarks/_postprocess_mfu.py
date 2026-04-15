"""Post-process an MBU CSV to add MFU + arithmetic-intensity columns.

Use when the MBU run was completed before mbu_analysis.py grew MFU columns.
Idempotent — re-running on a CSV that already has the columns is a no-op.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

A10G_PEAK_TFLOPS_FP16 = 125.0


def add_mfu(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "mfu_pct" in df.columns:
        return df
    M = df["M"].astype(float)
    H = df["hidden"].astype(float)
    time_s = df["time_us"].astype(float) / 1e6
    flops = 2.0 * M * (3 * H) * H
    df["achieved_tflops"] = flops / time_s / 1e12
    df["peak_tflops"] = A10G_PEAK_TFLOPS_FP16
    df["mfu_pct"] = df["achieved_tflops"] / A10G_PEAK_TFLOPS_FP16 * 100.0
    df["arith_intensity_flops_per_byte"] = flops / df["total_bytes"].astype(float)
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv", type=Path)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    df = add_mfu(df)
    out = args.out or args.csv
    df.to_csv(out, index=False)
    print(f"Wrote {out} ({len(df)} rows)")


if __name__ == "__main__":
    main()
