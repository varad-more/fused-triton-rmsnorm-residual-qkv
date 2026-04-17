"""torch.compile baseline for e2e decode.

Applies `torch.compile(mode='max-autotune')` to the stock Llama-3 model and
feeds it through the same `measure_decode()` harness used by
`e2e_decode.py`. This is the honest competitor: torch.compile fuses kernels
itself (including RMSNorm+matmul variants via Inductor), so we should
expect it to be competitive with our hand-written kernel -- and that's the
interesting comparison.

Usage:
    python benchmarks/torch_compile_baseline.py
    python benchmarks/torch_compile_baseline.py --batches 1 16 --decode-len 256
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch

from integration.llama3_patch import build_llama3_8b, remove_fused_patch
from e2e_decode import measure_decode


def compile_model(model, mode: str = "max-autotune", fullgraph: bool = False):
    """Wrap the model's forward in torch.compile.

    We compile `model.forward` (not the whole module) because HF's generate
    loop calls forward per decode step; compiling forward lets Inductor
    cache a specialized kernel for the decode (S=1) shape.

    `fullgraph=False` allows graph breaks -- HF's cache update has Python
    control flow that breaks compilation if we force fullgraph.
    """
    model.forward = torch.compile(model.forward, mode=mode, fullgraph=fullgraph)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 16])
    ap.add_argument("--prompt-len", type=int, default=128)
    ap.add_argument("--decode-len", type=int, default=256)
    ap.add_argument("--num-layers", type=int, default=None)
    ap.add_argument("--num-repeat", type=int, default=10)
    ap.add_argument(
        "--compile-mode",
        default="max-autotune-no-cudagraphs",
        help=(
            "torch.compile mode. `max-autotune` triggers the CUDA-graph path which "
            "crashes on HF generate's KV-cache update; use `max-autotune-no-cudagraphs` "
            "for a fair Inductor-only comparison."
        ),
    )
    ap.add_argument("--out", default="benchmarks/results/e2e_decode_torch_compile.csv")
    args = ap.parse_args()

    # Build the baseline model, compile it. No fused patch.
    model = build_llama3_8b(num_hidden_layers=args.num_layers, dtype=torch.float16)
    remove_fused_patch(model)     # no-op; defensive
    model = compile_model(model, mode=args.compile_mode)

    print(f"Compiled with mode={args.compile_mode!r}. First call will be slow.")

    results = []
    for B in args.batches:
        print(f"[torch.compile/{args.compile_mode}] B={B} ... ", end="", flush=True)
        r = measure_decode(
            model, B, args.prompt_len, args.decode_len,
            label=f"torch.compile/{args.compile_mode}",
            num_warmup=3,         # extra warmup to absorb compile time
            num_repeat=args.num_repeat,
        )
        results.append(r)
        print(
            f"{r.tok_per_s:7.1f} tok/s  "
            f"{r.ms_per_token:6.2f} ms/tok  "
            f"peak {r.peak_mem_gb:.2f} GB"
        )

    df = pd.DataFrame([asdict(r) for r in results])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nResults -> {out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
