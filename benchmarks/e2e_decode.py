"""End-to-end Llama-3-8B decode benchmark.

Measures tokens/sec, per-token latency, and peak memory for three configs:
    • `baseline`      -- stock HuggingFace LlamaForCausalLM
    • `fused`         -- our Triton kernel patched into each decoder layer
    • `fused_no_at`   -- fused kernel with autotune disabled (ablation)

A `torch.compile` baseline lives in `torch_compile_baseline.py` and wires
into the same `measure_decode()` helper here.

Methodology
-----------
Prompt length = 128 (prefill), decode length = 256 (new tokens).
Batch sizes = {1, 16}. Warmup = 2 runs, timed = median of 10.

Tokens/sec is measured over the *decode* phase only (256 new tokens per
sequence). Prefill latency is reported separately. We force exactly 256 new
tokens via `min_new_tokens=256, do_sample=False` -- no early stopping.

Model weights are random (built from LlamaConfig). Weight *values* don't
affect tok/s; only architecture (layer count, hidden dims, num_heads,
num_kv_heads, rope_theta) does.
"""
from __future__ import annotations

import argparse
import os
import statistics
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd
import torch

from integration.llama3_patch import (
    apply_fused_patch,
    build_llama3_8b,
    remove_fused_patch,
)


# ---------------------------------------------------------------------------
# Benchmark driver
# ---------------------------------------------------------------------------


@dataclass
class DecodeResult:
    config: str
    batch: int
    prompt_len: int
    decode_len: int
    prefill_ms: float
    decode_ms: float
    tok_per_s: float
    ms_per_token: float
    peak_mem_gb: float


def _generate_once(model, input_ids, decode_len: int):
    """One generate() call. Returns (prefill_ms, decode_ms)."""
    torch.cuda.synchronize()

    # Separate prefill (forward on the prompt) from decode (the N generate steps)
    # by running generate with max_new_tokens=1 first, then again with full length.
    # Simpler approach: just time the full generate and compute decode = total - prefill.
    prefill_start = torch.cuda.Event(enable_timing=True)
    prefill_end = torch.cuda.Event(enable_timing=True)
    full_end = torch.cuda.Event(enable_timing=True)

    prefill_start.record()
    _ = model.generate(
        input_ids,
        max_new_tokens=1,
        min_new_tokens=1,
        do_sample=False,
        use_cache=True,
        pad_token_id=0,
    )
    prefill_end.record()
    _ = model.generate(
        input_ids,
        max_new_tokens=decode_len,
        min_new_tokens=decode_len,
        do_sample=False,
        use_cache=True,
        pad_token_id=0,
    )
    full_end.record()
    torch.cuda.synchronize()

    prefill_ms = prefill_start.elapsed_time(prefill_end)
    full_ms = prefill_start.elapsed_time(full_end) - prefill_ms   # approx decode-only
    # `full_ms` is the SECOND generate call's full time (which includes prefill again).
    # Subtract measured prefill to isolate the N-step decode.
    decode_only_ms = full_ms - prefill_ms
    if decode_only_ms <= 0:
        # fallback: attribute (full - prefill) to decode assuming prefill cost stable
        decode_only_ms = max(full_ms * (decode_len / (decode_len + 1)), 0.0)
    return prefill_ms, decode_only_ms


def measure_decode(
    model,
    batch: int,
    prompt_len: int,
    decode_len: int,
    label: str,
    num_warmup: int = 2,
    num_repeat: int = 10,
    device: str = "cuda",
) -> DecodeResult:
    """Run generate(), return a DecodeResult with median-of-N timing."""
    torch.manual_seed(0)
    vocab = model.config.vocab_size
    input_ids = torch.randint(0, vocab, (batch, prompt_len), device=device)

    for _ in range(num_warmup):
        _generate_once(model, input_ids, decode_len)

    torch.cuda.reset_peak_memory_stats()

    prefill_ms_list = []
    decode_ms_list = []
    for _ in range(num_repeat):
        p, d = _generate_once(model, input_ids, decode_len)
        prefill_ms_list.append(p)
        decode_ms_list.append(d)

    peak_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    prefill_ms = statistics.median(prefill_ms_list)
    decode_ms = statistics.median(decode_ms_list)

    total_new_tokens = batch * decode_len
    tok_per_s = total_new_tokens / (decode_ms / 1000.0)
    ms_per_token = decode_ms / decode_len   # per-step latency (batched)

    return DecodeResult(
        config=label,
        batch=batch,
        prompt_len=prompt_len,
        decode_len=decode_len,
        prefill_ms=prefill_ms,
        decode_ms=decode_ms,
        tok_per_s=tok_per_s,
        ms_per_token=ms_per_token,
        peak_mem_gb=peak_mem_gb,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_grid(
    configs: list[str],
    batches: list[int],
    prompt_len: int,
    decode_len: int,
    num_layers: int | None = None,
    dtype: torch.dtype = torch.float16,
    num_repeat: int = 10,
) -> pd.DataFrame:
    """Build one model, benchmark each (config, batch) combination."""
    results: list[DecodeResult] = []

    # Build the model once; patch/unpatch in place to avoid 2x weight alloc
    model = build_llama3_8b(dtype=dtype, num_hidden_layers=num_layers)

    for cfg in configs:
        if cfg == "baseline":
            remove_fused_patch(model)
        elif cfg == "fused":
            os.environ.pop("TRITON_FUSED_DISABLE_AUTOTUNE", None)
            remove_fused_patch(model)
            apply_fused_patch(model)
        elif cfg == "fused_no_autotune":
            os.environ["TRITON_FUSED_DISABLE_AUTOTUNE"] = "1"
            # Reimport to pick up env-gated config; safest is a fresh patch.
            remove_fused_patch(model)
            apply_fused_patch(model)
        else:
            raise ValueError(f"unknown config: {cfg}")

        for B in batches:
            print(f"[{cfg}] B={B}  prompt={prompt_len} decode={decode_len} ... ", end="", flush=True)
            try:
                r = measure_decode(
                    model, B, prompt_len, decode_len, label=cfg, num_repeat=num_repeat
                )
                results.append(r)
                print(
                    f"{r.tok_per_s:7.1f} tok/s  "
                    f"{r.ms_per_token:6.2f} ms/tok  "
                    f"peak {r.peak_mem_gb:.2f} GB"
                )
            except torch.cuda.OutOfMemoryError:
                print("OOM — skipped")
                torch.cuda.empty_cache()

    return pd.DataFrame([asdict(r) for r in results])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 16])
    ap.add_argument("--prompt-len", type=int, default=128)
    ap.add_argument("--decode-len", type=int, default=256)
    ap.add_argument(
        "--configs",
        nargs="+",
        default=["baseline", "fused", "fused_no_autotune"],
        choices=["baseline", "fused", "fused_no_autotune"],
    )
    ap.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Override num_hidden_layers (default: 32, full Llama-3-8B)",
    )
    ap.add_argument("--num-repeat", type=int, default=10)
    ap.add_argument(
        "--out", default="benchmarks/results/e2e_decode.csv",
        help="Path to CSV output",
    )
    args = ap.parse_args()

    df = run_grid(
        configs=args.configs,
        batches=args.batches,
        prompt_len=args.prompt_len,
        decode_len=args.decode_len,
        num_layers=args.num_layers,
        num_repeat=args.num_repeat,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nResults -> {out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
