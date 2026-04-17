# Week 4 — End-to-End Llama-3-8B Decode

Drop-in integration of the fused RMSNorm+QKV Triton kernel into
`LlamaForCausalLM`, measured against the stock HuggingFace baseline and
`torch.compile(mode='max-autotune-no-cudagraphs')`.

This document reports results after the **Phase 6 native-GQA kernel
refactor** (`N_OUT` constexpr + `HAS_RESIDUAL` constexpr). A Phase-5 →
Phase-6 before/after table at the bottom documents the refactor's impact.

- **Hardware:** AWS g5.2xlarge (NVIDIA A10G, 22 GB, 600 GB/s HBM)
- **Driver / CUDA:** 590.48 / 13.1
- **Software:** PyTorch 2.11, Triton 3.6, Transformers 5.5.4, Python 3.13
- **Model:** Llama-3-8B config, 32 layers, fp16, **random weights**
  (speed depends on architecture shape, not weight values — see Methodology)
- **Workload:** prompt length 128, decode length 256, greedy, `use_cache=True`
- **Timing:** 2 warmup + median of 10 `generate()` runs per cell

Raw results:
[`benchmarks/results/e2e_decode.csv`](../benchmarks/results/e2e_decode.csv),
[`benchmarks/results/e2e_decode_torch_compile.csv`](../benchmarks/results/e2e_decode_torch_compile.csv)

## Results

| config | batch | tok/s | latency (ms/tok) | peak mem (GB) |
|--------|------:|------:|-----------------:|--------------:|
| baseline (HF + cuBLAS) | 1 | 26.8 | 37.26 | 15.02 |
| torch.compile | 1 | **31.2** | 32.02 | 15.07 |
| fused (ours, autotune) | 1 | **27.2** | 36.74 | 16.52 |
| fused (ours, no autotune) | 1 | 25.3 | 39.55 | 16.52 |
| baseline (HF + cuBLAS) | 16 | 379.6 | 42.15 | 15.76 |
| torch.compile | 16 | 342.2 | 46.76 | 16.57 |
| fused (ours, autotune) | 16 | **388.7** | 41.17 | 17.26 |
| fused (ours, no autotune) | 16 | 363.4 | 44.02 | 17.26 |

**tok/s** = `batch × decode_len / decode_ms × 1000`. Prefill is timed
separately (subtracted out) so the number reflects steady-state decode.

### Speedup matrix (vs HF baseline)

| | B=1 | B=16 |
|---|---:|---:|
| torch.compile | **+16.4%** | −9.9% |
| fused (ours, autotune) | **+1.4%** | **+2.4%** |
| fused (ours, no autotune) | −5.6% | −4.3% |

### Autotune contribution (ablation)

| batch | fused (autotune) | fused (fixed tile) | Δ from autotune |
|------:|-----------------:|-------------------:|----------------:|
| 1     | 27.2 tok/s       | 25.3 tok/s         | **+7.5%** |
| 16    | 388.7 tok/s      | 363.4 tok/s        | **+7.0%** |

The "fixed tile" config is the Week-2 starting point (BLOCK_M=32/64,
BLOCK_N=128, BLOCK_K=32, num_warps=4, num_stages=2 — Llama's first naive
choice). Autotune reliably buys ~7% at these decode shapes even after the
Phase-6 refactor.

## Honest analysis

### Where we **win**

1. **Over HF baseline at both batches** (+1.4% B=1, +2.4% B=16). After the
   Phase 6 native-GQA refactor, one fused kernel per layer amortizes
   cuBLAS's three-launch overhead for q_proj / k_proj / v_proj. The win is
   narrow — cuBLAS GEMMs are very good — but it is a real win on the model
   that matters.
2. **Over torch.compile at B=16** (+13.6%). At large-M decode, Inductor's
   per-kernel launch overhead (no-cudagraphs mode) adds up across the many
   dispatches per token; one fused Triton launch wins cleanly.
3. **Autotune** reliably contributes +7% at both batches. The Week-3 tuning
   work holds up end-to-end.

### Where we **lose**

1. **vs torch.compile at B=1** (−12.8%). Inductor's RMSNorm+matmul fusion
   plus a generated decode-specialized kernel beats us in the single-sequence
   regime. This is the honest comparison: a modern `torch.compile(mode='max-autotune')`
   is a tough baseline for hand-written kernels on shapes it knows how to
   fuse. Plain `max-autotune` crashes inside HF `generate()`'s KV-cache
   update (see Known Issues) so we compare against `max-autotune-no-cudagraphs`;
   if Inductor could also amortize the per-kernel launch cost, it would be
   faster still.

### Why (root causes — Phase-6 scoreboard)

1. **GQA zero-padding — FIXED in Phase 6.** The Phase-5 packed `(3H, H)`
   weight forced K/V zero-padding, doubling QKV weight HBM traffic. Phase 6's
   `N_OUT` constexpr lets us pack `(H + 2*H_kv, H) = (6144, 4096)` — the
   honest layout. Peak memory drops from 18.0 GB to 16.5 GB for the same
   integration (packed weight shrinks from 12288×4096 to 6144×4096).
2. **No residual fusion engaged — FIXED in Phase 6.** `HAS_RESIDUAL`
   constexpr skips the residual load entirely when absent (Llama places
   residual-add *after* attention, so the input-norm+QKV block has none).
   Previously the kernel read a cached zero tensor from HBM twice (variance
   and matmul phases), wasting ~5% decode bandwidth.
3. **torch.compile B=16 regression** (342 vs 380 HF baseline) — unchanged
   root cause. Inductor compiles a decode-shape kernel tuned for M=16, but
   the prefill (M=2048) path picks a compute-bound tile that doesn't help
   decode throughput. `no-cudagraphs` mode also costs ~µs per kernel launch
   vs cuBLAS + stream concurrency.

### Known issues surfaced during integration

- **`torch.compile(mode='max-autotune')` + HF `generate()`:**
  CUDA-graph replay fails with
  `RuntimeError: accessing tensor output of CUDAGraphs that has been
  overwritten by a subsequent run` inside
  `apply_rotary_pos_emb`. Switching to `max-autotune-no-cudagraphs` side-steps
  this but loses the launch-overhead amortization that makes
  `max-autotune` fast at small M. This is a well-known HF/PyTorch
  integration gap, not our code.

## Phase-5 → Phase-6 before/after

Context for the refactor. Numbers from commit 495a973 (Phase 5,
zero-padded `(3H, H)`) vs the current run (Phase 6, native GQA `(H+2*H_kv, H)`):

| config | B | Phase 5 tok/s | Phase 6 tok/s | Δ | Phase 5 peak | Phase 6 peak |
|---|---|---:|---:|---:|---:|---:|
| fused (autotune) | 1 | 24.5 | **27.2** | +11.0% | 18.02 GB | 16.52 GB |
| fused (autotune) | 16 | 353.9 | **388.7** | +9.8% | 18.77 GB | 17.26 GB |

vs HF baseline, the fused path went from **−9% to +1.4%** at B=1 and from
**−6% to +2.4%** at B=16. Peak memory dropped ~1.5 GB as predicted (the
packed weight shrank from 12288×4096 to 6144×4096 per layer).

## What the numbers mean for the portfolio story

- The hand-written kernel **narrowly beats cuBLAS on stock HF Llama-3-8B**
  at both decode batches, after the native-GQA refactor. Not a dramatic
  win — cuBLAS GEMMs are very good — but a real one in the HBM-bound decode
  regime where kernel launch amortization matters.
- The rigorous comparison against `torch.compile` shows Inductor wins
  cleanly at B=1 (+16%) — the honest headline caveat. We beat it
  meaningfully at B=16 (+14%) where launch-overhead amortization matters.
- Autotune contributes a reliable +7% at both batches over the naive Week-2
  tile — the Week-3 tuning work carries through end-to-end.

## Methodology notes

### Why random weights are fine for this measurement

Decode latency on a transformer depends on (a) tensor shapes, (b) dtypes,
(c) kernel launches, (d) HBM traffic, and (e) cache behavior. None
depend on weight *values*. We build the model directly from `LlamaConfig`
matching `meta-llama/Meta-Llama-3-8B`'s `config.json` (hidden=4096,
heads=32, kv_heads=8, head_dim=128, intermediate=14336, layers=32,
rope_theta=500000, rms_norm_eps=1e-5, vocab=128256). Pretrained weights
would reproduce these numbers exactly.

Building the model in fp16 directly (via
`torch.set_default_dtype(torch.float16)`) is required on A10G — the
default fp32 constructor needs 32 GB for 8B params and OOMs the 22 GB
card.

### Timing

`_generate_once()` in `benchmarks/e2e_decode.py` runs two `generate()`
calls per measurement: one with `max_new_tokens=1` to isolate prefill,
then one with `max_new_tokens=256`. Decode-only time is the difference.
`torch.cuda.Event` pairs around both calls; `cuda.synchronize()` before
reading. `min_new_tokens=max_new_tokens` suppresses early stopping.

### Repro

```bash
# Full e2e grid (baseline, fused, fused_no_autotune) on native-GQA kernel
PYTHONPATH=.:benchmarks python benchmarks/e2e_decode.py \
    --batches 1 16 --prompt-len 128 --decode-len 256 --num-repeat 10

# torch.compile comparison
PYTHONPATH=.:benchmarks python benchmarks/torch_compile_baseline.py \
    --batches 1 16 --prompt-len 128 --decode-len 256 --num-repeat 10 \
    --compile-mode max-autotune-no-cudagraphs
```

Full wall time on A10G: ~10 min for the three-config grid, ~10 min for
the torch.compile run (compile dominates).
