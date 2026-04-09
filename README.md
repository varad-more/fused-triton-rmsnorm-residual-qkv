# Fused RMSNorm + Residual + QKV Projection — Triton Kernel

Production-grade Triton kernel that fuses the residual add, RMSNorm, and
packed QKV projection into a single GPU launch — targeting the critical path
in decoder-only transformer inference (Llama-3-8B, Mistral-7B, Qwen2-7B).

**Target hardware:** AWS g5.2xlarge (NVIDIA A10G, 24 GB VRAM, 600 GB/s HBM).

## Why this matters

In standard transformer inference the sequence
`residual_add → RMSNorm → QKV_linear` issues three separate CUDA kernels,
each paying a full round-trip to HBM. Fusing them into one Triton program
eliminates two kernel-launch overheads and two redundant global memory passes,
which matters most at small-to-medium batch sizes where the workload is
memory-bandwidth-bound.

## Project roadmap

| Week | Milestone | Status |
|------|-----------|--------|
| 1 | PyTorch baseline + benchmark harness | done |
| 2 | Fused Triton kernel (forward) | **done** |
| 3 | Autotuning + roofline analysis | planned |
| 4 | Backward pass + end-to-end integration | planned |

## Repository layout

```
src/triton_fused_rmsnorm_qkv/
    baseline.py          # unfused PyTorch reference
    kernel.py            # (week 2) fused Triton kernel
benchmarks/
    harness.py           # torch.utils.benchmark-based harness
    results/             # CSV outputs
tests/
    test_correctness.py  # parametrized correctness suite
notebooks/               # exploration & analysis
```

## Quick start

```bash
# Install (requires uv)
make install

# Run correctness tests
make test

# Run benchmark grid
make benchmark
```

## Model configurations benchmarked

| Model | Hidden | Heads | Head dim |
|-------|--------|-------|----------|
| Llama-3-8B | 4096 | 32 | 128 |
| Mistral-7B | 4096 | 32 | 128 |
| Qwen2-7B | 3584 | 28 | 128 |

Shape grid: batch in {1, 4, 16}, seqlen in {128, 512, 2048}.

## Benchmark methodology

- **Timer:** `torch.utils.benchmark.Timer` (not `time.time`)
- **Sync:** explicit `torch.cuda.synchronize()` barriers
- **Warmup:** 10 iterations discarded
- **Measurement:** `blocked_autorange` with min 0.5 s run time
- **Output:** pandas DataFrame → `benchmarks/results/baseline.csv`

## Test results

Correctness suite: **44/44 passed** (CUDA, Python 3.12, PyTorch 2.6+cu124, Triton 3.2, A10G)

```
tests/test_correctness.py  44 passed in 58.29s

  TestBaselineCorrectness
    test_matches_manual_reference  8/8 passed  (fp16 + bf16 × 4 shapes)
    test_output_shapes             8/8 passed
    test_output_dtype_preserved    2/2 passed  (fp16 + bf16)
  TestEdgeCases
    test_zero_residual             1/1 passed
    test_unit_weight               1/1 passed
  TestFusedKernelCorrectness
    test_matches_baseline          8/8 passed  (fp16 + bf16 × 4 CUDA shapes)
    test_output_shapes             8/8 passed
    test_no_nans                   8/8 passed
```

## Kernel design (Phase 2)

The fused kernel uses a **2D grid** `(ceil(M/BLOCK_M), ceil(3H/BLOCK_N))` with
`tl.dot` (tensor cores) for the QKV projection.

Each program:
1. **Phase 1** — loads `(BLOCK_M, H)` of `hidden = x + r` in `BLOCK_K`-column
   chunks, accumulates sum-of-squares, computes `rstd[BLOCK_M]`.
2. **Phase 2** — re-reads the same hidden slice (L1-resident for small `BLOCK_M`)
   per K-tile, applies RMSNorm gain, then `tl.dot` with the
   `(BLOCK_K, BLOCK_N)` weight tile → accumulates output in fp32.

Default tile sizes: `BLOCK_M=16`, `BLOCK_N=128`, `BLOCK_K=64`
(autotuning is Phase 3).

### Performance profile

The kernel is designed for the **decode regime** (single new token per request,
`M = B ≤ BLOCK_M = 16`). In this regime `ceil(M/BLOCK_M) = 1`, so the weight
matrix is read exactly once, matching cuBLAS's amortisation.

For the prefill/training regime (large `M`), the weight is read
`ceil(M/BLOCK_M)` times — this amplification will be addressed in Phase 3
via autotuning of `BLOCK_M` and a persistent-kernel variant.

## Benchmark results (A10G, fp16)

### Decode step (S=1 — target regime)

| Model | B | Baseline (μs) | Fused (μs) | Speedup |
|-------|---|---------------|------------|---------|
| Llama-3-8B | 1 | 242 | 254 | 0.95× |
| Llama-3-8B | 4 | 234 | 256 | 0.91× |
| Llama-3-8B | 8 | 238 | 257 | 0.93× |
| Llama-3-8B | 16 | 258 | 259 | 1.00× |
| Qwen2-7B | 16 | 227 | 205 | **1.11×** |

Mean decode speedup: **0.96×** (essentially on-par; Phase 3 autotuning targets >1.0×).
Peak bandwidth achieved: **~430 GB/s** / 600 GB/s = 72% of A10G memory bandwidth.

### Prefill (S=128–2048 — reference, not the optimised regime)

| Model | B | S | Baseline (ms) | Fused (ms) | Speedup |
|-------|---|---|---------------|------------|---------|
| Llama-3-8B | 1 | 128 | 0.32 | 0.63 | 0.51× |
| Llama-3-8B | 1 | 2048 | 4.11 | 14.95 | 0.27× |
| Llama-3-8B | 16 | 2048 | 60.7 | 241.7 | 0.25× |

Prefill slowdown is due to weight matrix amplification (`ceil(M/BLOCK_M)` reads
instead of 1). Addressed in Phase 3.

## License

MIT
