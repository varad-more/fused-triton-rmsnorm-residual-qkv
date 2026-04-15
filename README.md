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
| 2 | Fused Triton kernel (forward) | done |
| 3 | Autotuning + MBU / Nsight analysis | **done** |
| 4 | Persistent-in-N rewrite + backward pass | planned |

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

## Phase 3 — autotuning + MBU / Nsight analysis

Full write-up: [`docs/week3_profiling.md`](docs/week3_profiling.md).

The kernel is now wrapped in `@triton.autotune` over the spec'd config space
(`BLOCK_M ∈ {32,64,128}`, `BLOCK_N ∈ {64,128,256}`, `BLOCK_K ∈ {16,32,64}`,
`num_warps ∈ {4,8}`, `num_stages ∈ {2,3,4}`), keyed on `(M, H, dtype)` with
per-shape pruning.

### MBU (decode, fp16, A10G — peak 600 GB/s)

| Model | B | time (μs) | achieved BW | **MBU** |
|---|---|---:|---:|---:|
| Llama-3-8B | 1 | 251.8 | 399.9 GB/s | **66.7 %** |
| Llama-3-8B | 16 | 256.7 | 394.7 GB/s | 65.8 % |
| Mistral-7B | 1 | 252.1 | 399.5 GB/s | 66.6 % |
| Qwen2-7B | 1 | 196.7 | 392.1 GB/s | 65.3 % |

**Decode MBU band: 63.6 – 66.7 %** — 3.3 pp short of the 70 % target.
Subtracting ~20 µs of launch + Timer overhead, *kernel-time* MBU is ~73 %.

### MFU (prefill, fp16, A10G — peak 125 TFLOPs)

Prefill shapes are compute-bound (AI ≫ 208 FLOPs/byte), so MBU is not the
right metric:

| Model | B | S | time (ms) | MFU |
|---|---|---:|---:|---:|
| Llama-3-8B | 1 | 2048 | 4.51 | **36.5 %** |
| Llama-3-8B | 16 | 2048 | 128.4 | 20.6 % |
| Qwen2-7B | 1 | 2048 | 3.56 | 35.5 % |

### Nsight Compute — Llama-3-8B B=1 S=2048

| Metric | Value |
|---|---:|
| `dram__bytes.sum.per_second` | **560.6 GB/s (93.4 % of peak)** |
| `l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum` | 16.1 GB |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | 41.3 % |
| `smsp__inst_executed.avg` | 2 469 120 inst / SM partition |

The kernel is saturating HBM at 93 % of peak, but the redundant-read
pattern (each `(pid_m, pid_n)` program re-reads its `x+r` row slice)
inflates actual DRAM traffic **~19×** over the theoretical minimum. This
is the architectural ceiling — a persistent-in-N rewrite (Phase 4) is the
path to ≥ 60 % MFU on prefill and > 70 % MBU on decode.

## License

MIT
