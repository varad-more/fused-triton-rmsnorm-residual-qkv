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
| 1 | PyTorch baseline + benchmark harness | **current** |
| 2 | Fused Triton kernel (forward) | planned |
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

## Results

*(to be populated after running on A10G)*

## License

MIT
