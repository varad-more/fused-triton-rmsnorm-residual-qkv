# fused-triton-rmsnorm-residual-qkv

A fresh implementation workspace for a fused RMSNorm + residual + QKV path using Triton.

## Current status

This repo contains the first working cut:

- Triton forward kernel for `residual + RMSNorm`
- PyTorch wrapper for fused `QKV` projection after normalization
- Reference implementation for correctness checks
- Unit tests against the reference path
- A small benchmark harness

## Scope of this first milestone

The forward path currently fuses:

1. residual add
2. RMSNorm statistics + scaling
3. a single packed QKV linear projection in PyTorch

That means the normalization step is Triton-backed now, while the packed projection is still routed through `torch.nn.functional.linear`.

## Why this layout

It gives us a clean, testable baseline before attempting the gnarlier version where RMSNorm and the packed QKV projection are fused deeper into a single Triton program or launch pipeline.

## Project layout

- `src/fused_triton_rmsnorm_residual_qkv/ops.py` - kernels and wrappers
- `tests/test_reference.py` - correctness checks
- `benchmarks/benchmark_forward.py` - quick perf harness

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[dev]
pytest -q
```

## Benchmark

```bash
python benchmarks/benchmark_forward.py --batch 8 --seq 512 --hidden 4096
```

## Next steps

- add backward support
- benchmark against PyTorch-native kernels
- decide whether to fuse the packed QKV projection fully in Triton
- add support for tensor-parallel / grouped projection layouts if the project spec needs them
