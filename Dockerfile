# Reproducible environment for the fused Triton kernel benchmarks.
#
# Target: NVIDIA A10G (Ampere, sm_86). Pins the exact stack used to generate
# the numbers in docs/week4_e2e.md:
#   PyTorch 2.11, Triton 3.6, Transformers 5.5.4, CUDA 13.1, Python 3.13
#
# Build:
#   docker build -t fused-triton-rmsnorm-qkv .
#
# Run the full benchmark grid (requires --gpus all + ≥22 GB VRAM):
#   docker run --rm --gpus all -v "$PWD/benchmarks/results:/app/benchmarks/results" \
#       fused-triton-rmsnorm-qkv make benchmark
#
# Interactive shell for reproducing individual commands from the README:
#   docker run --rm -it --gpus all fused-triton-rmsnorm-qkv bash
FROM nvidia/cuda:13.1.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.13 python3.13-dev python3.13-venv python3-pip \
        git build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.13 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.13 1

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/

# Pin exactly what generated the numbers in docs/week4_e2e.md.
RUN pip install --upgrade pip \
    && pip install torch==2.11.0 triton==3.6.0 \
    && pip install transformers==5.5.4 numpy pandas matplotlib \
    && pip install -e ".[dev]"

COPY benchmarks/ ./benchmarks/
COPY tests/ ./tests/
COPY integration/ ./integration/
COPY scripts/ ./scripts/
COPY docs/ ./docs/
COPY Makefile ./

ENV PYTHONPATH=/app/src:/app:/app/benchmarks

CMD ["bash"]
