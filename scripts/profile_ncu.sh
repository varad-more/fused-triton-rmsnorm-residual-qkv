#!/usr/bin/env bash
#
# Nsight Compute profile of the fused kernel on the spec'd shape:
#     model = Llama-3-8B  (hidden=4096)
#     batch = 1
#     seqlen = 2048
#     dtype = fp16
#
# Captured metrics:
#   dram__bytes.sum.per_second                                  → HBM throughput
#   l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum                → global-load bytes through L1
#   sm__warps_active.avg.pct_of_peak_sustained_active           → warp occupancy
#   smsp__inst_executed.avg                                     → executed inst per SM partition
#
# Output: ncu_report_<timestamp>.ncu-rep (binary) and parsed text in scripts/ncu/
#
# Requirements:
#   - NVIDIA Nsight Compute (`ncu` on PATH).
#     Install on Ubuntu:  sudo apt install -y nsight-compute-2024.x
#                or:      <CUDA_TOOLKIT>/bin/ncu
#   - GPU performance counter access:
#       * Either run as root (sudo).
#       * Or unblock for the current user via the kernel module option:
#           echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | \
#               sudo tee /etc/modprobe.d/nvidia-profiling.conf
#           sudo update-initramfs -u && sudo reboot
#
# Usage:
#   bash scripts/profile_ncu.sh
#

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${REPO_ROOT}/scripts/ncu"
mkdir -p "${OUT_DIR}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT="${OUT_DIR}/ncu_report_${TIMESTAMP}"

# Spec'd metrics (comma-separated for --metrics).
METRICS=(
  "dram__bytes.sum.per_second"
  "l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum"
  "sm__warps_active.avg.pct_of_peak_sustained_active"
  "smsp__inst_executed.avg"
)
METRICS_CSV=$(IFS=,; echo "${METRICS[*]}")

# Driver script invokes the fused kernel exactly once for the target shape.
# A small Python one-liner keeps the profiled launch isolated.
DRIVER_PY=$(cat <<'PY'
import torch
from triton_fused_rmsnorm_qkv.kernel import fused_rmsnorm_residual_qkv

torch.manual_seed(0)
B, S, H = 1, 2048, 4096
device = "cuda"
dtype = torch.float16

x = torch.randn(B, S, H, dtype=dtype, device=device)
r = torch.randn_like(x)
rw = torch.randn(H, dtype=dtype, device=device)
qw = torch.randn(3 * H, H, dtype=dtype, device=device)

# Warmup so autotune resolves before the profiled launch.
for _ in range(5):
    fused_rmsnorm_residual_qkv(x, r, rw, qw)
torch.cuda.synchronize()

# This is the launch we want ncu to capture.
fused_rmsnorm_residual_qkv(x, r, rw, qw)
torch.cuda.synchronize()
PY
)

if ! command -v ncu >/dev/null 2>&1; then
  echo "ERROR: 'ncu' not found on PATH. See header of this script for install instructions." >&2
  exit 127
fi

echo "Profiling Llama-3-8B  B=1  S=2048  fp16  → ${REPORT}.ncu-rep"

# --target-processes all          — capture all forked CUDA processes (PyTorch may use child workers)
# --kernel-name regex:.*fused_.*  — only the fused-kernel launches
# --launch-skip-before-match 0    — capture the very first matching launch (post-warmup)
# --launch-count 1                — single launch is enough for these metrics
# --replay-mode application       — re-run the app to collect each metric set; safe for short kernels
ncu \
  --target-processes all \
  --kernel-name "regex:.*fused_rmsnorm_residual_qkv_kernel.*" \
  --launch-skip-before-match 0 \
  --launch-count 1 \
  --metrics "${METRICS_CSV}" \
  --replay-mode application \
  --export "${REPORT}" \
  --force-overwrite \
  python -c "${DRIVER_PY}"

echo
echo "Parsed metrics:"
ncu --import "${REPORT}.ncu-rep" --csv > "${REPORT}.csv"
ncu --import "${REPORT}.ncu-rep" --print-summary per-kernel | tee "${REPORT}.txt"

echo
echo "Reports written to:"
echo "  ${REPORT}.ncu-rep   (open with 'ncu-ui')"
echo "  ${REPORT}.csv       (parsed metrics, machine-readable)"
echo "  ${REPORT}.txt       (parsed metrics, human-readable)"
