#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export LD_LIBRARY_PATH="$PWD/processing/runtime/nvidia-580.173.02/root/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
exec "$PWD/.venv/bin/python" -u -m tactilelatent.train_vjepa --config configs/joint_vjepa21_pair_32d_v1.json "$@"
