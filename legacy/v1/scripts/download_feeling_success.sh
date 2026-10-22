#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p datasets/evaluation_only/feeling_of_success
export OMP_NUM_THREADS=1
exec flock -n datasets/evaluation_only/feeling_of_success/download.lock \
  .venv/bin/python -u scripts/download_feeling_success.py "$@"
