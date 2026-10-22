#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p processing/runs/motiontrans_baseline_sequence
# The lock spans the entire sequence so a second launcher cannot duplicate it.
exec flock -n processing/runs/motiontrans_baseline_sequence/sequence.lock \
  .venv/bin/python -u scripts/train_motiontrans_sequence.py "$@"
