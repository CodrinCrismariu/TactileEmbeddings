#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

run_dir=processing/runs/joint_all_alternating_fresh_8k_50ep
.venv/bin/python -m tactilelatent.train_joint_alternating \
  --config configs/joint_all_alternating_fresh_8k_50ep.json "$@"

if [ ! -f "$run_dir/evaluations/vla_touch/results.json" ]; then
  .venv/bin/python -m tactilelatent.evaluate_holdout_decoder \
    --checkpoint "$run_dir/last.pt" --output "$run_dir/evaluations/vla_touch"
fi
if [ ! -f "$run_dir/evaluations/dataset_identity/results.json" ]; then
  .venv/bin/python -m tactilelatent.evaluate_dataset_classifier \
    --checkpoint "$run_dir/last.pt" --output "$run_dir/evaluations/dataset_identity"
fi
if [ ! -f "$run_dir/evaluations/baseline_comparison.json" ]; then
  .venv/bin/python scripts/compare_alternating_baseline.py
fi
