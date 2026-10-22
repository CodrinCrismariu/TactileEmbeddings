#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

run_dir=processing/runs/vla_touch_direct_vae32_50ep
.venv/bin/python -m tactilelatent.train_joint \
  --config configs/vla_touch_direct_vae32_50ep.json "$@"

if [ ! -f "$run_dir/evaluation/results.json" ]; then
  .venv/bin/python -m tactilelatent.evaluate_direct_vla_touch \
    --checkpoint "$run_dir/last.pt" --output "$run_dir/evaluation"
fi
if [ ! -f "$run_dir/evaluation/three_way_comparison.json" ]; then
  .venv/bin/python scripts/compare_vla_touch_three_way.py
fi
