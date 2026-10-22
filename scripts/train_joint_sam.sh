#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec .venv/bin/python -u -m tactilelatent.train_joint --config configs/joint_all_sam_holdout_motiontrans_fresh_8k_50ep.json "$@"
