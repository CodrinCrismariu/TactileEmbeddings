#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
.venv/bin/python -m tactilelatent.train_joint --config configs/joint_all_holdout_vla_8k_50ep.json "$@"
