#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec bash scripts/run_vjepa_pair.sh --config configs/joint_vjepa21_pair_32d_cached_v1.json --resume --adopt-feature-cache "$@"
