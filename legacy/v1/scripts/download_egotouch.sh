#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p datasets/EgoTouch download_logs

exec .venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="zhouzhoujy/EgoTouch",
    repo_type="dataset",
    revision="cfdbb0ac31cc2af4247943820aa250575e7e6637",
    local_dir="datasets/EgoTouch",
    allow_patterns=[
        "split.json",
        "*/*/*/chest.mp4",
        "*/*/*/left.mp4",
        "*/*/*/right.mp4",
        "*/*/*/pressure_grids.npz",
        "*/*/*/wilor_hands.json",
    ],
    max_workers=16,
)
PY
