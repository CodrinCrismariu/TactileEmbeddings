#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
.venv/bin/python -m tactilelatent.evaluate_uncertainty \
  --checkpoint processing/runs/egotouch_dinov3_vae_image_only_latent32/last.pt \
  --output processing/evaluations/egotouch_vae32_uncertainty.json \
  --split test_unseen --views ego all --samples 32 "$@"
