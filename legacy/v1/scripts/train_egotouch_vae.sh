#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
.venv/bin/python -m tactilelatent.train --config configs/egotouch_vae_image_only_latent32.json "$@"
