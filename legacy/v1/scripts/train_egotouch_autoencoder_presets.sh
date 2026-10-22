#!/usr/bin/env bash
# Sequential, matched small/medium/large convolutional autoencoder experiments.
# Run from any directory. Re-running resumes unfinished runs at epoch boundaries.
set -euo pipefail
cd "$(dirname "$0")/.."

suite=processing/runs/egotouch_ae_presets_z32
mkdir -p "$suite"
exec 9>"$suite/sequence.lock"
flock -n 9 || { echo "Autoencoder preset sequence is already running" >&2; exit 1; }
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
trap 'printf "failed: %s (exit %s)\n" "${preset:-initialization}" "$?" > "$suite/status.txt"' ERR

for preset in small medium large; do
    output="processing/runs/egotouch_ae_conv_${preset}_z32"
    resume=()
    if [[ -f "$output/last.pt" ]]; then
        resume=(--resume)
    fi
    printf 'running: %s\n' "$preset" > "$suite/status.txt"
    printf 'Starting %s at %s\n' "$preset" "$(date -u +%FT%TZ)"
    .venv/bin/python -u scripts/pretrain_egotouch_autoencoder.py \
        --architecture conv --size "$preset" --latent-dim 32 --output-mode sigmoid \
        --epochs 30 --batch-size 128 --learning-rate .001 --weight-decay .0001 \
        --frame-stride 2 --max-train-frames 100000 --max-val-frames 20000 \
        --workers 2 --seed 45 --device cuda --output "$output" \
        "${resume[@]}" >> "$suite/${preset}.log" 2>&1
    printf 'Finished %s at %s\n' "$preset" "$(date -u +%FT%TZ)"
done
printf 'complete\n' > "$suite/status.txt"
