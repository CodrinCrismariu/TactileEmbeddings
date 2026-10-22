#!/usr/bin/env bash
# Native sensor code size is INDEPENDENT of the future shared visual latent.
# 12 matched runs: three conv capacities x four tactile code dimensions.
set -euo pipefail
cd "$(dirname "$0")/.."
suite=processing/runs/egotouch_ae_sensor_code_grid_v1
mkdir -p "$suite"
exec 9>"$suite/sequence.lock"
flock -n 9 || { echo "Autoencoder grid is already running" >&2; exit 1; }
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
trap 'printf "failed: %s (exit %s)\n" "${run:-initialization}" "$?" > "$suite/status.txt"' ERR

for sensor_code_dim in 16 32 64 128; do
    for preset in small medium large; do
        run="${preset}_code${sensor_code_dim}"
        output="$suite/$run"
        resume=()
        if [[ -f "$output/last.pt" ]]; then
            resume=(--resume)
        fi
        printf 'running: %s\n' "$run" > "$suite/status.txt"
        printf 'Starting %s at %s\n' "$run" "$(date -u +%FT%TZ)"
        .venv/bin/python -u scripts/pretrain_egotouch_autoencoder.py \
            --architecture conv --size "$preset" --sensor-code-dim "$sensor_code_dim" --output-mode sigmoid \
            --epochs 30 --batch-size 128 --learning-rate .0001 --weight-decay .0001 \
            --frame-stride 2 --max-train-frames 100000 --max-val-frames 20000 \
            --workers 2 --seed 45 --device cuda --output "$output" \
            "${resume[@]}" >> "$suite/${run}.log" 2>&1
        printf 'Finished %s at %s\n' "$run" "$(date -u +%FT%TZ)"
        .venv/bin/python scripts/summarize_egotouch_autoencoder_grid.py --root "$suite"
    done
done
printf 'complete\n' > "$suite/status.txt"
