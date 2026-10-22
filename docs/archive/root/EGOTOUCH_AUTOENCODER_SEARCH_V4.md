> Historical record, relocated on October 21, 2026. Original path: `EGOTOUCH_AUTOENCODER_SEARCH_V4.md`.
> Results, proposals, and operational status below belong to their recorded experiment/date;
> use the [maintained project state](../../project_state.md) for the audited active workflow.

EgoTouch sigmoid autoencoder search (v4)
=======================================

Restarted from fresh weights with a single sigmoid pressure output for training
and inference. Earlier clipped and hurdle search artifacts are preserved.
The exact search is in [configs/egotouch_ae_search_v4.json](../../../legacy/v1/configs/egotouch_ae_search_v4.json).

- Screen 24 configurations for 60 epochs each (seed 45).
- Extend two finalists per sensor-code dimension to 200 total epochs (six models).
- Retrain the top two extended configurations for 200 epochs with seeds 46 and 47.

This totals 28 initializations and 3,080 epochs. Sensor-code dimensions are
64/128/256, independent of the shared visual latent. Twelve architecture trials
cross these dimensions with channels [32,64], [48,96], [64,128], [32,64,128].
Twelve additional trials vary learning rate, batch size, weight decay, contact
weight, or MAE weight around the medium 128D configuration.

Each epoch samples 200,000 stride-2 training frames. Validation uses all 80,987
stride-2 validation frames. Selection minimizes the mean of overall, contact,
and high-pressure MAE. AdamW starts at lr 1e-4 by default; ReduceLROnPlateau halves
the rate after eight patience epochs, with a floor of 1e-6. Test data is untouched.

The estimated full duration is 12–18 hours, depending on architecture throughput.
Checkpoint extension preserves optimizer, scheduler and RNG state. Source,
manifest and plan hashes protect resumes from mixing experiments.

Launch and monitor from the project root:

```bash
mkdir -p processing/runs/egotouch_ae_search_v4
nohup setsid .venv/bin/python -u scripts/tune_egotouch_autoencoder.py \
  > processing/runs/egotouch_ae_search_v4/nohup.log 2>&1 < /dev/null &
cat processing/runs/egotouch_ae_search_v4/status.json
tail -f processing/runs/egotouch_ae_search_v4/logs/medium_code64.log
```

Per-trial checkpoints and metrics live under trials/. comparison.csv updates after
completed runs; frozen selection files record promotion decisions.
seed_stability.json reports validation variation over three seeds for each
repeated configuration.
