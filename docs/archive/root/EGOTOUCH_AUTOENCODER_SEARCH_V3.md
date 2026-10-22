> Historical record, relocated on October 21, 2026. Original path: `EGOTOUCH_AUTOENCODER_SEARCH_V3.md`.
> Results, proposals, and operational status below belong to their recorded experiment/date;
> use the [maintained project state](../../project_state.md) for the audited active workflow.

Historical search: stopped at user request; partial artifacts preserved.
The active sigmoid restart is documented in [EGOTOUCH_AUTOENCODER_SEARCH_V4.md](EGOTOUCH_AUTOENCODER_SEARCH_V4.md).

The active EgoTouch search uses **one raw pressure output per taxel**, with
clipping to [0,1] for reconstruction metrics and inference. There is no separate
zero classifier, contact output branch, or binary classification loss.

The decoder's final layer is linear:

```text
sensor code -> decoder -> raw pressure
training loss: weighted MSE + MAE on raw pressure
inference: pressure = raw_pressure.clamp(0, 1)
```

Raw predictions <= 0 become exactly 0; predictions >= 1 become exactly 1.
Small positive pressures are retained, rather than rounded by a threshold.
Training against raw predictions preserves corrective gradients outside the
sensor range. Training through a hard clamp would have zero local gradient for
out-of-range outputs. Metrics always use clipped predictions, including during
training, while the optimization loss uses raw predictions.

For later RGB training with a frozen decoder, use `ae.decode(code, hard=False)`
to get raw pressure and preserve gradients into the visual adapter. For final
bounded inference use `ae.decode(code, hard=True)` or an eval-mode decoder.
The `hard` flag selects raw versus clipped decoding in this mode; it does not
invoke a classifier. The old sigmoid and abandoned hurdle modes remain loadable
for artifact compatibility, but new training defaults to `--output-mode clamp`.

Exact-zero capability does not guarantee correct zero decisions or calibrated
uncertainty. Reports include exact-zero recall, false-zero rate on positive
targets, and the zero-output miss rate on contact cells above .05. All metrics
exclude invalid sensor cells; ground-truth zero means exactly zero.

The plan is [configs/egotouch_ae_search_v3.json](../../../legacy/v1/configs/egotouch_ae_search_v3.json).
Sensor codes (64/128/256) remain independent of the shared visual latent and the
future trainable multilayer adapter between them.

| Stage | Budget |
| --- | --- |
| Screening | 26 fresh models, 60 epochs each, seed 45 |
| Extended training | Two finalists per code dimension (six models), continued to 200 total epochs |
| Seed repetition | Top two extended configurations, each retrained for 200 epochs with seeds 46 and 47 |

Total: 30 initializations and 3,200 planned epochs, preserving optimizer,
scheduler, RNG and sampling progress when extending finalists. The 26 screens
include 24 clipped-output candidates and two matched sigmoid controls; only
clipped-output candidates can advance. This is a targeted search, not a guarantee
of a globally optimal configuration.

The 24 candidates test all combinations of sensor code 64/128/256 and channels
[32,64], [48,96], [64,128], [32,64,128] (12 models), plus twelve single-factor
variants around the [32,64], 128D anchor:

- Learning rates: 3e-5, 5e-5, 2e-4, 3e-4 (baseline 1e-4).
- Batch sizes: 64, 256 (baseline 128).
- Weight decay: 0, .001 (baseline .0001).
- Contact-region loss weights: 1, 5 (baseline 3).
- Absolute-error weights: .25, 1 (baseline .5; squared-error weight is 1).

Each epoch uses 200,000 randomly selected stride-2 training frames. Validation
uses all 80,987 stride-2 validation frames, including every usable validation
episode. Test pressure files are never opened. All new trials start fresh.

The fixed model-selection score is:

```text
balanced_mae = (overall MAE + contact-region MAE + high-pressure-region MAE) / 3
```

This explicitly emphasizes contact fidelity and is not population MAE. The
three components remain visible independently. Contact is target > .05; high
pressure is target >= .5; the training contact weight applies above .1. Neither
the evaluation score nor these thresholds changes when tuning loss weights.
ReduceLROnPlateau halves LR after eight patience epochs without sufficient score
improvement, with minimum LR 1e-6. No early stopping is applied. The best model
is retained if later epochs worsen. Tiny smoke subsets without a region use
global MAE for that missing component; full validation has both regions.

```bash
mkdir -p processing/runs/egotouch_ae_search_v3
nohup setsid .venv/bin/python -u scripts/tune_egotouch_autoencoder.py --plan configs/egotouch_ae_search_v3.json --output processing/runs/egotouch_ae_search_v3 \
  > processing/runs/egotouch_ae_search_v3/nohup.log 2>&1 < /dev/null &

cat processing/runs/egotouch_ae_search_v3/status.json
tail -f processing/runs/egotouch_ae_search_v3/logs/medium_code64.log
```

Re-launching the same command skips completed work and resumes unfinished
epochs. A lock blocks duplicate supervisors. Changed source code, plan or data
manifest is rejected to avoid mixing implementations within one search.
Interrupted pre-checkpoint trial directories are archived before restarting.

The suite writes `plan.json`, source/manifest hashes in `provenance.json`,
`status.json`, per-run checkpoints and logs, and `comparison.csv` after each
completed stage/run. Frozen `screening_selection.json` and `repeat_selection.json`
record which candidates advance. `seed_stability.json` reports the final score
mean, sample SD and range over three seeds for each repeated configuration.
These describe validation selection, not untouched-test generalization.

The abandoned two-head v2 search was stopped at the user's request; partial
artifacts remain in `processing/runs/egotouch_ae_search_v2`. The initial sigmoid
grid also remains unchanged. Their scores have different selection/data budgets
and should not be treated as matched comparisons to this search.
