> Historical record, relocated on October 21, 2026. Original path: `EGOTOUCH_AUTOENCODER_SEARCH.md`.
> Results, proposals, and operational status below belong to their recorded experiment/date;
> use the [maintained project state](../../project_state.md) for the audited active workflow.

**Historical, stopped:** the user requested a single clipped pressure output.
The active search is [EGOTOUCH_AUTOENCODER_SEARCH_V3.md](EGOTOUCH_AUTOENCODER_SEARCH_V3.md).
Commands below describe the abandoned design; the launcher now defaults to v3.

The second EgoTouch autoencoder search extends the first grid with more training,
full validation, broader architectures and optimizer/loss settings. Sensor-code
dimension is independent of the shared visual latent: a later trainable multilayer
adapter will map the visual latent into whichever sensor code is selected.

The exact plan is [configs/egotouch_ae_search_v2.json](../../../legacy/v1/configs/egotouch_ae_search_v2.json).
The launcher is [scripts/tune_egotouch_autoencoder.py](../../../legacy/v1/scripts/tune_egotouch_autoencoder.py).

| Stage | Work |
| --- | --- |
| Screening | 30 fresh models (28 hurdle + 2 matched sigmoid controls), 60 epochs each, seed 45 |
| Extended training | Best two configurations per code dimension (six total), continued to 200 total epochs |
| Seed repetitions | Best two extended configurations, each trained fresh for 200 epochs with seeds 46 and 47 |

There are 34 distinct model initializations and 3,440 planned training epochs:
30*60 + 6*(200-60) + 4*200. Extended finalists preserve their exact optimizer,
scheduler, sampling-epoch and RNG state. Earlier 30-epoch grid artifacts remain
unchanged. This is a targeted search, not an exhaustive search for a global optimum.

The search space is:

- Sensor codes: 64, 128, 256 dimensions.
- Encoder channels: [32,64], [48,96], [64,128], [32,64,128], with mirrored decoders.
  All four architectures are tested at all three code dimensions (12 trials).
- At the medium [32,64], 128D anchor, twelve additional single-factor variants:
  learning rates 3e-5/5e-5/2e-4/3e-4; batch sizes 64/256; weight decay 0/.001;
  contact weights 1/5; absolute-error weights .25/1.0.
- The shared baseline is lr=1e-4, batch=128, weight decay=1e-4, contact weight=3,
  absolute-error weight=.5. The MSE weight stays 1. Contact weighting uses .1;
  contact evaluation uses .05, high-pressure evaluation uses .5.
- Four additional 128D variants change nonzero BCE weight from 1 to .3/3, or
  nonzero-probability threshold from .5 to .3/.7. Two medium 64D/128D sigmoid
  controls reproduce the old output/loss family with the new data budget.
  Controls are reported but excluded from promotion: exact-zero output is now
  a requirement for the selected decoder.

The default **hurdle decoder** emits two logits per hand/taxel: whether pressure
is strictly positive, and its positive magnitude. Training uses masked binary
cross entropy plus the weighted MSE/MAE magnitude loss on strictly positive valid
cells. Positive regression is normalized by positive-cell count; BCE uses all
valid cells. An all-zero batch has zero magnitude loss but still trains the gate.

Evaluation emits EXACT zero when predicted nonzero probability is below its
threshold; otherwise it emits the positive magnitude. This is a learned decision,
not rounding every small pressure to zero. Tiny positive targets remain positive
labels even below the .05 contact threshold. Invalid cells are masked in both
branches. The probabilities are not claimed calibrated.

Metrics and model selection use the hard output. Reports include exact-zero
recall, false-zero rate on genuinely positive cells, and the gate's miss rate on
contact cells above .05, so high zero recall cannot hide missed contacts. Later
RGB training should use `ae.decode(code, hard=False)` for a differentiable
probability-times-magnitude prediction through a frozen/eval decoder. Final
reconstruction uses `hard=True`.

Every epoch samples 200,000 frames from the existing train episodes, doubling
the first grid's per-epoch training exposure. All 80,987 stride-2 validation
frames are evaluated each epoch, rather than the old 20,000-frame subset.
Official test pressure files are never opened. All trials use the same split and
evaluation thresholds; source files and the data manifest are preserved.

Checkpoint selection, LR scheduling, and trial ranking use **balanced_mae**:

```text
(overall MAE + contact-region MAE + high-pressure-region MAE) / 3
```

This is an explicit equal weighting of three error summaries, not population
MAE. It emphasizes contact fidelity in sparse pressure data and does not change
when a trial's training-loss weights change. All three component errors, contact
F1, total parameter counts, and raw MAE remain visible in the comparison CSV.
For tiny smoke subsets only, absent contact/high-pressure regions fall back to
overall MAE. The full validation split contains both regions. This criterion
differs from the old grid's MAE criterion and validation subset, so their headline
scores must not be compared as if selection protocols were identical.

ReduceLROnPlateau halves the LR after eight patience epochs without sufficient
score improvement, down to 1e-6. There is no automatic early stopping: every
screened model gets its planned budget. The best checkpoint is retained even if
later epochs regress. Selection snapshots are written once before extending or
repeating candidates so restarts cannot silently change which trials advance.

Launch and monitor:

```bash
mkdir -p processing/runs/egotouch_ae_search_v2
nohup setsid .venv/bin/python -u scripts/tune_egotouch_autoencoder.py \
  > processing/runs/egotouch_ae_search_v2/nohup.log 2>&1 < /dev/null &

cat processing/runs/egotouch_ae_search_v2/status.json
tail -f processing/runs/egotouch_ae_search_v2/nohup.log
tail -f processing/runs/egotouch_ae_search_v2/logs/medium_code64.log
```

Repeat the same launcher command after an interruption; an exclusive lock refuses
a duplicate supervisor. Completed stages are skipped and partial trainers resume
from completed epochs. If a run was interrupted before its first checkpoint,
its partial files are archived before restarting it. A changed plan, implementation,
or manifest is rejected on restart; use a new output directory for changed work.

Run a small real-data CPU test of screening, extension, and repetition with:

```bash
.venv/bin/python scripts/tune_egotouch_autoencoder.py --smoke
```

Outputs are under `processing/runs/egotouch_ae_search_v2/`:

- `plan.json` and `provenance.json`: resolved search and file hashes.
- `status.json`: current stage, trial and log path, or completion/failure.
- `comparison.csv`: best-checkpoint metrics, updated after each completed stage/run.
- `screening_selection.json`: frozen screening results and six selected finalists.
- `repeat_selection.json`: frozen extended results and two repeated configurations.
- `seed_stability.json`: final validation-score means, sample SDs and ranges over
  the three training seeds. These SDs are not confidence intervals on new objects.
- `trials/<name>/`: normal autoencoder configs, best/last checkpoints and examples.
- `logs/<name>.log`: detailed per-epoch metrics, LR, and durations.

Final conclusions must distinguish raw-MAE, contact-quality, size/efficiency and
seed-stability tradeoffs. Repeated validation selection is model development;
no untouched-test or calibrated-uncertainty claim follows from winning this search.
