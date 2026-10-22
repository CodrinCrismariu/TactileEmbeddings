# Supported workflows

Use the project environment. Install once with `python -m pip install -e .`;
this provides `tl`. Help and command discovery are standard-library-only and do
not load models, download assets or create run folders.

## Configuration

`configs/experiment.json` references four ordinary JSON documents. There is no
configuration inheritance framework. Component filenames resolve beside the
experiment; artifact paths inside components resolve against `--project-root`.

- `model.json`: frozen backbone, attention, bottleneck and adapter architecture.
- `datasets.json`: selected sources, holdout, weights, prepared data and mirroring.
- `training.json`: seed, sampler cap, optimizer, SAM and run budget.
- `transfer.json`: frozen-readout splits, head, tuning and reporting protocol.

Native target shape, channel order, units, teacher code dimension and frozen
statistics come from source definitions and accepted artifact receipts. Changing
an architecture must create a new run. Do not edit historical configurations to
make them describe the current model.

## Running operations

From a private task directory, pass the project explicitly:

```bash
/absolute/project/.venv/bin/tl --project-root /absolute/project status
/absolute/project/.venv/bin/tl --project-root /absolute/project data discover
/absolute/project/.venv/bin/tl --project-root /absolute/project data verify --source egopressure
/absolute/project/.venv/bin/tl --project-root /absolute/project data cache
/absolute/project/.venv/bin/tl --project-root /absolute/project train --preflight-only
/absolute/project/.venv/bin/tl --project-root /absolute/project train
/absolute/project/.venv/bin/tl --project-root /absolute/project train --resume processing/runs/vjepa32_current_all_sources_core_v2
/absolute/project/.venv/bin/tl --project-root /absolute/project evaluate --run processing/runs/vjepa32_current_all_sources_core_v2 --epochs 10
/absolute/project/.venv/bin/tl --project-root /absolute/project view --host 127.0.0.1 --port 8080
```

`data cache` publishes small versioned consumer receipts for existing arrays; it
does not re-encode tactile outputs or rewrite historical receipts. An incompatible
cache fails validation. `train --preflight-only` performs disposable input and SAM
checks; it does not modify an existing run. Training itself creates a separate run
and requires accepted cache imports. Resume requires an unchanged v2 model,
configuration, data, implementation and optimizer/SAM/RNG state. Historical
checkpoints require the explicit, verified import described below; ordinary
resume never silently converts a historical file.

For a supported historical deterministic current-frame V-JEPA checkpoint, import
into the configured empty output directory once, then resume that new run:

```bash
tl train --import-checkpoint PATH_TO_PARENT/last.pt --import-only
tl train --resume PATH_TO_NEW_RUN
```

The parent checkpoint and completed history remain unchanged. Model weights,
AdamW moments, scheduler position, SAM counter and all RNG states are restored.
The same total epoch budget applies. See [compatibility](compatibility.md) for
the import checks and recorded continuation lineage.

`tl tune --plan PLAN --dry-run` validates an explicit candidate × seed budget.
A plan declares `candidates`, `seeds`, `budget.epochs_per_trial`,
`budget.max_trials`, and `output`. Omit `--dry-run` only when starting those trials
is intended. Evaluation freezes the representation but fits readout heads;
it is not a read-only score lookup. MotionTrans remains excluded from representation
training, source validation and code statistics.

## Viewing and compatibility

The current viewer uses the same registry and native target contract. Its
historical route preserves earlier reconstruction and prediction entries through
an isolated archived reader, including their original decoder selection.
Attention overlays display query-to-token weights, not calibrated uncertainty.
The training dashboard reads completed metrics and live progress separately.

## Active work and checkpoint safety

Inspect running jobs before changing their dependencies. The user authorized stopping the old run immediately for this migration. Four
completed epochs are preserved; the unfinished fifth epoch was discarded. The
migration verifies a separate sealed stop receipt before moving source files.
No new training starts automatically. A completed checkpoint is saved at an
epoch boundary; an arbitrary SIGTERM does not create a new checkpoint.

Logs live in run folders or `logs/`. Temporary scripts, command captures and
figures belong under `ai_work/scratch/<date>-<task>/`, with a short task note.
Keep protected backups and historical scientific records until explicitly reviewed.
