# Operational tools

`tl` is the maintained command surface. Model/dataset implementations live in
`src/tactilelatent/`; these command adapters orchestrate explicit workflows.
Historical experiment launchers under `legacy/v1/` are records, not fallback tools.

The [code standards](../docs/code_standards.md) apply to every maintained tool.
Each command documents its inputs, outputs and an example. Update the
[root command index](../README.md#maintained-command-index) when adding, renaming
or removing a command. Read-only checks live in `tactilelatent_tools/checks/`:

```bash
python -m tactilelatent_tools.checks.code_standards --project-root .
python -m tactilelatent_tools.checks.format_assets --project-root .
```

The asset check reports differences; `--write` explicitly applies formatting.
Install `.[dev]` for pinned formatters. Neither check loads models or datasets.

Use `tl --help` or any command's `--help`. Help imports only standard-library
modules and does not load datasets, touch the GPU, download or write artifacts.
From an uninstalled checkout, use `PYTHONPATH=src:tools .venv/bin/python -m
tactilelatent_tools` in place of `tl`.

| Command | Inputs | Significant effects |
|---|---|---|
| `train --config CONFIG` | Explicit four-part experiment; verified source artifacts | New current-core training run on the configured device |
| `train --resume RUN` | Same experiment and resumable v2 `last.pt` | Resumes optimizer/SAM/scheduler/RNG state; refuses incompatible history |
| `train --import-checkpoint PARENT/last.pt --import-only` | Complete supported legacy current-frame checkpoint; matching science/data | Verifies and writes a new schema-2 continuation run, parent hash and original metadata; no optimization until resumed |
| `train --preflight-only` | Native source clips and sensor teachers | Disposable native-input/SAM checks; no saved run weights changed |
| `evaluate --run RUN --epochs 10 --protocol motiontrans` | Saved current-frame checkpoint; held-out episodes and transfer config | Frozen feature extraction and fitted readout heads; writes evaluation artifacts |
| `evaluate … --protocol dataset-probe` | Trained source IDs, frozen features, separate episodes | Fits a source-identity classifier; excludes the holdout |
| `data discover / verify` | Dataset configuration and recorded source artifacts | Discovery/checksum inspection; no model training |
| `data import / cache --output PATH` | Verified prior artifacts and current consumer code | New import sidecars; original arrays/receipts retained |
| `data prepare / download --source ID` | Registered source provider and explicit inputs | Source-specific preparation/acquisition; inspect provider requirements |
| `view --host 127.0.0.1 --port 8080` | Configured sources, model artifacts and histories | HTTP serving; selected inference may use GPU |
| `status [--run RUN]` | Run `status.json` and latest complete metrics row | Read only; supports `--watch --interval 15` and finite `--samples N` |
| `tune --plan PLAN --dry-run` | Explicit candidates, seeds and finite budget | Read-only resolution; omit `--dry-run` to train exactly the declared trials |

Run operational tasks from a private `ai_work/scratch/<date>-<task>/` directory,
passing `tl --project-root /absolute/project …`. Experiment component paths are
relative to the experiment file; candidate and artifact paths resolve against the
project. Tool outputs belong in run/processing directories, never root log aliases.

A tuning plan must provide every selection explicitly:

```json
{
  "candidates": ["configs/experiment.json"],
  "seeds": [45],
  "budget": {"epochs_per_trial": 3, "max_trials": 1},
  "output": "processing/tuning/three_epoch_trial"
}
```

The candidate/seed product must fit `max_trials`. Each trial receives an immutable
expanded config with its seed, epoch budget and separate output. There are no
implicit seed repetitions, winner extensions or MotionTrans-based training
selection. An existing plan requires explicit `--resume`; changed plans or
candidate configurations require another output directory.
