# TactileLatent

Learn a shared representation of touch from video across heterogeneous sensors.
The maintained model uses frozen V-JEPA video tokens, trainable spatial/temporal
attention, a 32D current-frame latent, and sensor-specific prediction heads.
MotionTrans is held out for transfer evaluation. Reconstruction quality alone
establishes neither sensor invariance nor calibrated uncertainty.

Start with [the documentation guide](docs/README.md) and
[the architecture](docs/architecture.md). The
[original proposal](docs/proposal/main.pdf) and
[research history](docs/research_overview.md) preserve the distinction between
research goals, implemented behavior and experimental findings.

Read [the code standards](docs/code_standards.md) before editing maintained code.
They define documentation, naming, formatting, portable paths, package boundaries
and the checks required for a change.

## What to edit

| Change | Maintained location |
|---|---|
| Backbone, attention, latent and adapters | [src/tactilelatent/models/](src/tactilelatent/models/) |
| Model dimensions and attention settings | [configs/model.json](configs/model.json) |
| Selected sources, holdout and mirror policy | [configs/datasets.json](configs/datasets.json) |
| A source's loading, chronology and native targets | [src/tactilelatent/datasets/](src/tactilelatent/datasets/) |
| Downloads, artifact verification and preparation | [src/tactilelatent/dataset_management/](src/tactilelatent/dataset_management/) |
| Training budget and optimizer | [configs/training.json](configs/training.json) |
| Frozen transfer-head protocol | [configs/transfer.json](configs/transfer.json) |
| Operational commands for agents and people | [tools/tactilelatent_tools/](tools/tactilelatent_tools/) |

[configs/experiment.json](configs/experiment.json) explicitly selects the four
configuration files. Component paths are relative to that experiment file;
data/model/run paths inside components are relative to `--project-root`.
Each source enters training through the same registry/sample contract. Native
tactile resolution, signed values, masks and episode chronology stay source-specific.

## Commands

After installing the project into the environment (`python -m pip install -e
'.[test,video,acquisition]'`), use `tl`. With the existing environment and an
uninstalled source checkout, run from the project root:

```bash
PYTHONPATH=src:tools .venv/bin/python -m tactilelatent_tools --help
```

The declared menu is deliberately small:

```bash
tl --help
tl data discover
tl data verify --source egopressure
tl train --preflight-only
tl train
tl train --resume processing/runs/vjepa32_current_all_sources_core_v2
tl evaluate --run processing/runs/vjepa32_current_all_sources_core_v2 --epochs 10 --protocol motiontrans
tl status
tl view --host 127.0.0.1 --port 8080
```

Help performs no GPU, network or storage operations. Dataset preparation,
training, fitted evaluation heads and downloads have explicit effects. `status`
only reads run files. `tune --plan PLAN --dry-run` checks an explicit candidate,
seed and epoch budget without starting trials; see [workflows](docs/workflows.md).

Agents should run operations from a private `ai_work/scratch/<date>-<task>/`
workspace and pass `tl --project-root /absolute/path/to/TactileLatent …`.
Reusable code and canonical configurations are edited in their maintained
locations; scratch is not a restriction on normal source changes.

## Current core and preserved research

New training uses the maintained core and creates a separate run. Historical
commands and configurations are preserved under [legacy/v1/](legacy/v1/), with
research results and saved model artifacts remaining in their original data
locations. Historical training launchers are archived records, not supported
entry points. Viewer adapters preserve access to earlier model formats.

`datasets/`, `processing/`, `mano_models/` and `.venv/` stay in place. Runtime logs
belong in run/processing directories or `logs/`, never as root shortcuts.
Temporary preparation belongs under [ai_work/](ai_work/README.md); protected
backups must not be cleaned automatically. Read [AGENTS.md](AGENTS.md) before
modifying scientific behavior, data contracts or running jobs.

Operational options and explicit tuning-plan examples are documented in
[tools/README.md](tools/README.md). Import existing tactile-cache receipts once
with `tl data import --output processing/core_v2_cache_imports_standards_v1` before a new
production training run; this records new consumer compatibility without
recomputing the original sensor-code arrays.

To continue a supported historical current-frame V-JEPA run after the refactor,
use `tl train --import-checkpoint PATH_TO_PARENT/last.pt --import-only` once into
the configured empty output directory, then `tl train --resume PATH_TO_NEW_RUN`.
The import verifies architecture, data, teachers and the complete optimizer,
schedule, SAM and RNG state. It preserves the original run and records the parent
checkpoint identity; it does not restart the epoch count or optimizer. See
[checkpoint compatibility](docs/compatibility.md#checkpoints-and-initialization).

## Maintained command index

Update this index whenever a maintained executable or command handler is added,
renamed or removed. Research implementations remain in `src/`; operational
commands remain in `tools/`.

| Command or entrypoint | Implementation |
|---|---|
| `tl` argument parsing and dispatch | [CLI](tools/tactilelatent_tools/cli.py) |
| `python -m tactilelatent_tools` | [Module entrypoint](tools/tactilelatent_tools/__main__.py) |
| `tl train` | [Training command](tools/tactilelatent_tools/commands/train.py) |
| `tl evaluate` | [Evaluation command](tools/tactilelatent_tools/commands/evaluate.py) |
| `tl tune` | [Explicit tuning plans](tools/tactilelatent_tools/commands/tune.py) |
| `tl data` | [Dataset commands](tools/tactilelatent_tools/commands/data.py) |
| `tl view` | [Viewer command](tools/tactilelatent_tools/commands/view.py) |
| `tl status` | [Read-only run status](tools/tactilelatent_tools/commands/status.py) |
| `python -m tactilelatent_tools.checks.code_standards --project-root .` | [Standards check](tools/tactilelatent_tools/checks/code_standards.py) |
| `python -m tactilelatent_tools.checks.format_assets --project-root .` | [HTML/JS/CSS formatting check](tools/tactilelatent_tools/checks/format_assets.py) |
| Internal isolated historical viewer; launched by `tl view` | [Historical worker](src/tactilelatent/visualization/historical_worker.py) |
| Reconstruction, prediction and mirror pages; served by `tl view` | [Viewer JavaScript](src/tactilelatent/visualization/viewer.js) |
| Training curves; served by `tl view` | [Dashboard JavaScript](src/tactilelatent/visualization/training_dashboard.js) |
| `python examples/manual_attention_block.py` | [Educational attention example](examples/manual_attention_block.py) |

Install development tools with `python -m pip install -e '.[dev]'`. Run `ruff check
--no-cache .`, `ruff format --check .` and the two check commands above. Run the
maintained tests with `PYTHONPATH=src:tools python -m pytest -q`. See the
[standards](docs/code_standards.md#formatting-and-automated-checks) for scope,
formatting commands and provenance-sensitive maintenance.
