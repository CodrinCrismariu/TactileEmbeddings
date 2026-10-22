# Maintained command catalog

The supported command surface is `tl`; operations are reusable Python calls,
with thin argument handling in `tools/tactilelatent_tools/`. Historical fixed-run
launchers and their previous catalog are preserved in `legacy/v1/`.

| Command | Implementation | Effect |
|---|---|---|
| `tl train --config …` | `training.engine.train_experiment` | New/resumed joint optimization; explicit disposable preflight option |
| `tl train --import-checkpoint … --import-only` | `training.legacy_import` through the current engine | Verified legacy continuation state in a new run; original checkpoint preserved |
| `tl evaluate …` | `evaluation.transfer.evaluate_experiment` | Frozen-feature extraction and fitted transfer/classifier readouts |
| `tl tune --plan …` | `tactilelatent_tools.commands.tune` | Explicit bounded candidates/seeds; dry-run has no writes |
| `tl data …` | `dataset_management.api.manage` | Discover/verify, prepare/import receipts, or explicit acquisition |
| `tl view …` | `visualization.server.serve` | HTTP viewer with native renderers, attention and training history |
| `tl status …` | `tactilelatent_tools.commands.status` | Read-only run progress and last completed metrics |
| `python -m tactilelatent_tools.checks.code_standards --project-root .` | `checks.code_standards` | Read-only documentation, path, dependency and README checks |
| `python -m tactilelatent_tools.checks.format_assets --project-root .` | `checks.format_assets` | Read-only HTML/JS/CSS check; `--write` explicitly formats maintained assets |

Help imports no model, dataset, network or GPU library. Source-specific I/O and
acquisition conventions belong in dataset packages, not operational launchers.
Temporary investigations belong in a dated task workspace. Promote reusable
operations into these modules instead of growing a flat scripts directory.

The [root README command index](../README.md#maintained-command-index) links every
maintained executable and handler. Keep it synchronized when files change and
follow the [code standards](code_standards.md). Ruff handles Python formatting
and linting with pinned development versions in `pyproject.toml`.
