# Maintained code standards

These rules apply to `src/`, `tools/`, `tests/`, `examples/` and the five current
configuration files. They describe how to maintain the research implementation;
changes to the scientific experiment still require an explicit experiment.
Historical files in `legacy/`, recorded artifacts in `processing/`, downloaded
datasets, model assets and private agent work are outside the formatting scope.
Never bulk-format those records or rewrite their recorded hashes.

## Put each responsibility in its own place

| Responsibility | Location | Boundary |
|---|---|---|
| Tensor contracts and configuration reading | `src/tactilelatent/contracts.py`, `configuration.py` | Shared definitions; no operational launchers |
| Backbone, attention, latent, adapters and teachers | `src/tactilelatent/models/` | No source-name branches or dataset/training/tool imports |
| Source chronology, camera roles and target conventions | `src/tactilelatent/datasets/sources/` | A source adapter implements the common sample contract |
| Target decoding, normalization and sensor representations | `src/tactilelatent/datasets/codecs/` | Preserve native shape, units, signs and missing observations |
| Registry, preparation, caches, sampling and mirroring policy | `src/tactilelatent/dataset_management/` | Verify provenance; compose adapters explicitly |
| Training, objectives, SAM and checkpoints | `src/tactilelatent/training/` | Consume the registry; do not implement source readers |
| Reusable transfer and probe evaluation | `src/tactilelatent/evaluation/` | Keep source training and held-out evaluation separate |
| Viewer, rendering and attention displays | `src/tactilelatent/visualization/` | Historical inference uses an explicit compatibility adapter |
| Maintained commands and code-quality checks | `tools/tactilelatent_tools/` | Parse arguments and call reusable APIs; help has no side effects |
| Human-editable experiment choices | `configs/` | Model, datasets, training and transfer, selected by `experiment.json` |
| Temporary scripts, investigations and exploratory outputs | `ai_work/scratch/<date>-<task>/` | Never imported or executed by maintained code |

Do not add a new root script, a flat compatibility-module collection, or another
parallel configuration framework. Promote useful temporary automation into
`tools/` and reusable logic into the appropriate core package. Choose a descriptive
folder when a responsibility grows; avoid vague collections such as `misc/`.

## Write for the person reading the next change

- Use descriptive `snake_case.py` filenames and function/variable names, and
  `PascalCase` class names. Names such as `test2_final_v3.py`, `tmp.py` or `new.py`
  do not identify a maintained responsibility. Version recorded artifact schemas
  explicitly; do not encode a history of edits in implementation filenames.
- Start every Python module with a purpose docstring. Executable scripts,
  command handlers and checks must also have **Inputs**, **Outputs** and
  **Example** sections. Describe writes, fitted models and long-running services
  honestly. Use a real command, project-relative paths and placeholder arguments
  for user-supplied files. Package markers and test modules need only their purpose.
  Maintained JavaScript starts with an equivalent JSDoc header. Its example is the
  viewer command that serves the script rather than a direct browser-script run.
- Public research APIs explain tensor axes, expected units, mask meaning and
  returned values where these are not obvious. Distinguish an absent reading from
  a measured zero, the history from the supervised current frame, and standardized
  codes from native sensor measurements. State checkpoint or RNG constraints near
  the implementation they protect.
- Use one statement per line, conventional spacing, short logical blocks and
  early validation. Avoid semicolon-packed operations and one-line control flow.
  Keep helpers focused on one task. Comments explain a convention, risk or reason,
  rather than translating the next line into English.
- Prefer descriptive names over abbreviations. Mathematical names such as
  `q`, `k`, `v`, `mu` and `logvar` are appropriate when the formula and shapes are
  documented. Do not rename state-dict keys, registered modules or public artifact
  fields merely to satisfy a naming preference.
- Keep imports explicit. Optional heavy imports belong inside the operation that
  needs them so CLI discovery remains cheap. No training, downloading, dataset
  traversal or file writes at module import. Avoid wildcard imports and broad
  exception handlers that hide failed data or provenance checks.
- Add type annotations to new or substantially revised public interfaces when
  they clarify the contract; do not replace runtime tensor validation with hints.
  Prefer an existing contract/dataclass to another loosely specified dictionary.

For example, an operational module starts like this:

```python
"""Verify one registered source without starting training.

Inputs:
    Experiment configuration and source artifacts under --project-root.
Outputs:
    A verification report on stdout; existing artifacts are read only.
Example:
    tl data verify --source egopressure
"""
```

## Paths, configuration and artifacts

Do not hard-code a developer's home directory, mount point or local machine path.
Accept project roots and artifact locations through arguments/configuration and
resolve paths with `pathlib.Path`. Component references are relative to the
experiment file; data/run references are relative to the explicit project root.
HTTP routes and intentional operating-system interfaces such as `/proc` are not
machine-specific artifact paths. Tests may construct temporary absolute paths at
runtime; do not copy a real workstation path into a fixture.

Keep dataset identity routing in the registry/adapters. Models consume tensors,
camera roles and masks; a source identity selects a head afterward. Adding a source
using an existing target representation must not require a model or trainer edit.
No maintained component may import or run private `ai_work/` programs.

Preserve native target dimensions, synchronization, camera roles, mirroring,
source order, seeds, RNG consumption, state-dict keys and objectives during style
maintenance. Cache receipts pin exact implementation bytes: a formatting change
can require **new versioned consumer-import metadata**. Preserve original receipts,
producer bytes and arrays; never substitute a digest to force acceptance. See
[compatibility](compatibility.md) before changing a pinned consumer.

## Formatting and automated checks

Install `.[dev]` in a development environment. Tool versions and lint settings are
pinned in `pyproject.toml`; `.editorconfig` defines UTF-8, LF and indentation.
Python uses Ruff, 100-column wrapping and four spaces. JavaScript/CSS use the
pinned beautifiers and HTML uses djLint, all with two spaces. The asset checker
covers all three formats, including embedded CSS. Long URLs and strings may exceed the Python
wrapping target when splitting them would change their value.

Run these from the project root; the checks do not launch scientific work:

```bash
ruff check --no-cache .
ruff format --check .
python -m tactilelatent_tools.checks.code_standards --project-root .
python -m tactilelatent_tools.checks.format_assets --project-root .
PYTHONPATH=src:tools python -m pytest -q
```

For an uninstalled checkout, set `PYTHONPATH=src:tools` for the two module commands
as well. From a private task directory, pass absolute project paths and an explicit
`PYTHONPATH`; place test/cache/log outputs inside that task, not the repository root.

`ruff format .` and `python -m tactilelatent_tools.checks.format_assets
--project-root . --write` apply formatting. Inspect any lint fix before accepting it:
import order or an apparently unused value may protect initialization behavior.
The standards checker verifies module/script documentation, descriptive filenames,
machine-specific paths, statement layout, import boundaries and the README command
index, including the maintained browser scripts. Its checks are safeguards, not proof of readable design; review API clarity,
ownership and scientific equivalence too.

When adding, renaming or removing **any maintained executable or command handler**,
update the [root README command index](../README.md#maintained-command-index) in
the same change. Keep its direct source links valid and remove stale links. Update
the relevant package README and [script catalog](script_catalog.md) if behavior or
usage changes. The checker discovers entrypoints from Python main guards, the CLI
dispatch map and packaging metadata, rather than a fixed list of filenames.

Validate at the level of the change. For a broad formatting pass, run the complete
maintained suite including exact initialization, reconstruction and SAM parity.
Do not run training searches, downloads or expensive cache regeneration as style
validation. Never automatically delete uncertain files, old research or backups.
