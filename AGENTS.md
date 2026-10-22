# Working in TactileLatent

Read [README.md](README.md), [docs/README.md](docs/README.md), the selected
[experiment](configs/experiment.json), and the dated
[project state](docs/project_state.md) before choosing work.

Follow [docs/code_standards.md](docs/code_standards.md) for maintained code.
Every Python module needs a purpose docstring; executable modules and command
handlers also document Inputs, Outputs and an Example command. Use descriptive
names and folders, portable configured paths and one statement per line. Update
the root README command index whenever a maintained executable/handler is added,
renamed or removed. Run the documented Ruff, standards and asset checks; preserve
scientific behavior and version new consumer-import metadata when hashes change.

- Human-editable research implementation belongs in `src/tactilelatent/`.
  Operational commands belong in `tools/tactilelatent_tools/`; use the declared
  `tl` menu for training, evaluation, tuning, dataset management, viewing and status.
  Do not revive a historical launcher merely because its filename looks relevant.
- Model architecture is separate from dataset management. All selected sources
  enter through the registry and uniform sample contract. Add source-specific
  decoding, chronology, calibration and mirroring to the dataset package; do not
  add dataset-name branches throughout model/training code.
- Source selection and holdouts are explicit in `configs/datasets.json`; model
  choices are in `model.json`, budgets in `training.json`, transfer protocols in
  `transfer.json`. Update these canonical files instead of adding another root plan.
- For operational work, create a private `ai_work/scratch/<date>-<task>/`
  workspace. Run tools from there with `tl --project-root /absolute/project …`.
  Put temporary scripts, drafts, diagnostics and command captures in that task
  directory. Durable implementation may be edited normally in `src/` or `tools/`;
  scratch is not a permissions restriction.
- Maintain durable explanations in `docs/` and preserve historical findings in
  its archive. Promote reusable scratch work before maintained code depends on it.
  Maintained code must never import or execute disposable scratch files.
- `legacy/v1/` preserves historical code/configurations; it is not a supported
  training environment. Do not add compatibility-wrapper forests or restore old
  root commands. Historical checkpoint readers are explicit visualization adapters.
- Preserve `datasets/`, `processing/`, `mano_models/`, saved checkpoints, metrics,
  caches, native targets, provenance, splits, holdouts and seeds. Never rewrite a
  recorded source hash to make incompatible artifacts pass. Do not regenerate
  expensive artifacts under a cleanup label.
- Treat changes to architecture, state-dict keys, losses, normalization, camera
  roles, chronology, mirroring and evaluation as scientific changes. Verify native
  shapes and missing-versus-zero handling through dataset contract tests.
- Inspect running jobs/services before editing dependencies. Active jobs using
  historical paths must finish or be explicitly stopped with user authorization
  before those dependencies are physically relocated.
  Do not assume SIGTERM saves training; obtain authorization before interruption
  and explain unsaved progress. Use the documented migration/recovery procedure.
- Keep runtime logs under run/processing directories or `logs/`; no new root logs,
  log symlinks or temporary reports. Protected `ai_work/backups/` and uncertain
  research records must not be automatically deleted.
- Validate source changes with isolated tests and an explicit `PYTHONPATH`.
  Tool help must stay lazy and side-effect-free. Do not start training, downloads,
  compression or deletion just to validate cleanup.
- `.gitignore` only controls accidental tracking; it does not enforce these rules.
  Do not initialize Git, commit or publish without a user request.
