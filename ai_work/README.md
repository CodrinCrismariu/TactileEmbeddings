# Agent workspaces and protected recovery

Maintained research code is in `src/`, supported operations are in `tools/`,
canonical settings are in `configs/`, and durable explanations are in `docs/`.

Follow the [maintained code standards](../docs/code_standards.md) when promoting
work out of scratch. Give scripts descriptive names, document purpose/inputs/
outputs/example, pass paths explicitly and update the root command index when
adding a maintained entrypoint. Disposable audits do not belong in `src/` or `tools/`.

| Location | Purpose | Retention |
|---|---|---|
| `scratch/<date>-<task>/` | Private operational cwd, temporary diagnostics, drafts and captures | Review and promote durable work before cleanup |
| `refactor/<timestamp>/` | Isolated source preparation, audits, test outputs and migration plans | Keep while reviewing or deploying the migration |
| `backups/<timestamp>/` | Checksummed source archives and recovery manifests | Protected; never automatically delete |

Run `tl --project-root /absolute/path/to/TactileLatent …` from your private task
workspace. Set an explicit `--config` when testing another staged experiment.
Relative artifact paths still refer to the project, not the task working directory.

This convention keeps temporary operational material separate. It does not prevent
normal edits to maintained code. Promote useful implementation before another
maintained module imports it. No maintained package may depend on disposable
scratch or refactor copies.

Keep large datasets, checkpoints and generated research outputs in their existing
locations. Git ignores local task/refactor/backup contents; ignore rules alone do
not enforce behavior. [AGENTS.md](../AGENTS.md) establishes the working rules.
