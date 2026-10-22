# Research-core migration and recovery

**Deployed 2026-10-22 01:03 EDT.** Training was stopped immediately at the user's
request, preserving completed epoch 4 and discarding partial epoch 5. Both viewer
services were restored; current/historical HTTP health, 99 maintained tests and
all existing cache imports passed. No training was restarted.

The first attempt rolled back on an inherited policy path; the corrected resolver
passed an isolated archive-only simulation before successful reapplication.
The following records preserve the staged migration and its recovery gates.

Prepared October 22, 2026. This second refactor replaces the earlier alias-based
organization with physical maintained implementations. The first migration's
documents are preserved in `legacy/v1/docs/`.

## Prepared locations

- Source tree: `ai_work/refactor/20261022_000723/project/`.
- Untouched baseline: `ai_work/refactor/20261022_000723/baseline/`.
- Protected backup: `ai_work/backups/20261022_000723/`.
- Commands/audits: `ai_work/scratch/20261022-core-refactor-000723/`.
- Sealed migration, journal and live migration status:
  `ai_work/refactor/20261022_000723/`.

The source archive, file metadata and SHA-256 checksums protect the exact prior
state. The changeset lists every old/new file, mode and symlink target. Historical
source/configurations/scripts/tests are preserved under `legacy/v1/`.
Datasets, processing results, model assets and the environment are not relocated.

## Authorized immediate stop and cutover

The original plan waited for normal completion of all 30 epochs. On October 22,
the user first authorized stopping after the next saved epoch, then changed that
instruction to **“don't wait for epoch to finish just stop the training”**. This
latest instruction authorizes an immediate stop and deployment. At authorization,
four epochs were committed and epoch 5 was in progress. The preserved boundary is
therefore **4 completed epochs out of 30 configured**; unfinished epoch-5 work is
discarded. This is an intentional early stop, not a completed 30-epoch run.

Before replacing source dependencies, the stop gate verifies the latest committed
epoch-4 metrics/checkpoint (`last.pt` stores zero-based epoch 3) and confirms that
the training worker is stopped. A signal alone is not checkpoint evidence. An
incomplete save, conflicting process, or changed sealed file blocks migration.
The migration records the actual stopped-process and saved-checkpoint evidence.

Original run configurations, metrics, checkpoints, recorded hashes and scientific
receipts remain unchanged. The intentional stop is recorded separately; it does
not rewrite the historical run as having completed its configured budget. The
new schema-2 trainer cannot resume this historical checkpoint. The archived
checkpoint remains readable for historical inference and evaluation, and no
replacement training is started automatically.

After the verified stop, the coordinator stops only the viewer/monitor that were
active, rechecks other project workers and all sealed hashes, applies the file
changes with a recovery journal, installs the local package without network or
dependency changes, and switches viewer/status service entry points to maintained
tools. It runs CPU contract tests, imports small consumer cache receipts and
checks current/historical HTTP routes before restarting affected services.
Historical launcher service definitions are retained; autostart is disabled after
successful cutover so the stopped run does not restart against relocated code.
No new training, tuning, dataset download or tensor recaching is launched.

Visible root logs and the root downloader have already been relocated without
changing the live model/training dependencies. That limited cleanup is separate
from the pending source/configuration cutover. The status and journal determine
whether the remaining physical migration has actually been applied.

Read `cutover_status.json` and `deployment_journal.jsonl` under the refactor
work directory for the actual state; a staged tree alone is not proof of deployment.
The coordinator's stdout/stderr stays in the task workspace. Changed or uncertain
files are never overwritten to force deployment.

## Recovery

A validation failure stops the new viewer/monitor, reverses only sealed changed
paths from the protected archive, reinstalls the old local package and restores
the original affected service definitions. A partial application is recoverable;
later unexpected edits block automatic restoration instead of being overwritten.
Dataset/cache arrays and scientific receipts are outside rollback scope.

For manual inspection, `deploy_core.py --check` is read-only. `--apply` requires the authorized
immediate-stop and committed-checkpoint evidence plus stopped readers; `--rollback` requires
no active source-dependent workers and refuses divergent files. The helper lives
beside the sealed changeset. Never extract a full backup over a running project.
Keep backups, archived records and uncertain residual files until explicitly
reviewed. Do not run the first refactor's freeze/thaw instructions for this migration.
