# Organization migration and recovery

Prepared October 21, 2026, in an isolated source copy. The user subsequently authorized pausing training for this refactor. The approved pause uses the **kernel cgroup freezer**, verified on this machine before deployment. It suspends the same launcher/trainer/workers in memory; it does not terminate or restart training and does not trigger a new disk checkpoint.

## Review locations

All paths below are relative to the repository root:

- Staged project: `ai_work/refactor/20261021_231339/project/`.
- Untouched source baseline: `ai_work/refactor/20261021_231339/baseline/`.
- Checks and inventories: `test_outputs/`, `package_audit/`, `scripts_audit/` and `work/research_audit/` inside that refactor directory.
- Protected backup: `ai_work/backups/20261021_231339/`.
- Deployment changeset and old-to-new map: `changeset.json` and `old_to_new_paths.json` inside the refactor directory; the latter is also recorded with the backup.

The [README](../README.md), [architecture](architecture.md), [script catalog](script_catalog.md), and [compatibility guide](compatibility.md) are the reviewable result. [Verification](refactor_verification.md) records baseline/staged results, unavailable resources and the actual deployment observations.

## Changes and protected paths

Historical root reports move into `docs/archive/root/`; the original proposal moves into `docs/proposal/main.pdf`. Two obsolete guides are preserved under `docs/archive/superseded_guides/` before their maintained versions are updated. Five source namespaces provide canonical navigation. The loss implementation and dashboard implementation/assets physically move; old module names remain compatible. Three historical Python commands and the root downloader retain forwarding entry points to archived bodies.

The changeset is restricted to maintained source, package assets, documentation, tests, script entry points, `.gitignore` and working rules. It does not move or rewrite `datasets/`, `processing/`, `mano_models/`, `.venv/`, Git history, saved configurations, caches, checkpoints, metrics or provenance. All 81 source modules identified in saved hash receipts remain byte-identical.

The root downloader wrapper, `download_logs/`, `STORAGE_MIGRATION.log` and six historical log shortcuts remain explicit compatibility exceptions. Old commands still use these paths. The former root `0` download-status fragment is preserved in the [operations archive](archive/operations/README.md) and original backup. Uncertain records are retained, not automatically deleted.

## What the authorized pause preserves

The current trainer writes complete model/optimizer/SAM/scheduler/RNG checkpoints at **completed-epoch boundaries**. It has no signal-triggered checkpoint-and-stop hook. `systemctl stop`, SIGTERM or Ctrl+C would not guarantee a save; they are not used for the training pause.

The freezer suspends every process in `tactilelatent-joint-all-sources-v2.service` together, including its workers. Partial-epoch weights, gradients, sampler/RNG state and process memory remain resident. Normal freeze/thaw resumes that same process rather than reloading `last.pt`, so partial-epoch progress is preserved. **No fresh disk checkpoint is promised.** A process or machine failure during the pause could still lose work since the last completed checkpoint; the recorded `last.pt` is the recovery fallback.

Before freezing, the machine reported `CanFreeze=yes`. A transient dummy service was frozen/thawed successfully: its heartbeat stopped while frozen and advanced after thaw. Evidence is `test_outputs/freeze_mechanism.json` under the refactor directory. Deployment requires both systemd `FreezerState=frozen` and kernel `cgroup.events` reporting `frozen 1`; a stopped-looking PID alone is insufficient.

## Approved deployment procedure

The authorization to pause has already been given for this migration. Do not ask again solely because a service interruption is involved. Execute the sealed changeset and preserve the recorded process identity.

1. Record the training/viewer/monitor service states, launcher/trainer/worker PIDs and cgroup membership, current progress, latest completed metrics and checkpoint hash/epoch in the refactor audit output. At planning time epoch 3 was the completed-checkpoint fallback and epoch 4 was in progress; **re-read actual state before pausing**. This recording is read-only with respect to the research artifacts.

2. Freeze the training service, without stopping it:

   ```bash
   systemctl --user freeze tactilelatent-joint-all-sources-v2.service
   systemctl --user show tactilelatent-joint-all-sources-v2.service \
     -p ActiveState -p FreezerState -p CanFreeze -p MainPID -p ControlGroup
   training_cgroup="$(systemctl --user show tactilelatent-joint-all-sources-v2.service --property=ControlGroup --value)"
   cat "/sys/fs/cgroup${training_cgroup}/cgroup.events"
   ```

   Require the original MainPID and worker identities, systemd frozen state and kernel `frozen 1`. The service remains active while frozen. If the freezer is unavailable or verification fails, do not substitute a killing signal or proceed with deployment. Thaw any partially paused service and reassess.

3. Stop only the viewer and expansion monitor **if they were active in step 1**:

   ```bash
   systemctl --user stop tactilelatent-viewer.service
   systemctl --user stop tactilelatent-expansion-monitor.service
   ```

   This briefly interrupts HTTP viewing/progress monitoring. Do not stop the frozen training service. The acquisition queue and inactive historical jobs are not restarted. A new project Python/download/data-modification job blocks the deployment gate until reassessed.

4. Check and apply the sealed source change from outside the frozen training cgroup:

   ```bash
   .venv/bin/python ai_work/refactor/20261021_231339/deploy_refactor.py \
     --check --allow-frozen-training
   .venv/bin/python ai_work/refactor/20261021_231339/deploy_refactor.py \
     --apply --allow-frozen-training
   ```

   Check mode is read-only and controls no service. The helper checks the backup checksum, sealed staged/original file hashes, permissions/symlinks, protected paths and absence of other active project workers. The frozen path additionally verifies `CanFreeze=yes`, `FreezerState=frozen`, kernel frozen state, current MainPID and trainer/worker cgroup membership, plus inactive viewer/monitor. Compare those reported PIDs with the original pause receipt. Require the printed `deployment_ready` field to be `true`; check mode may report a blocked operational gate without a nonzero exit code. Apply independently rechecks the gate and blocks mutation if it fails. Apply writes individual files atomically and journals progress. Never replace recorded provenance hashes or regenerate caches to make this migration pass.

5. Run focused CPU checks and verify the checkpoint hash, pinned source bytes and training process identity remain unchanged:

   ```bash
   CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 \
     OPENBLAS_NUM_THREADS=1 PYTHONPATH="$PWD/src" .venv/bin/python -m pytest -q \
     tests/test_package_organization.py tests/test_script_archive_compatibility.py \
     tests/test_training_dashboard.py
   ```

   The staged tests and actual checkpoint equivalence checks have already run separately; these post-apply checks verify the installed paths. Do not run full training, GPU preflight, acquisition or code-cache generation as deployment validation.

6. Restart **only** the viewer/monitor services recorded active in step 1:

   ```bash
   systemctl --user start tactilelatent-expansion-monitor.service
   systemctl --user start tactilelatent-viewer.service
   ```

   Verify the root viewer, training monitor, reconstruction catalog and monitor heartbeat through read-only requests. No new joint-training process is launched.

7. Thaw the original training cgroup after successful checks:

   ```bash
   systemctl --user thaw tactilelatent-joint-all-sources-v2.service
   systemctl --user show tactilelatent-joint-all-sources-v2.service \
     -p ActiveState -p FreezerState -p MainPID
   ```

   Verify the **same** launcher/trainer PIDs, kernel `frozen 0`, continuing batch/epoch progress and absence of a duplicate worker. No training service restart or checkpoint reload occurs. Record pause duration and final service states. Do not leave training frozen because the maintenance shell is closing.

## Failure recovery and rollback

If apply or focused checks fail, keep training frozen and viewer/monitor stopped while restoring source paths:

```bash
.venv/bin/python ai_work/refactor/20261021_231339/deploy_refactor.py \
  --rollback --allow-frozen-training
```

The helper restores only changeset paths from the protected archive and removes only newly added files still matching staged hashes. Unexpected later edits block automatic rollback. It does not restore datasets, outputs or service definitions. Verify original files/paths before restarting only the viewer/monitor services paused for this task, then **thaw the same training service/PIDs**. Do not kill training or leave a half-applied source tree beneath active workers.

If the training process itself fails, freeze/thaw can no longer recover its RAM state. Preserve the failure evidence and resume separately from the last verified complete checkpoint using the same configuration and strict provenance, acknowledging any partial-epoch progress lost. Do not describe that recovery as lossless.

For manual recovery, verify `SHA256SUMS`, list the archive members, then extract into a **new empty directory**. The backup manifest records original paths, permissions, symlink targets and content checksums. Compare affected files against `changeset.json` before restoring individually. Never extract the whole archive over a running project. Backups and the relocation map remain protected until explicitly reviewed for removal.

## Alternative: deployment after normal completion

Without `--allow-frozen-training`, the helper retains its normal-completion gate: training `status.json` must say complete, 30 epochs must be committed, `last.pt` must contain stored epoch 29, the final metrics row must record epoch 30, and the service must be inactive with `ExecMainStatus=0`. Then pause/restart only the viewer/monitor around `--check`, `--apply` and focused tests. No training pause/thaw is needed and no new run is launched. This is an alternative to the already authorized freeze migration, not a claim that training has finished.
