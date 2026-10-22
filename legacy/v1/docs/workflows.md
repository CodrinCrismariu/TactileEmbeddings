# Setup and supported workflows

Run commands from the project root. The [script catalog](script_catalog.md) records every retained command, including archived/uncertain experiments and significant side effects. Never use an unfamiliar script's `--help` until its entry point is inspected: some historical scripts execute work at import time.

## Environment

The audited machine already has `.venv/`, datasets, frozen teachers and pretrained V-JEPA artifacts. Basic package setup on a fresh machine is:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test,video,acquisition]'
```

This uses the dependency ranges in `pyproject.toml`; the existing `requirements.lock.txt` records the local dependency snapshot. CUDA/driver compatibility is machine-specific. `expansion_runtime.configure()` in supported GPU launchers selects the existing kernel-matched NVIDIA runtime when present. Do not replace global driver libraries during a documentation/refactor task.

Installation does not download datasets, V-JEPA weights, official implementation, frozen sensor teachers or MANO assets. The current run pins its V-JEPA revision and checkpoint hash; look in `processing/vjepa2_1/` and `processing/upstream/vjepa2/`. The MANO OBJ assets stay in `mano_models/`, with their recorded research license and provenance.

Tests can run independently of GPU training, though some integration tests require local artifacts:

```bash
.venv/bin/python -m pytest -q
```

Separate pre-existing environment/artifact failures from changes caused by an edit. Use the refactor's staged verification instructions in [compatibility](compatibility.md) when testing an isolated source tree.

## Viewer and training monitor

First check whether the existing viewer service is already serving port 8080. On the audited Linux machine:

```bash
systemctl --user status tactilelatent-viewer.service --no-pager
```

To start a separate viewer when that port is free:

```bash
.venv/bin/python scripts/view_egotouch_autoencoder.py --host 127.0.0.1 --port 8080
```

The script installs acquisition, mirror-teacher and dashboard extensions around the existing HTTP viewer. Use http://127.0.0.1:8080/ locally or forward port 8080 through VS Code/SSH. It exposes:

- **Sensor reconstruction**: native measured targets and available frozen AE reconstructions.
- **Ego → tactile prediction**: trained visual model/epoch selection and A/B comparison.
- **Mirror quality check**: recorded versus consistently reflected targets/teachers, with shared scales.
- **Training monitor** (`?page=training`): saved epoch histories and sampled live batch losses.

Joint prediction can load GPU models; viewing curves alone loads no backbone. The dashboard's five-second refresh does not create fresh validation results. Read [dashboard semantics](TRAINING_DASHBOARD.md), [new source display conventions](ACQUIRED_DATASETS_VIEWER.md), [mirroring](BIMANUAL_MIRRORING.md) and [EgoPressure 3D UV](EGOPRESSURE_VISUALIZATION.md).

## Full joint training and resume

**These commands perform GPU work and write checkpoints. Do not run them alongside the existing job.** The verified current configuration is linked in [project state](project_state.md); historical configs remain reproducible but are not default recommendations.

```bash
# Audits data/teacher/backbone inputs and performs disposable training preflight.
.venv/bin/python scripts/train_all_sources_vjepa.py \
  --config configs/joint_vjepa21_all_sources_32d_mirror_v2.json --preflight-only

# Production launcher: preflight, then fresh start or resume if last.pt exists.
.venv/bin/python -u scripts/run_all_sources_joint.py \
  --config configs/joint_vjepa21_all_sources_32d_mirror_v2.json

# Explicit resume only after the previous worker has stopped.
.venv/bin/python -u scripts/train_all_sources_vjepa.py \
  --config configs/joint_vjepa21_all_sources_32d_mirror_v2.json --resume
```

Inputs include source manifests, compressed scene RGB, native tactile/marker targets, accepted frozen teachers, source-code caches, task-only EgoPressure selection and mirror policy/paired-code receipts. Preflight is not read-only: it may write audit/preparation output and execute test optimizer updates, which are discarded rather than promoted as a trained checkpoint. Never regenerate expensive caches simply to make a changed provenance hash pass.

Outputs are a resumable `last.pt`, source-selected `best.pt`, inference snapshots under `viewer_epochs/`, completed records in `metrics.jsonl`, and a transient `status.json`. The existing persistent user service runs the launcher; service definitions are local machine configuration, not portable package entry points.

## Sensor teachers, caching and acquisition

The current mirror-adapted teachers and paired target codes are complete. The [`mirror guide`](BIMANUAL_MIRRORING.md) identifies their exact policies/receipts and reconstruction tests. Preparing them again performs AE training/code generation; it is not required to open the viewer.

For additional datasets, `scripts/run_dataset_queue.py` selects `configs/data_queue_active.json`, not the historical frozen `data_queue_v2.json`. Its `run`/`source` modes download and process recordings, and even its `status` mode refreshes status documents. Consult the [catalog](script_catalog.md), [queue guide](DATA_DOWNLOAD_AND_PROCESSING_QUEUE.md), manifests, free-space reserve and licensing before acquisition. Compression/deletion commands require their explicit verification and authorization; reorganizing documentation does not authorize storage cleanup.

## Transfer and uncertainty evaluation

Frozen-encoder MotionTrans evaluation fits a separate target head; it is training a readout even though it does not update the visual encoder. Several historical scripts have fixed source run paths and distinct all-frame/current-frame contracts. Use the [catalog](script_catalog.md) and recorded evaluation configuration rather than substituting the active run into an older script without a compatibility check.

[`evaluate_vjepa_current_motiontrans.py`](../scripts/evaluate_vjepa_current_motiontrans.py) records the previous 32D/128D current-frame protocol, with three-hidden-layer 512-unit readouts, episode fit/tune partition, fit-only feature scaling and tune-selected learning rates. It is a historical comparison command with fixed model inputs, **not a verified general evaluator for the new 17-source run**. No new evaluation is launched by this reorganization.

The [research overview](research_overview.md) links completed protocols and results. Historical Gaussian coverage is substantially below nominal. Stochastic mixing, attention maps and reconstruction scores are not calibrated predictive uncertainty. The active deterministic model has no variance output.

## Active work and checkpoint safety

The audited trainer saves model/optimizer/SAM/scheduler/RNG at completed-epoch boundaries. It has no documented signal-triggered checkpoint-and-stop hook. SIGTERM, service stop or Ctrl+C must not be presented as a guaranteed save; partial-epoch updates can be lost.

For the October 21 refactor, the user authorized a machine-verified kernel cgroup freeze/thaw that preserves the same training processes and partial-epoch RAM state. It does not create a disk checkpoint; a process/machine failure could still lose progress since the last committed epoch. The exact approved checks, deployment and rollback are in [migration and recovery](refactor_migration.md).

Before applying future changes that affect a running trainer or viewer, prepare/test the concrete source changes in isolation and obtain any required approval with the exact pause/restart/rollback steps. Prefer an observed completed-epoch commit; do not promise a lossless pause without a supported mechanism. Inference-only viewer snapshots are not substitutes for a resumable checkpoint. Keep the original config and provenance consistent on resume.

Historical flat modules are sometimes hash-pinned; [compatibility](compatibility.md) documents why they remain and how canonical namespaces preserve old imports. Temporary work goes to `ai_work/scratch/<date>-<task>/`; backups under `ai_work/backups/` are protected recovery material, not scratch.
