# Script catalog

Observed **2026-10-21**, America/New_York. This covers all **135 pre-existing command files**:115 Python files and 20 shell files (including the root download worker). Classification is based on source/AST inspection, source/config/test references, service definitions and saved provenance. A historical classification does not establish that a script is obsolete or safe to delete.

Start with the [supported commands](../scripts/README.md). [Archived bodies and compatibility policy](../scripts/archive/README.md) explain the four physical moves. Other historical commands stay at their original paths where implementation hashes, self-hashing, services, tests, sibling imports or root-relative paths make relocation consequential.

No script was imported or executed to generate this catalog. Command templates below are explicitly labeled; checking syntax or finding an old result does not establish that its full dataset/checkpoint dependencies remain available. Paths are repository-relative. Most Python scripts expect the repository root as the working directory.

## Current command entry points

| Workflow | Command or entry point | Evidence and effect |
|---|---|---|
| Viewer, attention and training monitor | `.venv/bin/python scripts/view_egotouch_autoencoder.py --manifest processing/joint_alignment/manifest.json --host 127.0.0.1 --port 8080` | Active service command verified (service binds 0.0.0.0). Use the running viewer rather than starting a duplicate. |
| Integrated17-source joint training | `.venv/bin/python scripts/run_all_sources_joint.py --config configs/joint_vjepa21_all_sources_32d_mirror_v2.json` | Exact active service command inspected. Runs preflight, then starts/resumes GPU optimization. |
| Acquisition status | `.venv/bin/python scripts/run_dataset_queue.py status` | Parser inspected; not invoked in this audit. Even status updates aggregate reports. |
| Tactile-only reconstruction teachers | `.venv/bin/python scripts/train_acquisition_autoencoders.py --config configs/acquisition_autoencoders_v1.json` | Parser inspected; launches/resumes GPU optimization; no RGB backbone. |
| Missing mirrored teachers/codes | `.venv/bin/python scripts/prepare_all_mirrored_teachers.py` | Source inspected; may train decoders and write codes/receipts. Does not start joint training. |

The joint wrapper selects the deterministic 32D current-frame V-JEPA config explicitly. MotionTrans remains held out. Historical DINO, FGW, pooled-V-JEPA and FreeTacMan experiments are retained and cataloged; those launchers are not recommended as current training defaults.

## Execution hazards and compatibility exceptions

Do not run a directory-wide `--help` or import sweep. These seven files execute substantive work at import/top level, regardless of `--help`:

- `after_rgb_migration.py`: waits for conversion, then finalizes and can restart the viewer.
- `diagnose_vjepa_learning.py`: GPU diagnostic fits and report writes.
- `monitor_storage_migration.py`: polling loop and root log writes.
- `plot_vjepa_motiontrans_epochs.py`: immediate results loading and figure writes.
- `verify_cached_vjepa_training.py`: real GPU training updates and reports.
- `verify_vjepa_cache.py`: GPU inference and cache writes.
- `verify_vjepa_pair.py`: real GPU training updates and reports.

A `__main__` guard is not a blanket safety guarantee: several guarded launchers configure runtime/environment or replace loaders during import. Verification scripts can fit models or rewrite metadata. `migrate_jepa_rgb.py` changes source archives by default; `cleanup_storage.py --apply` deletes selected data/checkpoints; `retire_marker_images.py --apply` removes verified tactile image payloads. These are documented operations, not cleanup-validation commands.

The following 11 old paths have matching explicit SHA-256 receipts in saved JSON artifacts and remain unchanged: `build_zero_references.py`, `evaluate_alignment_checkpoint_sweep.py`, `evaluate_continuation_holdout.py`, `evaluate_invariance_holdout.py`, `evaluate_representation_screen.py`, `evaluate_vjepa_current_motiontrans.py`, `evaluate_vjepa_motiontrans.py`, `evaluate_vjepa_motiontrans_epochs.py`, `register_vjepa_transfer_viewer.py`, `run_continuation_study.py`, `start_vjepa_current128d.py`. Additional files hash their own `__file__` at runtime; their entries below identify this too. Never update an old receipt merely to accept moved implementation bytes.

Literal script imports in tests and subprocess entry points are another compatibility boundary. Active service paths `run_all_sources_joint.py`, `train_all_sources_vjepa.py`, `view_egotouch_autoencoder.py`, and `monitor_data_expansion.py` remain byte-for-byte unchanged.

Two historical package commands, `python -m tactilelatent.prepare_pilot` and `python -m tactilelatent.expand_pilot`, already fail during import because they request removed `train.check_storage`. No script directly calls them. This is a pre-existing blocked workflow, preserved rather than repaired under code cleanup.

## Complete inventory

Entries distinguish known effects from uncertain readiness. Imports and literal paths are static evidence, not a promise that dependencies exist or that an old run can resume. The complete machine-readable audit, including hashes, pin receipts, all arguments and service command lines, lives in `ai_work/refactor/20261021_231339/scripts_audit/inventory.json` and is protected refactor evidence rather than a runtime dependency.

### active workflow (observed) (4)

<details>
<summary><strong>scripts/monitor_data_expansion.py</strong> — Continuously publish a readable status page and aggregate JSON snapshot.</summary>

- Status: **active workflow (observed)**; high: service/process plus source inspection.
- Source: [scripts/monitor_data_expansion.py](../scripts/monitor_data_expansion.py).
- Inputs: Existing processing/data_expansion_v1 status/receipts.
- Outputs: Continuously rewritten aggregate status.json and STATUS.md.
- Effects: reads research/native artifacts and writes reports/derived outputs.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Service references: `tactilelatent-expansion-monitor.service`. Presence of a unit does not mean it is running.
- Literal paths/artifact names: `_status.json`, `baihu_deletions.jsonl`, `codes/egopressure_balanced_v2/index.json`, `processing/data_expansion_v1`, `source_teacher_overrides.json`, `status.json`, `teachers/egopressure_balanced_v2/quality.json`.
- Static imports: `datetime`, `json`, `pathlib`, `shutil`, `tactilelatent.aligned_frames`, `time`.

</details>

<details>
<summary><strong>scripts/run_all_sources_joint.py</strong> — Production preflight, then start/resume the single integrated joint run.</summary>

- Status: **active workflow (observed)**; high: service/process plus source inspection.
- Source: [scripts/run_all_sources_joint.py](../scripts/run_all_sources_joint.py).
- Inputs: Selected all-source config, native manifests, accepted teacher/code receipts.
- Outputs: Preflight and selected output run: checkpoints, metrics, live status.
- Effects: GPU optimization or fitted probes.
- Invocation: `.venv/bin/python scripts/run_all_sources_joint.py --config configs/joint_vjepa21_all_sources_32d_mirror_v2.json`. Exact active service command inspected; launches/resumes training. Not executed during refactor.
- Caution: Starts/resumes GPU training, always running preflight first. Do not launch a duplicate alongside the active service.
- Service references: `tactilelatent-joint-all-sources-v2.service`, `tactilelatent-joint-all-sources-v1.service`. Presence of a unit does not mean it is running.
- Inspected arguments: `parser.add_argument('--config', default='configs/joint_vjepa21_all_sources_32d_mirror_v2.json')`.
- Literal paths/artifact names: `configs/joint_vjepa21_all_sources_32d_mirror_v2.json`, `last.pt`.
- Static imports: `argparse`, `json`, `pathlib`, `subprocess`, `sys`.

</details>

<details>
<summary><strong>scripts/train_all_sources_vjepa.py</strong> — All seventeen native sensor layouts, verified mirroring, MotionTrans held out.</summary>

- Status: **active workflow (observed)**; high: service/process plus source inspection.
- Source: [scripts/train_all_sources_vjepa.py](../scripts/train_all_sources_vjepa.py).
- Inputs: All-source config; native RGB clips, tactile codes and masks.
- Outputs: Selected output run: per-epoch checkpoints, metrics and provenance.
- Effects: GPU optimization or fitted probes.
- Invocation: `.venv/bin/python scripts/train_all_sources_vjepa.py --config configs/joint_vjepa21_all_sources_32d_mirror_v2.json --preflight-only`. Parser is delegated to acquisition_training/train_vjepa_markers. Known production preflight command inspected; not executed in this audit.
- Literal paths/artifact names: `configs/joint_vjepa21_all_sources_32d_mirror_v2.json`.
- Referenced by: `configs/joint_vjepa21_all_sources_32d_mirror.json`, `configs/joint_vjepa21_all_sources_32d_mirror_v2.json`, `scripts/run_all_sources_joint.py`.
- Static imports: `sys`, `tactilelatent`, `tactilelatent.acquisition_training`, `tactilelatent.expansion_runtime`.

</details>

<details>
<summary><strong>scripts/view_egotouch_autoencoder.py</strong> — Launch the local EgoTouch reconstruction viewer.</summary>

- Status: **active workflow (observed)**; high: service/process plus source inspection.
- Source: [scripts/view_egotouch_autoencoder.py](../scripts/view_egotouch_autoencoder.py).
- Inputs: Manifests, RGB/targets, saved teachers/joint checkpoints, training histories.
- Outputs: HTTP viewer on selected host/port; lazy inference and response caches.
- Effects: viewer server or viewer registration.
- Invocation: `.venv/bin/python scripts/view_egotouch_autoencoder.py --manifest processing/joint_alignment/manifest.json --host 127.0.0.1 --port 8080`. Actual service command verified with host0.0.0.0; parser preserved. Use existing viewer while its service runs.
- Service references: `tactilelatent-viewer.service`. Presence of a unit does not mean it is running.
- CLI delegated to: `tactilelatent.autoencoder_viewer`.
- Referenced by: `src/tactilelatent/autoencoder_viewer.py`.
- Static imports: `tactilelatent.acquisition_joint_viewer`, `tactilelatent.autoencoder_viewer`, `tactilelatent.bilateral_runtime`, `tactilelatent.training_dashboard`.

</details>

### maintained specialized workflow (11)

<details>
<summary><strong>scripts/build_zero_references.py</strong> — Build source-wide references without rewriting any acquisition targets.</summary>

- Status: **maintained specialized workflow**; moderate: current source/config references; not executed by this audit.
- Source: [scripts/build_zero_references.py](../scripts/build_zero_references.py).
- Inputs: Native acquisition targets and explicitly reviewed no-contact frames.
- Outputs: processing/zero_reference_v1 fixed source-wide reference registry and audit.
- Effects: storage/registry modification; see source-specific note.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Provenance: Exact relative path/hash recorded in existing receipt(s).
- Inspected arguments: `parser.add_argument('--root', type=Path, default=Path('processing/data_queue_v2'))`; `parser.add_argument('--output', type=Path, default=Path('processing/zero_reference_v1'))`.
- Literal paths/artifact names: `.npz`, `audit.json`, `candidate_window_statistics.json`, `configs/data_queue_active.json`, `manifest.json`, `optical_candidates.json`, `processing/data_queue_v2`, `processing/zero_reference_v1`, `references.json`, `reviewed_numeric_frames.json`, `tacquad/manifest.json`.
- Path-sensitive code: line284: `(Path(__file__), Path('src/tactilelatent/zero_reference.py'))},`.
- Static imports: `argparse`, `collections`, `hashlib`, `json`, `numpy`, `pathlib`, `tactilelatent.aligned_frames`, `tactilelatent.tacquad_viewer`, `tactilelatent.zero_reference`.

</details>

<details>
<summary><strong>scripts/cache_sensor_codes.py</strong> — Encode frozen sensor teachers and persist per-frame tactile code caches.</summary>

- Status: **maintained specialized workflow**; moderate: current source/config references; not executed by this audit.
- Source: [scripts/cache_sensor_codes.py](../scripts/cache_sensor_codes.py).
- Inputs: Source/config/checkpoint references and CLI inputs below; delegated loaders resolve remaining native paths.
- Outputs: Reports, derived artifacts or configured run outputs named in source/CLI below; full data availability unverified.
- Effects: reads research/native artifacts and writes reports/derived outputs.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- CLI delegated to: `tactilelatent.sensor_code_cache`.
- Static imports: `tactilelatent.sensor_code_cache`.

</details>

<details>
<summary><strong>scripts/cache_sensor_targets.py</strong> — Prepare per-frame tactile target caches without training the visual model.</summary>

- Status: **maintained specialized workflow**; moderate: current source/config references; not executed by this audit.
- Source: [scripts/cache_sensor_targets.py](../scripts/cache_sensor_targets.py).
- Inputs: Source/config/checkpoint references and CLI inputs below; delegated loaders resolve remaining native paths.
- Outputs: Reports, derived artifacts or configured run outputs named in source/CLI below; full data availability unverified.
- Effects: reads research/native artifacts and writes reports/derived outputs.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- CLI delegated to: `tactilelatent.sensor_cache`.
- Static imports: `tactilelatent.sensor_cache`.

</details>

<details>
<summary><strong>scripts/prepare_all_mirrored_teachers.py</strong> — Train/cache missing mirror decoders only. Does not launch joint training.</summary>

- Status: **maintained specialized workflow**; moderate: current source/config references; not executed by this audit.
- Source: [scripts/prepare_all_mirrored_teachers.py](../scripts/prepare_all_mirrored_teachers.py).
- Inputs: configs/mirror_adapted_teachers_v4.json and mirror policyv3, original native targets.
- Outputs: processing/bimanual_mirror_v4: adapted teachers, original/mirrored codes, quality/ready receipts.
- Effects: GPU optimization or fitted probes.
- Invocation: `.venv/bin/python scripts/prepare_all_mirrored_teachers.py`. Source entrypoint verified, defaults may run GPU training; not launched. Add desired inspected CLI arguments or config.
- Literal paths/artifact names: `Some mirror teachers failed quality; inspect ready.json`.
- Static imports: `tactilelatent.bilateral_mirror_preparation`, `tactilelatent.expansion_runtime`.

</details>

<details>
<summary><strong>scripts/prepare_marker_targets.py</strong> — Extract native optical dot trajectories without changing source datasets.</summary>

- Status: **maintained specialized workflow**; moderate: current source/config references; not executed by this audit.
- Source: [scripts/prepare_marker_targets.py](../scripts/prepare_marker_targets.py).
- Inputs: Native optical tactile streams and original chronological manifest.
- Outputs: marker_displacement_v2 signed displacement, geometry, validity and provenance.
- Effects: reads research/native artifacts and writes reports/derived outputs.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Inspected arguments: `parser.add_argument('--manifest', default='processing/joint_alignment/manifest.json')`; `parser.add_argument('--output', type=Path, default=Path('processing/marker_displacement_v2'))`; `parser.add_argument('--workers', type=int, default=2)`; `parser.add_argument('--pilot', action='store_true', help='First/middle/last episodes per source, including validation')`.
- Literal paths/artifact names: `complete.json`, `episodes.jsonl`, `index.json`, `processing/joint_alignment/manifest.json`, `processing/marker_displacement_v2`, `provenance.json`, `status.json`, `tracks.npz`.
- Static imports: `argparse`, `collections`, `concurrent.futures`, `dataclasses`, `fcntl`, `json`, `numpy`, `os`, `pathlib`, `tactilelatent.aligned_frames`, `tactilelatent.marker_data`, `tactilelatent.marker_targets`, `time`.

</details>

<details>
<summary><strong>scripts/pretrain_egotouch_autoencoder.py</strong> — Train a configurable EgoTouch tactile reconstruction autoencoder.</summary>

- Status: **maintained specialized workflow**; moderate: current source/config references; not executed by this audit.
- Source: [scripts/pretrain_egotouch_autoencoder.py](../scripts/pretrain_egotouch_autoencoder.py).
- Inputs: Experiment config, native training/validation manifests, teacher/cache/backbone artifacts named below.
- Outputs: Reports, derived artifacts or configured run outputs named in source/CLI below; full data availability unverified.
- Effects: GPU optimization or fitted probes.
- Invocation: `.venv/bin/python scripts/pretrain_egotouch_autoencoder.py`. Source entrypoint verified, defaults may run GPU training; not launched. Add desired inspected CLI arguments or config.
- CLI delegated to: `tactilelatent.pretrain_egotouch_autoencoder`.
- Referenced by: `scripts/train_egotouch_autoencoder_grid.sh`, `scripts/train_egotouch_autoencoder_presets.sh`, `scripts/train_sensor_autoencoders.py`, `scripts/tune_egotouch_autoencoder.py`.
- Static imports: `tactilelatent.pretrain_egotouch_autoencoder`.

</details>

<details>
<summary><strong>scripts/pretrain_sensor_autoencoder.py</strong> — Train a source-specific native tactile reconstruction autoencoder.</summary>

- Status: **maintained specialized workflow**; moderate: current source/config references; not executed by this audit.
- Source: [scripts/pretrain_sensor_autoencoder.py](../scripts/pretrain_sensor_autoencoder.py).
- Inputs: Experiment config, native training/validation manifests, teacher/cache/backbone artifacts named below.
- Outputs: Reports, derived artifacts or configured run outputs named in source/CLI below; full data availability unverified.
- Effects: GPU optimization or fitted probes.
- Invocation: `.venv/bin/python scripts/pretrain_sensor_autoencoder.py`. Source entrypoint verified, defaults may run GPU training; not launched. Add desired inspected CLI arguments or config.
- CLI delegated to: `tactilelatent.pretrain_sensor_autoencoder`.
- Referenced by: `scripts/handover_sensor_autoencoder_search.py`, `scripts/train_sensor_autoencoders.py`, `tests/test_sensor_search_handover.py`.
- Static imports: `tactilelatent.pretrain_sensor_autoencoder`.

</details>

<details>
<summary><strong>scripts/run_dataset_queue.py</strong> — Run the approved downloads AND processing; never launches joint training.</summary>

- Status: **maintained specialized workflow**; moderate: current source/config references; not executed by this audit.
- Source: [scripts/run_dataset_queue.py](../scripts/run_dataset_queue.py).
- Inputs: configs/data_queue_active.json, source URLs/revisions and storage limits.
- Outputs: Downloaded native data, processed acquisition manifests, queue logs/status.
- Effects: network access / possible downloads.
- Invocation: `.venv/bin/python scripts/run_dataset_queue.py status`. Command and parser inspected; not executed because even status writes reports.
- Caution: Even status writes aggregate status reports. Import modifies queue.CONFIG and snapshot behavior.
- Service references: `tactilelatent-dataset-queue-v2.service`. Presence of a unit does not mean it is running.
- Inspected arguments: `p.add_argument('command', choices=['run', 'source', 'status'])`; `p.add_argument('source', nargs='?', choices=[s['id'] for s in json.loads(queue.CONFIG.read_text())['jobs']])`.
- Literal paths/artifact names: `configs/data_queue_active.json`.
- Referenced by: `src/tactilelatent/acquisition_queue.py`.
- Static imports: `argparse`, `json`, `os`, `pathlib`, `tactilelatent`.

</details>

<details>
<summary><strong>scripts/train_acquisition_autoencoders.py</strong> — Resumable queue of native tactile-vector teachers; no RGB/backbone loading.</summary>

- Status: **maintained specialized workflow**; moderate: current source/config references; not executed by this audit.
- Source: [scripts/train_acquisition_autoencoders.py](../scripts/train_acquisition_autoencoders.py).
- Inputs: configs/acquisition_autoencoders_v1.json, native vector targets, fixed zero registry.
- Outputs: Frozen numeric teachers, normalizations, native reconstruction metrics/checkpoints.
- Effects: GPU optimization or fitted probes.
- Invocation: `.venv/bin/python scripts/train_acquisition_autoencoders.py`. Source entrypoint verified, defaults may run GPU training; not launched. Add desired inspected CLI arguments or config.
- Service references: `tactilelatent-acquisition-autoencoders-v1.service`. Presence of a unit does not mean it is running.
- Inspected arguments: `parser.add_argument('--config', default='configs/acquisition_autoencoders_v1.json')`; `parser.add_argument('--job', help='Run one configured job; default runs the full resumable queue')`.
- Literal paths/artifact names: `.pt`, `best.pt`, `best_metrics.json`, `config.json`, `configs/acquisition_autoencoders_v1.json`, `configs/data_queue_active.json`, `epoch_000.pt`, `last.pt`, `manifest.json`, `metrics.jsonl`, `normalization.json`, `processing/data_queue_v2`, `processing/zero_reference_v1/references.json`, `provenance.json`, `quality.json`, `status.json`.
- Static imports: `argparse`, `fcntl`, `json`, `numpy`, `pathlib`, `random`, `tactilelatent.aligned_frames`, `tactilelatent.numeric_autoencoder`, `tactilelatent.pretrain_egotouch_autoencoder`, `time`, `torch`, `traceback`.

</details>

<details>
<summary><strong>scripts/train_egopressure_teacher.py</strong> — Repair EgoPressure AE training, preserving the original teacher and codes.</summary>

- Status: **maintained specialized workflow**; moderate: current source/config references; not executed by this audit.
- Source: [scripts/train_egopressure_teacher.py](../scripts/train_egopressure_teacher.py).
- Inputs: Task-only EgoPressure manifest, native224x224 UV pressure and normalization.
- Outputs: Selected output/code-output: repaired64D teacher, codes and quality checks.
- Effects: GPU optimization or fitted probes.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Provenance: Hashes own implementation at runtime; retain bytes/location.
- Service references: `tactilelatent-egopressure-ae-v2.service`. Presence of a unit does not mean it is running.
- Inspected arguments: `parser.add_argument('--epochs', type=int, default=30)`; `parser.add_argument('--train-frames', type=int, default=100000)`; `parser.add_argument('--val-frames', type=int, default=10000)`; `parser.add_argument('--output', type=Path, default=ROOT / 'teachers/egopressure_balanced_tasks_v1')`; `parser.add_argument('--code-output', type=Path, default=ROOT / 'codes/egopressure_balanced_tasks_v1')`.
- Literal paths/artifact names: `best.pt`, `best_metrics.json`, `config.json`, `index.json`, `input_statistics.json`, `last.pt`, `metrics.jsonl`, `pressure_normalization.json`, `quality.json`, `ready.json`, `sampling_index.json`, `source_teacher_overrides.json`, `status.json`.
- Path-sensitive code: line147: `config['implementation']={str(p):file_hash(p) for p in [Path(__file__),`.
- Referenced by: `scripts/prepare_egopressure_task_selection.py`, `tests/test_pressure_selection.py`.
- Static imports: `argparse`, `collections`, `fcntl`, `json`, `math`, `numpy`, `pathlib`, `pyarrow.parquet`, `tactilelatent.aligned_frames`, `tactilelatent.data_expansion`, `tactilelatent.expansion_models`, `tactilelatent.expansion_training`, `tactilelatent.pressure_selection`, `tactilelatent.pressure_teacher`, `tactilelatent.train_alignment`, `time`, `torch`.

</details>

<details>
<summary><strong>scripts/verify_all_mirrored_teachers.py</strong> — Validate frozen mirror decoders/codes and prepare joint data without training.</summary>

- Status: **maintained specialized workflow**; moderate: current source/config references; not executed by this audit.
- Source: [scripts/verify_all_mirrored_teachers.py](../scripts/verify_all_mirrored_teachers.py).
- Inputs: v4 accepted mirror teachers/caches; all-source configv2.
- Outputs: Fresh-code parity receipts and metadata; processing/joint_acquisitions_mirrored_v2 preparation.
- Effects: storage/registry modification; see source-specific note.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Caution: No joint optimizer, but rewrites accepted-teacher metadata/index receipts and prepares future joint data. Not read-only.
- Literal paths/artifact names: `configs/bimanual_mirror_policy_v3.json`, `configs/joint_vjepa21_all_sources_32d_mirror_v2.json`, `integration_verification.json`, `processing/bimanual_mirror_v4`, `ready.json`.
- Static imports: `json`, `pathlib`, `tactilelatent`, `tactilelatent.acquisition_training`, `tactilelatent.aligned_frames`, `tactilelatent.bilateral_mirror_preparation`, `tactilelatent.bimanual_mirror`, `tactilelatent.expansion_runtime`, `tactilelatent.mirror_cache_verification`, `tactilelatent.mirror_joint`, `torch`.

</details>

### specialized operational / historical utility (30)

<details>
<summary><strong>scripts/after_rgb_migration.py</strong> — Finish native-data checks and cache preparation when background conversion ends.</summary>

- Status: **specialized operational / historical utility**; uncertain readiness: purpose inspected; historical defaults require review.
- Source: [scripts/after_rgb_migration.py](../scripts/after_rgb_migration.py).
- Inputs: storage_v1/status.json.
- Outputs: Waits then invokes finalization, writes logs and may restart viewer indirectly.
- Effects: storage/registry modification; see source-specific note; TOP-LEVEL execution on import; never probe with --help.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Caution: Import immediately enters a completion poll and subsequently runs finalization.
- Literal paths/artifact names: `finalization.log`, `processing/storage_v1`, `status.json`.
- Static imports: `json`, `pathlib`, `subprocess`, `time`.

</details>

<details>
<summary><strong>scripts/cleanup_storage.py</strong> — Audited cleanup of retired FreeTacMan data and redundant experiment weights.</summary>

- Status: **specialized operational / historical utility**; uncertain readiness: purpose inspected; historical defaults require review.
- Source: [scripts/cleanup_storage.py](../scripts/cleanup_storage.py).
- Inputs: Historical FreeTacMan/experiment inventory and protected-artifact analysis.
- Outputs: Cleanup report; --apply recursively deletes selected historical data/checkpoints.
- Effects: storage/registry modification; see source-specific note.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Caution: Without --apply writes a deletion proposal report. --apply deletes selected paths recursively; not a general safe cleanup command.
- Inspected arguments: `parser.add_argument('--apply', action='store_true')`; `parser.add_argument('--data-only', action='store_true')`.
- Literal paths/artifact names: `*.pt`, `best.pt`, `best_metrics.json`, `config.json`, `datasets/OpenDriveLab__FreeTacMan`, `epoch_*.pt`, `last.pt`, `metrics.jsonl`, `processing/evaluations`, `processing/marker_displacement_v2/targets/freetacman`, `processing/runs`, `processing/sensor_ae_cache_v1`, `processing/sensor_code_cache_markers_v1/freetacman`, `processing/sensor_code_cache_v1/freetacman`, `processing/storage_v1/cleanup.json`, `processing/vjepa21_clip_cache_v1/freetacman`.
- Path-sensitive code: line18: `ROOT=Path.cwd();REPORT=Path('processing/storage_v1/cleanup.json')`.
- Static imports: `argparse`, `gc`, `json`, `pathlib`, `re`, `shutil`, `tactilelatent.aligned_frames`, `time`, `torch`.

</details>

<details>
<summary><strong>scripts/complete_baihu_teacher.py</strong> — Finish the authorized native four-surface AE, cache codes, then publish.</summary>

- Status: **specialized operational / historical utility**; uncertain readiness: purpose inspected; historical defaults require review.
- Source: [scripts/complete_baihu_teacher.py](../scripts/complete_baihu_teacher.py).
- Inputs: Source/config/checkpoint references and CLI inputs below; delegated loaders resolve remaining native paths.
- Outputs: Reports, derived artifacts or configured run outputs named in source/CLI below; full data availability unverified.
- Effects: GPU optimization or fitted probes.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Literal paths/artifact names: `best_metrics.json`, `index.json`, `preflight.json`, `quality.json`, `source_teacher_overrides.json`, `status.json`.
- Static imports: `fcntl`, `json`, `pathlib`, `tactilelatent.aligned_frames`, `tactilelatent.data_expansion`, `tactilelatent.expansion_models`, `tactilelatent.expansion_training`, `time`, `torch`.

</details>

<details>
<summary><strong>scripts/expand_datasets.py</strong> — Resumable data expansion. No joint-model training is launched.</summary>

- Status: **specialized operational / historical utility**; uncertain readiness: purpose inspected; historical defaults require review.
- Source: [scripts/expand_datasets.py](../scripts/expand_datasets.py).
- Inputs: Source/config/checkpoint references and CLI inputs below; delegated loaders resolve remaining native paths.
- Outputs: Reports, derived artifacts or configured run outputs named in source/CLI below; full data availability unverified.
- Effects: network access / possible downloads.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Inspected arguments: `parser.add_argument('job', choices=['downloads', 'egopressure', 'feel', 'baihu', 'teachers', 'publish', 'all'])`; `parser.add_argument('--workers', type=int, default=2)`; `parser.add_argument('--pilot-only', action='store_true')`.
- Literal paths/artifact names: `.log`.
- Path-sensitive code: line34: `p=subprocess.Popen([sys.executable,'-u',__file__,job,'--workers',str(args.workers)],`.
- Static imports: `argparse`, `fcntl`, `json`, `os`, `pathlib`, `subprocess`, `sys`, `tactilelatent.data_expansion`, `tactilelatent.expansion_baihu`, `tactilelatent.expansion_runtime`, `tactilelatent.expansion_training`, `time`.

</details>

<details>
<summary><strong>scripts/extend_fgw_pilot.py</strong> — Extend both FGW placements after the active worker saves its next epoch.</summary>

- Status: **specialized operational / historical utility**; uncertain readiness: purpose inspected; historical defaults require review.
- Source: [scripts/extend_fgw_pilot.py](../scripts/extend_fgw_pilot.py).
- Inputs: Source/config/checkpoint references and CLI inputs below; delegated loaders resolve remaining native paths.
- Outputs: Configured destination: parser.add_argument('--root', required=True).
- Effects: process/service control and run/config mutation; sends process signals.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Inspected arguments: `parser.add_argument('--root', required=True)`; `parser.add_argument('--after-epoch', type=int, required=True)`; `parser.add_argument('--epochs', type=int, required=True)`.
- Literal paths/artifact names: `.pt`, `checkpoint_before_extension.pt`, `configs/visual.json`, `controller.log`, `extension_status.json`, `last.pt`, `pilot.json`, `status.json`.
- Path-sensitive code: line58: `Path(__file__).parents[1] / 'src/tactilelatent/train_alignment.py')}`.
- Static imports: `argparse`, `json`, `os`, `pathlib`, `signal`, `subprocess`, `sys`, `tactilelatent.aligned_frames`, `tactilelatent.fgw_pilot`, `tactilelatent.train_alignment`, `time`, `torch`.

</details>

<details>
<summary><strong>scripts/extend_joint_alignment.py</strong> — Hand over the running alignment job after its next complete checkpoint.</summary>

- Status: **specialized operational / historical utility**; uncertain readiness: purpose inspected; historical defaults require review.
- Source: [scripts/extend_joint_alignment.py](../scripts/extend_joint_alignment.py).
- Inputs: Source/config/checkpoint references and CLI inputs below; delegated loaders resolve remaining native paths.
- Outputs: Reports, derived artifacts or configured run outputs named in source/CLI below; full data availability unverified.
- Effects: process/service control and run/config mutation.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Inspected arguments: `parser.add_argument('--config', required=True)`; `parser.add_argument('--service', required=True)`; `parser.add_argument('--pid', required=True, type=int)`; `parser.add_argument('--after-epoch', required=True, type=int)`; `parser.add_argument('--epochs', required=True, type=int)`.
- Literal paths/artifact names: `.pt`, `extension_status.json`, `last.pt`, `metrics.jsonl`.
- Path-sensitive code: line50: `Path(__file__).parents[1] / 'src/tactilelatent/train_alignment.py')}`.
- Static imports: `argparse`, `json`, `os`, `pathlib`, `subprocess`, `sys`, `tactilelatent.aligned_frames`, `tactilelatent.train_alignment`, `time`, `torch`.

</details>

<details>
<summary><strong>scripts/extend_vjepa_markers.py</strong> — Resume the marker-code trainer at its next committed epoch, with a longer LR tail.</summary>

- Status: **specialized operational / historical utility**; uncertain readiness: purpose inspected; historical defaults require review.
- Source: [scripts/extend_vjepa_markers.py](../scripts/extend_vjepa_markers.py).
- Inputs: Source/config/checkpoint references and CLI inputs below; delegated loaders resolve remaining native paths.
- Outputs: Reports, derived artifacts or configured run outputs named in source/CLI below; full data availability unverified.
- Effects: process/service control and run/config mutation; sends process signals.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Provenance: Hashes own implementation at runtime; retain bytes/location.
- Inspected arguments: `parser.add_argument('--request', required=True)`.
- Literal paths/artifact names: `.pt`, `config.json`, `extension_status.json`, `last.pt`, `nohup.log`, `preflight.json`.
- Path-sensitive code: line15: `from stop_spatial_vjepa_after_epoch import committed_record,identity`; line30: `if file_hash(Path(__file__))!=request['controller_sha256']:raise ValueError('Extension controller changed')`.
- Static imports: `argparse`, `fcntl`, `json`, `os`, `pathlib`, `signal`, `stop_spatial_vjepa_after_epoch`, `sys`, `tactilelatent.aligned_frames`, `tactilelatent.train_alignment`, `tactilelatent.train_vjepa_markers`, `time`, `torch`.

</details>

<details>
<summary><strong>scripts/finalize_rgb_migration.py</strong> — Verify preserved targets and prepare the next source-filtered joint run.</summary>

- Status: **specialized operational / historical utility**; uncertain readiness: purpose inspected; historical defaults require review.
- Source: [scripts/finalize_rgb_migration.py](../scripts/finalize_rgb_migration.py).
- Inputs: Completed RGB receipts, native targets and historical code caches.
- Outputs: Storage manifest/teacher/config/cache metadata; hard-links code arrays; restarts viewer service.
- Effects: storage/registry modification; see source-specific note; restarts viewer service.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Literal paths/artifact names: `audit.json`, `configs/joint_vjepa21_spatial_storage_32d.json`, `configs/vjepa_live_teacher_normalization.json`, `finalization_status.json`, `index.json`, `manifest.json`, `marker_teachers.json`, `native_target_checks.json`, `normalization.json`, `preparation.json`, `processing/joint_alignment/audit.json`, `processing/joint_alignment/marker_teachers_v1.json`, `processing/joint_alignment/teachers.json`, `processing/marker_displacement_storage_v1/targets`, `processing/marker_displacement_v2/targets`, `processing/marker_displacement_v2/targets/index.json`, `processing/sensor_code_cache_markers_v1`, `processing/sensor_code_cache_storage_v1`, `processing/storage_v1`, `registry.json`, `status.json`, `teachers.json`.
- Referenced by: `scripts/after_rgb_migration.py`.
- Static imports: `hashlib`, `json`, `numpy`, `os`, `pathlib`, `subprocess`, `tactilelatent.aligned_frames`, `tactilelatent.marker_ae_data`, `tactilelatent.marker_code_cache`, `tactilelatent.sensor_code_cache`, `tactilelatent.train_vjepa_markers`, `time`.

</details>

<details>
<summary><strong>scripts/finish_alignment_and_evaluate_motiontrans.py</strong> — Stop the alignment service after a saved epoch and evaluate held-out MotionTrans.</summary>

- Status: **specialized operational / historical utility**; uncertain readiness: purpose inspected; historical defaults require review.
- Source: [scripts/finish_alignment_and_evaluate_motiontrans.py](../scripts/finish_alignment_and_evaluate_motiontrans.py).
- Inputs: Source/config/checkpoint references and CLI inputs below; delegated loaders resolve remaining native paths.
- Outputs: Configured destination: parser.add_argument('--output', required=True).
- Effects: process/service control and run/config mutation.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Inspected arguments: `parser.add_argument('--run', required=True)`; `parser.add_argument('--service', required=True)`; `parser.add_argument('--pid', type=int, required=True)`; `parser.add_argument('--epoch', type=int, required=True)`; `parser.add_argument('--output', required=True)`.
- Literal paths/artifact names: `.pt`, `/evaluation.log`, `checkpoint.pt`, `comparison.json`, `evaluation.log`, `launch.json`, `metrics.jsonl`, `results.json`, `stop_status.json`.
- Static imports: `argparse`, `json`, `os`, `pathlib`, `subprocess`, `sys`, `tactilelatent.aligned_frames`, `time`, `torch`.

</details>

<details>
<summary><strong>scripts/finish_continuation_and_probe.py</strong> — Honor the epoch-22 stop, finish its probes, then screen frozen representations.</summary>

- Status: **specialized operational / historical utility**; uncertain readiness: purpose inspected; historical defaults require review.
- Source: [scripts/finish_continuation_and_probe.py](../scripts/finish_continuation_and_probe.py).
- Inputs: Source/config/checkpoint references and CLI inputs below; delegated loaders resolve remaining native paths.
- Outputs: Configured destination: parser.add_argument('--root', type=Path, default=Path('processing/runs/continuation_32d_v1')); parser.add_argument('--output', type=Path, default=Path('processing/evaluations/motiontrans_representation_screen_v1')).
- Effects: process/service control and run/config mutation; sends process signals.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Inspected arguments: `parser.add_argument('--root', type=Path, default=Path('processing/runs/continuation_32d_v1'))`; `parser.add_argument('--output', type=Path, default=Path('processing/evaluations/motiontrans_representation_screen_v1'))`.
- Literal paths/artifact names: `evaluations/epoch_022/summary.json`, `finish_status.json`, `logs/finish_epoch_022.log`, `metrics.jsonl`, `nohup.log`, `processing/evaluations/motiontrans_representation_screen_v1`, `processing/runs/continuation_32d_v1`, `status.json`, `stop_after_epoch_022.json`, `stop_status.json`, `summary.json`, `viewer_epochs/epoch_022.pt`.
- Static imports: `argparse`, `datetime`, `json`, `os`, `pathlib`, `signal`, `subprocess`, `sys`, `tactilelatent.aligned_frames`, `time`.

</details>

<details>
<summary><strong>scripts/handover_sensor_autoencoder_search.py</strong> — Switch a search plan after its live trial finishes, preserving all checkpoints.</summary>

- Status: **specialized operational / historical utility**; uncertain readiness: purpose inspected; historical defaults require review.
- Source: [scripts/handover_sensor_autoencoder_search.py](../scripts/handover_sensor_autoencoder_search.py).
- Inputs: Source/config/checkpoint references and CLI inputs below; delegated loaders resolve remaining native paths.
- Outputs: Configured destination: parser.add_argument('--output', type=Path, required=True).
- Effects: process/service control and run/config mutation; sends process signals.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Inspected arguments: `parser.add_argument('--plan', type=Path, required=True)`; `parser.add_argument('--output', type=Path, required=True)`; `parser.add_argument('--coordinator-pid', type=int)`; `parser.add_argument('--trial-pid', type=int)`; `parser.add_argument('--resume-wait', action='store_true', help='Resume a recorded handover after the old coordinator has exited')`; `parser.add_argument('--code-changes', type=Path, help='Verified old/new hashes and archived old code for authorized implementation changes')`.
- Literal paths/artifact names: `*_finalists.json`, `*_repeat.json`, `code_changes.json`, `configs/sensor_autoencoder_search_`, `four_candidate_migration.json`, `plan.json`, `provenance.json`.
- Path-sensitive code: line66: `root = Path(__file__).resolve().parents[1]`; line67: `os.chdir(root)`.
- Referenced by: `tests/test_sensor_search_handover.py`.
- Static imports: `argparse`, `fcntl`, `hashlib`, `json`, `os`, `pathlib`, `shutil`, `signal`, `sys`, `time`.

</details>

<details>
<summary><strong>scripts/migrate_jepa_rgb.py</strong> — Resumable, verified, camera-only H.264 conversion and archive compaction.</summary>

- Status: **specialized operational / historical utility**; uncertain readiness: purpose inspected; historical defaults require review.
- Source: [scripts/migrate_jepa_rgb.py](../scripts/migrate_jepa_rgb.py).
- Inputs: Native camera streams and source archives.
- Outputs: 384px H.264 camera replacements and storage ledger; default compacts/replaces source archives.
- Effects: storage/registry modification; see source-specific note.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Inspected arguments: `parser.add_argument('--workers', type=int, default=4)`; `parser.add_argument('--sources', nargs='+')`; `parser.add_argument('--no-compact', action='store_true')`.
- Literal paths/artifact names: `.compact.partial.tar`, `.json`, `.mp4`, `.partial.mp4`, `conversion_status.json`, `original_manifest.json`, `processing/storage_v1`, `registry.json`.
- Referenced by: `scripts/run_rgb_migration.py`.
- Static imports: `PIL`, `argparse`, `av`, `collections`, `concurrent.futures`, `fcntl`, `fractions`, `h5py`, `hashlib`, `io`, `json`, `numpy`, `pathlib`, `shutil`, `tactilelatent.aligned_frames`, `tactilelatent.sensor_data`, `tactilelatent.storage`, `tarfile`, `time`, `torch`, `torch.nn.functional`, `zipfile`.

</details>

<details>
<summary><strong>scripts/monitor_bimanual_mirror.py</strong> — SSH-independent progress log for the supervised mirror preparation job.</summary>

- Status: **specialized operational / historical utility**; uncertain readiness: purpose inspected; historical defaults require review.
- Source: [scripts/monitor_bimanual_mirror.py](../scripts/monitor_bimanual_mirror.py).
- Inputs: Source/config/checkpoint references and CLI inputs below; delegated loaders resolve remaining native paths.
- Outputs: Configured destination: p.add_argument('--root', type=Path, default=Path('processing/bimanual_mirror_v1')).
- Effects: reads research/native artifacts and writes reports/derived outputs.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Inspected arguments: `p.add_argument('--root', type=Path, default=Path('processing/bimanual_mirror_v1'))`; `p.add_argument('--interval', type=float, default=30.0)`.
- Literal paths/artifact names: `background_result.json`, `failure.log`, `monitor_status.json`, `processing/bimanual_mirror_v1`, `processing/data_expansion_v1/teachers/egopressure_balanced_v2/status.json`, `ready.json`, `status.json`, `verification/report.json`.
- Static imports: `argparse`, `fcntl`, `json`, `pathlib`, `subprocess`, `tactilelatent.aligned_frames`, `time`.

</details>

<details>
<summary><strong>scripts/monitor_storage_migration.py</strong> — Continuously summarize the live storage migration in a workspace log.</summary>

- Status: **specialized operational / historical utility**; uncertain readiness: purpose inspected; historical defaults require review.
- Source: [scripts/monitor_storage_migration.py](../scripts/monitor_storage_migration.py).
- Inputs: storage_v1 conversion/status JSON.
- Outputs: Appends root STORAGE_MIGRATION.log until completion/failure.
- Effects: TOP-LEVEL execution on import; never probe with --help.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Caution: Import starts a potentially long polling loop and appends a root log.
- Literal paths/artifact names: `STORAGE_MIGRATION.log`, `conversion_status.json`, `processing/storage_v1`, `status.json`.
- Static imports: `datetime`, `json`, `pathlib`, `time`, `zoneinfo`.

</details>

<details>
<summary><strong>scripts/prepare_bimanual_mirror.py</strong> — Prepare verified bilateral teachers/codes; never launch joint training.</summary>

- Status: **specialized operational / historical utility**; uncertain readiness: purpose inspected; historical defaults require review.
- Source: [scripts/prepare_bimanual_mirror.py](../scripts/prepare_bimanual_mirror.py).
- Inputs: Source/config/checkpoint references and CLI inputs below; delegated loaders resolve remaining native paths.
- Outputs: Configured destination: p.add_argument('--output', type=Path, default=ROOT).
- Effects: GPU optimization or fitted probes.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Service references: `tactilelatent-bimanual-mirror-v1.service`. Presence of a unit does not mean it is running.
- Inspected arguments: `p.add_argument('--output', type=Path, default=ROOT)`; `p.add_argument('--manifest', default=MANIFEST)`; `p.add_argument('--policy', default=POLICY)`; `p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')`; `p.add_argument('--train-frames', type=int, default=100000)`; `p.add_argument('--val-frames', type=int, default=10000)`.
- Static imports: `argparse`, `pathlib`, `tactilelatent.mirror_cache_verification`, `tactilelatent.mirror_preparation`.

</details>

<details>
<summary><strong>scripts/prepare_egopressure_task_selection.py</strong> — Exclude calibration episodes, reuse codes, and evaluate task-only validation.</summary>

- Status: **specialized operational / historical utility**; uncertain readiness: purpose inspected; historical defaults require review.
- Source: [scripts/prepare_egopressure_task_selection.py](../scripts/prepare_egopressure_task_selection.py).
- Inputs: EgoPressure manifests, cached codes and frozen teacher.
- Outputs: Task-only manifest/cache index/quality and source-teacher override metadata.
- Effects: storage/registry modification; see source-specific note.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Inspected arguments: `p.add_argument('--root', type=Path, default=Path('processing/data_expansion_v1'))`; `p.add_argument('--device', default='cuda')`.
- Literal paths/artifact names: `calibration_support_audit.json`, `codes/egopressure_balanced_v2/index.json`, `egopressure_manifest.json`, `index.json`, `processing/data_expansion_v1`, `quality.json`, `sampling_index.json`, `source_teacher_overrides.json`, `source_teacher_overrides_before.json`, `status.json`, `teachers/egopressure_balanced_v2/best.pt`.
- Static imports: `argparse`, `collections`, `importlib.util`, `json`, `numpy`, `pathlib`, `tactilelatent.aligned_frames`, `tactilelatent.expansion_models`, `tactilelatent.pressure_selection`, `time`, `torch`.

</details>

<details>
<summary><strong>scripts/prepare_gel_mirror.py</strong> — Audit/adapt the confirmed Unit/Baihu mirrors and cache both native variants.</summary>

- Status: **specialized operational / historical utility**; uncertain readiness: purpose inspected; historical defaults require review.
- Source: [scripts/prepare_gel_mirror.py](../scripts/prepare_gel_mirror.py).
- Inputs: Source/config/checkpoint references and CLI inputs below; delegated loaders resolve remaining native paths.
- Outputs: Configured destination: p.add_argument('--output', type=Path, default=ROOT).
- Effects: GPU optimization or fitted probes.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Service references: `tactilelatent-bimanual-mirror-v2.service`. Presence of a unit does not mean it is running.
- Inspected arguments: `p.add_argument('--output', type=Path, default=ROOT)`; `p.add_argument('--policy', default=POLICY)`; `p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')`; `p.add_argument('--train-frames', type=int, default=100000)`; `p.add_argument('--val-frames', type=int, default=10000)`.
- Literal paths/artifact names: `Some confirmed sources failed teacher/cache checks; see ready.json`.
- Static imports: `argparse`, `pathlib`, `tactilelatent.expansion_runtime`, `tactilelatent.gel_mirror_preparation`.

</details>

<details>
<summary><strong>scripts/prepare_joint_acquisitions.py</strong> — Prepare frozen numeric codes and valid chronological histories for joint use.</summary>

- Status: **specialized operational / historical utility**; uncertain readiness: purpose inspected; historical defaults require review.
- Source: [scripts/prepare_joint_acquisitions.py](../scripts/prepare_joint_acquisitions.py).
- Inputs: Prepared acquisition manifests, native targets and selected teachers.
- Outputs: Original acquisition_joint.ROOT data/code/teacher registry; not the newer v2 root.
- Effects: reads research/native artifacts and writes reports/derived outputs.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Static imports: `fcntl`, `tactilelatent.acquisition_joint`, `tactilelatent.expansion_runtime`.

</details>

<details>
<summary><strong>scripts/repair_marker_startup.py</strong> — Repair an untrackable startup without shifting or inventing tactile frames.</summary>

- Status: **specialized operational / historical utility**; uncertain readiness: purpose inspected; historical defaults require review.
- Source: [scripts/repair_marker_startup.py](../scripts/repair_marker_startup.py).
- Inputs: Historical Slot/Slot_164 native optical data and marker target directory.
- Outputs: Replaces one episode marker artifact and optional completion/index metadata.
- Effects: storage/registry modification; see source-specific note.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Caution: Episode Slot/Slot_164 is hard-coded; this is a one-off repair, not generic marker extraction.
- Provenance: Hashes own implementation at runtime; retain bytes/location.
- Inspected arguments: `parser.add_argument('--root', default='processing/marker_displacement_v2/targets')`; `parser.add_argument('--wait-and-reconcile', action='store_true')`.
- Literal paths/artifact names: `complete.json`, `index.json`, `processing/joint_alignment/manifest.json`, `processing/marker_displacement_v2/targets`, `status.json`, `tracks.npz`.
- Path-sensitive code: line53: `repair=dict(reason='Corrupted white/black startup frames on the first surface',script_sha256=file_hash(__file__),`.
- Static imports: `argparse`, `json`, `numpy`, `pathlib`, `tactilelatent.aligned_frames`, `tactilelatent.marker_data`, `tactilelatent.marker_targets`, `time`.

</details>

<details>
<summary><strong>scripts/retire_marker_images.py</strong> — Retire only verified FTP marker-source tactile pixels, preserving all else.</summary>

- Status: **specialized operational / historical utility**; uncertain readiness: purpose inspected; historical defaults require review.
- Source: [scripts/retire_marker_images.py](../scripts/retire_marker_images.py).
- Inputs: Verified FTP marker targets/codes and source image archives.
- Outputs: Receipts and atomic compacted archives/indexes; --apply removes tactile image payloads.
- Effects: storage/registry modification; see source-specific note.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Caution: Without --apply it still creates reports/preflight records and may run consumer verification. --apply changes raw archives.
- Inspected arguments: `parser.add_argument('--apply', action='store_true')`.
- Literal paths/artifact names: `*.json`, `.json`, `audit.json`, `complete.json`, `configs/joint_vjepa21_spatial_current_32d.json`, `datasets/AllenBi21__VLA-Touch/raw_data`, `implementation.json`, `manifest.json`, `plan.json`, `preflight.json`, `preparation.json`, `previous_implementation.json`, `previous_registry.json`, `processing/marker_displacement_storage_v1/targets`, `processing/sensor_code_cache_storage_v1`, `processing/storage_v1`, `progress.jsonl`, `registry.json`, `retirement.json`, `status.json`, `teachers.json`, `tracks.npz`, `verification.json`.
- Referenced by: `tests/test_marker_storage.py`.
- Static imports: `argparse`, `fcntl`, `hashlib`, `io`, `json`, `numpy`, `os`, `pathlib`, `shutil`, `tactilelatent.aligned_frames`, `tactilelatent.expansion_runtime`, `tactilelatent.marker_autoencoder`, `tactilelatent.marker_code_cache`, `tactilelatent.marker_code_model`, `tactilelatent.marker_data`, `tactilelatent.marker_storage`, `tactilelatent.marker_viewer`, `tactilelatent.sensor_code_cache`, `tactilelatent.train_alignment`, `tactilelatent.train_vjepa_markers`, `tarfile`, `time`, `torch`, `zipfile`.

</details>

<details>
<summary><strong>scripts/run_rgb_migration.py</strong> — Run verified source conversions sequentially so reclaimed space arrives early.</summary>

- Status: **specialized operational / historical utility**; uncertain readiness: purpose inspected; historical defaults require review.
- Source: [scripts/run_rgb_migration.py](../scripts/run_rgb_migration.py).
- Inputs: Fixed historical eleven-source list and native camera streams.
- Outputs: Sequential migrations, storage_v1 status and logs; rewrites/deletes verified RGB originals.
- Effects: storage/registry modification; see source-specific note.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Literal paths/artifact names: `.log`, `processing/storage_v1`, `status.json`.
- Static imports: `json`, `os`, `pathlib`, `subprocess`, `tactilelatent.aligned_frames`, `time`.

</details>

<details>
<summary><strong>scripts/start_vjepa_current.py</strong> — Launch a fresh deterministic current-frame run with a configured latent size.</summary>

- Status: **specialized operational / historical utility**; uncertain readiness: purpose inspected; historical defaults require review.
- Source: [scripts/start_vjepa_current.py](../scripts/start_vjepa_current.py).
- Inputs: Experiment config, native training/validation manifests, teacher/cache/backbone artifacts named below.
- Outputs: Reports, derived artifacts or configured run outputs named in source/CLI below; full data availability unverified.
- Effects: reads research/native artifacts and writes reports/derived outputs.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Inspected arguments: `parser.add_argument('--config', type=Path, required=True)`.
- Literal paths/artifact names: `last.pt`, `launch_status.json`, `nohup.log`, `preflight.json`, `preflight.log`, `processing/storage_v1/status.json`.
- Static imports: `argparse`, `fcntl`, `json`, `os`, `pathlib`, `subprocess`, `tactilelatent.aligned_frames`, `time`.

</details>

<details>
<summary><strong>scripts/start_vjepa_current128d.py</strong> — Run the requested fresh 128D experiment after RGB migration is verified.</summary>

- Status: **specialized operational / historical utility**; uncertain readiness: purpose inspected; historical defaults require review.
- Source: [scripts/start_vjepa_current128d.py](../scripts/start_vjepa_current128d.py).
- Inputs: Experiment config, native training/validation manifests, teacher/cache/backbone artifacts named below.
- Outputs: Reports, derived artifacts or configured run outputs named in source/CLI below; full data availability unverified.
- Effects: reads research/native artifacts and writes reports/derived outputs.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Provenance: Exact relative path/hash recorded in existing receipt(s).
- Literal paths/artifact names: `configs/joint_vjepa21_spatial_current_128d.json`, `finalization_status.json`, `last.pt`, `launch_status.json`, `nohup.log`, `preflight.json`, `preflight.log`, `processing/storage_v1`, `status.json`.
- Static imports: `fcntl`, `json`, `os`, `pathlib`, `subprocess`, `sys`, `tactilelatent.aligned_frames`, `time`.

</details>

<details>
<summary><strong>scripts/stop_spatial_vjepa_after_epoch.py</strong> — Stop a running live trainer after its complete checkpoint/validation log.</summary>

- Status: **specialized operational / historical utility**; uncertain readiness: purpose inspected; historical defaults require review.
- Source: [scripts/stop_spatial_vjepa_after_epoch.py](../scripts/stop_spatial_vjepa_after_epoch.py).
- Inputs: Source/config/checkpoint references and CLI inputs below; delegated loaders resolve remaining native paths.
- Outputs: Reports, derived artifacts or configured run outputs named in source/CLI below; full data availability unverified.
- Effects: process/service control and run/config mutation; sends process signals.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Inspected arguments: `parser.add_argument('--run', type=Path, required=True)`.
- Literal paths/artifact names: `.pt`, `last.pt`, `nohup.log`, `status.json`, `stop_controller_status.json`, `stop_request.json`, `stop_status.json`.
- Static imports: `argparse`, `fcntl`, `json`, `os`, `pathlib`, `signal`, `tactilelatent.aligned_frames`, `time`, `torch`.

</details>

<details>
<summary><strong>scripts/train_baihu_autoencoder_repair.py</strong> — Repair the collapsed Baihu teacher; preserve old checkpoints and targets.</summary>

- Status: **specialized operational / historical utility**; uncertain readiness: purpose inspected; historical defaults require review.
- Source: [scripts/train_baihu_autoencoder_repair.py](../scripts/train_baihu_autoencoder_repair.py).
- Inputs: Experiment config, native training/validation manifests, teacher/cache/backbone artifacts named below.
- Outputs: Configured destination: parser.add_argument('--output', type=Path, default=ROOT / 'teachers/baihu_native_v3').
- Effects: GPU optimization or fitted probes.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Provenance: Hashes own implementation at runtime; retain bytes/location.
- Inspected arguments: `parser.add_argument('--output', type=Path, default=ROOT / 'teachers/baihu_native_v3')`; `parser.add_argument('--prepare-codes', action='store_true')`.
- Literal paths/artifact names: `best.pt`, `best_metrics.json`, `complete.json`, `config.json`, `epoch_000.pt`, `index.json`, `last.pt`, `manifest.json`, `metrics.jsonl`, `progress.jsonl`, `quality.json`, `sampling.json`, `source_teacher_overrides.json`, `status.json`.
- Path-sensitive code: line46: `implementation={str(p):file_hash(p) for p in [Path(__file__),`.
- Static imports: `argparse`, `fcntl`, `json`, `numpy`, `pathlib`, `tactilelatent.aligned_frames`, `tactilelatent.baihu_ae_data`, `tactilelatent.baihu_autoencoder`, `tactilelatent.data_expansion`, `tactilelatent.expansion_training`, `tactilelatent.train_alignment`, `time`, `torch`.

</details>

<details>
<summary><strong>scripts/train_balanced_mirror_autoencoders.py</strong> — Continue all verified mirror AEs with paired native targets and parity checks.</summary>

- Status: **specialized operational / historical utility**; uncertain readiness: purpose inspected; historical defaults require review.
- Source: [scripts/train_balanced_mirror_autoencoders.py](../scripts/train_balanced_mirror_autoencoders.py).
- Inputs: Experiment config, native training/validation manifests, teacher/cache/backbone artifacts named below.
- Outputs: Reports, derived artifacts or configured run outputs named in source/CLI below; full data availability unverified.
- Effects: GPU optimization or fitted probes.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Service references: `tactilelatent-bimanual-mirror-v3.service`. Presence of a unit does not mean it is running.
- Inspected arguments: `parser.add_argument('--config', default='configs/mirror_balanced_training_v3.json')`.
- Literal paths/artifact names: `configs/joint_vjepa21_spatial_expanded_32d_mirror_v2.json`, `configs/mirror_balanced_training_v3.json`.
- Static imports: `argparse`, `json`, `pathlib`, `subprocess`, `sys`, `tactilelatent.expansion_runtime`, `tactilelatent.mirror_balanced_training`.

</details>

<details>
<summary><strong>scripts/validate_rgb_migration.py</strong> — Check native targets, converted camera masks, and indexed video offsets.</summary>

- Status: **specialized operational / historical utility**; uncertain readiness: purpose inspected; historical defaults require review.
- Source: [scripts/validate_rgb_migration.py](../scripts/validate_rgb_migration.py).
- Inputs: Source/config/checkpoint references and CLI inputs below; delegated loaders resolve remaining native paths.
- Outputs: Reports, derived artifacts or configured run outputs named in source/CLI below; full data availability unverified.
- Effects: reads research/native artifacts and writes reports/derived outputs.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Literal paths/artifact names: `manifest.json`, `native_target_checks.json`, `offset.mp4`, `processing/storage_v1`, `validation.json`.
- Static imports: `av`, `hashlib`, `json`, `numpy`, `pathlib`, `tactilelatent`, `tactilelatent.aligned_frames`, `tempfile`, `torch`, `unittest.mock`.

</details>

<details>
<summary><strong>scripts/verify_bimanual_mirror.py</strong> — Disposable real-RGB V-JEPA/SAM updates for prepared bilateral sources.</summary>

- Status: **specialized operational / historical utility**; uncertain readiness: purpose inspected; historical defaults require review.
- Source: [scripts/verify_bimanual_mirror.py](../scripts/verify_bimanual_mirror.py).
- Inputs: Source/config/checkpoint references and CLI inputs below; delegated loaders resolve remaining native paths.
- Outputs: Reports, derived artifacts or configured run outputs named in source/CLI below; full data availability unverified.
- Effects: GPU optimization or fitted probes.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Caution: Disposable V-JEPA/SAM optimization on real RGB; no full joint checkpoint, but does use GPU.
- Service references: `tactilelatent-bimanual-mirror-v1.service`, `tactilelatent-bimanual-mirror-v2.service`. Presence of a unit does not mean it is running.
- Inspected arguments: `p.add_argument('--preparation', default='processing/bimanual_mirror_v1')`; `p.add_argument('--base-config', default='configs/joint_vjepa21_spatial_current_32d.json')`.
- Literal paths/artifact names: `available_joint_config.json`, `configs/joint_vjepa21_spatial_current_32d.json`, `processing/bimanual_mirror_v1`, `processing/runs/joint_vjepa21_spatial_`, `ready.json`, `report.json`.
- Referenced by: `scripts/train_balanced_mirror_autoencoders.py`.
- Static imports: `argparse`, `copy`, `json`, `matplotlib`, `matplotlib.pyplot`, `numpy`, `pathlib`, `tactilelatent`, `tactilelatent.aligned_frames`, `tactilelatent.expansion_joint`, `tactilelatent.expansion_runtime`, `tactilelatent.marker_code_model`, `tactilelatent.marker_data`, `tactilelatent.mirror_joint`, `tactilelatent.sam`, `tactilelatent.vjepa_model`, `tactilelatent.vjepa_spatial`, `time`, `torch`.

</details>

<details>
<summary><strong>scripts/verify_gel_mirror_mapping.py</strong> — Verify applied gel mappings on native data and the running mirror viewer.</summary>

- Status: **specialized operational / historical utility**; uncertain readiness: purpose inspected; historical defaults require review.
- Source: [scripts/verify_gel_mirror_mapping.py](../scripts/verify_gel_mirror_mapping.py).
- Inputs: Source/config/checkpoint references and CLI inputs below; delegated loaders resolve remaining native paths.
- Outputs: Configured destination: p.add_argument('--output', type=Path, default=Path('processing/bimanual_mirror_v2/mapping_verification.json')).
- Effects: reads research/native artifacts and writes reports/derived outputs.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Inspected arguments: `p.add_argument('--url', default='http://localhost:8080')`; `p.add_argument('--output', type=Path, default=Path('processing/bimanual_mirror_v2/mapping_verification.json'))`.
- Literal paths/artifact names: `configs/bimanual_mirror_policy_v2.json`, `processing/bimanual_mirror_v2/historical_before.json`, `processing/bimanual_mirror_v2/mapping_verification.json`, `processing/data_expansion_v1/manifest.json`, `processing/storage_v1/manifest.json`.
- Static imports: `PIL`, `argparse`, `io`, `json`, `numpy`, `pathlib`, `tactilelatent.aligned_frames`, `tactilelatent.bimanual_mirror`, `tactilelatent.mirror_preparation`, `torch`, `urllib.parse`, `urllib.request`.

</details>

<details>
<summary><strong>scripts/verify_joint_acquisitions.py</strong> — Fresh teacher/cache parity, all-source validation, and held-out exclusion.</summary>

- Status: **specialized operational / historical utility**; uncertain readiness: purpose inspected; historical defaults require review.
- Source: [scripts/verify_joint_acquisitions.py](../scripts/verify_joint_acquisitions.py).
- Inputs: Original all-source config v1 and new acquisition teachers/codes.
- Outputs: Verification report from fresh-code parity and GPU inference with random task weights; no optimizer or full joint run.
- Effects: GPU backbone/teacher inference and reconstruction validation; writes verification reports; no optimizer updates.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Caution: Uses a random task model for integration validation; its metrics do not measure trained model quality.
- Literal paths/artifact names: `configs/joint_vjepa21_all_sources_32d_mirror.json`, `verification.json`.
- Static imports: `json`, `numpy`, `pathlib`, `tactilelatent`, `tactilelatent.acquisition_data`, `tactilelatent.acquisition_joint`, `tactilelatent.acquisition_training`, `tactilelatent.aligned_frames`, `tactilelatent.expansion_runtime`, `tactilelatent.marker_data`, `tactilelatent.vjepa_model`, `torch`.

</details>

### diagnostic / exploratory retained (24)

<details>
<summary><strong>scripts/analyze_marker_dynamics.py</strong> — Episode bootstraps and long-trajectory summaries of the frozen audit.</summary>

- Status: **diagnostic / exploratory retained**; purpose verified statically; rerun resource availability not established.
- Source: [scripts/analyze_marker_dynamics.py](../scripts/analyze_marker_dynamics.py).
- Inputs: Existing native data/checkpoints/results selected by the literals and arguments below.
- Outputs: Reports, derived artifacts or configured run outputs named in source/CLI below; full data availability unverified.
- Effects: sends process signals.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Literal paths/artifact names: `clip_metrics.json`, `extended_summary.json`, `freetacman_traces.json`, `processing/research/marker_dynamics_v1`, `task_coverage.json`.
- Path-sensitive code: line9: `from audit_marker_dynamics import summarize`.
- Static imports: `audit_marker_dynamics`, `collections`, `json`, `numpy`, `pathlib`, `tactilelatent.aligned_frames`.

</details>

<details>
<summary><strong>scripts/audit_baihu_autoencoder_repair.py</strong> — Compare frozen Baihu teachers and ablate frame codes on the same episodes.</summary>

- Status: **diagnostic / exploratory retained**; purpose verified statically; rerun resource availability not established.
- Source: [scripts/audit_baihu_autoencoder_repair.py](../scripts/audit_baihu_autoencoder_repair.py).
- Inputs: Existing native data/checkpoints/results selected by the literals and arguments below.
- Outputs: Reports, derived artifacts or configured run outputs named in source/CLI below; full data availability unverified.
- Effects: reads research/native artifacts and writes reports/derived outputs.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Literal paths/artifact names: `full_validation_comparison.json`, `processing/audits/baihu_autoencoder_investigation_v1`, `processing/data_expansion_v1/teachers/baihu_native_v2/best.pt`, `processing/data_expansion_v1/teachers/baihu_native_v3/best.pt`.
- Static imports: `collections`, `json`, `numpy`, `pathlib`, `tactilelatent.aligned_frames`, `tactilelatent.baihu_ae_data`, `tactilelatent.expansion_models`, `tactilelatent.expansion_training`, `torch`.

</details>

<details>
<summary><strong>scripts/audit_bimanual_mirror_mappings.py</strong> — Read-only audit of native geometry; this never promotes a mirror mapping.</summary>

- Status: **diagnostic / exploratory retained**; purpose verified statically; rerun resource availability not established.
- Source: [scripts/audit_bimanual_mirror_mappings.py](../scripts/audit_bimanual_mirror_mappings.py).
- Inputs: Existing native data/checkpoints/results selected by the literals and arguments below.
- Outputs: Configured destination: parser.add_argument('--output', default='processing/audits/bimanual_mirror_mapping_v2').
- Effects: reads research/native artifacts and writes reports/derived outputs.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Provenance: Hashes own implementation at runtime; retain bytes/location.
- Inspected arguments: `parser.add_argument('--output', default='processing/audits/bimanual_mirror_mapping_v2')`.
- Literal paths/artifact names: `*.json`, `audit.json`, `complete.json`, `configs/bimanual_mirror_policy_v1.json`, `datasets/OpenGraphLabs-Research__ego-tactile-manipulation`, `meta/info.json`, `meta/opengraph/episode_manifest.json`, `native_episode_checks.json`, `processing/audits/bimanual_mirror_mapping_v2`, `processing/bimanual_mirror_v1/codes/egotouch/index.json`, `processing/bimanual_mirror_v1/ready.json`, `processing/bimanual_mirror_v1/teachers/egotouch/best.pt`, `processing/data_expansion_v1/baihu_available.json`, `processing/marker_displacement_storage_v1/targets`, `processing/storage_v1/manifest.json`, `status.json`, `tracks.npz`.
- Path-sensitive code: line228: `script_sha256=file_hash(__file__))`.
- Static imports: `argparse`, `collections`, `concurrent.futures`, `json`, `numpy`, `pathlib`, `pyarrow.parquet`, `tactilelatent.aligned_frames`, `tactilelatent.bimanual_mirror`, `tactilelatent.marker_data`, `tactilelatent.mirror_preparation`, `time`, `torch`.

</details>

<details>
<summary><strong>scripts/audit_compressed_vjepa_predictions.py</strong> — Small paired source-validation check of compression with a frozen checkpoint.</summary>

- Status: **diagnostic / exploratory retained**; purpose verified statically; rerun resource availability not established.
- Source: [scripts/audit_compressed_vjepa_predictions.py](../scripts/audit_compressed_vjepa_predictions.py).
- Inputs: Existing native data/checkpoints/results selected by the literals and arguments below.
- Outputs: Reports, derived artifacts or configured run outputs named in source/CLI below; full data availability unverified.
- Effects: reads research/native artifacts and writes reports/derived outputs.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Literal paths/artifact names: `configs/joint_vjepa21_spatial_marker_codes_32d.json`, `prediction_compression_check.json`, `processing/research/visual_resize`, `processing/runs/joint_vjepa21_spatial_marker_codes_32d_v1/epoch_009.pt`.
- Static imports: `av`, `io`, `json`, `numpy`, `pathlib`, `tactilelatent.aligned_frames`, `tactilelatent.marker_code_model`, `tactilelatent.marker_data`, `tactilelatent.marker_targets`, `tactilelatent.train_vjepa_markers`, `tactilelatent.vjepa_model`, `time`, `torch`.

</details>

<details>
<summary><strong>scripts/audit_ftp_extra_schema.py</strong> — Summarize paired visual/tactile arrays in indexed FTP-1 Zarr groups.</summary>

- Status: **diagnostic / exploratory retained**; purpose verified statically; rerun resource availability not established.
- Source: [scripts/audit_ftp_extra_schema.py](../scripts/audit_ftp_extra_schema.py).
- Inputs: Existing native data/checkpoints/results selected by the literals and arguments below.
- Outputs: Configured destination: parser.add_argument('--output', default='processing/audits/ftp_extra_schema.json').
- Effects: reads research/native artifacts and writes reports/derived outputs.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Inspected arguments: `parser.add_argument('groups', nargs='+')`; `parser.add_argument('--output', default='processing/audits/ftp_extra_schema.json')`; `parser.add_argument('--sample-frames', action='store_true')`.
- Literal paths/artifact names: `.json`, `.tar`, `datasets/MJJJJ1064__FTP-1-Dataset`, `processing/audits/ftp_extra_schema.json`, `processing/indexes`.
- Static imports: `argparse`, `collections`, `json`, `numpy`, `pathlib`, `tactilelatent.tar_zarr`, `zarr`.

</details>

<details>
<summary><strong>scripts/audit_joint_sources.py</strong> — Verify the six selected local dataset scopes against pinned HF revisions.</summary>

- Status: **diagnostic / exploratory retained**; purpose verified statically; rerun resource availability not established.
- Source: [scripts/audit_joint_sources.py](../scripts/audit_joint_sources.py).
- Inputs: Existing native data/checkpoints/results selected by the literals and arguments below.
- Outputs: Configured destination: parser.add_argument('--output', default='processing/audits/joint_six_integrity.json').
- Effects: network access / possible downloads.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Inspected arguments: `parser.add_argument('--hash', action='store_true', help='hash all LFS files, not just check size')`; `parser.add_argument('--output', default='processing/audits/joint_six_integrity.json')`.
- Literal paths/artifact names: `*/*/*/chest.mp4`, `*/*/*/left.mp4`, `*/*/*/pressure_grids.npz`, `*/*/*/right.mp4`, `datasets/EgoTouch`, `datasets/HustleHard__EgoTactile`, `datasets/MJJJJ1064__FTP-1-Dataset`, `datasets/OpenDriveLab__FreeTacMan`, `datasets/OpenGraphLabs-Research__ego-tactile-manipulation`, `datasets/rayxsong__opentouch`, `processing/audits/joint_six_integrity.json`, `split.json`.
- Referenced by: `src/tactilelatent/prepare_joint.py`.
- Static imports: `argparse`, `fnmatch`, `hashlib`, `huggingface_hub`, `json`, `pathlib`, `time`.

</details>

<details>
<summary><strong>scripts/audit_marker_dynamics.py</strong> — Read-only dynamics audit of frozen marker-code visual models and teachers.</summary>

- Status: **diagnostic / exploratory retained**; purpose verified statically; rerun resource availability not established.
- Source: [scripts/audit_marker_dynamics.py](../scripts/audit_marker_dynamics.py).
- Inputs: Existing native data/checkpoints/results selected by the literals and arguments below.
- Outputs: Reports, derived artifacts or configured run outputs named in source/CLI below; full data availability unverified.
- Effects: reads research/native artifacts and writes reports/derived outputs.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Provenance: Hashes own implementation at runtime; retain bytes/location.
- Literal paths/artifact names: `.npz`, `.pt`, `clip_metrics.json`, `freetacman_traces.json`, `processing/research/marker_dynamics_v1`, `processing/runs/joint_vjepa21_spatial_marker_codes_32d_v1`, `provenance.json`, `status.json`, `summary.json`.
- Path-sensitive code: line103: `script_sha256=file_hash(__file__),data=states[30]['provenance']['data'],teacher_registry_sha256=file_hash(cfg['marker_teachers']),`.
- Static imports: `PIL`, `collections`, `json`, `matplotlib`, `matplotlib.pyplot`, `numpy`, `os`, `pathlib`, `tactilelatent.aligned_frames`, `tactilelatent.alignment_data`, `tactilelatent.frozen_probe`, `tactilelatent.marker_autoencoder`, `tactilelatent.marker_code_model`, `tactilelatent.marker_data`, `tactilelatent.marker_targets`, `tactilelatent.sensor_code_cache`, `tactilelatent.vjepa_model`, `tactilelatent.vjepa_spatial`, `time`, `torch`, `torch.utils.data`.

</details>

<details>
<summary><strong>scripts/audit_marker_temporal_inputs.py</strong> — Temporal-input and shifted-window checks of frozen FreeTacMan prediction.</summary>

- Status: **diagnostic / exploratory retained**; purpose verified statically; rerun resource availability not established.
- Source: [scripts/audit_marker_temporal_inputs.py](../scripts/audit_marker_temporal_inputs.py).
- Inputs: Existing native data/checkpoints/results selected by the literals and arguments below.
- Outputs: Reports, derived artifacts or configured run outputs named in source/CLI below; full data availability unverified.
- Effects: reads research/native artifacts and writes reports/derived outputs.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Literal paths/artifact names: `temporal_input_checks.json`, `viewer_epochs/epoch_030.pt`.
- Path-sensitive code: line16: `from audit_marker_dynamics import ROOT, RUN, sums, summarize`.
- Static imports: `audit_marker_dynamics`, `json`, `numpy`, `pathlib`, `tactilelatent.aligned_frames`, `tactilelatent.alignment_data`, `tactilelatent.marker_autoencoder`, `tactilelatent.marker_code_model`, `tactilelatent.marker_data`, `tactilelatent.vjepa_model`, `tactilelatent.vjepa_spatial`, `torch`, `torch.utils.data`.

</details>

<details>
<summary><strong>scripts/audit_mirror_previews.py</strong> — Exercise real native target hypotheses without enabling augmentation.</summary>

- Status: **diagnostic / exploratory retained**; purpose verified statically; rerun resource availability not established.
- Source: [scripts/audit_mirror_previews.py](../scripts/audit_mirror_previews.py).
- Inputs: Existing native data/checkpoints/results selected by the literals and arguments below.
- Outputs: Configured destination: parser.add_argument('--output', type=Path, default=Path('processing/audits/bimanual_mirror_mapping_v3')).
- Effects: reads research/native artifacts and writes reports/derived outputs.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Inspected arguments: `parser.add_argument('--url', default='http://127.0.0.1:8080')`; `parser.add_argument('--output', type=Path, default=Path('processing/audits/bimanual_mirror_mapping_v3'))`.
- Literal paths/artifact names: `configs/bimanual_mirror_policy_v1.json`, `native_preview_checks.json`, `processing/audits/bimanual_mirror_mapping_v3`, `processing/bimanual_mirror_v1/teachers/egotouch/quality.json`, `processing/data_expansion_v1/baihu_available.json`, `processing/storage_v1/implementation.json`, `processing/storage_v1/manifest.json`.
- Static imports: `PIL`, `argparse`, `io`, `json`, `numpy`, `pathlib`, `tactilelatent.aligned_frames`, `tactilelatent.bimanual_mirror`, `tactilelatent.mirror_mapping_review`, `tactilelatent.mirror_preparation`, `torch`, `urllib.error`, `urllib.parse`, `urllib.request`.

</details>

<details>
<summary><strong>scripts/audit_opentouch_format.py</strong> — Read-only comparison of OpenTouch HDF5, loaders, caches and upstream layout.</summary>

- Status: **diagnostic / exploratory retained**; purpose verified statically; rerun resource availability not established.
- Source: [scripts/audit_opentouch_format.py](../scripts/audit_opentouch_format.py).
- Inputs: Existing native data/checkpoints/results selected by the literals and arguments below.
- Outputs: Reports, derived artifacts or configured run outputs named in source/CLI below; full data availability unverified.
- Effects: reads research/native artifacts and writes reports/derived outputs.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Literal paths/artifact names: `audit.json`, `examples.json`, `handLayoutNewest_meshid.json`, `index.json`, `processing/audits/opentouch_format`, `processing/joint_alignment/manifest.json`, `processing/joint_alignment/teachers.json`, `processing/sensor_ae_cache_native_v2`, `processing/sensor_ae_cache_v1`, `processing/sensor_code_cache_v1`, `upstream_tree.json`.
- Static imports: `collections`, `h5py`, `json`, `numpy`, `pathlib`, `tactilelatent.aligned_frames`, `tactilelatent.sensor_autoencoder`, `tactilelatent.sensor_code_cache`, `time`, `torch`.

</details>

<details>
<summary><strong>scripts/audit_saved_visual_resize.py</strong> — Read-only sample audit of 384-pixel visual inputs across the saved manifest.</summary>

- Status: **diagnostic / exploratory retained**; purpose verified statically; rerun resource availability not established.
- Source: [scripts/audit_saved_visual_resize.py](../scripts/audit_saved_visual_resize.py).
- Inputs: Existing native data/checkpoints/results selected by the literals and arguments below.
- Outputs: Reports, derived artifacts or configured run outputs named in source/CLI below; full data availability unverified.
- Effects: reads research/native artifacts and writes reports/derived outputs.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Literal paths/artifact names: `processing/joint_alignment/manifest.json`, `processing/research/visual_resize`, `sample_benchmark.json`.
- Static imports: `PIL`, `av`, `collections`, `h5py`, `io`, `json`, `numpy`, `pathlib`, `statistics`, `tactilelatent.aligned_frames`, `tactilelatent.sensor_data`, `time`, `torch`.

</details>

<details>
<summary><strong>scripts/audit_saved_visual_video.py</strong> — Small in-memory H.264 resize/storage pilot; does not rewrite source datasets.</summary>

- Status: **diagnostic / exploratory retained**; purpose verified statically; rerun resource availability not established.
- Canonical historical body: [scripts/archive/audit_saved_visual_video.py](../scripts/archive/audit_saved_visual_video.py); original command is a compatibility shim.
- Inputs: Existing native data/checkpoints/results selected by the literals and arguments below.
- Outputs: Reports, derived artifacts or configured run outputs named in source/CLI below; full data availability unverified.
- Effects: reads research/native artifacts and writes reports/derived outputs.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Literal paths/artifact names: `processing/joint_alignment/manifest.json`, `processing/research/visual_resize`, `video_benchmark.json`.
- Static imports: `av`, `collections`, `io`, `json`, `numpy`, `pathlib`, `statistics`, `tactilelatent.aligned_frames`, `time`, `torch`.

</details>

<details>
<summary><strong>scripts/audit_visual_video_quality.py</strong> — Measure RGB reconstruction error and true lossless-video sizes on real clips.</summary>

- Status: **diagnostic / exploratory retained**; purpose verified statically; rerun resource availability not established.
- Canonical historical body: [scripts/archive/audit_visual_video_quality.py](../scripts/archive/audit_visual_video_quality.py); original command is a compatibility shim.
- Inputs: Existing native data/checkpoints/results selected by the literals and arguments below.
- Outputs: Reports, derived artifacts or configured run outputs named in source/CLI below; full data availability unverified.
- Effects: reads research/native artifacts and writes reports/derived outputs.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Literal paths/artifact names: `processing/joint_alignment/manifest.json`, `processing/research/visual_resize`, `quality_lossless_benchmark.json`.
- Static imports: `av`, `collections`, `io`, `json`, `numpy`, `pathlib`, `tactilelatent.aligned_frames`, `time`, `torch`.

</details>

<details>
<summary><strong>scripts/audit_vla_touch_dataset_order.py</strong> — Recover tactile chronology by exact pixels, preserving the FTP archives.</summary>

- Status: **diagnostic / exploratory retained**; purpose verified statically; rerun resource availability not established.
- Source: [scripts/audit_vla_touch_dataset_order.py](../scripts/audit_vla_touch_dataset_order.py).
- Inputs: Existing native data/checkpoints/results selected by the literals and arguments below.
- Outputs: Configured destination: parser.add_argument('--output', type=Path, default=Path('processing/audits/vla_touch_native_dynamics/dataset_frame_order.json')).
- Effects: reads research/native artifacts and writes reports/derived outputs.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Inspected arguments: `parser.add_argument('--manifest', type=Path, default=Path('processing/joint_all/manifest_sensor_roles_corrected_v2.json'))`; `parser.add_argument('--raw-root', type=Path, default=Path('datasets/AllenBi21__VLA-Touch/raw_data'))`; `parser.add_argument('--output', type=Path, default=Path('processing/audits/vla_touch_native_dynamics/dataset_frame_order.json'))`.
- Literal paths/artifact names: `datasets/AllenBi21__VLA-Touch/raw_data`, `processing/audits/vla_touch_native_dynamics/dataset_frame_order.json`, `processing/joint_all/manifest_sensor_roles_corrected_v2.json`.
- Static imports: `PIL`, `argparse`, `collections`, `hashlib`, `io`, `json`, `numpy`, `pathlib`, `re`, `tactilelatent.native_optical`, `tactilelatent.sensor_data`, `time`, `zipfile`.

</details>

<details>
<summary><strong>scripts/audit_vla_touch_order.py</strong> — Match raw numbered GelSight JPEGs to the authors' processed H5 frames.</summary>

- Status: **diagnostic / exploratory retained**; purpose verified statically; rerun resource availability not established.
- Source: [scripts/audit_vla_touch_order.py](../scripts/audit_vla_touch_order.py).
- Inputs: Existing native data/checkpoints/results selected by the literals and arguments below.
- Outputs: Configured destination: parser.add_argument('--output', type=Path, default=Path('processing/audits/vla_touch_native_dynamics')).
- Effects: network access / possible downloads.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Inspected arguments: `parser.add_argument('--url', default='https://huggingface.co/datasets/AllenBi21/VLA-Touch/resolve/main/raw_data/water_cup.zip')`; `parser.add_argument('--size', type=int, default=1286454671)`; `parser.add_argument('--local-zip', type=Path)`; `parser.add_argument('--episode', type=int, default=0)`; `parser.add_argument('--h5', type=Path, default=Path('processing/audits/vla_touch_native_dynamics/upstream_watercup_episode_0.h5'))`; `parser.add_argument('--output', type=Path, default=Path('processing/audits/vla_touch_native_dynamics'))`.
- Literal paths/artifact names: `processing/audits/vla_touch_native_dynamics`, `processing/audits/vla_touch_native_dynamics/upstream_watercup_episode_0.h5`, `raw_order_comparison.json`.
- Static imports: `PIL`, `argparse`, `h5py`, `hashlib`, `io`, `json`, `numpy`, `pathlib`, `re`, `urllib.request`, `zipfile`.

</details>

<details>
<summary><strong>scripts/compare_alternating_baseline.py</strong> — Summarize the completed alternating run against the immutable baseline.</summary>

- Status: **diagnostic / exploratory retained**; purpose verified statically; rerun resource availability not established.
- Source: [scripts/compare_alternating_baseline.py](../scripts/compare_alternating_baseline.py).
- Inputs: Existing native data/checkpoints/results selected by the literals and arguments below.
- Outputs: Reports, derived artifacts or configured run outputs named in source/CLI below; full data availability unverified.
- Effects: reads research/native artifacts and writes reports/derived outputs.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Literal paths/artifact names: `baseline_comparison.json`, `evaluations/dataset_identity/results.json`, `evaluations/vla_touch/results.json`, `processing/evaluations/baselines/joint_50ep_vla_touch_v1`, `processing/evaluations/dataset_identity_probe/results.json`, `processing/evaluations/vla_touch_frozen_decoder/results.json`, `processing/runs/joint_all_alternating_fresh_8k_50ep`.
- Referenced by: `scripts/train_joint_alternating_fresh.sh`.
- Static imports: `json`, `pathlib`.

</details>

<details>
<summary><strong>scripts/compare_current_vjepa_sources.py</strong> — Compare an old eight-output model and a current-frame model on identical inputs.</summary>

- Status: **diagnostic / exploratory retained**; purpose verified statically; rerun resource availability not established.
- Source: [scripts/compare_current_vjepa_sources.py](../scripts/compare_current_vjepa_sources.py).
- Inputs: Existing native data/checkpoints/results selected by the literals and arguments below.
- Outputs: Configured destination: parser.add_argument('--output', type=Path, required=True).
- Effects: reads research/native artifacts and writes reports/derived outputs.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Inspected arguments: `parser.add_argument('--epoch', type=int, default=6)`; `parser.add_argument('--output', type=Path, required=True)`; `parser.add_argument('--extra', action='append', default=[], metavar='NAME=RUN@EPOCH', help='Additional marker-code checkpoint scored on the same clips')`.
- Literal paths/artifact names: `.pt`, `processing/runs/joint_vjepa21_spatial_current128d_v1`, `processing/runs/joint_vjepa21_spatial_marker_codes_32d_v1`, `results.json`, `status.json`.
- Static imports: `argparse`, `collections`, `json`, `pathlib`, `tactilelatent.aligned_frames`, `tactilelatent.joint_data`, `tactilelatent.marker_code_model`, `tactilelatent.marker_data`, `tactilelatent.marker_targets`, `tactilelatent.train_alignment`, `tactilelatent.train_vjepa_markers`, `tactilelatent.vjepa_model`, `tactilelatent.vjepa_spatial`, `time`, `torch`, `torch.utils.data`.

</details>

<details>
<summary><strong>scripts/compare_sam_motiontrans.py</strong> — Compare matched final-epoch frozen MotionTrans decoder evaluations.</summary>

- Status: **diagnostic / exploratory retained**; purpose verified statically; rerun resource availability not established.
- Canonical historical body: [scripts/archive/compare_sam_motiontrans.py](../scripts/archive/compare_sam_motiontrans.py); original command is a compatibility shim.
- Inputs: Existing native data/checkpoints/results selected by the literals and arguments below.
- Outputs: Reports, derived artifacts or configured run outputs named in source/CLI below; full data availability unverified.
- Effects: reads research/native artifacts and writes reports/derived outputs.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Literal paths/artifact names: `non_sam_comparison.json`, `processing/runs`, `results.json`.
- Static imports: `json`, `pathlib`.

</details>

<details>
<summary><strong>scripts/compare_vla_touch_three_way.py</strong> — Compare direct VLA_touch training with both frozen-encoder adaptations.</summary>

- Status: **diagnostic / exploratory retained**; purpose verified statically; rerun resource availability not established.
- Source: [scripts/compare_vla_touch_three_way.py](../scripts/compare_vla_touch_three_way.py).
- Inputs: Existing native data/checkpoints/results selected by the literals and arguments below.
- Outputs: Reports, derived artifacts or configured run outputs named in source/CLI below; full data availability unverified.
- Effects: reads research/native artifacts and writes reports/derived outputs.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Literal paths/artifact names: `evaluation/results.json`, `evaluation/three_way_comparison.json`, `evaluations/vla_touch/results.json`, `processing/evaluations/baselines/joint_50ep_vla_touch_v1`, `processing/evaluations/vla_touch_frozen_decoder/results.json`, `processing/runs/joint_all_alternating_fresh_8k_50ep`, `processing/runs/vla_touch_direct_vae32_50ep`.
- Referenced by: `scripts/train_vla_touch_direct.sh`.
- Static imports: `json`, `pathlib`.

</details>

<details>
<summary><strong>scripts/diagnose_vjepa_learning.py</strong> — Bounded source-only diagnostics on copies; no production training resumes.</summary>

- Status: **diagnostic / exploratory retained**; purpose verified statically; rerun resource availability not established.
- Source: [scripts/diagnose_vjepa_learning.py](../scripts/diagnose_vjepa_learning.py).
- Inputs: Source/config/checkpoint references and CLI inputs below; delegated loaders resolve remaining native paths.
- Outputs: Reports, derived artifacts or configured run outputs named in source/CLI below; full data availability unverified.
- Effects: GPU optimization or fitted probes; TOP-LEVEL execution on import; never probe with --help.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Caution: Import immediately runs GPU diagnostic fits on copies, not just analysis; old production model is not modified.
- Literal paths/artifact names: `configs/joint_vjepa21_pair_32d_cached_v1.json`, `epoch_000.pt`, `epoch_030.pt`, `partial.json`, `processing/diagnostics/vjepa_learning_v1`, `processing/runs/joint_alignment_holdout_motiontrans/viewer_epochs/epoch_030.pt`, `results.json`.
- Static imports: `collections`, `json`, `numpy`, `pathlib`, `tactilelatent.aligned_frames`, `tactilelatent.alignment_model`, `tactilelatent.joint_data`, `tactilelatent.sam`, `tactilelatent.train_alignment`, `tactilelatent.train_vjepa`, `tactilelatent.vjepa_cache`, `tactilelatent.vjepa_model`, `time`, `torch`, `torch.utils.data`.

</details>

<details>
<summary><strong>scripts/investigate_baihu_autoencoder.py</strong> — Bounded matched diagnostics for Baihu's collapsed native marker AE.</summary>

- Status: **diagnostic / exploratory retained**; purpose verified statically; rerun resource availability not established.
- Source: [scripts/investigate_baihu_autoencoder.py](../scripts/investigate_baihu_autoencoder.py).
- Inputs: Source/config/checkpoint references and CLI inputs below; delegated loaders resolve remaining native paths.
- Outputs: Configured destination: p.add_argument('--output', type=Path, default=Path('processing/audits/baihu_autoencoder_investigation_v1')).
- Effects: GPU optimization or fitted probes.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Inspected arguments: `p.add_argument('--output', type=Path, default=Path('processing/audits/baihu_autoencoder_investigation_v1'))`; `p.add_argument('--epochs', type=int, default=8)`; `p.add_argument('--overfit-steps', type=int, default=400)`.
- Literal paths/artifact names: `.json`, `.pt`, `pilot_results.json`, `pilot_sampling.json`, `processing/audits/baihu_autoencoder_investigation_v1`, `processing/data_expansion_v1/baihu_available.json`.
- Static imports: `argparse`, `collections`, `copy`, `json`, `numpy`, `pathlib`, `tactilelatent.aligned_frames`, `tactilelatent.baihu_autoencoder`, `tactilelatent.data_expansion`, `tactilelatent.expansion_models`, `time`, `torch`.

</details>

<details>
<summary><strong>scripts/smoke_joint_alignment.py</strong> — Real-data smoke: native pairing -> teacher codes -> DINO -> joint SAM step.</summary>

- Status: **diagnostic / exploratory retained**; purpose verified statically; rerun resource availability not established.
- Source: [scripts/smoke_joint_alignment.py](../scripts/smoke_joint_alignment.py).
- Inputs: Source/config/checkpoint references and CLI inputs below; delegated loaders resolve remaining native paths.
- Outputs: Configured destination: parser.add_argument('--output', default='processing/joint_alignment/smoke').
- Effects: GPU optimization or fitted probes.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Inspected arguments: `parser.add_argument('--device', default='cuda')`; `parser.add_argument('--output', default='processing/joint_alignment/smoke')`.
- Literal paths/artifact names: `audit.json`, `config.json`, `configs/joint_alignment_holdout_motiontrans.json`, `last.pt`, `manifest.json`, `processing/joint_alignment/smoke`, `result.json`.
- Static imports: `argparse`, `json`, `pathlib`, `subprocess`, `sys`, `tactilelatent.aligned_frames`, `tactilelatent.audit_alignment`, `tactilelatent.sensor_code_cache`, `torch`.

</details>

<details>
<summary><strong>scripts/smoke_joint_six.py</strong> — Decode representative train/val clips and run one six-head VAE forward.</summary>

- Status: **diagnostic / exploratory retained**; purpose verified statically; rerun resource availability not established.
- Source: [scripts/smoke_joint_six.py](../scripts/smoke_joint_six.py).
- Inputs: Source/config/checkpoint references and CLI inputs below; delegated loaders resolve remaining native paths.
- Outputs: Configured destination: parser.add_argument('--output', default='processing/audits/joint_six_smoke.json').
- Effects: reads research/native artifacts and writes reports/derived outputs.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Inspected arguments: `parser.add_argument('--samples-per-source', type=int, default=20)`; `parser.add_argument('--output', default='processing/audits/joint_six_smoke.json')`.
- Literal paths/artifact names: `processing/audits/joint_six_smoke.json`, `processing/runs/egotouch_dinov3_vae_image_only_latent32/last.pt`.
- Static imports: `argparse`, `json`, `numpy`, `pathlib`, `tactilelatent.joint_data`, `tactilelatent.joint_model`, `tactilelatent.model`, `tactilelatent.prepare_joint`, `tactilelatent.train`, `torch`.

</details>

<details>
<summary><strong>scripts/smoke_motiontrans_evaluation.py</strong> — Exercise both MotionTrans evaluation paths with real smoke checkpoints.</summary>

- Status: **diagnostic / exploratory retained**; purpose verified statically; rerun resource availability not established.
- Source: [scripts/smoke_motiontrans_evaluation.py](../scripts/smoke_motiontrans_evaluation.py).
- Inputs: Source/config/checkpoint references and CLI inputs below; delegated loaders resolve remaining native paths.
- Outputs: Reports, derived artifacts or configured run outputs named in source/CLI below; full data availability unverified.
- Effects: GPU optimization or fitted probes.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Literal paths/artifact names: `/last.pt`, `_features.pt`, `processing/joint_all/manifest.json`, `processing/runs/`, `processing/runs/motiontrans_baseline_sequence/evaluation_smoke.json`.
- Static imports: `json`, `pathlib`, `tactilelatent.evaluate_direct_vla_touch`, `tactilelatent.evaluate_holdout_decoder`, `tactilelatent.frozen_probe`, `tactilelatent.joint_data`, `torch`.

</details>

### historical experiment / compatibility retained (66)

<details>
<summary><strong>scripts/benchmark_vjepa_pair.py</strong> — Time complete frozen-feature + SAM training updates on matched real clips.</summary>

- Status: **historical experiment / compatibility retained**; inferred from targeted historical configs/runs; not classified as disposable.
- Source: [scripts/benchmark_vjepa_pair.py](../scripts/benchmark_vjepa_pair.py).
- Inputs: Source/config/checkpoint references and CLI inputs below; delegated loaders resolve remaining native paths.
- Outputs: Reports, derived artifacts or configured run outputs named in source/CLI below; full data availability unverified.
- Effects: GPU optimization or fitted probes.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Literal paths/artifact names: `configs/joint_vjepa21_pair_32d_v1.json`, `processing/vjepa2_1/benchmark.json`.
- Static imports: `gc`, `json`, `pathlib`, `tactilelatent.aligned_frames`, `tactilelatent.alignment_data`, `tactilelatent.alignment_model`, `tactilelatent.model`, `tactilelatent.train`, `tactilelatent.train_alignment`, `tactilelatent.train_vjepa`, `tactilelatent.vjepa_model`, `time`, `torch`, `torch.utils.data`.

</details>

<details>
<summary><strong>scripts/cache_vjepa_clips.sh</strong> — Launch the original pooled V-JEPA clip-feature cache producer.</summary>

- Status: **historical experiment / compatibility retained**; inferred from targeted historical configs/runs; not classified as disposable.
- Source: [scripts/cache_vjepa_clips.sh](../scripts/cache_vjepa_clips.sh).
- Inputs: Original pooled-clip cache CLI manifest/config and pretrained V-JEPA.
- Outputs: Pooled per-clip visual feature partitions; incompatible with no-spatial-pooling current path.
- Effects: reads research/native artifacts and writes reports/derived outputs.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Literal paths/artifact names: `processing/runtime/nvidia-580.173.02/root/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH`.
- Static imports: `tactilelatent.vjepa_cache`.

</details>

<details>
<summary><strong>scripts/download_egotouch.sh</strong> — Download the pinned original EgoTouch subset, including hand-pose metadata.</summary>

- Status: **historical experiment / compatibility retained**; inferred from targeted historical configs/runs; not classified as disposable.
- Source: [scripts/download_egotouch.sh](../scripts/download_egotouch.sh).
- Inputs: Pinned zhouzhoujy/EgoTouch release allowlist.
- Outputs: datasets/EgoTouch and download_logs; can reacquire assets intentionally removed later.
- Effects: network access / possible downloads.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Caution: Historical allowlist includes wilor_hands.json; rerunning can restore pose assets intentionally cleaned up.
- Literal paths/artifact names: `datasets/EgoTouch`.
- Static imports: `huggingface_hub`.

</details>

<details>
<summary><strong>scripts/download_feeling_success.py</strong> — Acquire all Feeling of Success grasps as a compact, evaluation-only copy.</summary>

- Status: **historical experiment / compatibility retained**; inferred from targeted historical configs/runs; not classified as disposable.
- Source: [scripts/download_feeling_success.py](../scripts/download_feeling_success.py).
- Inputs: Official Feeling of Success archive/index.
- Outputs: Compact datasets/evaluation_only/feeling_of_success plus evaluation manifest/status.
- Effects: network access / possible downloads.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Inspected arguments: `parser.add_argument('--output', default='datasets/evaluation_only/feeling_of_success')`.
- Literal paths/artifact names: `datasets/evaluation_only/feeling_of_success`, `feeling_of_success.json`, `manifest.json`, `processing/evaluation_datasets`, `source.json`, `status.json`.
- Path-sensitive code: line29: `ROOT = Path(__file__).resolve().parents[1]`; line217: `os.chdir(ROOT)`.
- Referenced by: `scripts/download_feeling_success.sh`, `tests/test_feeling_success_download.py`.
- Static imports: `argparse`, `h5py`, `io`, `json`, `numpy`, `os`, `pathlib`, `shutil`, `struct`, `subprocess`, `tactilelatent.frozen_probe`, `time`, `urllib.request`, `zipfile`, `zlib`.

</details>

<details>
<summary><strong>scripts/download_feeling_success.sh</strong> — Run the evaluation-only Feeling of Success downloader under an exclusive lock.</summary>

- Status: **historical experiment / compatibility retained**; inferred from targeted historical configs/runs; not classified as disposable.
- Source: [scripts/download_feeling_success.sh](../scripts/download_feeling_success.sh).
- Inputs: Source/config/checkpoint references and CLI inputs below; delegated loaders resolve remaining native paths.
- Outputs: Reports, derived artifacts or configured run outputs named in source/CLI below; full data availability unverified.
- Effects: network access / possible downloads.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Literal paths/artifact names: `datasets/evaluation_only/feeling_of_success`, `datasets/evaluation_only/feeling_of_success/download.lock`.

</details>

<details>
<summary><strong>scripts/download_ftp_extra.py</strong> — Download complete selected FTP-1 archives without exceeding the data cap.</summary>

- Status: **historical experiment / compatibility retained**; inferred from targeted historical configs/runs; not classified as disposable.
- Source: [scripts/download_ftp_extra.py](../scripts/download_ftp_extra.py).
- Inputs: Pinned FTP-1 archive groups and aggregate storage budget.
- Outputs: Native datasets/MJJJJ1064__FTP-1-Dataset archives plus download report.
- Effects: network access / possible downloads.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Inspected arguments: `parser.add_argument('groups', nargs='+', choices=GROUPS)`; `parser.add_argument('--workers', type=int, default=3)`; `parser.add_argument('--report', default='processing/audits/ftp_extra_download.json')`.
- Literal paths/artifact names: `.tar`, `datasets/MJJJJ1064__FTP-1-Dataset`, `processing/audits/ftp_extra_download.json`.
- Static imports: `argparse`, `concurrent.futures`, `hashlib`, `huggingface_hub`, `json`, `pathlib`, `time`.

</details>

<details>
<summary><strong>scripts/evaluate_16d_pair.py</strong> — Evaluate the matched 16D pair on MotionTrans at one completed epoch.</summary>

- Status: **historical experiment / compatibility retained**; inferred from targeted historical configs/runs; not classified as disposable.
- Source: [scripts/evaluate_16d_pair.py](../scripts/evaluate_16d_pair.py).
- Inputs: Frozen historical model checkpoint(s), established MotionTrans episode split, native held-out data; see paths/arguments.
- Outputs: Configured destination: parser.add_argument('--root', default='processing/runs/joint_16d_pair_v1').
- Effects: GPU optimization or fitted probes; writes evaluation features/checkpoints/results; holdout protocol sensitive.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Inspected arguments: `parser.add_argument('--root', default='processing/runs/joint_16d_pair_v1')`; `parser.add_argument('--epoch', type=int, default=10)`.
- Literal paths/artifact names: `.log`, `processing/runs/joint_16d_pair_v1`, `status.json`, `summary.json`.
- Static imports: `argparse`, `fcntl`, `json`, `os`, `pathlib`, `subprocess`, `sys`, `tactilelatent.aligned_frames`, `time`.

</details>

<details>
<summary><strong>scripts/evaluate_alignment_checkpoint_sweep.py</strong> — Evaluate requested saved alignment epochs on held-out MotionTrans.</summary>

- Status: **historical experiment / compatibility retained**; inferred from targeted historical configs/runs; not classified as disposable.
- Source: [scripts/evaluate_alignment_checkpoint_sweep.py](../scripts/evaluate_alignment_checkpoint_sweep.py).
- Inputs: Frozen historical model checkpoint(s), established MotionTrans episode split, native held-out data; see paths/arguments.
- Outputs: Reports, derived artifacts or configured run outputs named in source/CLI below; full data availability unverified.
- Effects: GPU optimization or fitted probes; writes evaluation features/checkpoints/results; holdout protocol sensitive.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Provenance: Exact relative path/hash recorded in existing receipt(s).
- Inspected arguments: `parser.add_argument('--run', default='processing/runs/joint_alignment_holdout_motiontrans')`; `parser.add_argument('--epochs', type=int, nargs='+', help='Defaults to all saved epochs divisible by 10')`; `parser.add_argument('--report-name', default='motiontrans_checkpoint_sweep', help='Separate report folder within evaluations; individual head results remain reusable')`; `parser.add_argument('--head-epochs', type=int, default=100)`; `parser.add_argument('--patience', type=int, default=10)`; `parser.add_argument('--samples', type=int, default=64)`; `parser.add_argument('--seed', type=int, default=45)`; `parser.add_argument('--workers', type=int, default=4)`; `parser.add_argument('--checkpoint-kind', choices=('auto', 'viewer'), default='auto', help='Explicitly use viewer weights when sharing a precomputed snapshot/cache')`; `parser.add_argument('--heads', nargs='+', choices=tuple(HEADS), default=['shallow', 'deep'], help='Head presets; default retains the original one/two hidden layers')`.
- Literal paths/artifact names: `.pt`, `/evaluation.log`, `checkpoint.pt`, `decoder.pt`, `epoch_*.pt`, `evaluation.log`, `launch.json`, `metrics.jsonl`, `processing/runs/joint_alignment_holdout_motiontrans`, `results.json`, `status.json`, `summary.csv`, `summary.json`.
- Path-sensitive code: line126: `subprocess.run(['/usr/bin/python3', str(Path(__file__).with_name(`.
- Referenced by: `scripts/evaluate_16d_pair.py`, `scripts/evaluate_continuation_holdout.py`, `scripts/evaluate_invariance_holdout.py`, `scripts/run_continuation_study.py`, `tests/test_motiontrans_sweep.py`, `tests/test_paired_probe_cache.py`.
- Static imports: `argparse`, `csv`, `fcntl`, `json`, `numpy`, `os`, `pathlib`, `subprocess`, `sys`, `tactilelatent.aligned_frames`, `tactilelatent.frozen_probe`, `time`, `torch`.

</details>

<details>
<summary><strong>scripts/evaluate_continuation_holdout.py</strong> — Full MotionTrans probes with shared DINO extraction across continuation arms.</summary>

- Status: **historical experiment / compatibility retained**; inferred from targeted historical configs/runs; not classified as disposable.
- Source: [scripts/evaluate_continuation_holdout.py](../scripts/evaluate_continuation_holdout.py).
- Inputs: Frozen historical model checkpoint(s), established MotionTrans episode split, native held-out data; see paths/arguments.
- Outputs: Configured destination: parser.add_argument('--root', type=Path, default=Path('processing/runs/continuation_32d_v1')).
- Effects: GPU optimization or fitted probes; writes evaluation features/checkpoints/results; holdout protocol sensitive.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Provenance: Exact relative path/hash recorded in existing receipt(s).
- Inspected arguments: `parser.add_argument('--root', type=Path, default=Path('processing/runs/continuation_32d_v1'))`; `parser.add_argument('--epoch', type=int, required=True)`.
- Literal paths/artifact names: `.log`, `.pt`, `_features.pt`, `checkpoint.pt`, `evaluations/epoch_007/summary.json`, `partial_results.json`, `processing/runs/continuation_32d_v1`, `status.json`, `summary.json`.
- Path-sensitive code: line13: `from evaluate_invariance_holdout import paired_difference`.
- Referenced by: `scripts/finish_continuation_and_probe.py`, `scripts/run_continuation_study.py`.
- Static imports: `argparse`, `evaluate_invariance_holdout`, `gc`, `json`, `os`, `pathlib`, `subprocess`, `sys`, `tactilelatent.aligned_frames`, `tactilelatent.alignment_data`, `tactilelatent.alignment_model`, `tactilelatent.frozen_probe`, `tactilelatent.model`, `tactilelatent.paired_probe_cache`, `time`, `torch`.

</details>

<details>
<summary><strong>scripts/evaluate_current_sam_motiontrans.py</strong> — Snapshot a completed SAM epoch and evaluate without interrupting its trainer.</summary>

- Status: **historical experiment / compatibility retained**; inferred from targeted historical configs/runs; not classified as disposable.
- Source: [scripts/evaluate_current_sam_motiontrans.py](../scripts/evaluate_current_sam_motiontrans.py).
- Inputs: Frozen historical model checkpoint(s), established MotionTrans episode split, native held-out data; see paths/arguments.
- Outputs: Reports, derived artifacts or configured run outputs named in source/CLI below; full data availability unverified.
- Effects: GPU optimization or fitted probes; writes evaluation features/checkpoints/results; holdout protocol sensitive.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Literal paths/artifact names: `.pt`, `baseline_comparison.json`, `checkpoint.pt`, `current_motiontrans_evaluation.json`, `evaluation.log`, `evaluations/motiontrans/results.json`, `last.pt`, `processing/runs/joint_all_holdout_motiontrans_fresh_8k_50ep`, `processing/runs/joint_all_sam_holdout_motiontrans_fresh_8k_50ep`, `results.json`, `status.json`.
- Path-sensitive code: line15: `check_storage_budget(Path.cwd(), additional=1_000_000_000)`.
- Static imports: `json`, `os`, `pathlib`, `subprocess`, `sys`, `tactilelatent.sam`, `time`, `torch`.

</details>

<details>
<summary><strong>scripts/evaluate_egotouch_vae_uncertainty.sh</strong> — Evaluate the historical EgoTouch VAE uncertainty on its unseen test split.</summary>

- Status: **historical experiment / compatibility retained**; inferred from targeted historical configs/runs; not classified as disposable.
- Source: [scripts/evaluate_egotouch_vae_uncertainty.sh](../scripts/evaluate_egotouch_vae_uncertainty.sh).
- Inputs: Frozen historical model checkpoint(s), established MotionTrans episode split, native held-out data; see paths/arguments.
- Outputs: Reports, derived artifacts or configured run outputs named in source/CLI below; full data availability unverified.
- Effects: GPU stochastic inference and calibration evaluation; writes historical uncertainty metrics; no head fitting.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Literal paths/artifact names: `processing/evaluations/egotouch_vae32_uncertainty.json`, `processing/runs/egotouch_dinov3_vae_image_only_latent32/last.pt`.
- Static imports: `tactilelatent.evaluate_uncertainty`.

</details>

<details>
<summary><strong>scripts/evaluate_fgw_current.py</strong> — Run requested transfer and identity probes against an immutable FGW snapshot.</summary>

- Status: **historical experiment / compatibility retained**; inferred from targeted historical configs/runs; not classified as disposable.
- Source: [scripts/evaluate_fgw_current.py](../scripts/evaluate_fgw_current.py).
- Inputs: Frozen historical model checkpoint(s), established MotionTrans episode split, native held-out data; see paths/arguments.
- Outputs: Reports, derived artifacts or configured run outputs named in source/CLI below; full data availability unverified.
- Effects: GPU optimization or fitted probes; writes evaluation features/checkpoints/results; holdout protocol sensitive.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Inspected arguments: `parser.add_argument('--request', required=True)`.
- Literal paths/artifact names: `/evaluation.log`, `evaluation.log`, `processing/runs/joint_alignment_holdout_motiontrans/evaluations`, `results.json`, `status.json`, `summary.json`.
- Static imports: `argparse`, `fcntl`, `json`, `pathlib`, `subprocess`, `sys`, `tactilelatent.aligned_frames`, `time`.

</details>

<details>
<summary><strong>scripts/evaluate_invariance_holdout.py</strong> — Matched MotionTrans evaluation while the invariance study remains paused.</summary>

- Status: **historical experiment / compatibility retained**; inferred from targeted historical configs/runs; not classified as disposable.
- Source: [scripts/evaluate_invariance_holdout.py](../scripts/evaluate_invariance_holdout.py).
- Inputs: Frozen historical model checkpoint(s), established MotionTrans episode split, native held-out data; see paths/arguments.
- Outputs: Configured destination: parser.add_argument('--root', default='processing/runs/invariance_16d_v1').
- Effects: GPU optimization or fitted probes; writes evaluation features/checkpoints/results; holdout protocol sensitive.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Provenance: Exact relative path/hash recorded in existing receipt(s).
- Inspected arguments: `parser.add_argument('--root', default='processing/runs/invariance_16d_v1')`; `parser.add_argument('--epoch', type=int, default=15)`; `parser.add_argument('--runs', nargs='+', default=['control', 'relational'], choices=['control', 'relational', 'meta', 'meta_control'], help='Completed conditions to compare with the same held-out protocol')`; `parser.add_argument('--heads', nargs='+', choices=['shallow', 'deep', 'deeper', 'deepest'], default=['shallow', 'deep'])`; `parser.add_argument('--report-tag', default='', help='Separate aggregate report folder, e.g. deeper_heads')`.
- Literal paths/artifact names: `.log`, `.pt`, `_features.pt`, `checkpoint.pt`, `partial_results.json`, `pause_status.json`, `processing/runs/invariance_16d_v1`, `status.json`, `summary.json`.
- Referenced by: `scripts/run_continuation_study.py`, `tests/test_paired_probe_cache.py`.
- Static imports: `argparse`, `fcntl`, `gc`, `json`, `numpy`, `os`, `pathlib`, `subprocess`, `sys`, `tactilelatent.aligned_frames`, `tactilelatent.alignment_data`, `tactilelatent.alignment_model`, `tactilelatent.frozen_probe`, `tactilelatent.model`, `tactilelatent.paired_probe_cache`, `time`, `torch`.

</details>

<details>
<summary><strong>scripts/evaluate_previous_vjepa_current_targets.py</strong> — Compare the previous 8-query JEPA to the current models on matched targets.</summary>

- Status: **historical experiment / compatibility retained**; inferred from targeted historical configs/runs; not classified as disposable.
- Source: [scripts/evaluate_previous_vjepa_current_targets.py](../scripts/evaluate_previous_vjepa_current_targets.py).
- Inputs: Frozen historical model checkpoint(s), established MotionTrans episode split, native held-out data; see paths/arguments.
- Outputs: Configured destination: parser.add_argument('--output', type=Path, required=True).
- Effects: GPU optimization or fitted probes; writes evaluation features/checkpoints/results; holdout protocol sensitive.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Provenance: Hashes own implementation at runtime; retain bytes/location.
- Inspected arguments: `parser.add_argument('--current', type=Path, default=Path('processing/evaluations/motiontrans_vjepa21_current_epoch030_v1'))`; `parser.add_argument('--previous', type=Path, default=Path('processing/runs/joint_vjepa21_spatial_marker_codes_32d_v1/viewer_epochs/epoch_030.pt'))`; `parser.add_argument('--output', type=Path, required=True)`.
- Literal paths/artifact names: `_sample_contract.json`, `checkpoints/previous32_epoch_030.pt`, `probes/previous32/results.json`, `processing/evaluations/motiontrans_vjepa21_current_epoch030_v1`, `processing/runs/joint_vjepa21_spatial_marker_codes_32d_v1/viewer_epochs/epoch_030.pt`, `progress.jsonl`, `provenance.json`, `status.json`, `summary.json`.
- Path-sensitive code: line29: `from evaluate_vjepa_current_motiontrans import digest_tensor`; line98: `implementation={p:file_hash(p) for p in [__file__,'scripts/evaluate_vjepa_current_motiontrans.py',`.
- Static imports: `argparse`, `datetime`, `evaluate_vjepa_current_motiontrans`, `fcntl`, `gc`, `json`, `os`, `pathlib`, `shutil`, `tactilelatent.aligned_frames`, `tactilelatent.alignment_data`, `tactilelatent.frozen_probe`, `tactilelatent.marker_code_model`, `tactilelatent.representation_probe`, `tactilelatent.storage`, `tactilelatent.vjepa_model`, `tactilelatent.vjepa_spatial`, `time`, `torch`, `torch.utils.data`.

</details>

<details>
<summary><strong>scripts/evaluate_representation_screen.py</strong> — One-seed frozen DINO/random/trained-trunk readout comparison on MotionTrans.</summary>

- Status: **historical experiment / compatibility retained**; inferred from targeted historical configs/runs; not classified as disposable.
- Source: [scripts/evaluate_representation_screen.py](../scripts/evaluate_representation_screen.py).
- Inputs: Frozen historical model checkpoint(s), established MotionTrans episode split, native held-out data; see paths/arguments.
- Outputs: Configured destination: parser.add_argument('--output', type=Path, default=Path('processing/evaluations/motiontrans_representation_screen_v1')).
- Effects: GPU optimization or fitted probes; writes evaluation features/checkpoints/results; holdout protocol sensitive.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Provenance: Exact relative path/hash recorded in existing receipt(s).
- Inspected arguments: `parser.add_argument('--output', type=Path, default=Path('processing/evaluations/motiontrans_representation_screen_v1'))`.
- Literal paths/artifact names: `.pt`, `_targets.pt`, `epoch_100.pt`, `initialization_audit.json`, `processing/evaluations/motiontrans_representation_screen_v1`, `processing/joint_alignment/manifest.json`, `processing/runs/joint_alignment_holdout_motiontrans`, `provenance.json`, `reconstructed_epoch000.pt`, `results.json`, `status.json`, `summary.json`, `untrained_initialization.json`, `viewer_epochs/epoch_007.pt`.
- Referenced by: `scripts/finish_continuation_and_probe.py`.
- Static imports: `argparse`, `gc`, `json`, `os`, `pathlib`, `tactilelatent.aligned_frames`, `tactilelatent.alignment_data`, `tactilelatent.alignment_model`, `tactilelatent.frozen_probe`, `tactilelatent.model`, `tactilelatent.representation_probe`, `tactilelatent.train`, `tactilelatent.train_alignment`, `time`, `torch`, `torch.utils.data`.

</details>

<details>
<summary><strong>scripts/evaluate_vjepa_current_motiontrans.py</strong> — Matched MotionTrans transfer for the final current-frame 32D/128D encoders.</summary>

- Status: **historical experiment / compatibility retained**; inferred from targeted historical configs/runs; not classified as disposable.
- Source: [scripts/evaluate_vjepa_current_motiontrans.py](../scripts/evaluate_vjepa_current_motiontrans.py).
- Inputs: Frozen historical model checkpoint(s), established MotionTrans episode split, native held-out data; see paths/arguments.
- Outputs: Configured destination: parser.add_argument('--output', type=Path, required=True).
- Effects: GPU optimization or fitted probes; writes evaluation features/checkpoints/results; holdout protocol sensitive.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Provenance: Exact relative path/hash recorded in existing receipt(s); Hashes own implementation at runtime; retain bytes/location.
- Inspected arguments: `parser.add_argument('--epoch', type=int, default=30)`; `parser.add_argument('--output', type=Path, required=True)`.
- Literal paths/artifact names: `.pt`, `_sample_contract.json`, `_targets.pt`, `_viewer_registration.json`, `processing/evaluations/motiontrans_vjepa21_best_v1/features`, `processing/runs/joint_vjepa21_spatial_current128d_v1`, `processing/runs/joint_vjepa21_spatial_current32d_v1`, `progress.jsonl`, `provenance.json`, `results.json`, `status.json`, `summary.json`.
- Path-sensitive code: line156: `implementation={p:file_hash(p) for p in [__file__, 'src/tactilelatent/representation_probe.py',`.
- Referenced by: `scripts/evaluate_previous_vjepa_current_targets.py`.
- Static imports: `argparse`, `datetime`, `fcntl`, `gc`, `hashlib`, `json`, `os`, `pathlib`, `register_vjepa_transfer_viewer`, `shutil`, `tactilelatent.aligned_frames`, `tactilelatent.alignment_data`, `tactilelatent.frozen_probe`, `tactilelatent.marker_code_model`, `tactilelatent.representation_probe`, `tactilelatent.vjepa_model`, `tactilelatent.vjepa_spatial`, `time`, `torch`, `torch.utils.data`.

</details>

<details>
<summary><strong>scripts/evaluate_vjepa_marker_comparison.py</strong> — Matched MotionTrans probes: marker-code epochs 20/30 and image-code epoch 20.</summary>

- Status: **historical experiment / compatibility retained**; inferred from targeted historical configs/runs; not classified as disposable.
- Source: [scripts/evaluate_vjepa_marker_comparison.py](../scripts/evaluate_vjepa_marker_comparison.py).
- Inputs: Frozen historical model checkpoint(s), established MotionTrans episode split, native held-out data; see paths/arguments.
- Outputs: Reports, derived artifacts or configured run outputs named in source/CLI below; full data availability unverified.
- Effects: GPU optimization or fitted probes; writes evaluation features/checkpoints/results; holdout protocol sensitive.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Provenance: Hashes own implementation at runtime; retain bytes/location.
- Literal paths/artifact names: `.pt`, `_targets.pt`, `_viewer_registration.json`, `processing/evaluations/motiontrans_vjepa21_best_v1/features`, `processing/evaluations/motiontrans_vjepa21_marker_epochs020_030_v1`, `processing/runs`, `provenance.json`, `results.json`, `source_validation_epoch020.json`, `status.json`, `summary.json`.
- Path-sensitive code: line119: `implementation={p:file_hash(p) for p in [__file__, 'src/tactilelatent/representation_probe.py',`.
- Static imports: `fcntl`, `gc`, `json`, `os`, `pathlib`, `register_vjepa_transfer_viewer`, `shutil`, `tactilelatent.aligned_frames`, `tactilelatent.alignment_data`, `tactilelatent.frozen_probe`, `tactilelatent.marker_code_model`, `tactilelatent.representation_probe`, `tactilelatent.vjepa_model`, `tactilelatent.vjepa_spatial`, `time`, `torch`, `torch.utils.data`.

</details>

<details>
<summary><strong>scripts/evaluate_vjepa_marker_epoch.py</strong> — Evaluate a frozen marker-code V-JEPA checkpoint with the established probe.</summary>

- Status: **historical experiment / compatibility retained**; inferred from targeted historical configs/runs; not classified as disposable.
- Source: [scripts/evaluate_vjepa_marker_epoch.py](../scripts/evaluate_vjepa_marker_epoch.py).
- Inputs: Frozen historical model checkpoint(s), established MotionTrans episode split, native held-out data; see paths/arguments.
- Outputs: Configured destination: parser.add_argument('--output', type=Path, required=True).
- Effects: GPU optimization or fitted probes; writes evaluation features/checkpoints/results; holdout protocol sensitive.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Provenance: Hashes own implementation at runtime; retain bytes/location.
- Inspected arguments: `parser.add_argument('--epoch', type=int, required=True)`; `parser.add_argument('--run', type=Path, default=Path('processing/runs/joint_vjepa21_spatial_marker_codes_32d_v1'))`; `parser.add_argument('--output', type=Path, required=True)`.
- Literal paths/artifact names: `.pt`, `_targets.pt`, `processing/evaluations/motiontrans_vjepa21_best_v1/features`, `processing/evaluations/motiontrans_vjepa21_marker_epochs020_030_v1/summary.json`, `processing/runs/joint_vjepa21_spatial_marker_codes_32d_v1`, `provenance.json`, `results.json`, `status.json`, `summary.json`, `viewer_registration.json`.
- Path-sensitive code: line84: `implementation={p:file_hash(p) for p in [__file__, 'src/tactilelatent/representation_probe.py',`.
- Static imports: `argparse`, `fcntl`, `gc`, `json`, `os`, `pathlib`, `register_vjepa_transfer_viewer`, `shutil`, `tactilelatent.aligned_frames`, `tactilelatent.alignment_data`, `tactilelatent.frozen_probe`, `tactilelatent.marker_code_model`, `tactilelatent.representation_probe`, `tactilelatent.vjepa_model`, `tactilelatent.vjepa_spatial`, `time`, `torch`, `torch.utils.data`.

</details>

<details>
<summary><strong>scripts/evaluate_vjepa_motiontrans.py</strong> — MotionTrans-only transfer of frozen best V-JEPA task representations.</summary>

- Status: **historical experiment / compatibility retained**; inferred from targeted historical configs/runs; not classified as disposable.
- Source: [scripts/evaluate_vjepa_motiontrans.py](../scripts/evaluate_vjepa_motiontrans.py).
- Inputs: Frozen historical model checkpoint(s), established MotionTrans episode split, native held-out data; see paths/arguments.
- Outputs: Reports, derived artifacts or configured run outputs named in source/CLI below; full data availability unverified.
- Effects: GPU optimization or fitted probes; writes evaluation features/checkpoints/results; holdout protocol sensitive.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Provenance: Exact relative path/hash recorded in existing receipt(s).
- Literal paths/artifact names: `.pt`, `_targets.pt`, `best.pt`, `complete.json`, `index.json`, `processing/evaluations/motiontrans_representation_screen_v1`, `processing/evaluations/motiontrans_vjepa21_best_v1`, `processing/joint_alignment/manifest.json`, `processing/runs/joint_vjepa21_pair_32d_v1`, `processing/vjepa21_clip_cache_v1`, `provenance.json`, `results.json`, `status.json`, `summary.json`.
- Static imports: `fcntl`, `gc`, `json`, `os`, `pathlib`, `tactilelatent.aligned_frames`, `tactilelatent.alignment_data`, `tactilelatent.frozen_probe`, `tactilelatent.representation_probe`, `tactilelatent.vjepa_cache`, `tactilelatent.vjepa_model`, `time`, `torch`, `torch.utils.data`.

</details>

<details>
<summary><strong>scripts/evaluate_vjepa_motiontrans_epochs.py</strong> — Deterministic V-JEPA checkpoint transfer sweep with the deeper head only.</summary>

- Status: **historical experiment / compatibility retained**; inferred from targeted historical configs/runs; not classified as disposable.
- Source: [scripts/evaluate_vjepa_motiontrans_epochs.py](../scripts/evaluate_vjepa_motiontrans_epochs.py).
- Inputs: Frozen historical model checkpoint(s), established MotionTrans episode split, native held-out data; see paths/arguments.
- Outputs: Reports, derived artifacts or configured run outputs named in source/CLI below; full data availability unverified.
- Effects: GPU optimization or fitted probes; writes evaluation features/checkpoints/results; holdout protocol sensitive.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Provenance: Exact relative path/hash recorded in existing receipt(s).
- Inspected arguments: `parser.add_argument('--worker-epochs', nargs='+', type=int)`.
- Literal paths/artifact names: `.json`, `.log`, `.pt`, `_targets.pt`, `complete.json`, `index.json`, `processing/evaluations/motiontrans_representation_screen_v1`, `processing/evaluations/motiontrans_vjepa21_best_v1`, `processing/evaluations/motiontrans_vjepa21_deterministic_epochs_v1`, `processing/joint_alignment/manifest.json`, `processing/runs/joint_vjepa21_pair_32d_v1/deterministic`, `processing/vjepa21_clip_cache_v1`, `provenance.json`, `results.json`, `status.json`, `summary.json`.
- Path-sensitive code: line191: `p=subprocess.Popen([sys.executable,'-u',__file__,'--worker-epochs',*map(str,epochs)],`.
- Static imports: `argparse`, `fcntl`, `gc`, `json`, `os`, `pathlib`, `subprocess`, `sys`, `tactilelatent.aligned_frames`, `tactilelatent.alignment_data`, `tactilelatent.frozen_probe`, `tactilelatent.representation_probe`, `tactilelatent.vjepa_cache`, `tactilelatent.vjepa_model`, `time`, `torch`, `torch.utils.data`.

</details>

<details>
<summary><strong>scripts/evaluate_vjepa_spatial_motiontrans.py</strong> — Evaluate one frozen spatial-attention checkpoint with the established probe.</summary>

- Status: **historical experiment / compatibility retained**; inferred from targeted historical configs/runs; not classified as disposable.
- Source: [scripts/evaluate_vjepa_spatial_motiontrans.py](../scripts/evaluate_vjepa_spatial_motiontrans.py).
- Inputs: Frozen historical model checkpoint(s), established MotionTrans episode split, native held-out data; see paths/arguments.
- Outputs: Configured destination: parser.add_argument('--output', type=Path, required=True).
- Effects: GPU optimization or fitted probes; writes evaluation features/checkpoints/results; holdout protocol sensitive.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Provenance: Hashes own implementation at runtime; retain bytes/location.
- Inspected arguments: `parser.add_argument('--epoch', type=int, default=15)`; `parser.add_argument('--run', type=Path, default=Path('processing/runs/joint_vjepa21_spatial_live_32d_v1'))`; `parser.add_argument('--output', type=Path, required=True)`.
- Literal paths/artifact names: `.pt`, `_targets.pt`, `processing/evaluations/motiontrans_representation_screen_v1/summary.json`, `processing/evaluations/motiontrans_vjepa21_best_v1`, `processing/runs/joint_vjepa21_spatial_live_32d_v1`, `provenance.json`, `results.json`, `status.json`, `summary.json`, `viewer_registration.json`.
- Path-sensitive code: line86: `implementation={p:file_hash(p) for p in [__file__, 'src/tactilelatent/representation_probe.py',`.
- Static imports: `argparse`, `fcntl`, `gc`, `json`, `os`, `pathlib`, `register_vjepa_transfer_viewer`, `shutil`, `tactilelatent.aligned_frames`, `tactilelatent.alignment_data`, `tactilelatent.frozen_probe`, `tactilelatent.representation_probe`, `tactilelatent.vjepa_model`, `tactilelatent.vjepa_spatial`, `time`, `torch`, `torch.utils.data`.

</details>

<details>
<summary><strong>scripts/index_ftp_archives.py</strong> — Index FTP-1 tar members so Zarr can be read without archive extraction.</summary>

- Status: **historical experiment / compatibility retained**; inferred from targeted historical configs/runs; not classified as disposable.
- Source: [scripts/index_ftp_archives.py](../scripts/index_ftp_archives.py).
- Inputs: Source/config/checkpoint references and CLI inputs below; delegated loaders resolve remaining native paths.
- Outputs: Reports, derived artifacts or configured run outputs named in source/CLI below; full data availability unverified.
- Effects: reads research/native artifacts and writes reports/derived outputs.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Inspected arguments: `parser.add_argument('groups', nargs='+')`.
- Literal paths/artifact names: `.json`, `.tar`, `datasets/MJJJJ1064__FTP-1-Dataset`, `processing/indexes`.
- Static imports: `argparse`, `json`, `pathlib`, `tarfile`.

</details>

<details>
<summary><strong>scripts/inspect_ftp_remote.py</strong> — Inspect FTP-1 tar/Zarr metadata using HTTP byte ranges, without downloading GBs.</summary>

- Status: **historical experiment / compatibility retained**; inferred from targeted historical configs/runs; not classified as disposable.
- Source: [scripts/inspect_ftp_remote.py](../scripts/inspect_ftp_remote.py).
- Inputs: Pinned remote FTP tar/Zarr HTTP byte ranges.
- Outputs: Printed/header inspection metadata; limited network reads.
- Effects: network access / possible downloads.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Inspected arguments: `parser.add_argument('groups', nargs='+')`; `parser.add_argument('--max-headers', type=int, default=500)`.
- Literal paths/artifact names: `.tar`.
- Static imports: `argparse`, `json`, `urllib.request`.

</details>

<details>
<summary><strong>scripts/launch_joint_sam.py</strong> — Detached supervisor for the new SAM experiment; never touches old runs.</summary>

- Status: **historical experiment / compatibility retained**; inferred from targeted historical configs/runs; not classified as disposable.
- Source: [scripts/launch_joint_sam.py](../scripts/launch_joint_sam.py).
- Inputs: Source/config/checkpoint references and CLI inputs below; delegated loaders resolve remaining native paths.
- Outputs: Reports, derived artifacts or configured run outputs named in source/CLI below; full data availability unverified.
- Effects: GPU optimization or fitted probes.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Inspected arguments: `parser.add_argument('--resume', action='store_true')`.
- Literal paths/artifact names: `configs/joint_all_sam_holdout_motiontrans_fresh_8k_50ep.json`, `last.pt`, `nohup.log`, `status.json`, `tensorboard_bridge.log`.
- Path-sensitive code: line21: `used = check_storage_budget(Path.cwd(), additional=6_000_000_000)`.
- Static imports: `argparse`, `json`, `pathlib`, `subprocess`, `sys`, `tactilelatent.sam`.

</details>

<details>
<summary><strong>scripts/plot_fgw_study.py</strong> — Export transfer comparisons; error bars describe seed variation, not CIs.</summary>

- Status: **historical experiment / compatibility retained**; inferred from targeted historical configs/runs; not classified as disposable.
- Source: [scripts/plot_fgw_study.py](../scripts/plot_fgw_study.py).
- Inputs: Existing native data/checkpoints/results selected by the literals and arguments below.
- Outputs: Reports, derived artifacts or configured run outputs named in source/CLI below; full data availability unverified.
- Effects: reads research/native artifacts and writes reports/derived outputs.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Referenced by: `src/tactilelatent/fgw_study.py`.
- Static imports: `json`, `matplotlib`, `matplotlib.pyplot`, `pathlib`, `sys`.

</details>

<details>
<summary><strong>scripts/plot_motiontrans_checkpoint_sweep.py</strong> — Plot the saved sweep report using the system matplotlib installation.</summary>

- Status: **historical experiment / compatibility retained**; inferred from targeted historical configs/runs; not classified as disposable.
- Source: [scripts/plot_motiontrans_checkpoint_sweep.py](../scripts/plot_motiontrans_checkpoint_sweep.py).
- Inputs: Existing native data/checkpoints/results selected by the literals and arguments below.
- Outputs: Reports, derived artifacts or configured run outputs named in source/CLI below; full data availability unverified.
- Effects: reads research/native artifacts and writes reports/derived outputs.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Referenced by: `scripts/evaluate_alignment_checkpoint_sweep.py`.
- Static imports: `json`, `matplotlib`, `matplotlib.pyplot`, `pathlib`, `sys`.

</details>

<details>
<summary><strong>scripts/plot_vjepa_motiontrans_epochs.py</strong> — Plot the requested deterministic checkpoint-transfer sweep.</summary>

- Status: **historical experiment / compatibility retained**; inferred from targeted historical configs/runs; not classified as disposable.
- Source: [scripts/plot_vjepa_motiontrans_epochs.py](../scripts/plot_vjepa_motiontrans_epochs.py).
- Inputs: Existing native data/checkpoints/results selected by the literals and arguments below.
- Outputs: Reports, derived artifacts or configured run outputs named in source/CLI below; full data availability unverified.
- Effects: TOP-LEVEL execution on import; never probe with --help.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Caution: Import immediately reads old results and writes plot files; no parser handles --help.
- Literal paths/artifact names: `processing/evaluations/motiontrans_vjepa21_deterministic_epochs_v1`, `summary.json`.
- Static imports: `json`, `matplotlib`, `matplotlib.pyplot`, `pathlib`.

</details>

<details>
<summary><strong>scripts/prepare_joint_alignment.py</strong> — Prepare, pin, and audit the new joint experiment without starting training.</summary>

- Status: **historical experiment / compatibility retained**; inferred from targeted historical configs/runs; not classified as disposable.
- Source: [scripts/prepare_joint_alignment.py](../scripts/prepare_joint_alignment.py).
- Inputs: Source/config/checkpoint references and CLI inputs below; delegated loaders resolve remaining native paths.
- Outputs: Configured destination: parser.add_argument('--output', default=MANIFEST).
- Effects: reads research/native artifacts and writes reports/derived outputs.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Inspected arguments: `parser.add_argument('--original', default='processing/joint_all/manifest_sensor_roles_corrected_v2.json')`; `parser.add_argument('--frame-order', default='processing/audits/vla_touch_native_dynamics/dataset_frame_order.json')`; `parser.add_argument('--camera-roles', default='configs/freetacman_camera_roles.json')`; `parser.add_argument('--output', default=MANIFEST)`; `parser.add_argument('--registry', default=REGISTRY)`; `parser.add_argument('--audit', default='processing/joint_alignment/audit.json')`.
- Literal paths/artifact names: `configs/freetacman_camera_roles.json`, `processing/audits/vla_touch_native_dynamics/dataset_frame_order.json`, `processing/joint_alignment/audit.json`, `processing/joint_all/manifest_sensor_roles_corrected_v2.json`.
- Static imports: `argparse`, `json`, `tactilelatent.aligned_frames`, `tactilelatent.audit_alignment`, `tactilelatent.sensor_code_cache`.

</details>

<details>
<summary><strong>scripts/prepare_relational_floors.py</strong> — Audit frozen teacher round-trip errors in standardized sensor-code space.</summary>

- Status: **historical experiment / compatibility retained**; inferred from targeted historical configs/runs; not classified as disposable.
- Source: [scripts/prepare_relational_floors.py](../scripts/prepare_relational_floors.py).
- Inputs: Historical frozen teacher round-trip outputs and standardized codes.
- Outputs: Per-sensor relational normalization floor audit, without changing teacher targets.
- Effects: reads research/native artifacts and writes reports/derived outputs.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- CLI delegated to: `tactilelatent.prepare_relational`.
- Static imports: `tactilelatent.prepare_relational`.

</details>

<details>
<summary><strong>scripts/register_vjepa_transfer_viewer.py</strong> — Register completed V-JEPA probes for viewing. No fitting or evaluation sweep.</summary>

- Status: **historical experiment / compatibility retained**; inferred from targeted historical configs/runs; not classified as disposable.
- Source: [scripts/register_vjepa_transfer_viewer.py](../scripts/register_vjepa_transfer_viewer.py).
- Inputs: Completed old V-JEPA MotionTrans probe features/results/head checkpoints.
- Outputs: Viewer registration and inference checkpoints; does not fit the probes.
- Effects: viewer server or viewer registration.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Provenance: Exact relative path/hash recorded in existing receipt(s).
- Literal paths/artifact names: `decoder.pt`, `head.pt`, `normalization.pt`, `processing/evaluations/motiontrans_vjepa21_best_v1`, `processing/evaluations/motiontrans_vjepa21_deterministic_epochs_v1`, `provenance.json`, `results.json`, `viewer_registration.json`.
- Referenced by: `scripts/evaluate_vjepa_current_motiontrans.py`, `scripts/evaluate_vjepa_marker_comparison.py`, `scripts/evaluate_vjepa_marker_epoch.py`, `scripts/evaluate_vjepa_spatial_motiontrans.py`.
- Static imports: `json`, `pathlib`, `tactilelatent.aligned_frames`, `tactilelatent.frozen_probe`, `tactilelatent.train_alignment`, `tactilelatent.viewer_transfer`, `torch`.

</details>

<details>
<summary><strong>scripts/resume_vjepa_cached.sh</strong> — Resume the original deterministic/stochastic V-JEPA pair using its pooled clip cache.</summary>

- Status: **historical experiment / compatibility retained**; inferred from targeted historical configs/runs; not classified as disposable.
- Source: [scripts/resume_vjepa_cached.sh](../scripts/resume_vjepa_cached.sh).
- Inputs: Source/config/checkpoint references and CLI inputs below; delegated loaders resolve remaining native paths.
- Outputs: Reports, derived artifacts or configured run outputs named in source/CLI below; full data availability unverified.
- Effects: reads research/native artifacts and writes reports/derived outputs.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Literal paths/artifact names: `configs/joint_vjepa21_pair_32d_cached_v1.json`.

</details>

<details>
<summary><strong>scripts/run_continuation_study.py</strong> — Detached four-arm queue; clean epoch boundaries alternate training and probes.</summary>

- Status: **historical experiment / compatibility retained**; inferred from targeted historical configs/runs; not classified as disposable.
- Source: [scripts/run_continuation_study.py](../scripts/run_continuation_study.py).
- Inputs: Experiment config, native training/validation manifests, teacher/cache/backbone artifacts named below.
- Outputs: Configured destination: parser.add_argument('--root', type=Path, default=Path('processing/runs/continuation_32d_v1')).
- Effects: GPU optimization or fitted probes; sends process signals.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Provenance: Exact relative path/hash recorded in existing receipt(s).
- Inspected arguments: `parser.add_argument('action', choices=('prepare', 'run'))`; `parser.add_argument('--root', type=Path, default=Path('processing/runs/continuation_32d_v1'))`; `parser.add_argument('--parallel', type=int, default=2)`.
- Literal paths/artifact names: `.json`, `.log`, `initialization.json`, `last.pt`, `metrics.jsonl`, `plan.json`, `processing/runs/continuation_32d_v1`, `status.json`, `summary.json`.
- Static imports: `argparse`, `fcntl`, `json`, `os`, `pathlib`, `shutil`, `signal`, `subprocess`, `sys`, `tactilelatent.aligned_frames`, `tactilelatent.train_continuation`, `time`.

</details>

<details>
<summary><strong>scripts/run_fgw_study.py</strong> — Launch the configured fused Gromov-Wasserstein latent alignment study.</summary>

- Status: **historical experiment / compatibility retained**; inferred from targeted historical configs/runs; not classified as disposable.
- Source: [scripts/run_fgw_study.py](../scripts/run_fgw_study.py).
- Inputs: Experiment config, native training/validation manifests, teacher/cache/backbone artifacts named below.
- Outputs: Reports, derived artifacts or configured run outputs named in source/CLI below; full data availability unverified.
- Effects: GPU optimization or fitted probes.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- CLI delegated to: `tactilelatent.fgw_study`.
- Static imports: `tactilelatent.fgw_study`.

</details>

<details>
<summary><strong>scripts/run_invariance_study.py</strong> — Detached, resumable queue for the five explicitly selected invariance runs.</summary>

- Status: **historical experiment / compatibility retained**; inferred from targeted historical configs/runs; not classified as disposable.
- Source: [scripts/run_invariance_study.py](../scripts/run_invariance_study.py).
- Inputs: Experiment config, native training/validation manifests, teacher/cache/backbone artifacts named below.
- Outputs: Configured destination: parser.add_argument('--root', type=Path, default=ROOT).
- Effects: GPU optimization or fitted probes; sends process signals.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Inspected arguments: `parser.add_argument('action', choices=('prepare', 'run'))`; `parser.add_argument('--root', type=Path, default=ROOT)`; `parser.add_argument('--parallel', type=int, default=2)`.
- Literal paths/artifact names: `.json`, `.log`, `configs/invariance_16d_`, `initialization.json`, `last.pt`, `metrics.jsonl`, `plan.json`, `processing/runs/invariance_16d_v1`, `status.json`.
- Static imports: `argparse`, `fcntl`, `json`, `os`, `pathlib`, `shutil`, `signal`, `subprocess`, `sys`, `tactilelatent.aligned_frames`, `tactilelatent.train_invariance`, `time`.

</details>

<details>
<summary><strong>scripts/run_vjepa_markers.sh</strong> — Launch spatial V-JEPA joint training using marker-code tactile supervision.</summary>

- Status: **historical experiment / compatibility retained**; inferred from targeted historical configs/runs; not classified as disposable.
- Source: [scripts/run_vjepa_markers.sh](../scripts/run_vjepa_markers.sh).
- Inputs: Experiment config, native training/validation manifests, teacher/cache/backbone artifacts named below.
- Outputs: Reports, derived artifacts or configured run outputs named in source/CLI below; full data availability unverified.
- Effects: reads research/native artifacts and writes reports/derived outputs.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Literal paths/artifact names: `processing/runtime/nvidia-580.173.02/root/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH`.
- Referenced by: `scripts/start_vjepa_current.py`, `scripts/start_vjepa_current128d.py`.
- Static imports: `tactilelatent.train_vjepa_markers`.

</details>

<details>
<summary><strong>scripts/run_vjepa_pair.sh</strong> — Launch the original pooled V-JEPA deterministic/stochastic training pair.</summary>

- Status: **historical experiment / compatibility retained**; inferred from targeted historical configs/runs; not classified as disposable.
- Source: [scripts/run_vjepa_pair.sh](../scripts/run_vjepa_pair.sh).
- Inputs: Experiment config, native training/validation manifests, teacher/cache/backbone artifacts named below.
- Outputs: Reports, derived artifacts or configured run outputs named in source/CLI below; full data availability unverified.
- Effects: reads research/native artifacts and writes reports/derived outputs.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Literal paths/artifact names: `configs/joint_vjepa21_pair_32d_v1.json`, `processing/runtime/nvidia-580.173.02/root/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH`.
- Referenced by: `scripts/resume_vjepa_cached.sh`.
- Static imports: `tactilelatent.train_vjepa`.

</details>

<details>
<summary><strong>scripts/run_vjepa_spatial_live.sh</strong> — Launch the historical eight-output spatial-attention V-JEPA model.</summary>

- Status: **historical experiment / compatibility retained**; inferred from targeted historical configs/runs; not classified as disposable.
- Source: [scripts/run_vjepa_spatial_live.sh](../scripts/run_vjepa_spatial_live.sh).
- Inputs: Experiment config, native training/validation manifests, teacher/cache/backbone artifacts named below.
- Outputs: Reports, derived artifacts or configured run outputs named in source/CLI below; full data availability unverified.
- Effects: reads research/native artifacts and writes reports/derived outputs.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Literal paths/artifact names: `configs/joint_vjepa21_spatial_live_32d.json`, `processing/runtime/nvidia-580.173.02/root/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH`.
- Static imports: `tactilelatent.train_vjepa_spatial`.

</details>

<details>
<summary><strong>scripts/screen_source_transfer.py</strong> — Correlate saved source validation with held-out transfer; no training.</summary>

- Status: **historical experiment / compatibility retained**; inferred from targeted historical configs/runs; not classified as disposable.
- Source: [scripts/screen_source_transfer.py](../scripts/screen_source_transfer.py).
- Inputs: Source/config/checkpoint references and CLI inputs below; delegated loaders resolve remaining native paths.
- Outputs: Configured destination: parser.add_argument('--output', default='processing/continuation_32d_v1/source_screen').
- Effects: reads research/native artifacts and writes reports/derived outputs.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Inspected arguments: `parser.add_argument('--run', default='processing/runs/joint_alignment_holdout_motiontrans')`; `parser.add_argument('--output', default='processing/continuation_32d_v1/source_screen')`.
- Literal paths/artifact names: `metrics.jsonl`, `processing/continuation_32d_v1/source_screen`, `processing/runs/joint_alignment_holdout_motiontrans`, `results.csv`, `results.json`, `summary.json`.
- Static imports: `argparse`, `csv`, `json`, `numpy`, `pathlib`, `scipy.stats`.

</details>

<details>
<summary><strong>scripts/snapshot_joint_baseline.py</strong> — Make a non-overwriting, checksummed reference for future VLA_touch runs.</summary>

- Status: **historical experiment / compatibility retained**; inferred from targeted historical configs/runs; not classified as disposable.
- Source: [scripts/snapshot_joint_baseline.py](../scripts/snapshot_joint_baseline.py).
- Inputs: Historical VLA-holdout run/evaluation/checkpoint bundle.
- Outputs: Checksummed non-overwriting baseline under evaluations/baselines/joint_50ep_vla_touch_v1.
- Effects: reads research/native artifacts and writes reports/derived outputs.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Literal paths/artifact names: `baseline.json`, `processing/evaluations/baselines/joint_50ep_vla_touch_v1`, `processing/evaluations/dataset_identity_probe/classifier.pt`, `processing/evaluations/dataset_identity_probe/results.json`, `processing/evaluations/vla_touch_frozen_decoder/decoder.pt`, `processing/evaluations/vla_touch_frozen_decoder/examples/first.png`, `processing/evaluations/vla_touch_frozen_decoder/examples/most_uncertain.png`, `processing/evaluations/vla_touch_frozen_decoder/examples/worst_error.png`, `processing/evaluations/vla_touch_frozen_decoder/results.json`, `processing/joint_all/manifest.json`, `processing/runs/joint_all_holdout_vla_8k_50ep/config.json`, `processing/runs/joint_all_holdout_vla_8k_50ep/last.pt`, `processing/runs/joint_all_holdout_vla_8k_50ep/metrics.jsonl`.
- Path-sensitive code: line10: `ROOT = Path(__file__).resolve().parents[1]`.
- Static imports: `hashlib`, `json`, `pathlib`, `shutil`, `tempfile`.

</details>

<details>
<summary><strong>scripts/summarize_egotouch_autoencoder_grid.py</strong> — Collect completed and partial autoencoder ablations into one CSV table.</summary>

- Status: **historical experiment / compatibility retained**; inferred from targeted historical configs/runs; not classified as disposable.
- Source: [scripts/summarize_egotouch_autoencoder_grid.py](../scripts/summarize_egotouch_autoencoder_grid.py).
- Inputs: Source/config/checkpoint references and CLI inputs below; delegated loaders resolve remaining native paths.
- Outputs: Configured destination: parser.add_argument('--root', default='processing/runs/egotouch_ae_sensor_code_grid_v1').
- Effects: reads research/native artifacts and writes reports/derived outputs.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Inspected arguments: `parser.add_argument('--root', default='processing/runs/egotouch_ae_sensor_code_grid_v1')`.
- Literal paths/artifact names: `best_metrics.json`, `comparison.csv`, `config.json`, `metrics.jsonl`, `processing/runs/egotouch_ae_sensor_code_grid_v1`, `summary.json`.
- Referenced by: `scripts/train_egotouch_autoencoder_grid.sh`.
- Static imports: `argparse`, `csv`, `json`, `pathlib`.

</details>

<details>
<summary><strong>scripts/tensorboard_joint_live.py</strong> — Mirror a running joint-training log into TensorBoard without changing training.</summary>

- Status: **historical experiment / compatibility retained**; inferred from targeted historical configs/runs; not classified as disposable.
- Source: [scripts/tensorboard_joint_live.py](../scripts/tensorboard_joint_live.py).
- Inputs: Selected joint run status/checkpoints/metrics.
- Outputs: TensorBoard event stream and live scalar summaries; may wait continuously.
- Effects: sends process signals.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Inspected arguments: `parser.add_argument('run_dir', type=Path)`; `parser.add_argument('--pid', type=int, help='Exit after this training process exits')`; `parser.add_argument('--poll-seconds', type=float, default=2)`.
- Literal paths/artifact names: `metrics.jsonl`, `nohup.log`.
- Referenced by: `scripts/launch_joint_sam.py`, `scripts/train_motiontrans_sequence.py`.
- Static imports: `argparse`, `json`, `os`, `pathlib`, `time`, `torch.utils.tensorboard`.

</details>

<details>
<summary><strong>scripts/train_16d_pair.py</strong> — Run one matched 16D control and one visual-FGW model concurrently.</summary>

- Status: **historical experiment / compatibility retained**; inferred from targeted historical configs/runs; not classified as disposable.
- Source: [scripts/train_16d_pair.py](../scripts/train_16d_pair.py).
- Inputs: Experiment config, native training/validation manifests, teacher/cache/backbone artifacts named below.
- Outputs: Configured destination: parser.add_argument('--root', default=ROOT).
- Effects: GPU optimization or fitted probes; sends process signals.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Inspected arguments: `parser.add_argument('action', choices=('prepare', 'run'))`; `parser.add_argument('--root', default=ROOT)`.
- Literal paths/artifact names: `.json`, `.log`, `configs/joint_alignment_holdout_motiontrans.json`, `initialization.json`, `last.pt`, `matched_initialization.json`, `metrics.jsonl`, `plan.json`, `processing/runs/joint_16d_pair_v1`, `status.json`.
- Static imports: `argparse`, `dataclasses`, `fcntl`, `json`, `os`, `pathlib`, `shutil`, `subprocess`, `sys`, `tactilelatent.aligned_frames`, `tactilelatent.fgw_mixer`, `time`.

</details>

<details>
<summary><strong>scripts/train_egotouch_autoencoder_grid.sh</strong> — Run the original twelve capacity/code-size EgoTouch autoencoder ablations.</summary>

- Status: **historical experiment / compatibility retained**; inferred from targeted historical configs/runs; not classified as disposable.
- Source: [scripts/train_egotouch_autoencoder_grid.sh](../scripts/train_egotouch_autoencoder_grid.sh).
- Inputs: Experiment config, native training/validation manifests, teacher/cache/backbone artifacts named below.
- Outputs: Reports, derived artifacts or configured run outputs named in source/CLI below; full data availability unverified.
- Effects: GPU optimization or fitted probes.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Literal paths/artifact names: `processing/runs/egotouch_ae_sensor_code_grid_v1`.

</details>

<details>
<summary><strong>scripts/train_egotouch_autoencoder_presets.sh</strong> — Run the original small/medium/large 32D EgoTouch autoencoder ablation.</summary>

- Status: **historical experiment / compatibility retained**; inferred from targeted historical configs/runs; not classified as disposable.
- Source: [scripts/train_egotouch_autoencoder_presets.sh](../scripts/train_egotouch_autoencoder_presets.sh).
- Inputs: Experiment config, native training/validation manifests, teacher/cache/backbone artifacts named below.
- Outputs: Reports, derived artifacts or configured run outputs named in source/CLI below; full data availability unverified.
- Effects: GPU optimization or fitted probes.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Literal paths/artifact names: `processing/runs/egotouch_ae_conv_${preset}_z32`, `processing/runs/egotouch_ae_presets_z32`.
- Static imports: `any`.

</details>

<details>
<summary><strong>scripts/train_egotouch_vae.sh</strong> — Launch the original single-source DINO EgoTouch stochastic32D experiment.</summary>

- Status: **historical experiment / compatibility retained**; inferred from targeted historical configs/runs; not classified as disposable.
- Source: [scripts/train_egotouch_vae.sh](../scripts/train_egotouch_vae.sh).
- Inputs: Experiment config, native training/validation manifests, teacher/cache/backbone artifacts named below.
- Outputs: Reports, derived artifacts or configured run outputs named in source/CLI below; full data availability unverified.
- Effects: GPU optimization or fitted probes.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Literal paths/artifact names: `configs/egotouch_vae_image_only_latent32.json`.
- Static imports: `tactilelatent.train`.

</details>

<details>
<summary><strong>scripts/train_expanded_vjepa.py</strong> — Expanded twelve-source entrypoint; reuse the existing SAM training loop.</summary>

- Status: **historical experiment / compatibility retained**; inferred from targeted historical configs/runs; not classified as disposable.
- Source: [scripts/train_expanded_vjepa.py](../scripts/train_expanded_vjepa.py).
- Inputs: Experiment config, native training/validation manifests, teacher/cache/backbone artifacts named below.
- Outputs: Reports, derived artifacts or configured run outputs named in source/CLI below; full data availability unverified.
- Effects: GPU optimization or fitted probes.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Literal paths/artifact names: `configs/joint_vjepa21_spatial_expanded_32d.json`.
- Referenced by: `configs/joint_vjepa21_spatial_expanded_32d.json`, `src/tactilelatent/expansion_training.py`.
- Static imports: `sys`, `tactilelatent`, `tactilelatent.expansion_joint`, `tactilelatent.expansion_runtime`.

</details>

<details>
<summary><strong>scripts/train_freetacman_available_markers.py</strong> — One configurable AE on a fixed snapshot of completed FreeTacMan tracks.</summary>

- Status: **historical experiment / compatibility retained**; inferred from targeted historical configs/runs; not classified as disposable.
- Source: [scripts/train_freetacman_available_markers.py](../scripts/train_freetacman_available_markers.py).
- Inputs: Experiment config, native training/validation manifests, teacher/cache/backbone artifacts named below.
- Outputs: Reports, derived artifacts or configured run outputs named in source/CLI below; full data availability unverified.
- Effects: GPU optimization or fitted probes.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Inspected arguments: `parser.add_argument('--config', default='configs/marker_freetacman_available_16d.json')`.
- Literal paths/artifact names: `available_manifest.json`, `complete.json`, `configs/marker_freetacman_available_16d.json`, `pinned_snapshot.json`, `plan.json`, `processing/runs/marker_autoencoders_v1`, `selection.json`, `status.json`, `teachers.json`.
- Referenced by: `tests/test_freetacman_available.py`.
- Static imports: `argparse`, `collections`, `datetime`, `fcntl`, `json`, `os`, `pathlib`, `shutil`, `tactilelatent.aligned_frames`, `tactilelatent.marker_ae_data`, `tactilelatent.marker_data`, `tactilelatent.train_marker_autoencoders`, `time`, `torch`.

</details>

<details>
<summary><strong>scripts/train_invariance.py</strong> — Run the optional invariance objectives; inference architecture is unchanged.</summary>

- Status: **historical experiment / compatibility retained**; inferred from targeted historical configs/runs; not classified as disposable.
- Source: [scripts/train_invariance.py](../scripts/train_invariance.py).
- Inputs: Experiment config, native training/validation manifests, teacher/cache/backbone artifacts named below.
- Outputs: Reports, derived artifacts or configured run outputs named in source/CLI below; full data availability unverified.
- Effects: GPU optimization or fitted probes.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- CLI delegated to: `tactilelatent.train_invariance`.
- Referenced by: `src/tactilelatent/train_invariance.py`.
- Static imports: `tactilelatent.train_invariance`.

</details>

<details>
<summary><strong>scripts/train_joint_alignment.py</strong> — Train joint alignment</summary>

- Status: **historical experiment / compatibility retained**; inferred from targeted historical configs/runs; not classified as disposable.
- Source: [scripts/train_joint_alignment.py](../scripts/train_joint_alignment.py).
- Inputs: Experiment config, native training/validation manifests, teacher/cache/backbone artifacts named below.
- Outputs: Reports, derived artifacts or configured run outputs named in source/CLI below; full data availability unverified.
- Effects: GPU optimization or fitted probes.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- CLI delegated to: `tactilelatent.train_alignment`.
- Referenced by: `scripts/extend_joint_alignment.py`, `scripts/smoke_joint_alignment.py`.
- Static imports: `tactilelatent.train_alignment`.

</details>

<details>
<summary><strong>scripts/train_joint_all_8k_50ep.sh</strong> — Launch the historical DINO joint50epoch study with VLA Touch held out.</summary>

- Status: **historical experiment / compatibility retained**; inferred from targeted historical configs/runs; not classified as disposable.
- Source: [scripts/train_joint_all_8k_50ep.sh](../scripts/train_joint_all_8k_50ep.sh).
- Inputs: Experiment config, native training/validation manifests, teacher/cache/backbone artifacts named below.
- Outputs: Reports, derived artifacts or configured run outputs named in source/CLI below; full data availability unverified.
- Effects: GPU optimization or fitted probes.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Literal paths/artifact names: `configs/joint_all_holdout_vla_8k_50ep.json`.
- Static imports: `tactilelatent.train_joint`.

</details>

<details>
<summary><strong>scripts/train_joint_all_holdout.sh</strong> — Launch the historical DINO joint study with VLA Touch held out.</summary>

- Status: **historical experiment / compatibility retained**; inferred from targeted historical configs/runs; not classified as disposable.
- Source: [scripts/train_joint_all_holdout.sh](../scripts/train_joint_all_holdout.sh).
- Inputs: Experiment config, native training/validation manifests, teacher/cache/backbone artifacts named below.
- Outputs: Reports, derived artifacts or configured run outputs named in source/CLI below; full data availability unverified.
- Effects: GPU optimization or fitted probes.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Literal paths/artifact names: `configs/joint_all_holdout_vla_vae32.json`.
- Static imports: `tactilelatent.train_joint`.

</details>

<details>
<summary><strong>scripts/train_joint_alternating_fresh.sh</strong> — Train historical alternating DINO baseline, then fit holdout and classifier probes.</summary>

- Status: **historical experiment / compatibility retained**; inferred from targeted historical configs/runs; not classified as disposable.
- Source: [scripts/train_joint_alternating_fresh.sh](../scripts/train_joint_alternating_fresh.sh).
- Inputs: Experiment config, native training/validation manifests, teacher/cache/backbone artifacts named below.
- Outputs: Reports, derived artifacts or configured run outputs named in source/CLI below; full data availability unverified.
- Effects: GPU optimization or fitted probes.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Literal paths/artifact names: `configs/joint_all_alternating_fresh_8k_50ep.json`, `processing/runs/joint_all_alternating_fresh_8k_50ep`.
- Static imports: `tactilelatent.evaluate_dataset_classifier`, `tactilelatent.evaluate_holdout_decoder`, `tactilelatent.train_joint_alternating`.

</details>

<details>
<summary><strong>scripts/train_joint_grl_fresh.sh</strong> — Train historical gradient-reversal DINO baseline, then fit evaluation probes.</summary>

- Status: **historical experiment / compatibility retained**; inferred from targeted historical configs/runs; not classified as disposable.
- Source: [scripts/train_joint_grl_fresh.sh](../scripts/train_joint_grl_fresh.sh).
- Inputs: Experiment config, native training/validation manifests, teacher/cache/backbone artifacts named below.
- Outputs: Reports, derived artifacts or configured run outputs named in source/CLI below; full data availability unverified.
- Effects: GPU optimization or fitted probes.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Literal paths/artifact names: `configs/joint_all_grl_fresh_8k_50ep.json`, `processing/runs/joint_all_grl_fresh_8k_50ep`.
- Static imports: `tactilelatent.evaluate_dataset_classifier`, `tactilelatent.evaluate_holdout_decoder`, `tactilelatent.train_joint_adversarial`.

</details>

<details>
<summary><strong>scripts/train_joint_sam.sh</strong> — Launch the historical DINO SAM joint baseline with MotionTrans held out.</summary>

- Status: **historical experiment / compatibility retained**; inferred from targeted historical configs/runs; not classified as disposable.
- Source: [scripts/train_joint_sam.sh](../scripts/train_joint_sam.sh).
- Inputs: Experiment config, native training/validation manifests, teacher/cache/backbone artifacts named below.
- Outputs: Reports, derived artifacts or configured run outputs named in source/CLI below; full data availability unverified.
- Effects: GPU optimization or fitted probes.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Literal paths/artifact names: `configs/joint_all_sam_holdout_motiontrans_fresh_8k_50ep.json`.
- Static imports: `tactilelatent.train_joint`.

</details>

<details>
<summary><strong>scripts/train_joint_six_vae.sh</strong> — Launch the original six-source joint stochastic32D DINO model.</summary>

- Status: **historical experiment / compatibility retained**; inferred from targeted historical configs/runs; not classified as disposable.
- Source: [scripts/train_joint_six_vae.sh](../scripts/train_joint_six_vae.sh).
- Inputs: Experiment config, native training/validation manifests, teacher/cache/backbone artifacts named below.
- Outputs: Reports, derived artifacts or configured run outputs named in source/CLI below; full data availability unverified.
- Effects: GPU optimization or fitted probes.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Literal paths/artifact names: `configs/joint_six_vae32.json`.
- Static imports: `tactilelatent.train_joint`.

</details>

<details>
<summary><strong>scripts/train_mirrored_vjepa.py</strong> — Opt-in bilateral mirroring; ordinary checkpoints need no inference changes.</summary>

- Status: **historical experiment / compatibility retained**; inferred from targeted historical configs/runs; not classified as disposable.
- Source: [scripts/train_mirrored_vjepa.py](../scripts/train_mirrored_vjepa.py).
- Inputs: Experiment config, native training/validation manifests, teacher/cache/backbone artifacts named below.
- Outputs: Reports, derived artifacts or configured run outputs named in source/CLI below; full data availability unverified.
- Effects: GPU optimization or fitted probes.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Literal paths/artifact names: `configs/joint_vjepa21_spatial_current_32d_mirror.json`.
- Referenced by: `configs/joint_vjepa21_spatial_current_32d_mirror.json`, `configs/joint_vjepa21_spatial_expanded_32d_mirror.json`, `configs/joint_vjepa21_spatial_expanded_32d_mirror_v2.json`, `scripts/verify_bimanual_mirror.py`.
- Static imports: `json`, `pathlib`, `sys`, `tactilelatent`, `tactilelatent.expansion_joint`, `tactilelatent.expansion_runtime`, `tactilelatent.mirror_joint`.

</details>

<details>
<summary><strong>scripts/train_motiontrans_sequence.py</strong> — Sequential fresh joint baseline, heldout adaptation and direct-sensor control.</summary>

- Status: **historical experiment / compatibility retained**; inferred from targeted historical configs/runs; not classified as disposable.
- Source: [scripts/train_motiontrans_sequence.py](../scripts/train_motiontrans_sequence.py).
- Inputs: Experiment config, native training/validation manifests, teacher/cache/backbone artifacts named below.
- Outputs: Reports, derived artifacts or configured run outputs named in source/CLI below; full data availability unverified.
- Effects: GPU optimization or fitted probes.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Inspected arguments: `parser.add_argument('--smoke', action='store_true', help='Only one training/validation clip per source, in separate runs')`.
- Literal paths/artifact names: `.json`, `/evaluation/results.json`, `/evaluations/motiontrans/results.json`, `comparison.json`, `configs/`, `evaluation.log`, `last.pt`, `nohup.log`, `processing/joint_all/manifest.json`, `processing/runs/`, `processing/runs/motiontrans_baseline_sequence`, `results.json`, `status.json`, `tensorboard_bridge.log`.
- Path-sensitive code: line19: `ROOT = Path(__file__).resolve().parents[1]`.
- Referenced by: `scripts/train_motiontrans_sequence.sh`.
- Static imports: `argparse`, `json`, `pathlib`, `subprocess`, `sys`, `tactilelatent.frozen_probe`, `torch`.

</details>

<details>
<summary><strong>scripts/train_motiontrans_sequence.sh</strong> — Lock and launch the historical DINO baseline/transfer/direct-control sequence.</summary>

- Status: **historical experiment / compatibility retained**; inferred from targeted historical configs/runs; not classified as disposable.
- Source: [scripts/train_motiontrans_sequence.sh](../scripts/train_motiontrans_sequence.sh).
- Inputs: Experiment config, native training/validation manifests, teacher/cache/backbone artifacts named below.
- Outputs: Reports, derived artifacts or configured run outputs named in source/CLI below; full data availability unverified.
- Effects: GPU optimization or fitted probes.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Literal paths/artifact names: `processing/runs/motiontrans_baseline_sequence`, `processing/runs/motiontrans_baseline_sequence/sequence.lock`.
- Referenced by: `scripts/train_motiontrans_sequence.py`.

</details>

<details>
<summary><strong>scripts/train_sensor_autoencoders.py</strong> — Resumable tactile-only AE searches for all native sensor layouts.</summary>

- Status: **historical experiment / compatibility retained**; inferred from targeted historical configs/runs; not classified as disposable.
- Source: [scripts/train_sensor_autoencoders.py](../scripts/train_sensor_autoencoders.py).
- Inputs: Experiment config, native training/validation manifests, teacher/cache/backbone artifacts named below.
- Outputs: Configured destination: parser.add_argument('--output', default='processing/runs/sensor_autoencoders_native_v2').
- Effects: GPU optimization or fitted probes.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Provenance: Hashes own implementation at runtime; retain bytes/location.
- Inspected arguments: `parser.add_argument('--manifest', default='processing/joint_all/manifest.json')`; `parser.add_argument('--cache', default='processing/sensor_ae_cache_v1')`; `parser.add_argument('--output', default='processing/runs/sensor_autoencoders_native_v2')`; `parser.add_argument('--plan', help='Optional explicit JSON plan; overrides generated source/budget settings')`; `parser.add_argument('--sources', nargs='+', choices=ORDER, default=ORDER)`; `parser.add_argument('--screen-epochs', type=int, default=30)`; `parser.add_argument('--final-epochs', type=int, default=100)`; `parser.add_argument('--smoke', action='store_true')`.
- Literal paths/artifact names: `.log`, `_finalists.json`, `_repeat.json`, `best.pt`, `best_by_source.json`, `best_metrics.json`, `comparison.csv`, `config.json`, `last.pt`, `metrics.jsonl`, `plan.json`, `processing/joint_all/manifest.json`, `processing/runs/sensor_autoencoders_native_v2`, `processing/sensor_ae_cache_v1`, `provenance.json`, `seed_stability.json`, `smoke_manifest.json`, `status.json`.
- Path-sensitive code: line24: `ROOT = Path(__file__).resolve().parents[1]`; line182: `args = parser.parse_args(); os.chdir(ROOT)`; line214: `files = [Path(__file__), Path(args.manifest), *[ROOT / "src/tactilelatent" / name for name in`.
- Referenced by: `scripts/handover_sensor_autoencoder_search.py`, `tests/test_sensor_search_handover.py`.
- Static imports: `argparse`, `csv`, `fcntl`, `hashlib`, `json`, `numpy`, `os`, `pathlib`, `subprocess`, `sys`, `tactilelatent.joint_all_config`, `tactilelatent.native_optical`, `tactilelatent.sensor_cache`, `time`, `torch`.

</details>

<details>
<summary><strong>scripts/train_vla_touch_direct.sh</strong> — Train the historical direct VLA Touch baseline and write comparison evaluations.</summary>

- Status: **historical experiment / compatibility retained**; inferred from targeted historical configs/runs; not classified as disposable.
- Source: [scripts/train_vla_touch_direct.sh](../scripts/train_vla_touch_direct.sh).
- Inputs: Experiment config, native training/validation manifests, teacher/cache/backbone artifacts named below.
- Outputs: Reports, derived artifacts or configured run outputs named in source/CLI below; full data availability unverified.
- Effects: GPU optimization or fitted probes.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Literal paths/artifact names: `configs/vla_touch_direct_vae32_50ep.json`, `processing/runs/vla_touch_direct_vae32_50ep`.
- Static imports: `tactilelatent.evaluate_direct_vla_touch`, `tactilelatent.train_joint`.

</details>

<details>
<summary><strong>scripts/tune_egotouch_autoencoder.py</strong> — Sequential 60-epoch screening -> 200-epoch finalists -> seed repetitions.</summary>

- Status: **historical experiment / compatibility retained**; inferred from targeted historical configs/runs; not classified as disposable.
- Source: [scripts/tune_egotouch_autoencoder.py](../scripts/tune_egotouch_autoencoder.py).
- Inputs: Experiment config, native training/validation manifests, teacher/cache/backbone artifacts named below.
- Outputs: Configured destination: parser.add_argument('--output', default='processing/runs/egotouch_ae_search_v4').
- Effects: GPU optimization or fitted probes.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Provenance: Hashes own implementation at runtime; retain bytes/location.
- Inspected arguments: `parser.add_argument('--plan', default='configs/egotouch_ae_search_v4.json')`; `parser.add_argument('--output', default='processing/runs/egotouch_ae_search_v4')`; `parser.add_argument('--smoke', action='store_true', help='Tiny CPU search testing all three stages')`.
- Literal paths/artifact names: `.log`, `best_metrics.json`, `comparison.csv`, `config.json`, `configs/egotouch_ae_search_v4.json`, `last.pt`, `metrics.jsonl`, `plan.json`, `processing/joint_all/manifest.json`, `processing/runs/egotouch_ae_search_v4`, `provenance.json`, `repeat_selection.json`, `screening_selection.json`, `seed_stability.json`, `status.json`, `summary.json`.
- Path-sensitive code: line21: `ROOT = Path(__file__).resolve().parents[1]`; line116: `os.chdir(ROOT)`; line134: `files = [Path(__file__), ROOT / "src/tactilelatent/tactile_autoencoder.py",`.
- Static imports: `argparse`, `csv`, `fcntl`, `hashlib`, `json`, `os`, `pathlib`, `re`, `subprocess`, `sys`, `time`, `torch`.

</details>

<details>
<summary><strong>scripts/verify_cached_vjepa_training.py</strong> — Check complete cache integrity, source exclusion, real SAM steps and decoding.</summary>

- Status: **historical experiment / compatibility retained**; inferred from targeted historical configs/runs; not classified as disposable.
- Source: [scripts/verify_cached_vjepa_training.py](../scripts/verify_cached_vjepa_training.py).
- Inputs: Source/config/checkpoint references and CLI inputs below; delegated loaders resolve remaining native paths.
- Outputs: Reports, derived artifacts or configured run outputs named in source/CLI below; full data availability unverified.
- Effects: GPU optimization or fitted probes; TOP-LEVEL execution on import; never probe with --help.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Caution: Top-level GPU optimization and report writes occur on import, including when invoked with --help.
- Literal paths/artifact names: `.pt`, `configs/joint_vjepa21_pair_32d_cached_v1.json`, `pair_state.pt`, `processing/vjepa2_1/cached_training_preflight.json`.
- Static imports: `json`, `pathlib`, `tactilelatent.aligned_frames`, `tactilelatent.joint_data`, `tactilelatent.train_alignment`, `tactilelatent.train_vjepa`, `tactilelatent.vjepa_cached_training`, `tactilelatent.vjepa_model`, `time`, `torch`, `torch.utils.data`.

</details>

<details>
<summary><strong>scripts/verify_vjepa_cache.py</strong> — Compare live full-video features and predictions with on-disk cache reads.</summary>

- Status: **historical experiment / compatibility retained**; inferred from targeted historical configs/runs; not classified as disposable.
- Source: [scripts/verify_vjepa_cache.py](../scripts/verify_vjepa_cache.py).
- Inputs: Source/config/checkpoint references and CLI inputs below; delegated loaders resolve remaining native paths.
- Outputs: Reports, derived artifacts or configured run outputs named in source/CLI below; full data availability unverified.
- Effects: TOP-LEVEL execution on import; never probe with --help.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Caution: Top-level backbone GPU inference and visual-cache partition writes occur on import, including --help.
- Literal paths/artifact names: `configs/joint_vjepa21_pair_32d_v1.json`, `epoch_000.pt`, `index.json`, `processing/runs/joint_vjepa21_pair_32d_v1`, `processing/vjepa2_1/cache_verification`, `processing/vjepa2_1/cache_verification.json`.
- Static imports: `json`, `numpy`, `pathlib`, `tactilelatent.aligned_frames`, `tactilelatent.alignment_data`, `tactilelatent.vjepa_cache`, `tactilelatent.vjepa_model`, `torch`, `torch.utils.data`, `types`.

</details>

<details>
<summary><strong>scripts/verify_vjepa_pair.py</strong> — Production-data preflight: all sources, frozen gradients, native decoding.</summary>

- Status: **historical experiment / compatibility retained**; inferred from targeted historical configs/runs; not classified as disposable.
- Source: [scripts/verify_vjepa_pair.py](../scripts/verify_vjepa_pair.py).
- Inputs: Source/config/checkpoint references and CLI inputs below; delegated loaders resolve remaining native paths.
- Outputs: Reports, derived artifacts or configured run outputs named in source/CLI below; full data availability unverified.
- Effects: GPU optimization or fitted probes; TOP-LEVEL execution on import; never probe with --help.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Caution: Top-level GPU optimization and report writes occur on import, including --help.
- Literal paths/artifact names: `configs/joint_vjepa21_pair_32d_v1.json`, `processing/vjepa2_1/preflight.json`.
- Static imports: `json`, `pathlib`, `tactilelatent.aligned_frames`, `tactilelatent.joint_data`, `tactilelatent.train_alignment`, `tactilelatent.train_vjepa`, `tactilelatent.vjepa_model`, `time`, `torch`, `torch.utils.data`.

</details>

<details>
<summary><strong>scripts/view_sensor_autoencoders.py</strong> — Launch the viewer for all datasets and native sensor types.</summary>

- Status: **historical experiment / compatibility retained**; inferred from targeted historical configs/runs; not classified as disposable.
- Source: [scripts/view_sensor_autoencoders.py](../scripts/view_sensor_autoencoders.py).
- Inputs: Original sensor-AE viewer manifests and checkpoint discovery.
- Outputs: Legacy HTTP viewer without the added acquisition/mirror/dashboard install hooks.
- Effects: viewer server or viewer registration.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Caution: Historical basic launcher omits extension hooks installed by view_egotouch_autoencoder.py; use the latter for the maintained viewer.
- CLI delegated to: `tactilelatent.autoencoder_viewer`.
- Static imports: `tactilelatent.autoencoder_viewer`.

</details>

<details>
<summary><strong>download_egotouch_worker.sh</strong> — Download the historical alternative EgoTouch HDF5 mirror and remove its budget reservation.</summary>

- Status: **historical experiment / compatibility retained**; inferred from targeted historical configs/runs; not classified as disposable.
- Canonical historical body: [scripts/archive/download_egotouch_worker.sh](../scripts/archive/download_egotouch_worker.sh); original command is a compatibility shim.
- Inputs: Hugging Face zhenyuxie-zhzh/EgoTouch_hdf5 (unversioned default).
- Outputs: datasets/zhenyuxie-zhzh__EgoTouch_hdf5, download_logs; deletes its reservation marker.
- Effects: network access / possible downloads.
- Invocation: no current recommendation; preserve the old path and review its parser/config before rerunning.
- Caution: Default Hugging Face revision is not pinned. Historical download target differs from the current raw EgoTouch release; do not queue automatically.
- Literal paths/artifact names: `datasets/.reservations/zhenyuxie-zhzh__EgoTouch_hdf5.bytes`, `datasets/zhenyuxie-zhzh__EgoTouch_hdf5`, `download_logs/download.log`, `download_logs/zhenyuxie-zhzh__EgoTouch_hdf5.log`.

</details>

