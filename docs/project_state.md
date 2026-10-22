# Project state — October 22, 2026

The research-core refactor was deployed to the main working directory on
**2026-10-22 01:03 EDT**. Actual implementations now live in `src/tactilelatent/`,
operational commands in `tools/tactilelatent_tools/`, the five editable settings
in `configs/`, and historical source/scripts/configs/tests in `legacy/v1/`.
Root logs are under `logs/`; temporary agent work is under `ai_work/`.

At the user's request, joint training resumed on **2026-10-22 at 01:25 EDT** in
`processing/runs/vjepa32_current_all_sources_core_v2/`, using the maintained trainer
and `tactilelatent-joint-core.service`. It continues at **epoch 5 of 30**, with
the deterministic 32D V-JEPA model, 17 training sources, verified bimanual mirroring
and MotionTrans held out. The initial live batches have finite losses and the
service remains active across SSH disconnects. Follow `status.json`, the viewer's
Training monitor or `logs/training/joint-core.log` for current progress.

The original run preserved four completed epochs; its unfinished fifth epoch was
discarded during the authorized stop. An explicit verified import retained its
12,236 AdamW/SAM updates, learning-rate schedule, all RNG states, validation and
elapsed time. The original run, checkpoint and scientific receipts remain
unchanged. The new run records its parent in `checkpoint_import.json` and retains
an `imported_checkpoint.pt` anchor. Ordinary schema-2 resume stays strict; the
supported historical family now has a separately verified import path.

The reconstruction/prediction viewer is active on port 8080. Current and historical
catalog checks passed. The maintained suite passed 99 tests after cutover, and
all 15 existing cache-import sidecars were published without regenerating arrays.
The preserved original run is in
`processing/runs/joint_vjepa21_all_sources32d_mirror_v2/`.
Its raw status file is the final pre-stop observation, not evidence of a live job;
the continuation's status file is authoritative for the current process.
Verified stop and deployment receipts are under
`ai_work/refactor/20261022_000723/`.

The maintained defaults are explicit in `configs/experiment.json`, `model.json`,
`datasets.json`, `training.json` and `transfer.json`. They retain 17 sources,
MotionTrans holdout, FreeTacMan/TacQuad/T-Dex exclusions and the existing mirrored
teacher/code selections. The continuation has a distinct output path and was
started explicitly. Historical launchers/configs are archived under `legacy/v1/`.

Continuation checks passed **182 maintained tests**, GPU updates for all 17
sources, strict import plus ordinary schema-2 restoration, exact saved-checkpoint
SAM parity, and the entire train/validation window ordering. Epoch 5 retains the
original 48,930-batch sampling schedule. Evidence is under
`ai_work/scratch/20261022-resume-joint/audits/`; recovery material is protected under
`ai_work/backups/20261022-resume-joint/`. The prior four epochs took approximately
50–65 minutes each, which initially suggested 22–28 hours for the remaining 26.
The first 701 resumed batches took 89 seconds (about 7.8 batches/second), slower
than that historical average. If sustained, the startup rate suggests roughly
40–50 hours remaining. This is an early estimate; a completed resumed epoch is
needed to include validation and establish steady throughput.

## Artifact authority

- Native chronological source manifest: `processing/data_expansion_v1/manifest.json`.
- Compressed scene metadata: `processing/storage_v1/registry.json`.
- Accepted teachers and model statistics: `processing/joint_acquisitions_mirrored_v2/`.
- Mirror policy: `legacy/v1/configs/bimanual_mirror_policy_v3.json`.
- Verified mirrored teachers/codes: `processing/bimanual_mirror_v4/ready.json`.
- Marker/native source artifacts: existing `processing/marker_displacement_storage_v1/`
  and `processing/data_queue_v2/` paths.
- Consumer imports after the original refactor: `processing/core_v2_cache_imports/`, preserved unchanged.
- Current consumer imports after code-standard formatting: `processing/core_v2_cache_imports_standards_v1/`. Original arrays, teachers and receipts stay untouched.

The live viewer uses the uniform registry. Its explicit historical viewer
preserves prior architectures, model variants and source-specific decoder choices.
The CPU verification covers actual saved model/teacher reconstruction and native
samples, not a new scientific experiment or end-to-end GPU speed benchmark.
Read [verification](refactor_verification.md) for evidence and limitations.

## Code standards maintenance — October 22, 2026

The maintained Python and viewer assets now follow the
[code standards](code_standards.md). Scientific modules have readable statement
layout and public tensor/mask/unit documentation; command scripts document inputs,
outputs and example usage. Pinned development formatters, a read-only standards
checker and the root README command index enforce future maintenance boundaries.

The complete maintained suite passed **156 tests**, including initialization/RNG,
SAM replay, checkpoint and native-data contract checks. Eight active real-data
original/mirrored samples across five sources matched their previous sample,
cached-code and frozen-teacher reconstruction bytes exactly. HTML structure and
JavaScript/CSS tokens were preserved. No new scientific experiment was launched.
The 15 consumer-import sidecars were versioned for the new code bytes; old imports,
original receipts, cached arrays, teacher weights and historical code are intact.
Verification and recovery material are under
`ai_work/scratch/20261022-code-standards/` and
`ai_work/backups/20261022-code-standards/` respectively. Training was stopped at
the end of that maintenance; the later continuation above is now active.
