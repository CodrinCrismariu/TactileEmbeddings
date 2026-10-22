# Project state and artifact authority

**Observed October 21, 2026, 11:15 PM EDT (America/New_York).** This is an audit snapshot, not live process state. Status files and metrics can advance while this document is being read.

## Active experiment observed

The user service `tactilelatent-joint-all-sources-v2.service` was active. Its launcher ran `scripts/run_all_sources_joint.py --config configs/joint_vjepa21_all_sources_32d_mirror_v2.json`; trainer PID 4125969 was validating epoch 3, with epochs 1 and 2 committed. The separate `tactilelatent-viewer.service` was serving port 8080. These observations do not authorize pausing either service during maintenance.

| Setting | Observed value / evidence |
|---|---|
| Config | [`joint_vjepa21_all_sources_32d_mirror_v2.json`](../configs/joint_vjepa21_all_sources_32d_mirror_v2.json) |
| Output | [`processing/runs/joint_vjepa21_all_sources32d_mirror_v2/`](../processing/runs/joint_vjepa21_all_sources32d_mirror_v2/) |
| Task model | Frozen V-JEPA 2.1 B/16, full spatial tokens, two trainable attention blocks, deterministic 32D current-frame latent |
| Initialization | Fresh task weights, seed 45; pretrained frozen backbone |
| Schedule | 30 configured epochs, SAM AdamW, batch 2, accumulation 16 |
| Source scope | 17 source heads; MotionTrans excluded from source fitting/validation |
| Augmentation | 50% eligible-clip mirroring for EgoTouch, OpenGraph, Unit Bimanual, RDP Bimanual, Baihu |
| Disabled objectives | No KL, Gaussian sampling, FGW, relational, retention, meta-learning, or static episode context |
| Visual processing | Eight chronological 384×384 RGB frames per available view; no persistent V-JEPA visual-token cache |

Completed records at observation:

| Epoch | Equal-source validation supervision score | Mean-latent effective rank | Epoch seconds |
|---:|---:|---:|---:|
| 1 | 0.72355293 | 3.8347 | 3008.96 |
| 2 | 0.63616946 | 5.2875 | 3025.20 |

This score averages code-supervision/direct-numeric losses, not native physical MAE or MotionTrans transfer. Native metrics have source-specific units and cannot be ranked across sensors. Effective rank describes spread and does not establish invariance or predict transfer by itself. No MotionTrans transfer result for this active run was established in this audit.

An epoch is capped: up to 8,000 eligible clips per source, with cycling coverage rather than a complete pass through every large source. The run's [coverage receipt](../processing/runs/joint_vjepa21_all_sources32d_mirror_v2/clip_coverage.json) records 748,784 eligible clips, 97,857 sampled per epoch and 48,930 microbatches. Full coverage requires 36 configured epochs because of T-Rex; 30 reaches about 83.6% of its eligible clips. Mirroring is random and does not guarantee both versions of every clip.

## Current target and teacher scope

The 17 sources are EgoTouch, EgoTactile, OpenGraph, OpenTouch, FTP ViTaMIn, Unit, Unit Bimanual, VLA Touch, RDP, RDP Bimanual, Baihu, EgoPressure, FEEL Lab, FEEL Kitchen, HRDexDB Allegro V5, HRDexDB Inspire F1 and T-Rex.

The prepared [model configuration](../processing/joint_acquisitions_mirrored_v2/model_config.json) has 13 sensor-code heads and four direct target heads. FEEL Lab (5), Allegro (4), RDP (1) and RDP Bimanual (2) have no unnecessary autoencoder. Four optical sources use marker displacement: Unit, Unit Bimanual, VLA Touch and Baihu. FreeTacMan, TacQuad and T-Dex remain excluded at the user's request. EgoPressure uses native 224×224 hand UV pressure only and excludes calibration-routine episodes.

[Mirror preparation v4](../processing/bimanual_mirror_v4/ready.json) is complete for all five enabled mirror sources. EgoTouch/Baihu reuse accepted adapted teachers; OpenGraph/Unit Bimanual use shared-hand 16D autoencoders; RDP Bimanual swaps scalar targets. OpenGraph's within-finger reflection is a recorded user-requested display-coordinate policy, **not independently calibrated hardware geometry**.

## Which artifact answers which question?

- [`status.json`](../processing/runs/joint_vjepa21_all_sources32d_mirror_v2/status.json): transient stage/batch progress; cross-check live process/service identity.
- [`metrics.jsonl`](../processing/runs/joint_vjepa21_all_sources32d_mirror_v2/metrics.jsonl): completed-epoch losses, native validation, latent statistics and mirror diagnostics.
- [`config.json`](../processing/runs/joint_vjepa21_all_sources32d_mirror_v2/config.json): exact run settings; compare with the named source config rather than changing it beneath resume.
- [`provenance.json`](../processing/runs/joint_vjepa21_all_sources32d_mirror_v2/provenance.json): hashes, backbone, targets, frame mapping, caches and implementation identities.
- [`last.pt`](../processing/runs/joint_vjepa21_all_sources32d_mirror_v2/last.pt): latest committed resumable model/optimizer/SAM/scheduler/RNG state; per-epoch viewer snapshots are inference artifacts, not equivalent resume checkpoints.
- [`launch.json`](../processing/runs/joint_vjepa21_all_sources32d_mirror_v2/launch.json): requested launch, service, preflight and initial-state evidence.
- [`teachers.json`](../processing/joint_acquisitions_mirrored_v2/teachers.json) and [mirror v4 receipt](../processing/bimanual_mirror_v4/ready.json): actual teacher/cached-code bindings.

A legacy text field `tactile_encoding` in this run's provenance still summarizes an earlier ten-source stage. The actual 17-source configuration/model registry and acquisition/mirror bindings establish scope. Do not edit a pinned artifact to make its summary current.

## Historical, stopped, and uncertain

The [research archive](archive/README.md) distinguishes old DINO direct-output/code-supervision experiments, FGW tests, 16D relational/meta-learning studies, and pooled/spatial V-JEPA studies. A saved `status.json` without a live matching process is not evidence that an old run remains active. Failed reconstruction attempts, paused trajectory work and unexecuted broad sweeps are not successful completed experiments.

MotionTrans has repeatedly informed model design and is a development holdout, not an untouched final test set. Calibrated uncertainty remains unresolved; the active deterministic model provides none. Historical Gaussian sampling and stochastic mixing did not establish reliable coverage. Do not interpret source reconstruction improvements as demonstrated transfer gains.
