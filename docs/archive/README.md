# Research and workflow archive

These records preserve the earlier root reports and their scientific qualifications. They include completed results, incomplete/stopped experiments, proposals and point-in-time operational notes. **A filename ending in a newer version does not establish current authority.** Use [project state](../project_state.md), [architecture](../architecture.md), [workflows](../workflows.md) and the [research overview](../research_overview.md) for maintained instructions.

Moved documents keep their original basenames under `root/`; a short archive notice and repaired relative Markdown links are the only editorial changes. Paths written literally in old command examples describe the historical repository layout. Refer to the [script catalog](../script_catalog.md) before executing them. Saved metrics, caches, checkpoints and provenance remain in the original `processing/` locations rather than being duplicated.

The [August 24, 2026 proposal](../proposal/main.pdf) is preserved separately under `docs/proposal/`. Its hypotheses and provisional timeline are not completed findings.

## Project history, proposal context and earlier plans

- [TactileLatent: image-only tactile latent learning](root/README.md) — original `README.md`.
- [TactileLatent: goals, implementation history, and results](root/PROJECT_HISTORY_AND_RESULTS.md) — original `PROJECT_HISTORY_AND_RESULTS.md`.
- [PROJECT_ANALYSIS.md](root/PROJECT_ANALYSIS.md) — original `PROJECT_ANALYSIS.md`.
- [EgoTouch image-only VAE research plan](root/TRAINING_PLAN.md) — original `TRAINING_PLAN.md`.

## Historical DINO model and source-training protocols

- [The current tactile model, taught from first principles](root/ARCHITECTURE_GUIDE.md) — original `ARCHITECTURE_GUIDE.md`.
- [How the training code works](root/TRAINING_GUIDE.md) — original `TRAINING_GUIDE.md`.
- [Twelve-source training with one source held out](root/JOINT_ALL_TRAINING.md) — original `JOINT_ALL_TRAINING.md`.
- [MotionTrans sensor-disjoint baseline and direct control](root/MOTIONTRANS_BASELINE_TRAINING.md) — original `MOTIONTRANS_BASELINE_TRAINING.md`.
- [MotionTrans transfer with a larger frozen-encoder decoder](root/MOTIONTRANS_LARGE_DECODER.md) — original `MOTIONTRANS_LARGE_DECODER.md`.
- [Fresh joint SAM experiment](root/SAM_JOINT_TRAINING.md) — original `SAM_JOINT_TRAINING.md`.
- [Fresh joint training with dataset gradient reversal](root/ADVERSARIAL_TRAINING.md) — original `ADVERSARIAL_TRAINING.md`.
- [Alternating dataset-adversarial training](root/ALTERNATING_ADVERSARIAL_TRAINING.md) — original `ALTERNATING_ADVERSARIAL_TRAINING.md`.
- [Direct VLA_touch image-to-tactile experiment](root/VLA_TOUCH_DIRECT_TRAINING.md) — original `VLA_TOUCH_DIRECT_TRAINING.md`.
- [Joint training with frozen tactile-encoder supervision](root/JOINT_SENSOR_ALIGNMENT.md) — original `JOINT_SENSOR_ALIGNMENT.md`.
- [Feeling of Success: frozen joint-baseline evaluation](root/FEELING_SUCCESS_EVALUATION.md) — original `FEELING_SUCCESS_EVALUATION.md`.

## Tactile teachers, ablations and target changes

- [Small real-data plumbing check: 16 frames per split, CPU, one epoch.](root/EGOTOUCH_AUTOENCODER.md) — original `EGOTOUCH_AUTOENCODER.md`.
- [EGOTOUCH_AUTOENCODER_SEARCH.md](root/EGOTOUCH_AUTOENCODER_SEARCH.md) — original `EGOTOUCH_AUTOENCODER_SEARCH.md`.
- [EGOTOUCH_AUTOENCODER_SEARCH_V3.md](root/EGOTOUCH_AUTOENCODER_SEARCH_V3.md) — original `EGOTOUCH_AUTOENCODER_SEARCH_V3.md`.
- [EGOTOUCH_AUTOENCODER_SEARCH_V4.md](root/EGOTOUCH_AUTOENCODER_SEARCH_V4.md) — original `EGOTOUCH_AUTOENCODER_SEARCH_V4.md`.
- [EGOTOUCH_AUTOENCODER_VIEWER.md](root/EGOTOUCH_AUTOENCODER_VIEWER.md) — original `EGOTOUCH_AUTOENCODER_VIEWER.md`.
- [Keep autograd enabled through the frozen decoder to train the adapter/encoder.](root/SENSOR_AUTOENCODER_PRETRAINING.md) — original `SENSOR_AUTOENCODER_PRETRAINING.md`.
- [Marker autoencoders and frozen-code joint supervision](root/MARKER_AUTOENCODERS.md) — original `MARKER_AUTOENCODERS.md`.
- [Optical tactile targets: tracked dot displacement](root/MARKER_DISPLACEMENT_TARGETS.md) — original `MARKER_DISPLACEMENT_TARGETS.md`.

## Invariance, FGW and transfer studies

- [How FGW alignment and mixing works in TactileLatent](root/FGW_ALIGNMENT_AND_MIXING.md) — original `FGW_ALIGNMENT_AND_MIXING.md`.
- [Self-supervised FGW study](root/FGW_STUDY.md) — original `FGW_STUDY.md`.
- [Matched 16D latent experiment](root/TRAINING_16D_PAIR.md) — original `TRAINING_16D_PAIR.md`.
- [Invariance objectives on the existing 16D model](root/INVARIANCE_TRAINING.md) — original `INVARIANCE_TRAINING.md`.
- [Shared/private tactile representation: implementation plan](root/SHARED_PRIVATE_META_LEARNING_PLAN.md) — original `SHARED_PRIVATE_META_LEARNING_PLAN.md`.
- [Fresh deterministic 16D control](root/DETERMINISTIC_CONTROL.md) — original `DETERMINISTIC_CONTROL.md`.
- [Epoch-7 continuation experiment](root/CONTINUATION_TRANSFER_EXPERIMENT.md) — original `CONTINUATION_TRANSFER_EXPERIMENT.md`.
- [Frozen representation screen and epoch-22 stopping decision](root/FROZEN_REPRESENTATION_SCREEN.md) — original `FROZEN_REPRESENTATION_SCREEN.md`.

## Historical V-JEPA architectures and caching

- [Frozen V-JEPA 2.1: deterministic and stochastic 32D joint models](root/VJEPA_32D_EXPERIMENT.md) — original `VJEPA_32D_EXPERIMENT.md`.
- [V-JEPA attention readout for per-frame tactile latents](root/VJEPA_ATTENTION_PROPOSAL.md) — original `VJEPA_ATTENTION_PROPOSAL.md`.
- [Full-clip V-JEPA feature cache](root/VJEPA_FEATURE_CACHE.md) — original `VJEPA_FEATURE_CACHE.md`.
- [Full-spatial V-JEPA deterministic joint training](root/VJEPA_SPATIAL_LIVE_RUN.md) — original `VJEPA_SPATIAL_LIVE_RUN.md`.
- [V-JEPA joint training with cached marker codes](root/VJEPA_MARKER_CODE_RUN.md) — original `VJEPA_MARKER_CODE_RUN.md`.

## Data acquisition/expansion snapshots

- [Baihu, EgoPressure and FEEL integration](root/DATA_EXPANSION.md) — original `DATA_EXPANSION.md`.
- [Additional FTP-1 data](root/FTP_EXTRA_DATASETS.md) — original `FTP_EXTRA_DATASETS.md`.
- [Joint datasets: sensors, inputs, and prediction targets](root/JOINT_DATASETS.md) — original `JOINT_DATASETS.md`.

## Superseded maintained guides

- [Bimanual mirroring v1 guide](superseded_guides/BIMANUAL_MIRRORING.md): sources then blocked and the original adaptation/preparation stage. Current [mirror guide](../BIMANUAL_MIRRORING.md) records the five-source v3-policy/v4-teacher state.
- [Acquired-datasets measured-only guide](superseded_guides/ACQUIRED_DATASETS_VIEWER.md): early visualization integration before later teachers/joint heads. Current [source viewer guide](../ACQUIRED_DATASETS_VIEWER.md) records later capability.

## Later results retained at their artifact locations

- [Current-frame 32D versus 128D, epoch 30](../../processing/evaluations/motiontrans_vjepa21_current_epoch030_v1/REPORT.md): three-hidden-layer readouts; paired difference unresolved.
- [Matched previous-versus-current V-JEPA epoch 30](../../processing/evaluations/motiontrans_vjepa21_current_vs_previous_epoch030_v1/REPORT.md): one-target-per-clip matched comparison; no clear transfer gain.
- [Marker interaction dynamics audit](../../processing/research/marker_dynamics_v1/REPORT.md): weak fine dynamics in FreeTacMan despite a useful teacher; source subsequently removed by the user.
- [Grokking audit](../../processing/research/grokking_audit_v1/REPORT.md): no convincing delayed generalization transition in inspected historical curves.
- [Mirror-adapted teacher report](../../processing/bimanual_mirror_v4/REPORT.md): teacher/code preparation snapshot; its statement that joint training was stopped was true at preparation, before the later authorized launch.

Existing investigation notes kept at stable `docs/` paths are indexed in the [documentation guide](../README.md). Their status language is historical; their findings are not silently rewritten to match later policy choices.

The refactor backup includes a timestamped checksum manifest and the document relocation map. Research records and uncertain files must not be automatically deleted.

## Preserved operational fragments

[Operations archive](operations/README.md) preserves the small former root `0` download-status fragment and explains retained historical log paths. These are operational records, not research results.
