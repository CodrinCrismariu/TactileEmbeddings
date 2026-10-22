# Documentation guide

Read in this order:

1. [Research overview](research_overview.md): question, hypotheses, completed evidence and unresolved claims.
2. [Project state](project_state.md): dated active-job observation, exact configurations and artifact authority.
3. [Architecture](architecture.md): how video, attention, latents, teachers and decoders fit together.
4. [Workflows](workflows.md): setup, viewing, training/resume and evaluation boundaries.
5. [Script catalog](script_catalog.md): every retained command, its status, inputs, outputs and effects.
6. [Compatibility](compatibility.md): canonical modules, historical imports and source-hash exceptions.

## Data and visualization references

- [Current bimanual mirroring](BIMANUAL_MIRRORING.md): eligible sources, geometry conventions, adapted decoders and cache provenance.
- [Acquired datasets in the viewer](ACQUIRED_DATASETS_VIEWER.md): native targets, reconstruction models, fixed scales and missing observations.
- [Training monitor](TRAINING_DASHBOARD.md): live batch samples versus completed-epoch metrics.
- [Attention block walkthrough](attention_block_walkthrough.md) and the [manual implementation](../examples/manual_attention_block.py).
- [EgoPressure 3D UV visualization](EGOPRESSURE_VISUALIZATION.md) and [task-only selection](EGOPRESSURE_DATA_SELECTION.md).
- [Fixed sensor zero references](FIXED_SENSOR_ZERO_REFERENCES.md): limitations on physical calibration; includes explicitly historical removed sources.
- [FEEL hand ownership](FEEL_HAND_OWNERSHIP.md): Lab right hand versus Kitchen left/right mappings.
- [Download/processing queue](DATA_DOWNLOAD_AND_PROCESSING_QUEUE.md): operational reference; consult live receipts before assuming a job remains active.

## Organization, verification and recovery

- [Refactor migration and recovery](refactor_migration.md): prepared changes, active-service safety gates, approval, deployment and rollback.
- [Refactor verification](refactor_verification.md): baseline versus staged tests, unavailable resources and compatibility checks.
- [Persistent working rules](../AGENTS.md) and [local agent work boundaries](../ai_work/README.md).

## Historical material

The [archive index](archive/README.md) preserves the former root reports and superseded maintained guides. The [original proposal](proposal/main.pdf) is a research proposal, not a description of completed experiments.

Existing investigation notes retain their original paths for reproducibility:
[mirror mapping investigation](BIMANUAL_MAPPING_INVESTIGATION.md),
[user-reviewed gel mapping v2](GEL_MIRROR_MAPPING_V2.md),
[mirror-teacher continuation v3](MIRROR_AUTOENCODER_CONTINUATION_V3.md), and
[Baihu autoencoder investigation](BAIHU_AUTOENCODER_INVESTIGATION.md).
Their statements about “active”, blocked sources, or unstarted jobs describe their recorded stage. The maintained [project state](project_state.md) and [mirror guide](BIMANUAL_MIRRORING.md) explain the later v4 preparation.

When adding a durable result, update the relevant maintained document or archive a dated experiment report with its configuration, artifacts and limitations. Do not infer authority from filename recency or duplicate the current instructions in another root report.
