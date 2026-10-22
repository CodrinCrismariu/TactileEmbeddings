# Bimanual mirroring and adapted decoders

**Maintained scope, October 21, 2026:** the current 17-source joint run enables five mirror transformations using `legacy/v1/configs/bimanual_mirror_policy_v3.json` and `processing/bimanual_mirror_v4/ready.json`. The [previous v1 guide](archive/superseded_guides/BIMANUAL_MIRRORING.md), [gel v2 review](GEL_MIRROR_MAPPING_V2.md), and [v3 continuation](MIRROR_AUTOENCODER_CONTINUATION_V3.md) remain historical evidence, including former blocked mappings.

For each eligible clip, mirroring is sampled with probability 0.5 **before wrist dropout** and frozen V-JEPA encoding. All eight RGB frames reflect horizontally; left/right wrist images and availability masks exchange slots. The tactile targets, validity and geometry transform consistently without changing chronology, magnitude or native resolution. This teaches output equivariance, not identical latents for both handednesses. SAM replays exactly the same transformed inputs/features/supervision.

| Source | Defined transformation | Adapted reconstruction |
|---|---|---|
| EgoTouch | Swap hands and horizontally flip both native pressure grids | Accepted 256D teacher handles both versions |
| OpenGraph | Swap hands and reverse four columns within each named 4×4 finger grid | Shared-hand AE, full 16D code = two 8D hand codes |
| Unit Bimanual | Swap two surfaces and reflect horizontal marker coordinates/vectors | Shared-hand marker AE, full 16D = two 8D hand codes |
| Baihu | Four-surface mapping `[3,2,1,0]`; reflect horizontal geometry/vectors | Accepted 32D four-surface marker teacher |
| RDP Bimanual | Swap two scalar force targets, preserving their existing signs/scales | Direct head; no autoencoder |

For marker reflection, reference `u'=1-u`, `v'=v`, displacement `dx'=-dx`, `dy'=dy`. Marker identity, surface membership, native spacing, unequal counts, padding and missing tracks stay attached. The mapping is involutive. Both original and mirrored targets are encoded freshly; arbitrary sensor-code coordinates are never permuted to fake mirrored codes.

**Evidence matters:** EgoTouch uses fixed official masks. Unit/Baihu horizontal reflection and gripper correspondence were visually reviewed by the user. OpenGraph uses the user's explicitly requested displayed-grid reflection; policy metadata states `hardware_geometry_verified=false`. Calling the operational mapping verified does not turn it into an independently calibrated physical wiring map. FEEL Kitchen and T-Rex have no enabled mirror version because their sensor/channel/axis maps remain unresolved. Single-hand sources, including EgoPressure, remain original-only.

## Decoder quality and code caches

The same adapted encoder/decoder handles **both** orientations. Historical original-only teachers remain available; the viewer prefers accepted mirror-adapted models.

| Teacher | Original MAE | Mirrored MAE | Units |
|---|---:|---:|---|
| EgoTouch | 0.00325482 | 0.00335874 | Existing normalized pressure |
| OpenGraph | 0.00862824 | 0.00862824 | Existing normalized taxels |
| Unit Bimanual | 0.14448642 | 0.14448642 | Native pixels |
| Baihu | 0.08737078 | 0.08798229 | Native pixels |

These are teacher reconstruction audits, **not joint visual prediction errors**, and different units must not be ranked together. OpenGraph trained 90 epochs, selected 80; Unit trained 100, selected 99. Both preserve their 16D code sizes and use shared per-hand weights/canonical coordinates. EgoTouch/Baihu reuse qualifying v3 teachers. Selection required absolute reconstruction quality, original-teacher retention, zero-baseline checks and parity; equal errors alone were insufficient.

Paired code caches contain separately encoded original/transformed measurements with equal-orientation training-only statistics. Receipts pin teacher, target, policy and implementation hashes. Fresh producer-batch encodings reproduced checked cached codes with zero maximum difference. The combined v4 receipt includes unchanged historical per-source receipts where applicable rather than relabeling their provenance.

Artifacts:

- [Policy](../legacy/v1/configs/bimanual_mirror_policy_v3.json) and [teacher preparation config](../legacy/v1/configs/mirror_adapted_teachers_v4.json).
- [Combined readiness](../processing/bimanual_mirror_v4/ready.json), [quality report](../processing/bimanual_mirror_v4/REPORT.md), [integration verification](../processing/bimanual_mirror_v4/integration_verification.json).
- New [OpenGraph quality](../processing/bimanual_mirror_v4/teachers/opengraph/quality.json) and [Unit Bimanual quality](../processing/bimanual_mirror_v4/teachers/ftp_unit_bimanual/quality.json).
- [Viewer verification](../processing/bimanual_mirror_v4/viewer_verification.json).

`prepare_all_mirrored_teachers.py` trains/builds these artifacts; do not rerun it to view them. `verify_all_mirrored_teachers.py` verifies cached codes and prepares data/teacher metadata, but is not a production joint-training launcher. Read the [script catalog](script_catalog.md) before either command.

## Viewer and validation

Open **Mirror quality check** (`?page=mirror`) on port 8080. It compares measured original/transformed targets with each selected teacher's independent reconstruction, alongside reflected RGB/wrist roles. Marker grids show positions and signed arrows; shared scales avoid amplifying one orientation's small fluctuations.

Ordinary source validation stays unaugmented. The joint trainer additionally logs original/mirrored native reconstruction and prediction-equivariance error; these do not replace ordinary source checkpoint selection. No transport solver, mirror flag or tactile target is required for ordinary joint inference. See [architecture](architecture.md) and [current run](project_state.md).
