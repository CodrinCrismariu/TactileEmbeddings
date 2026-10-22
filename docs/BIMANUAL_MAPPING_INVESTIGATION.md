# Bimanual mirror integration and physical mapping investigation

Updated 2026-10-19. The investigation was performed locally and against the
official releases; no correspondence was supplied by the user.

**OpenGraph, Unit Bimanual, and Baihu now have executable target/geometry
previews in the mirror viewer. They are not verified training augmentations.**
The two accepted v1 sources, their teacher, and paired code caches are preserved.

## What is available

Open **Mirror quality check** in the running viewer. All five bilateral sources
are selectable. EgoTouch and RDP use their verified mappings. The other three
display an orange **UNVERIFIED MAPPING PREVIEW** notice and a hypothesis selector.

| Source | Executable preview | Physical evidence still absent |
|---|---|---|
| OpenGraph | Eight within-finger channel hypotheses, hand swap, five labeled 4×4 acquisition grids per hand, fresh selected-AE reconstruction | Across-finger orientation and the recorded release's thumb convention |
| Unit Bimanual | Hand swap, four local-axis hypotheses, transformed reference positions and signed vectors, fresh marker AE | Image axes and fingertip ownership in the actual left/right mounts |
| Baihu | Both candidate jaw correspondences and independently selected axis flips for each surface pair; all four surfaces; platform shown | Correspondence and local axes for each platform and mount revision |

The hypothesis families are deliberately limited. For example, they do not
cover every possible 90-degree sensor rotation or every finger-specific wiring
revision. The first dropdown choice is merely a preview, not a preferred or
automatically inferred mapping. Selections are never saved into training policy.

RGB is horizontally reflected and wrist roles and availability are exchanged.
Native tactile shapes, chronology, pressure magnitudes, marker spacing, and
missing observations are preserved. A marker reflection transforms the reference
position affinely and the signed displacement linearly. The teacher receives
those transformed measurements and geometry; reconstructed images/vectors and
arbitrary sensor-code coordinates are never substituted for teacher inference.

The existing preparation and joint trainer still reject these three sources
because the training policy remains blocked. Low AE error does not establish
the correct physical map. Viewing or switching a hypothesis cannot bypass that
gate. No new joint run was launched.

## Evidence searched and obtained

The raw source files and public API responses are retained under
`processing/audits/bimanual_mirror_mapping_v3/`. The previous full native-target
inventory audit remains under `bimanual_mirror_mapping_v2/`.

### OpenGraph

- Local release: both labeled 80-channel schemas and all 144 raw tactile
  streams. Named fingers remain `index, middle, pinky, ring, thumb`. SDK wire
  order must not be applied a second time to these labeled arrays.
- [Official SDK orientation implementation](https://github.com/OpenGraphLabs/oglo-python/blob/26ba0e812ab1065accfdcaf17ec4d3223a232ad3/src/oglo/_frame.py):
  columns run along the finger; the left thumb has a measured column reversal;
  the across-finger row direction is explicitly unmeasured. Its September
  measurement does not certify the June dataset's export convention.
- [Official SDK PR 8](https://github.com/OpenGraphLabs/oglo-python/pull/8)
  identifies `OpenGraphLabs/oglo-hardware#41` and Rev-T KiCad as the relevant
  hardware evidence. That repository and issue return 404 both anonymously
  and through the connected GitHub account. No private CAD content was obtained.
- All public OpenGraph repositories and relevant SDK issue/PR descriptions
  were searched. Local home/Documents searches found no matching hardware files.
- NVIDIA IsaacTeleop, revision
  `17a3eacd49a8ee86ba37666c311f79c24b3fb9bd`: its
  [heatmap implementation](https://github.com/NVIDIA/IsaacTeleop/blob/17a3eacd49a8ee86ba37666c311f79c24b3fb9bd/examples/oglo_tactile/oglo_heatmap.py)
  instead describes *rows* as the fingertip direction. This conflicts with the
  SDK's measured column description. Display code cannot resolve that conflict.

### Unit Bimanual

- All 380 retained marker episodes have established hand ownership and native
  240×320 geometry. Original tactile imagery stays retired.
- [UniT recorder](https://github.com/ZhengtongXu/UniT/blob/52a286520b09708934b25c77aa826360d72c79db/UniT/utils/tactile_recorder.py),
  its complete public repository tree/issues, and the associated
  [GelSight driver](https://github.com/raghavauppuluri13/gsmini/blob/4ab09f6ac7c38d598f0280492a1a2ac45c6eed6c/gsmini/gsdevice.py)
  establish direct image recording and crop/resize behavior. They contain no
  bilateral mounting transform.
- [Official project photos and collection demonstrations](https://zhengtongxu.github.io/unit-website/)
  identify the physical setup but do not annotate image x/y axes or certify
  mirrored fingertip correspondence.
- [Manufacturer CAD references](https://github.com/gelsightinc/gsrobotics/blob/main/README.md)
  describe generic sensor/adapter geometry. A generic adapter is not evidence
  that the recorded UniT robots used that mounting and orientation.

### Baihu

- Previous audit: 3,013 marker episodes; four explicit stream identities;
  2,564,143 chronological frames; D-WHEEL, PIKA, and QINGLOONG.
- The [official conversion directory](https://ai.atomgit.com/openloong/visuo-tactile/tree/main/VISUO-TACTILE%E6%95%B0%E6%8D%AE%E4%BD%BF%E7%94%A8%E6%8C%87%E5%8D%97/data_convert)
  was retrieved through the site's public repository API after inspecting its
  frontend. All eight guide/conversion files were saved with Git blob/commit
  metadata. They copy the four named streams without a reflection calibration.
  A synthetic-zero fallback in that converter is not used by our native loader.
- The [official acquisition schematic](https://huaqianxi.github.io/V-Touch/)
  identifies ViTai GF225 sensors across the three platforms. It does not label
  their image axes.
- Manufacturer [GF225 documentation](https://docs.vitai.site/en/docs/sdk-usage/api/gf225/),
  calibration docs, and six public [ViTai repositories](https://github.com/ViTai-Tech)
  were checked; 42 small source/documentation files were saved at pinned
  revisions. UMI configuration assigns sensor serial numbers to left/right
  fingers but does not supply the dataset's per-platform image mounting map.
- The 6,213 retained camera parameter files describe scene cameras, with
  identity extrinsics. They are not tactile-image extrinsics.

## Baihu teacher repair and preparation

The four-surface architecture passes real native optimization updates for all
three platforms and unequal marker counts. The production trainer's validation
path had a real broadcasting error: a `[frames, markers]` validity mask was
used against `[frames, markers, 2]` vectors without expanding its final axis.
This is fixed and covered by a missing-observation regression test.

The authorized 30-epoch, seed-45, 32D native Baihu teacher runs separately in
`processing/data_expansion_v1/teachers/baihu_native_v2/`. Historical folders are
preserved. Its service is `tactilelatent-baihu-teacher-native-v2.service`; its
continuous log is `processing/data_expansion_v1/baihu_teacher_native_v2.log`.
The viewer exposes saved best checkpoints during this preparation.

After 30 epochs, it must beat native zero-prediction MAE by at least 10% before
its codes can be cached and published. Failure leaves it unpromoted. Successful
completion caches only native 32D sensor codes, preserves the active UV-only,
task-only EgoPressure selection, and publishes the prepared expanded source
configuration. This does not enable a physical Baihu mirror or start joint training.

**This attempt finished all 30 epochs and failed that quality gate.** Best
validation MAE was **0.266962 px**, versus **0.266991 px** for zero displacement
(only about 0.011% better). No Baihu codes or expanded-source publication were
created from it, and no further search was launched. Its checkpoint remains
inspectable in the viewer with a **native quality failed** label. The validation
broadcasting bug was repaired. A subsequent investigation identified a collapsed
code projection and poor batch mixing; **native v3 now reconstructs successfully**
with validation MAE 0.087232 px. Its 30-epoch teacher passed quality checks and
all 3,013 episodes have verified frozen codes. See
[Baihu autoencoder investigation](BAIHU_AUTOENCODER_INVESTIGATION.md).
This resolves teacher readiness; the physical mirror mappings remain unverified.

Cache integrity checks replay the actual 128-frame producer chunks. This avoids
confusing batch-dependent CUDA attention kernels with corrupted stored codes;
the numerical tolerance remains unchanged.

## Acceptance and remaining limit

The new geometry renderer and backend handle taxels and marker sets, including
unequal surface counts, masks, fresh teacher inference, and native pixel units.
Real browser checks cover all five source selections; snapshots and results
are under `bimanual_mirror_mapping_v3/viewer/`. Training eligibility is a
separate field from preview availability.

`scripts/audit_mirror_previews.py` also passed six real episode checks: OpenGraph,
both Unit tasks, and all three Baihu platforms. It checked 112 hypothesis round
trips on native data, fresh teacher reconstruction, exact reflection of decoded
RGB for every available camera slot, and rejection of unverified mappings by
default. All 12 pinned storage implementation files and the original EgoTouch
mirror teacher/policy hashes still match. Results are in
`bimanual_mirror_mapping_v3/native_preview_checks.json`. These checks establish
software correctness for each selected hypothesis, not its physical truth.
The focused regression suite passed **50 tests**, including missing-marker
validation and producer-batch cache replay.

Completing physical verification requires independent evidence associating
known anatomical/sensor locations and positive directions across the actual
paired mounts, for every relevant hardware revision. No such complete evidence
was found in the searched accessible release. Inferring it from a symmetric dot
grid, a round trip, or reconstruction loss would not resolve the ambiguity.
Consequently, full training integration of these three sources remains incomplete.
# Current gel mapping update

The Unit Bimanual and Baihu blocks described in this investigation apply to
the historical v1 policy. The user subsequently confirmed horizontal reflection
for both sources and exchange of Baihu jaw sides. The viewer now uses the
separate v2 policy. See [the confirmed mappings and teacher/cache preparation](GEL_MIRROR_MAPPING_V2.md).
OpenGraph remains unresolved. Original v1 evidence, teachers and caches are retained.

