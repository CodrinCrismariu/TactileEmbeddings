> Historical record, relocated on October 21, 2026. Original path: `JOINT_DATASETS.md`.
> Results, proposals, and operational status below belong to their recorded experiment/date;
> use the [maintained project state](../../project_state.md) for the audited active workflow.

# Joint datasets: sensors, inputs, and prediction targets

Updated 2026-10-01. This document describes the **12 sources actually indexed
by the current project**, rather than every dataset downloaded on disk.
The authoritative manifest is [processing/joint_all/manifest.json](../../../processing/joint_all/manifest.json).
Target definitions are in [joint_all_config.py](../../../legacy/v1/src/tactilelatent/joint_all_config.py)
and [prepare_joint.py](../../../legacy/v1/src/tactilelatent/prepare_joint.py); decoding and
normalization are in [joint_data.py](../../../legacy/v1/src/tactilelatent/joint_data.py).

The **current design** trains the non-adversarial joint baseline on **11
sources including VLA_touch**, and holds out **`ftp_motiontrans`**. A separate
MotionTrans-only model provides a direct-training comparison. Both start
with fresh learned weights; only frozen DINOv3 is pretrained. The earlier
VLA_touch-holdout experiments remain preserved historical results, not the
current unseen-sensor benchmark. All experiments use images as input and a shared
32-dimensional Gaussian latent. Each source has a decoder for its own target
layout. Tactile labels supervise training; they are not encoder inputs.
Hand pose, robot actions, joint state, IMU, and force/torque readings are not
additional encoder inputs.

## What the sensor types mean

- **Pressure arrays / taxels:** numerical readings from discrete tactile
  sensing locations on a hand or glove. A taxel is a tactile sensing element.
  A grid and a flattened vector are two storage layouts for these readings.
- **Optical tactile sensors:** a camera records the appearance or deformation
  of a contact surface. The target is the sensor's RGB image. Predicting that
  image does not directly predict pressure in calibrated physical units.
- **Gripper-force measurements:** numerical gripper-force channels, rather
  than a spatial pressure image.
- **Torque measurements:** numerical torque channels from fingers/thumb,
  rather than force, pressure, or optical appearance.

Names below describe modalities supported by the local source cards and
array schemas. The loader does not establish whether every pressure device
is resistive, capacitive, or piezoelectric, or verify a manufacturer for every
optical device; those details should not be inferred from array shape alone.

## All current sources and what we predict

Shapes are **per sampled frame**, ordered `[channels, height, width]`.
A full target clip adds an eight-frame dimension: `[8,C,H,W]`.

| Source / project key | Sensor type and recorded tactile modality | Target used by this project | Target shape; values per frame | Visual input views |
| --- | --- | --- | --- | --- |
| [EgoTouch](https://huggingface.co/datasets/zhouzhoujy/EgoTouch) — `egotouch` | Wearable tactile pressure sensing on both hands; released left/right pressure grids | Left and right normalized pressure maps from `left_pressure_grid` and `right_pressure_grid` | `[2,21,21]`; 882 pressure values | Chest/main + left wrist + right wrist |
| [EgoTactile](https://huggingface.co/datasets/HustleHard/EgoTactile) — `egotactile` | Full-hand tactile pressure glove; numerical sensor readings | The **right-hand stored 256-value array**, `RH.sensor_256`, kept as a vector | `[1,1,256]`; 256 sensor values | Egocentric RGB |
| [OpenTouch](https://huggingface.co/datasets/rayxsong/opentouch) — `opentouch` | Human-hand tactile pressure array | Right-hand `right_pressure` map; the current loader does not predict a left-hand map | `[1,16,16]`; 256 pressure values | Egocentric RGB |
| [OpenGraph ego-tactile](https://huggingface.co/datasets/OpenGraphLabs-Research/ego-tactile-manipulation) — `opengraph` | OGLO tactile gloves on both hands; five fingers × 4×4 taxels = 80 channels per hand | Two 80-value hand vectors; finger-grid layout remains flattened | `[2,1,80]`; 160 sensor values | Egocentric RGB |
| [FreeTacMan](https://huggingface.co/datasets/OpenDriveLab/FreeTacMan) — `freetacman` | Two optical, camera-based tactile surfaces in its robot-free manipulation collection system | Both tactile RGB videos, resized to 32×32 and concatenated by channel | `[6,32,32]`; 6,144 RGB values | Wrist/scene RGB |
| [FTP-1 Unit_Bimanual](https://huggingface.co/datasets/MJJJJ1064/FTP-1-Dataset/tree/main/Unit_Bimanual) — `ftp_unit_bimanual` | Bilateral optical gripper tactile sensors | `left_tactile_data_gripper` + `right_tactile_data_gripper`; one RGB tactile image per side, resized to 32×32 | `[6,32,32]`; 6,144 RGB values | Main + both wrists |
| [FTP-1 Unit](https://huggingface.co/datasets/MJJJJ1064/FTP-1-Dataset/tree/main/Unit) — `ftp_unit` | Optical gripper tactile sensor | `right_tactile_data_gripper`; one RGB tactile image resized to 32×32 | `[3,32,32]`; 3,072 RGB values | Main + right wrist |
| [FTP-1 VLA_touch](https://huggingface.co/datasets/MJJJJ1064/FTP-1-Dataset/tree/main/VLA_touch) — `ftp_vla_touch` | Optical gripper tactile sensor | `right_tactile_data_gripper`; one RGB tactile image resized to 32×32 | `[3,32,32]`; 3,072 RGB values | Main + right wrist |
| [FTP-1 RDP](https://huggingface.co/datasets/MJJJJ1064/FTP-1-Dataset/tree/main/RDP) — `ftp_rdp` | Archive includes optical GelSight/MCTac streams and a numerical gripper-force stream | **Only** `left_tactile_data_gripperforce_flexivgripper`; optical streams are not current prediction targets | `[1,1,1]`; one force value | Ego/main + left wrist |
| [FTP-1 RDP_Bimanual](https://huggingface.co/datasets/MJJJJ1064/FTP-1-Dataset/tree/main/RDP_Bimanual) — `ftp_rdp_bimanual` | Archive includes bilateral optical GelSight/MCTac streams and numerical gripper-force streams | **Only** left and right `*_tactile_data_gripperforce_flexivgripper` channels | `[2,1,1]`; two force values | Ego/main + both wrists |
| [FTP-1 MotionTrans](https://huggingface.co/datasets/MJJJJ1064/FTP-1-Dataset/tree/main/MotionTrans) — `ftp_motiontrans` | Numerical finger/thumb torque measurements | `right_tactile_data_fingertorque` + `right_tactile_data_thumbtorque`; four finger channels and two thumb channels | `[6,1,1]`; six torque values | Ego/main RGB |
| [FTP-1 ViTaMIn](https://huggingface.co/datasets/MJJJJ1064/FTP-1-Dataset/tree/main/ViTaMIn) — `ftp_vitamin` | Two optical tactile images in the right-gripper sensor array | `right_tactile_data_gripper`; two RGB sensor images concatenated into six channels, resized to 32×32 | `[6,32,32]`; 6,144 RGB values | Right wrist RGB |

**Six RGB channels do not always mean two hands.** Unit_Bimanual combines
left and right sensor images. ViTaMIn's two images come from the
right-gripper array. FreeTacMan combines its two tactile surfaces.

**EgoTactile schema caveat:** its local dataset card describes 162 physical
sensing locations, but the downloaded records consumed by our loader expose
`RH.sensor_256` with 256 entries. The present model predicts all 256 stored
entries. We have not established a physical 162-to-256 mapping, padding
convention, or calibrated force interpretation for every entry.

## Normalization and the meaning of the predicted numbers

| Sources | Transformation used by the loader | Meaning of a prediction |
| --- | --- | --- |
| EgoTouch | Released grid / 1.0 | Released normalized pressure value; no extra physical-unit conversion |
| EgoTactile | Stored sensor value / 255 | Source-local normalized sensor reading |
| OpenTouch | Stored pressure / 3072 | Source-local normalized pressure reading |
| OpenGraph | Stored taxel reading / 4095 | Source-local normalized taxel reading |
| FreeTacMan and optical FTP sources | RGB / 255, then resize to 32×32 where needed | Optical tactile RGB intensity in [0,1] |
| RDP and RDP_Bimanual | `0.5 + 0.5*tanh(raw_force/30)` | Bounded representation of a signed force reading |
| MotionTrans | `0.5 + 0.5*tanh(raw_torque/1000)` | Bounded representation of a signed torque reading |

The force and torque transform scales, 30 and 1000, are project settings,
not verified sensor calibration constants. Zero raw force/torque maps to
0.5; negative readings map below 0.5 and positive readings above it.
Finite transformed values can be inverted with
`scale * atanh(2*prediction - 1)`, but values close to 0 or 1 are sensitive
to small prediction errors. The loader does not establish SI units.

A value of 0.5 means different things across these sources. Likewise,
normalized MAE cannot be compared directly between pressure grids, optical
RGB, force, and torque as if they were the same physical quantity. For
VLA_touch, reported MAE, CRPS, sample spread, and interval coverage measure
**optical RGB prediction**, not calibrated pressure uncertainty.

## Current usable splits and clip counts

These counts come from the current manifest and `JointClipDataset`.
Train/validation assignments are episode-disjoint. Clips sample eight
frames at stride two, spanning 15 source frames; adjacent clips advance by
that full span and do not overlap within an episode.

| Source | Train episodes | Validation episodes | Train clips | Validation clips |
| --- | ---: | ---: | ---: | ---: |
| EgoTouch | 1,458 | 174 | 109,041 | 10,710 |
| EgoTactile | 692 | 75 | 18,590 | 1,986 |
| OpenTouch | 2,552 | 406 | 17,748 | 2,742 |
| OpenGraph | 68 | 4 | 8,769 | 441 |
| FreeTacMan | 5,439 | 186 | 97,124 | 3,720 |
| FTP Unit_Bimanual | 354 | 26 | 5,792 | 440 |
| FTP Unit | 155 | 15 | 2,285 | 211 |
| FTP VLA_touch | 338 | 42 | 1,523 | 183 |
| FTP RDP | 134 | 6 | 3,955 | 177 |
| FTP RDP_Bimanual | 47 | 3 | 1,436 | 92 |
| FTP MotionTrans | 1,601 | 167 | 18,613 | 1,972 |
| FTP ViTaMIn | 680 | 67 | 21,551 | 2,190 |

EgoTouch additionally retains 179 `test_seen` and 85 `test_unseen` episodes,
excluded from the train/validation counts above. Its published split lists
332 episode paths without the required files in the pinned release; our
manifest includes the 1,896 usable episodes across all published splits.

Most other splits are project-created rather than official benchmark
splits: OpenTouch groups whole HDF5 recordings, FreeTacMan groups whole
task categories, and the other sources use stable episode-level hashes.
Four unusable FreeTacMan video groups are excluded for missing cameras,
short duration, or mismatched stream lengths. OpenGraph is the public
72-episode sample. FTP groups are selected complete archives, not the
entire multi-terabyte FTP-1 corpus.

## Common model input and target contract

```text
rgb          [8, 3 camera slots, 3 RGB channels, 224, 224]
view_mask    [8, 3] booleans identifying available cameras
target       [8, C, H, W] from the source-specific table above
target_mask  [8, C, H, W] valid readings/pixels
source       source identifier used to select decoder and loss
episode_id   source-local episode identifier
```

Camera slots are main/ego/chest (0), left wrist (1), and right wrist (2);
absent views are masked. The source identifier routes the tactile decoder
and loss; it is not supplied as a feature to the visual encoder.
The learned encoder follows frozen DINOv3-B/16 patch features with camera
fusion and temporal/spatial transformers, then predicts `mu` and
`logvar` for a 32D Gaussian at each sampled time.

Each non-EgoTouch decoder maps `32 -> 512 -> C*H*W` with a sigmoid output.
EgoTouch keeps its specialized pressure decoder and pressure loss.
Other sources use masked squared error plus 0.5 times absolute error,
alongside the standard-normal latent KL penalty. These differing targets
are the reason for source-specific decoders.

## Current experiments and evaluation roles

The current sensor-disjoint protocol is documented in
[MOTIONTRANS_BASELINE_TRAINING.md](MOTIONTRANS_BASELINE_TRAINING.md).
MotionTrans's six-channel InspireHand torque target is excluded completely
from joint encoder training and its validation. The new 32→512→6 decoder
is fitted only after freezing the joint encoder, using MotionTrans train
episodes; final evaluation uses all 1,972 MotionTrans validation clips.
The direct control trains its encoder and decoder on MotionTrans train
episodes only. Both use 50 epochs, a maximum of 8,000 clips per source per
epoch, and 64-draw final evaluation. There is no dataset classifier or GRL
in this new baseline. RDP gripper force remains a training target, so this
is an unseen torque-sensor/embodiment test, not exclusion of every numerical
state modality.

The following table records **historical** VLA_touch experiments:

| Experiment | Sources used to train the encoder | VLA_touch use |
| --- | --- | --- |
| Preserved original joint baseline | Eleven sources; VLA_touch excluded | Encoder frozen; a new decoder fitted on VLA_touch train clips, evaluated on all 183 validation clips |
| Completed alternating adversarial run | Same eleven sources, with dataset classifiers and reversed encoder gradients | Same frozen-encoder adaptation protocol |
| Completed VLA_touch-only run | Only VLA_touch train clips; fresh learned encoder and decoder, frozen pretrained DINOv3 | Encoder and decoder trained jointly for 50 epochs; final evaluation on the same 183 validation clips |

VLA_touch, Unit and Unit_Bimanual use GelSight Mini, so the old VLA_touch
holdout tests dataset transfer within familiar sensor hardware, not a new
sensor. See the [FTP-1 sensor inventory](https://arxiv.org/html/2606.13102v1).
Old results are not comparable directly to MotionTrans torque metrics.

The original baseline is preserved in
`processing/evaluations/baselines/joint_50ep_vla_touch_v1/`.
The three-way results are in
`processing/runs/vla_touch_direct_vae32_50ep/evaluation/three_way_comparison.json`.
All final evaluations compare `z=mu` with 64 Gaussian draws. Their intervals
describe empirical predictions from the fitted decoder; they do not measure
decoder-weight uncertainty or prove that every plausible tactile outcome
is represented. Sharing a latent does not establish dataset invariance.

## Storage, provenance, and further documentation

Media stays in its existing dataset directories. FTP Zarr arrays are read
from indexed tar archives without maintaining a second extracted copy.
The archives still include ancillary robot fields that this pipeline does
not consume. Existing file-hash, schema, and decoding audits are under
`processing/audits/`; indexes are under `processing/indexes/`.

- [MOTIONTRANS_BASELINE_TRAINING.md](MOTIONTRANS_BASELINE_TRAINING.md): current sensor-disjoint baseline and sequential direct control.
- [JOINT_ALL_TRAINING.md](JOINT_ALL_TRAINING.md): preserved historical VLA_touch-holdout training and frozen-head protocol.
- [FTP_EXTRA_DATASETS.md](FTP_EXTRA_DATASETS.md): selected FTP archives and schema audits.
- [ALTERNATING_ADVERSARIAL_TRAINING.md](ALTERNATING_ADVERSARIAL_TRAINING.md): alternating classifier experiment.
- [VLA_TOUCH_DIRECT_TRAINING.md](VLA_TOUCH_DIRECT_TRAINING.md): direct VLA_touch training comparison.

The local source cards state CC BY-NC 4.0 for EgoTactile, CC BY 4.0 for
OpenTouch and OpenGraph, and MIT for FreeTacMan. The checked EgoTouch and
FTP-1 cards do not establish a dataset license; a code license does not
automatically license its data.
