# Bimanual mirroring with frozen sensor-code supervision

EgoPressure keeps its native 224×224 hand-relative UV pressure target. There is
no raw-grid prediction head, raw-grid teacher, or raw-grid code cache. Its
ongoing pressure-teacher repair is independent of this change.

## Training behavior

The optional `mirror_augmentation` configuration selects a physical mirror
variant with probability 0.5 for each eligible eight-frame clip, before wrist
dropout and live V-JEPA encoding. All frames use the same choice. The ego image
is flipped horizontally; wrist views are flipped and exchanged along with
their camera masks. The current target changes consistently; frame IDs,
episode boundaries, sample counts, source weights and native resolution stay
the same. No extra visual feature pass or disk RGB copy is required for the
augmentation.

This teaches prediction equivariance: mirror the interaction and the tactile
prediction should transform accordingly. There is no latent-equality loss.
Different handedness/location can remain represented in the latent.

Every source needs an explicit, versioned sensor-coordinate policy. EgoTouch
swaps hands and flips the 21×21 grids; its upstream fixed physical sensor masks
prove that correspondence. RDP Bimanual swaps its two scalar force channels;
the recorded sign/scale, including measured zero encoded as 0.5, is retained.
OpenGraph's within-finger taxel axes, Unit Bimanual's tactile-camera axes, and
Baihu's within-gripper/platform mappings are currently **blocked**, not guessed.
Two tactile pads on one right gripper are not a bilateral-hand dataset.

The [2026-10-19 mapping review](../processing/audits/bimanual_mirror_mapping_v2/REPORT.md)
checks all 380 Unit Bimanual and 3,013 Baihu marker episodes and all 144 raw
OpenGraph streams. Their stored identities and geometry are consistent, but
the remaining physical reflection maps are unresolved. The OGLO SDK explicitly
leaves the across-finger **row** orientation unmeasured; its columns run along
the finger, with a separate left-thumb correction. Baihu includes D-WHEEL,
PIKA, and QINGLOONG, requiring platform-specific mounting evidence. The review
does not change the pinned v1 policy or its prepared teachers/codes.

## Interactive visual quality check

The viewer's **Mirror quality check** tab (`?page=mirror`, port 8080) shows
recorded and transformed targets at the same frame with shared color scales.
The selected frozen autoencoder runs separately on each input. The mirrored
reconstruction is not a reflection of the original reconstruction. Select the
accepted `egotouch AE · mirror-adapted teacher` or a historical EgoTouch model
to inspect their behavior. RDP Bimanual displays swapped scalar measurements
and their original-unit tables without introducing an unnecessary autoencoder.

The mirrored scene preview horizontally reflects RGB and exchanges wrist
camera roles. Its separate camera selector includes only the transformed
available slots. Dataset/model controls, episode search, frame slider, exact
value hover, and error metrics work on the paired views. Sources with unresolved
geometry are described as unavailable and cannot be selected on this tab.
The sensor reconstruction and joint prediction tabs retain their existing roles.

The marker implementation supports explicitly verified per-surface horizontal
and/or vertical reflections. Reference coordinates, signed displacement and
surface identity transform together. Horizontal reflection uses `u'=1-u`,
`dx'=-dx`; missing tracks stay missing. Padded points stay padded. An unverified
rotation/axis swap is rejected because normalized UV and pixel-displacement
units need native aspect-ratio handling.

## Teacher and cache preparation

Run `.venv/bin/python scripts/prepare_bimanual_mirror.py`, or use the persistent
user service `tactilelatent-bimanual-mirror-v1.service`.

The preparation audits each verified source on identical original/mirrored
validation frames. If mirrored MAE exceeds original MAE by more than 10%, it
fine-tunes a separate teacher for 10 epochs, seed 45, with 50% original/mirrored
examples, AdamW at 1e-4 and cosine decay to 1e-6. Architecture, code dimension,
native targets and historical weights are preserved. Defaults sample at most
100,000 training frames and 10,000 fixed validation frames per epoch, spread
across source episodes. There is no search or automatic extension.

Only checkpoints preserving original MAE within 10% and reducing mirrored MAE
by at least 10% are accepted. Pressure teachers must also beat the zero
baseline globally and on measured positive cells. Unadapted teachers still
must pass the pressure-collapse checks. Unsupported or failed sources are
reported explicitly and cannot be requested by a production run.

Accepted teachers encode original and transformed *native targets* in frozen
FP32 evaluation mode. Latent coordinates themselves are never swapped or
flipped. Per-episode caches hold `[2, frames, sensor_code_dim]` float32 codes,
support masks and exact chronological frame indices. Training-only statistics
give both variants equal weight. Artifacts pin teacher, coordinate policy,
native target hashes, frame contracts and implementation hashes. Interrupted
episodes resume safely; fresh encodings must reproduce cached samples.

Fresh-code verification replays the original encoder batches (128 frames plus
each episode's final remainder). CUDA convolutions can produce slightly
different float32 outputs for different batch shapes; checking three isolated
frames previously caused a false failure. The corrected verifier keeps the
same 2e-5 relative/absolute tolerance and records the batch/backend contract
and its own implementation hash separately, preserving teacher/cache-builder
provenance. The initial EgoTouch cache reproduced exactly with the original
batches; neither its values nor the teacher weights needed replacement.

Artifacts live under `processing/bimanual_mirror_v1/`: `status.json`,
`progress.jsonl`, `policy_audit.json`, `teachers/`, `codes/` and `ready.json`.
Historical teachers/caches are untouched. Native RGB/tactile data are not
duplicated or downsampled. MotionTrans and FreeTacMan cannot enter preparation.

SSH-independent progress is also recorded by `scripts/monitor_bimanual_mirror.py`
in `processing/bimanual_mirror_v1/nohup.log` and `monitor_status.json`. The user
service has lingering enabled, so closing SSH does not interrupt teacher
preparation or its subsequent disposable GPU verification. The monitor writes
`background_result.json` when the service finishes or fails.

## Joint configuration and verification

The new entrypoint is `scripts/train_mirrored_vjepa.py`. Two configurations
describe the existing ten-source and future expanded twelve-source studies:

- `configs/joint_vjepa21_spatial_current_32d_mirror.json`
- `configs/joint_vjepa21_spatial_expanded_32d_mirror.json`

Both deliberately fail preflight if any requested source has unresolved
geometry/teacher quality. The expanded configuration also retains existing
Baihu/EgoPressure publication gates. No fallback silently removes a requested
bilateral source.

After preparation, `scripts/verify_bimanual_mirror.py` writes an explicitly
partial `available_joint_config.json` for ready sources, plus disposable real
V-JEPA/SAM updates and paired RGB/target/AE reconstruction plots. These updates
do not save or launch a production joint-training model. Its report lists all
blocked sources.

Ordinary source validation stays unaugmented. A separate paired diagnostic
reports original/mirrored reconstruction and decoded prediction-equivariance
MAE. It does not affect checkpoint selection. SAM replay retains the selected
transformation, supervised codes and detached frozen features. Existing
checkpoint loading and inference architecture remain unchanged; no mirror or
transport solver is required for inference.

## Interpreting EgoPressure UV spots

A UV map is a cut-and-flattened texture atlas, not an image of the touchpad or
a fingertip detector. One contact crossing a chart seam can produce separated
patches. The authors' [single-finger figure, panels j/k](https://arxiv.org/html/2409.02224v2/F4.png)
shows this directly. The viewer now identifies its atlas representation.

The audit at `processing/audits/egopressure_uv_layout_v1/` checks the user-reported
`p_021/p_021_push-away_5x_left`, viewer frame 198 / native frame 199. The viewer
matches the released UV values exactly after existing normalization. At a 5%
peak threshold, excluding components smaller than four cells, the raw pad has
five main regions and the UV atlas ten. This supports chart fragmentation;
exact patch-to-finger assignments require the matching licensed MANO UV mesh,
which is not installed. No raw measurements become training targets.
