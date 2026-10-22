> Historical record, relocated on October 21, 2026. Original path: `README.md`.
> Results, proposals, and operational status below belong to their recorded experiment/date;
> use the [maintained project state](../../project_state.md) for the audited active workflow.

# TactileLatent: image-only tactile latent learning

The new joint-training pipeline predicts **frozen tactile-autoencoder codes**:
shared visual encoder → 32D Gaussian latent → sensor-specific MLP → sensor code.
Frozen tactile encoders supply supervision; frozen decoders reconstruct native
outputs during evaluation. MotionTrans remains held out. See
[JOINT_SENSOR_ALIGNMENT.md](JOINT_SENSOR_ALIGNMENT.md) for audited frame pairing,
resumable target-code caching, smoke checks, training, and transfer evaluation.
The full joint run is prepared and has not been started.

For separate **tactile-only EgoTouch autoencoder pretraining**, see
[EGOTOUCH_AUTOENCODER.md](EGOTOUCH_AUTOENCODER.md). The new script supports
convolutional/MLP size and bottleneck ablations, and exports a reusable decoder
without changing the existing visual models or runs.

For an interactive video/frame viewer of original and reconstructed hand pressure,
run `.venv/bin/python scripts/view_egotouch_autoencoder.py` and open
http://127.0.0.1:8080. See [viewer instructions](EGOTOUCH_AUTOENCODER_VIEWER.md).
Click **Ego → tactile prediction** for scene-to-touch inference, independent
model/epoch selectors, and simultaneous A/B comparison. Saved checkpoints from
ongoing runs appear automatically; future joint trainers retain each epoch's
inference weights.

For tactile autoencoder pretraining and viewer dataset switching,
see [SENSOR_AUTOENCODER_PRETRAINING.md](SENSOR_AUTOENCODER_PRETRAINING.md).
The current selection has nine autoencoder teachers; the one-/two-value RDP
outputs use direct numeric heads, and the six-value MotionTrans sensor is held out.

The preceding experiment is a **50-epoch SAM+AdamW joint run** with the same
11 sources and MotionTrans holdout. See [SAM_JOINT_TRAINING.md](SAM_JOINT_TRAINING.md)
for optimizer details, checkpoint resumption and monitoring. The non-SAM
baseline and its evaluations remain preserved for comparison.

The **earlier baseline** is a fresh 50-epoch, non-adversarial joint baseline
on 11 sources, including VLA_touch, with **MotionTrans held out as an unseen
torque sensor**. A MotionTrans-only model is trained afterwards for comparison.
Both retain frozen pretrained DINOv3 and the image-only 32D Gaussian VAE.
See [MOTIONTRANS_BASELINE_TRAINING.md](MOTIONTRANS_BASELINE_TRAINING.md) for
the sequential launcher, evaluation and monitoring commands, and
[JOINT_DATASETS.md](JOINT_DATASETS.md) for all 12 sensor/target definitions.
Old VLA_touch and EgoTouch experiments remain preserved, not active defaults.

An additional evaluation-only benchmark tests the frozen joint baseline on
**Feeling of Success grasp-success labels**, using pre-lift RGB and disjoint
objects. See [FEELING_SUCCESS_EVALUATION.md](FEELING_SUCCESS_EVALUATION.md)
for its separate protocol, uncertainty metrics and reproducible command.

## Historical EgoTouch experiment and architecture reference

`main.pdf` proposes learning a tactile latent from egocentric video across
datasets with sensor-specific decoders. The **earlier established experiment** was
EgoTouch only: one pressure decoder, no dataset-invariance module, and an
image-only 32-dimensional Gaussian-latent VAE-style predictor. Pressure is
a training target, never an encoder input; hand pose is not used. A separate
six-source joint VAE **data pipeline and optional trainer** are now prepared;
see `JOINT_DATASETS.md`. An optional twelve-source run with one source held
out is historically described in `JOINT_ALL_TRAINING.md`. These do not alter the EgoTouch
checkpoint or results.

For a teaching-oriented explanation with exact per-layer shapes, weight
matrices, parameter counts, and a worked data-flow walkthrough, see
`ARCHITECTURE_GUIDE.md`. For a line-by-line explanation of the training loop,
loss, hyperparameters, logs, and checkpoint behavior, see `TRAINING_GUIDE.md`.

## Data and model, in order

| Stage | Shape for one 8-frame clip | What happens |
| --- | --- | --- |
| RGB | `8 × 3 cameras × 3 × 224 × 224` | Chest, left wrist, right wrist. Ego-only evaluation masks the wrists. |
| DINOv3-B/16 | `8 × 3 × 196 patches × 768` | Frozen encoder; no CLS/register tokens passed onward. |
| Cross-view fusion | `8 × 196 × 768` | Two attention layers over three camera summaries produce camera weights; weighted patch features are fused. |
| Space-time transformer | `8 × 196 × 768` | Three blocks: attention across time for each patch, then across patches for each frame. |
| Gaussian head | `8 × (32 mean + 32 log-variance)` | One diagonal Gaussian latent distribution per frame. |
| Sampled/mean latent | `8 × 32` | Sample during training; use mean by default at evaluation. |
| Pressure decoder | `8 × 2 hands × 21 × 21` | MLP `32 → 512 → 882`, sigmoid, normalized pressure. |

The backbone has 85,660,416 **frozen** parameters. The historical EgoTouch model after it has
44,385,587 **trainable** parameters, for 130,046,003 total. Its fp32 weights
alone are about 496 MiB; checkpoints are larger because they include optimizer
state. Of the trainable weights, 14.18M are in cross-view attention and 28.36M
are in the space-time transformer. The 32D bottleneck is small in *latent
coordinates* but does not make the visual encoder small.

For a clip, the model receives 8 frames sampled at stride 2 and predicts eight
bilateral pressure maps. A clip spans 15 source-video frames from first to
last, and clips are non-overlapping. Temporal attention is **bidirectional**:
predictions for earlier positions can see later images in that same clip. The
last position sees only images at or before its own time. This is not a
streaming-causal implementation.

The training loss is contact-weighted `MSE + 0.5 × MAE`, plus `0.01 ×` spatial
total variation and an annealed KL penalty (maximum weight `0.001`) toward
`N(0,I)`. AdamW uses an effective batch size of 288 through gradient
accumulation. Wrist cameras are randomly dropped during training with
probability 0.3. That EgoTouch config specifies 20 epochs. Evaluation reports
pressure MAE, contact IoU/F1 and volumetric IoU; the uncertainty evaluator
also reports CRPS, interval coverage, sample variation and related diagnostics.

Scientifically, this is **VAE-style**, not the conventional ELBO for a
conditional generative model: its Gaussian encoder sees images only, not the
observed pressure. Sampling it provides multiple outputs, but the objective
alone does not guarantee a calibrated pressure distribution or distinct
physical modes. Treat uncertainty metrics as measurements, not as proof that
the latent variance represents the true ambiguity of force.

## Code map

- `src/tactilelatent/aligned_frames.py`, `audit_alignment.py`: canonical scene/
  target ordering, native target resolution, and source-data audit.
- `src/tactilelatent/sensor_code_cache.py`: pinned frozen tactile encoders,
  resumable per-frame codes, and training-only code statistics.
- `src/tactilelatent/alignment_data.py`, `alignment_model.py`, `train_alignment.py`:
  visual-to-sensor-code training and explicit native decoder evaluation.
- `configs/joint_alignment_holdout_motiontrans.json`: new joint-training defaults.
- `src/tactilelatent/data.py`: raw EgoTouch MP4/pressure loader; an older HDF5
  adapter is retained but not used by active training.
- `src/tactilelatent/media.py`: frame decoding.
- `src/tactilelatent/model.py`: frozen DINOv3 wrapper, view/space-time model,
  Gaussian head and pressure decoder.
- `src/tactilelatent/losses.py`: training objective and point metrics.
- `src/tactilelatent/train.py`: one training loop and validation.
- `src/tactilelatent/evaluate.py`: seen/unseen point evaluation.
- `src/tactilelatent/evaluate_uncertainty.py`: sampling and calibration metrics.
- `src/tactilelatent/infer.py`: latent and pressure export.
- `src/tactilelatent/prepare_joint.py`, `joint_data.py`, `tar_zarr.py`:
  six-source indexing and lazy native-modality loading.
- `src/tactilelatent/joint_model.py`, `train_joint.py`: image-only
  VAE with one shared encoder and source-specific decoders.
- `scripts/train_motiontrans_sequence.sh`: preceding joint baseline and direct
  MotionTrans control, trained sequentially and evaluated automatically.
- `src/tactilelatent/evaluate_holdout_decoder.py`, `evaluate_direct_sensor.py`:
  frozen-head transfer and direct-sensor evaluation, including 64-draw metrics.
- `configs/egotouch_vae_image_only_latent32.json`: established EgoTouch config;
  `configs/joint_six_vae32.json` is the separate optional joint experiment.
- `ARCHITECTURE_GUIDE.md`: teaching walkthrough of model shapes and sizes.
- `TRAINING_GUIDE.md`: teaching walkthrough of the training code and settings.
- `JOINT_DATASETS.md`: current twelve-source sensor scopes, formats and splits.
- `MOTIONTRANS_BASELINE_TRAINING.md`: earlier sensor-disjoint baseline protocol.
- `scripts/train_egotouch_vae.sh`: training launcher.
- `scripts/evaluate_egotouch_vae_uncertainty.sh`: uncertainty evaluation launcher.
- `TRAINING_PLAN.md`: remaining research and verification steps.

`prepare_pilot.py`, `expand_pilot.py`, and `audit.py` are data utilities, not
part of this training path. Older experiment checkpoints and metrics under
`processing/runs/` remain on disk for provenance but are not active configs.

## Historical EgoTouch commands

These commands are for the old EgoTouch experiment, not the current baseline.
The historical config initializes its model
weights from the existing image-only 32D VAE checkpoint (epoch index 12), but
starts a new optimizer and a separate 20-epoch run. To train:

```bash
bash scripts/train_egotouch_vae.sh
```

To continue that new run after interruption:

```bash
bash scripts/train_egotouch_vae.sh --resume
```

To evaluate the existing VAE checkpoint on official test splits:

```bash
.venv/bin/python -m tactilelatent.evaluate \
  --checkpoint processing/runs/egotouch_dinov3_vae_image_only_latent32/last.pt \
  --output processing/evaluations/egotouch_vae32_point.json
bash scripts/evaluate_egotouch_vae_uncertainty.sh
```

Run tests with `.venv/bin/pytest -q`.

The simplified loader accepts the historical image-only VAE checkpoint by
discarding weights of modules that were stored but never used in its VAE
forward pass. Other historical experiment types are not supported by the
active evaluator. No old checkpoints or datasets have been deleted.
