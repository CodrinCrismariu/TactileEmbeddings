> Historical record, relocated on October 21, 2026. Original path: `JOINT_SENSOR_ALIGNMENT.md`.
> Results, proposals, and operational status below belong to their recorded experiment/date;
> use the [maintained project state](../../project_state.md) for the audited active workflow.

# Joint training with frozen tactile-encoder supervision

Scene frames go through frozen DINO, shared spatial/temporal layers, a 32D
Gaussian bottleneck, and a sensor-specific MLP. The MLP predicts the code from
the selected pretrained tactile encoder. Training minimizes standardized code
MSE plus Gaussian KL; no tactile decoder is called in a training step.

The new experiment starts fresh, excludes MotionTrans from shared training,
and keeps one-/two-channel RDP outputs as direct numeric heads. The Gaussian
sampling mechanism is retained; its intervals are not automatically calibrated.

## Prepare and verify

```bash
.venv/bin/python scripts/prepare_joint_alignment.py
```

This creates `processing/joint_alignment/manifest.json`, pins nine completed
teacher checkpoints, and writes `audit.json`. Historical manifests, checkpoints,
and autoencoder training loaders remain unchanged.

The schema-2 manifest records the corrected FreeTacMan camera roles, native
tactile shapes, and every VLA chronological-to-stored tactile index. VLA scene
images are read from their original numbered JPEGs, not inferred from tactile
motion. Main camera1 and wrist camera2 retain their documented roles; the wrist
is rotated 90 degrees counterclockwise after scene preprocessing. Only scene
RGB is resized to 224x224 for DINO. Targets keep every source pixel and their
declared sensor order. Source-local ADC/intensity normalization remains intact.

Audit checks cover all local video headers, FTP episode boundaries, OpenTouch
demo lengths, native target reads, every VLA tactile frame against its original
JPEG, and the complete numbered scene-file inventories. Hardware synchronization
is not independently measured. A missing permutation or incompatible source
snapshot prevents code caching and training.

## Cache targets once

```bash
.venv/bin/python scripts/cache_sensor_codes.py
```

All 4,893,180 frames from nine datasets need 3.99 GiB of float32 codes plus
about 5 MiB of validity flags. With pinned teachers and metadata, reserve
5 GiB of additional disk. The measured serial budget is roughly 3-4 hours;
FreeTacMan native video decoding dominates. Two bounded decoding workers feed
one FP32 encoder worker. FreeTacMan uses 32-frame staging chunks and encoder
microbatches of eight; targets are never resized. No DINO or decoder runs here.

Caches include all published splits, including EgoTouch tests. Only training
frames contribute to means/stds; the training dataset never includes tests.
Each code corresponds to an exact episode and chronological frame index.
Interrupted builds resume automatically at completed episode boundaries.
Changed teachers, source metadata, preprocessing, or builder code require a
fresh cache directory. Run `--help` for source/device/worker options.

## Verify and train

```bash
.venv/bin/python scripts/smoke_joint_alignment.py
.venv/bin/python scripts/train_joint_alignment.py
# Resume the same configuration:
.venv/bin/python scripts/train_joint_alignment.py --resume
```

The real-data smoke uses isolated caches and smaller trainable transformer
layers while preserving real DINO features, native targets, teachers, Gaussian
sampling, SAM, and both numeric heads. It does not start the 50-epoch run.

Edit `configs/joint_alignment_holdout_motiontrans.json` to ablate shared latent
width or `adapter_hidden_dims`. Sensor-code widths come from teacher checkpoints
and stay independent of the shared bottleneck. These ablations reuse the same
codes. Choosing a different tactile autoencoder requires new codes for that
sensor. Default adapters are 32 -> 128 -> 128 -> sensor code, without a final
sigmoid; the retained pretrained decoders produce bounded native outputs.

Training defaults: 50 epochs; balanced source clips capped at 8,000/source/epoch;
batch two; accumulation sixteen; SAM-AdamW, rho .05; learning rate 5e-5; weight
decay .05; KL ramps to .001 over ten epochs. Final partial accumulation groups
use their actual size. SAM reuses detached DINO features and teacher codes with
identical Gaussian samples and dropout masks.

Checkpoints store model type, configuration, train-only code normalization,
pinned teachers, source/cache signatures, optimizer/SAM/scheduler, and RNG state.
Validation uses the common mean, decodes predictions at native shape, and reports
code MSE, native MAE, optical PSNR, contact metrics, temporal-change MAE, the
frozen AE reconstruction, and a training-mean-code baseline. Best selection is
the equally weighted mean of source supervision losses, excluding KL.

The run was extended to **200 total epochs**, handing over after epoch 40.
The resumed trainer runs as the user service
`tactilelatent-joint-alignment-200.service`; its handover is recorded in
`processing/runs/joint_alignment_holdout_motiontrans/extension_status.json`.
The extension preserves weights, optimizer/SAM and RNG state. Cosine decay
continues from the saved learning rate to 1e-6 at epoch 200, without restarting
warmup. The resolved schedule anchor is saved in the run's `config.json`.
After handover, use that resolved configuration for any subsequent resume:

```bash
.venv/bin/python scripts/train_joint_alignment.py \
  --config processing/runs/joint_alignment_holdout_motiontrans/config.json \
  --resume --device cuda
```

Only start this resume command when the training service is stopped. Existing
checkpoints, best-model selection, and per-epoch viewer snapshots stay in the
same run directory.

The current instruction is to **stop after epoch 100** and evaluate MotionTrans.
`tactilelatent-alignment-motiontrans-epoch100.service` waits for the complete
epoch-100 checkpoint, stops the trainer, and snapshots that exact checkpoint.
It fits `32 -> 512 -> 6` and `32 -> 512 -> 512 -> 6` transfer heads, sharing
one frozen-feature cache. Each head uses an episode-disjoint fit/tune split,
up to 100 head-training epochs with patience ten, followed by refitting on all
training episodes for the selected duration. Validation episodes never select
head-training duration. Accuracy and empirical uncertainty diagnostics use
64 latent samples, with separate scalar metrics for all six MotionTrans channels.
No MotionTrans autoencoder is involved.

Status is in `processing/runs/joint_alignment_holdout_motiontrans/stop_status.json`.
Results and feature caches are under that run's `evaluations/motiontrans_epoch_100*`.
The original checkpoint's 200-epoch schedule is preserved as experiment history;
the evaluation explicitly permits this completed epoch via `--allow-intermediate`.
Both fitted heads become selectable on the viewer's MotionTrans transfer entry
once their evaluations finish.

To evaluate all saved checkpoints divisible by ten, with the same two heads:

```bash
.venv/bin/python scripts/evaluate_alignment_checkpoint_sweep.py
```

The active sweep runs as `tactilelatent-motiontrans-checkpoint-sweep.service`
for epochs 10, 20, ..., 100. Valid completed evaluations (including epoch 100)
are reused, and interrupted feature extraction resumes from completed split
caches. Each epoch has its own cache because the shared encoder weights differ.
One cache serves both heads for that epoch. The sweep keeps a process lock,
checks checkpoint/data/head provenance, and saves each result before continuing.
Progress, logs, CSV/JSON tables and final PNG/PDF plots are under
`processing/runs/joint_alignment_holdout_motiontrans/evaluations/motiontrans_checkpoint_sweep`.
The head budget remains 100 epochs with patience ten, seed 45 and 64 samples.
All successfully completed heads appear automatically in the viewer.
Cross-checkpoint rankings use MotionTrans validation data; they do not constitute
an independent test-set assessment after checkpoint selection.

A system driver update replaced the userspace NVIDIA libraries while kernel
580.173.02 remained loaded. Evaluation uses a local extraction of matching
580.173.02 libraries via `LD_LIBRARY_PATH` under `processing/runtime/`, with
no system driver changes. New GPU processes need this environment until the
kernel and system libraries match again.

## Transfer and dataset-identity probes

Existing evaluators dispatch the new checkpoint type and canonical loader:

```bash
.venv/bin/python -m tactilelatent.evaluate_holdout_decoder \
  --checkpoint processing/runs/joint_alignment_holdout_motiontrans/best.pt \
  --manifest processing/joint_alignment/manifest.json \
  --source ftp_motiontrans --hidden-dims 16 \
  --output processing/evaluations/alignment_motiontrans
.venv/bin/python -m tactilelatent.evaluate_dataset_classifier \
  --checkpoint processing/runs/joint_alignment_holdout_motiontrans/best.pt \
  --manifest processing/joint_alignment/manifest.json \
  --heldout-source ftp_motiontrans \
  --output processing/evaluations/alignment_identity
```

These fit episode-disjoint heads on frozen common latents. Sampling-based
coverage diagnostics do not replace a separate uncertainty-calibration stage.
The viewer uses the same frame-order validator and prefers the canonical
manifest on its next startup. Canonical VLA scene previews retain native
640x480 pixels; its tactile previews and reconstructions remain 320x240.
The viewer's **Ego → tactile prediction** page runs visual checkpoints through
their pinned frozen decoders and compares two selected models/epochs on the
same frame. `save_viewer_epochs: true` saves lightweight inference snapshots
after every epoch; full optimizer resumption still uses `last.pt`. See
[viewer instructions](EGOTOUCH_AUTOENCODER_VIEWER.md) for camera and epoch controls.
