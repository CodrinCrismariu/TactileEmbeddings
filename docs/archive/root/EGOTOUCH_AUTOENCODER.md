> Historical record, relocated on October 21, 2026. Original path: `EGOTOUCH_AUTOENCODER.md`.
> Results, proposals, and operational status below belong to their recorded experiment/date;
> use the [maintained project state](../../project_state.md) for the audited active workflow.

EgoTouch tactile autoencoder pretraining
======================================

Train a deterministic **pressure -> latent code -> pressure** autoencoder before
using its frozen decoder in a future RGB predictor. This script contains no DINO,
RGB loading, temporal model, adversary, Gaussian sampling, or KL penalty.
It prepares a decoder; it does not change or launch existing visual experiments.

Each example is one frame of bilateral pressure, shape `[2,21,21]`. Both hands
share one code. Non-finite cells are zero-filled and accompanied by two validity
mask channels at the encoder input. Missing cells do not contribute to loss or
metrics. The decoder receives only the latent vector and predicts `[2,21,21]`
in `[0,1]`. The default is a **single sigmoid pressure output**, used for both
training and inference. Optional clamp and legacy hurdle modes remain available
for reproducing earlier experiments. There are no skip connections bypassing
the code. The active search is documented in
[EGOTOUCH_AUTOENCODER_SEARCH_V4.md](EGOTOUCH_AUTOENCODER_SEARCH_V4.md).

Run from the repository root with the existing `.venv`:

```bash
# Small real-data plumbing check: 16 frames per split, CPU, one epoch.
# Automatically appends _smoke to the output directory.
.venv/bin/python scripts/pretrain_egotouch_autoencoder.py \
  --output processing/runs/egotouch_ae_conv_medium_z32 --smoke

# Default convolutional run: 30 epochs, 32D code, 100k frames/epoch.
.venv/bin/python scripts/pretrain_egotouch_autoencoder.py \
  --output processing/runs/egotouch_ae_conv_medium_z32

# Resume at the last completed epoch: keep settings; optionally increase --epochs.
.venv/bin/python scripts/pretrain_egotouch_autoencoder.py \
  --output processing/runs/egotouch_ae_conv_medium_z32 --resume
```

Network-size ablations
----------------------

`--sensor-code-dim` (alias `--latent-dim`) changes the **sensor-specific tactile
code**, independently of network capacity and independently of the earlier 32D
shared visual latent. The equal default dimension was only an initial setting,
not a constraint tying those representations together.

The intended later RGB architecture is:

```text
DINO -> camera/time/space encoder -> shared visual latent (e.g. 32D)
     -> trainable sensor-specific MLP with hidden layers
     -> sensor code (16 / 32 / 64 / 128D, selected by this ablation)
     -> pretrained frozen tactile decoder -> pressure maps
```

The trainable MLP adapter belongs to later visual training. It is not needed in
tactile autoencoder pretraining, which starts directly from pressure maps.

`--size` selects a readable preset; explicit widths override the selected preset.
Encoder and decoder sizes are mirrored; the decoder's final output is always
two 21x21 maps. Actual parameter counts are printed and saved in `summary.json`.

| Size | Convolutional encoder channels | MLP encoder hidden widths |
| --- | --- | --- |
| small | 16, 32 | 128 |
| medium (default) | 32, 64 | 512 |
| large | 32, 64, 128 | 1024, 512 |

Convolutions use 3x3 kernels, stride 2, and GELU. The default encoder reduces
21x21 -> 11x11 -> 6x6, then projects to the latent vector. The decoder projects
back to 64x6x6 and uses nearest-neighbor resize plus convolutions at exact 11x11
and 21x21 sizes. The MLP flattens the pressure/mask input and uses GELU hidden
layers. Neither architecture uses dropout, batch normalization, or stochastic
latents, keeping comparisons and later frozen decoding straightforward.

```bash
# Change bottleneck size only.
.venv/bin/python scripts/pretrain_egotouch_autoencoder.py \
  --latent-dim 16 --output processing/runs/egotouch_ae_conv_medium_z16

# Change network size only.
.venv/bin/python scripts/pretrain_egotouch_autoencoder.py \
  --size large --output processing/runs/egotouch_ae_conv_large_z32

# Explicit convolutional widths; number of entries controls depth.
.venv/bin/python scripts/pretrain_egotouch_autoencoder.py \
  --channels 16 32 64 --latent-dim 64 \
  --output processing/runs/egotouch_ae_conv_16_32_64_z64

# MLP comparator or custom MLP widths.
.venv/bin/python scripts/pretrain_egotouch_autoencoder.py \
  --architecture mlp --hidden-dims 512 256 --latent-dim 32 \
  --output processing/runs/egotouch_ae_mlp_512_256_z32

# Disable frame caps for a full-data run (also use a new output).
.venv/bin/python scripts/pretrain_egotouch_autoencoder.py \
  --max-train-frames 0 --max-val-frames 0 \
  --output processing/runs/egotouch_ae_conv_medium_z32_full
```

The completed initial grid tested **small/medium/large x sensor code 16/32/64/128**: twelve
separate 30-epoch runs. Seed, frame caps, stride, batch size, loss, and epoch
budget are identical. Compare code dimensions at fixed capacity, and capacities
at fixed code dimension. All CLI options are available with `--help`.

```bash
mkdir -p processing/runs/egotouch_ae_sensor_code_grid_v1
nohup setsid bash scripts/train_egotouch_autoencoder_grid.sh \
  > processing/runs/egotouch_ae_sensor_code_grid_v1/nohup.log 2>&1 < /dev/null &
```

This historical grid uses sigmoid output. It runs sequentially on CUDA with a lock against duplicate launchers.
Each completed run updates `comparison.csv` in the grid directory. Per-run logs
are adjacent to run directories, and `status.txt` identifies the active run.
Re-running resumes existing checkpoints with the same settings.
The earlier fixed-code sequence was stopped and its partial artifacts retained
in `processing/runs/egotouch_ae_presets_z32` and its original run directories.

Data, loss, and evaluation
--------------------------

The default `processing/joint_all/manifest.json` already contains the usable
EgoTouch episodes and preserves official train/val/test assignments. The trainer
selects only EgoTouch train and val rows, rejects duplicate IDs/target paths,
and reads only `pressure_grids.npz`. Test pressure files are not opened.
Paths in the manifest follow the project's repository-root-relative convention.
Frame counts and shapes are checked when each NPZ is opened.

Frames are independent, sampled at stride 2 through each whole episode. This is
**not** the visual trainer's eight-frame, non-overlapping clip sampling protocol.
At most 100,000 uniformly selected training frames are used each epoch; the
subset changes with `seed + epoch`. Validation uses a fixed uniform subset of
at most 20,000 frames. A cap of 0 uses all indexed frames. Same seeds and sampling
arguments give the same subsets across architectures. Episode order and frame
order within each episode are shuffled in training; batches remain episode-local
to avoid repeatedly decompressing NPZ files. Short final episode batches are
retained. These are not fully globally shuffled frame batches. Two NPZ episodes
are cached per worker, rather than loading the entire corpus into memory.

Training uses AdamW (lr .0001, weight decay .0001), batch size up to 128, and
gradient clipping at norm 1. The learning rate is constant. Loss is:

```text
sum(valid * weight * (squared_error + 0.5 * absolute_error)) / sum(valid)
weight = 3 when target > 0.1, otherwise 1
```

This loss applies to all valid cells, including actual zeros. The default
`--output-mode sigmoid` applies the regression loss to sigmoid predictions.
`--mae-weight` changes .5 in the regression formula. Optional clamp mode trains
on raw linear outputs and clips at evaluation; legacy hurdle options exist for
earlier artifact reuse.

`--contact-weight 1` removes contact emphasis. There is no total variation or KL.
Reconstruction metrics use
global valid-cell counts. Validation reports MAE, MSE, contact MAE, high-pressure
MAE (target >= .5), contact F1/IoU, and an all-zero predictor MAE. Contact metrics
use .05 by default; this differs intentionally from the .1 loss threshold.
Metrics with no applicable cells are null. Errors are in released normalized
pressure units, not calibrated SI units.

The best checkpoint is selected by **validation MAE** on the fixed subset.
Inspect contact and high-pressure errors as well: low global MAE alone can hide
poor contact reconstruction. `--selection-metric balanced_mae` instead weights
global/contact/high-pressure MAE equally. Exact-zero recall, false-zero rate on
positive cells, and zero-output misses on contact cells expose degenerate solutions.
An exact-zero-capable output does not establish calibrated predictive uncertainty.
`--scheduler plateau` can reduce LR as the selected validation score plateaus.

Saved artifacts and decoder reuse
---------------------------------

Each output directory contains:

- `config.json`: resolved architecture, sampling, optimizer, and manifest hash.
- `summary.json`: parameter counts and available frame/episode counts.
- `metrics.jsonl`: train/val metrics after each completed epoch (one-based epoch).
- `last.pt`: full state for epoch-boundary resume, including optimizer and RNG.
- `best.pt`: best model plus separate encoder/decoder states and model config.
- `best_metrics.json`: metrics for the selected model.
- `reconstructions.png`: selected model, first up to four frames of the first
  validation batch. Rows are frames; columns are true left/right, reconstructed
  left/right, absolute-error left/right. All panels use grayscale [0,1]; invalid
  cells are magenta. These fixed examples are qualitative, not representative
  aggregate results.

Nonempty output directories are refused unless `--resume` is supplied.
Resume requires identical settings and manifest contents, except that `--epochs`
may be increased to extend a run while preserving optimizer/scheduler/RNG state.
For historical sigmoid runs, explicitly pass `--output-mode sigmoid` and their
original learning rate when resuming; their output architecture cannot be changed
in-place. The active sigmoid search starts fresh and preserves earlier checkpoints.
Checkpoints store zero-based epoch indices. NPZ content changes are not detected
by the manifest hash; preserve the audited source files for reproducibility.

```python
import torch
from tactilelatent.tactile_autoencoder import TactileAutoencoder

saved = torch.load("processing/runs/egotouch_ae_conv_medium_z32/best.pt",
                   map_location="cpu", weights_only=False)
ae = TactileAutoencoder(**saved["model_config"])
ae.load_state_dict(saved["model"])
ae.eval().requires_grad_(False)

# During later visual training, measured pressure supplies teacher codes only.
# pressure and valid have shape [B, 2, 21, 21].
with torch.no_grad():
    target_code = ae.encode(pressure, valid)

# visual_adapter is a future trainable RGB-latent -> tactile-code module.
predicted_code = visual_adapter(visual_latent)
prediction = ae.decode(predicted_code, hard=False)  # Sigmoid output; keep autograd ON.
# Gradients pass through ae.decoder to predicted_code, but decoder weights stay frozen.
# In sigmoid mode, hard=True and hard=False produce the same output.
```

Implementation: `tactile_autoencoder.py` defines the two architectures;
`pretrain_egotouch_autoencoder.py` contains pressure loading, sampling, loss,
metrics, and the explicit training loop. The script in `scripts/` is a thin
launcher. Historical sigmoid models remain loadable; new checkpoints store output
mode, gate threshold, and scheduler settings explicitly.
