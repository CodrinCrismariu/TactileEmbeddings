> Historical record, relocated on October 21, 2026. Original path: `TRAINING_GUIDE.md`.
> Results, proposals, and operational status below belong to their recorded experiment/date;
> use the [maintained project state](../../project_state.md) for the audited active workflow.

# How the training code works

This is a reading companion to `src/tactilelatent/train.py`. The model itself
is taught step by step in `ARCHITECTURE_GUIDE.md`. The active config is
`configs/egotouch_vae_image_only_latent32.json`. Nothing in this guide starts
training; the commands at the end do so only if you run them.

## The whole training process in one paragraph

`main()` reads the config, indexes EgoTouch clips, creates DataLoaders, loads
frozen DINOv3 and the trainable pressure model, and optionally initializes
the model from the earlier image-only 32D VAE checkpoint. In each epoch,
`train_one_epoch()` predicts pressure from images, compares it with measured
pressure, backpropagates through the trainable model, and updates its weights.
Then `evaluate()` measures mean-latent predictions on validation clips with
chest-only and all-camera input. Finally it appends metrics and saves
`last.pt`. This repeats for the configured 20 epochs unless interrupted.

## Where to read the code

| Question | Function/file |
| --- | --- |
| What does one dataset item contain? | `RawEgoTouch` in `data.py` |
| How do RGB images become DINO features? | `features()` in `train.py` and `DinoBackbone` in `model.py` |
| What does the learned model predict? | `EgoTouchLatentModel.forward()` in `model.py` |
| How is a prediction scored for training? | `objective()` and `standard_normal_kl()` in `losses.py` |
| When are gradients and weights updated? | `train_one_epoch()` in `train.py` |
| Why does learning rate change? | `make_scheduler()` in `train.py` |
| How are point metrics calculated? | `evaluate()` in `train.py`, `metric_counts()` and `summarize()` in `losses.py` |
| How is a run resumed? | checkpoint logic near the end of `main()` in `train.py` |

## One microbatch, with real shapes

The active DataLoader gives **two clips** at a time. For each clip, it selects
eight synchronized times and three camera images. The arrays are:

```text
RGB images:          [2,8,3,3,224,224]
Camera availability: [2,8,3]
Measured pressure:   [2,8,2,21,21]
Valid-cell mask:     [2,8,2,21,21]
```

There are 48 RGB images in a full microbatch (`2×8×3`). `features()` flattens
the first three dimensions, runs DINO in chunks of at most eight images, and
restores the arrangement `[2,8,3,196,768]`: 196 patch descriptions of 768
numbers for each image. DINO has no gradients and never gets optimizer
updates.

The model fuses cameras, reasons across time and space, produces a 32D
Gaussian at each of the eight times, samples it during training, and decodes
pressure `[2,8,2,21,21]`. The measured pressure is **not** fed into the model;
it is used only for `objective(prediction, target, mask, ...)`.

### What is the loss trying to teach?

For each valid pressure cell, the model pays a squared-error plus half an
absolute-error penalty. Cells whose measured pressure exceeds `0.1` count
three times as much as lower-pressure cells. This fights the temptation to
predict zero everywhere when contact is sparse. Invalid target cells are
ignored, even if stored as NaN.

Two additional terms are added:

- **Spatial TV:** a small penalty for sharp jumps between neighboring
  predicted pressure cells. Its coefficient is `0.01`.
- **Latent KL:** regularizes the 32D Gaussian toward a standard normal. Its
  coefficient `beta` rises from 0 to `0.001` over the first ten epochs.

In symbols, the implementation is approximately
`loss = weighted_pressure_error + 0.01×TV + beta×mean_KL`. `beta` controls
regularization strength; it is **not** a probability. The loss is intended to
make useful predictions, not to prove that Gaussian samples are perfectly
calibrated physical possibilities.

### When does one optimizer update occur?

`batch_size=2` means only two clips are held in the model for one forward and
backward pass. With `accumulation=144`, the code sums gradients from 144 such
microbatches before calling `optimizer.step()`. Nominally, one update uses
`2×144=288` clips. This reduces GPU memory compared with placing 288 clips in
one batch. If an epoch ends with fewer than 144 microbatches remaining, that
last update uses a smaller group; the code still divides each loss by 144,
so that final group's gradient is proportionally smaller.

Before each update, gradient norm is clipped to 1.0. AdamW then updates the
**44.39 million trainable parameters after DINO**. Its learning rate warms up
for two epochs and then follows a cosine-shaped decline from the configured
peak toward `0.000001`. `weight_decay=0.05` is AdamW regularization, separate
from the Gaussian KL term.

## What each main config setting means

| Setting | Current value | Plain-language meaning |
| --- | ---: | --- |
| `root` | `datasets/EgoTouch` | Where the official split and per-episode files live. |
| `frames`, `stride`, `size` | `8`, `2`, `224` | Eight frames sampled every two source frames and resized to 224×224. |
| `batch_size`, `accumulation` | `2`, `144` | Two clips at a time, nominally 288 clips per optimizer update. |
| `epochs` | `20` | At most 20 passes through the indexed training windows in this new run. |
| `width`, `heads` | `768`, `8` | Trainable transformer token width and attention heads. |
| `cross_view_layers` | `2` | Number of camera-summary transformer layers. |
| `temporal_layers` | `3` | Number of divided time/space transformer blocks. |
| `latent_dim` | `32` | Numbers in one sampled latent per time. |
| `stochastic` | `true` | Sample the latent in training; default to its mean in evaluation. |
| `wrist_dropout` | `0.3` | Randomly hide each wrist camera during training. |
| `learning_rate` | `0.00005` | Peak AdamW learning rate. |
| `warmup_epochs` | `2` | Gradually raise learning rate before cosine decay. |
| `beta_max`, `beta_warmup_epochs` | `0.001`, `10` | Maximum Gaussian KL weight and its ramp length. |
| `pressure_threshold` | `0.1` | Defines high-pressure cells for training-loss weighting. |
| `evaluation_threshold` | `0.05` | Defines contact for validation IoU/F1. |
| `train_windows`, `validation_windows` | `null`, `null` | Use all indexed windows; integers would select evenly spaced subsets. |

JSON has no comments, so the config stays machine-readable while this table
explains its choices.

## Validation and log interpretation

At the end of each epoch, `evaluate()` switches the trainable model into
evaluation mode. That disables dropout and makes `forward()` decode the
Gaussian **mean** rather than a random sample. It measures the same model
twice: `ego` allows chest camera only; `all` allows chest and both wrists.
For each it reports:

- `mae`: average absolute pressure-cell error; lower is better.
- `contact_iou` and `contact_f1`: overlap of predicted versus measured cells
  above the evaluation threshold; higher is better.
- `volumetric_iou`: overlap of pressure *magnitudes*, not just contact labels;
  higher is better.
- `zero`: the same metrics for a predictor that always outputs zero pressure.
  It is a sanity-check baseline, not another learned model.

Every `log_every=10` training microbatches, a JSON line prints the batch loss,
unweighted KL, mean latent standard deviation, current `beta`, and current
learning rate. `latent_std` describes **spread in latent coordinates**, not
directly the pressure-map standard deviation. Each completed epoch prints a
record with average training loss, validation metrics, and elapsed time.

## Files and restarting

The new run writes inside
`processing/runs/egotouch_dinov3_vae_only_latent32/`:

- `config.json`: the exact configuration used.
- `metrics.jsonl`: one JSON record per completed epoch.
- `last.pt`: the latest completed epoch's model, AdamW, scheduler, config,
  and random-number states. It is written via a temporary file and rename.

The active config's `init_checkpoint` points to an **earlier image-only 32D
VAE checkpoint**. For a new output directory, this copies the model weights
but starts a fresh optimizer, schedule and epoch counter. It is a weight
warm-start, **not** a resume and **not** training from random initialization.
`--resume`, by contrast, reads this new run's own `last.pt` and continues at
the next epoch with optimizer/scheduler state restored. The code refuses to
overwrite an existing `last.pt` unless `--resume` is specified.

Run tests without starting training:

```bash
.venv/bin/pytest -q
```

Start the configured training run only when ready:

```bash
bash scripts/train_egotouch_vae.sh
```

Resume that same run after interruption:

```bash
bash scripts/train_egotouch_vae.sh --resume
```
