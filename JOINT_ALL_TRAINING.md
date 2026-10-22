# Twelve-source training with one source held out

**Historical design:** this document records the completed VLA_touch-holdout
experiments. The current holdout is **MotionTrans**, with VLA_touch included
in joint training. Use [MOTIONTRANS_BASELINE_TRAINING.md](MOTIONTRANS_BASELINE_TRAINING.md)
for the new fresh baseline and sequential MotionTrans-only control. Old
checkpoints and their immutable baseline are retained unchanged. The old
VLA_touch holdout was not sensor-disjoint: Unit and Unit_Bimanual also use
GelSight Mini. Commands below reproduce the historical design, not the
current experiment.

This is the same image-only VAE architecture as the previous joint run:
frozen DINOv3-B/16 patch features, learned view fusion and temporal/spatial
transformers, a shared 32-dimensional Gaussian latent, and a native decoder
for each **training** source. No hand pose, tactile-informed encoder, CVAE,
or dataset adversary was added.

`processing/joint_all/manifest.json` contains the original six sources plus
the six additional FTP-1 groups, 14,953 usable episodes total. The default
holdout is `ftp_vla_touch` (380 episodes). The trainer derives its 11-source
train/validation loaders from the manifest and excludes every VLA_touch
episode. It does not construct a VLA_touch decoder or use VLA_touch metrics
for checkpoint selection. The other sources retain their own validation
episodes; EgoTouch's published test splits remain excluded from training.

## Targets used in this run

- EgoTouch, EgoTactile, OpenTouch, OpenGraph, FreeTacMan, and FTP-1
  Unit_Bimanual retain the targets documented in `JOINT_DATASETS.md`.
- FTP-1 Unit and ViTaMIn predict optical tactile images resized to 32×32.
- FTP-1 RDP and RDP_Bimanual predict recorded numerical gripper force
  (one and two channels respectively). Their optical tactile arrays are
  retained on disk but are not auxiliary targets in this first run.
- FTP-1 MotionTrans predicts four finger-torque and two thumb-torque values.
- Numerical force and torque use a bounded, signed, source-local tanh
  transform before the sigmoid decoder: `0.5 + 0.5*tanh(raw/30)` for force
  and `0.5 + 0.5*tanh(raw/1000)` for torque. These are **not common physical
  units** with the pressure maps or optical images. The values 30 and 1000
  are explicit initial scale choices, not calibrated sensor specifications.

Each training mini-batch contains one source. The initial run capped each
source at 512 eight-frame windows per epoch and completed 8 epochs before
being stopped. Its checkpoint is retained at
`processing/runs/joint_all_holdout_vla_vae32/last.pt`.

The completed successor used **up to 8,000 clips per source per epoch**, 50 new
epochs, batch size 2, and gradient accumulation 16. It starts from all
learned encoder and decoder weights in that eight-epoch joint checkpoint,
but uses a **fresh optimizer and 50-epoch learning-rate schedule**. The
large-source sampler walks shuffled clips without repeats until it has
covered the source, then reshuffles. Smaller sources contribute all their
available windows once per epoch. This yields 69,468 training clips and
34,735 microbatches per epoch. VLA_touch remains excluded throughout.

## Commands

The manifest and sampled array audit are already prepared. Rebuild the
manifest after changing source files or schema:

```bash
.venv/bin/python -m tactilelatent.prepare_joint_all
```

Run a small plumbing check (one batch per training source):

```bash
.venv/bin/python -m tactilelatent.train_joint --config configs/joint_all_holdout_vla_smoke.json
```

The 8k/50-epoch run was launched under `nohup` with:

```bash
bash scripts/train_joint_all_8k_50ep.sh
```

Monitor it with:

```bash
tail -f processing/runs/joint_all_holdout_vla_8k_50ep/nohup.log
```

Resume the same 8k/50-epoch run after an interruption with:

```bash
bash scripts/train_joint_all_8k_50ep.sh --resume
```

The completed run wrote `processing/runs/joint_all_holdout_vla_8k_50ep/last.pt`
and `metrics.jsonl`. The earlier smoke run passed all 11 sources and
validation; it is not an accuracy result. Frozen-encoder adaptation and
evaluation are separate from this training script and documented below.

You can change `heldout_source` in the configuration to another source,
but use a new output directory. If holding out EgoTouch, remove the
EgoTouch-pretrained initialization checkpoint or the trainer will reject
the configuration as leakage.

## Frozen-encoder evaluations after the 50-epoch run

These independent commands load the final `last.pt` checkpoint, freeze DINO
and the shared image-to-Gaussian encoder, and train only a new head. They cache
32D latent means and log-variances under their output directories; a cache
from a different checkpoint or manifest is rejected.

```bash
.venv/bin/python -m tactilelatent.evaluate_holdout_decoder
.venv/bin/python -m tactilelatent.evaluate_dataset_classifier
```

The first command fits a new 32→512→3072 optical-tactile head on VLA_touch's
train episodes. The second fits a 32→512→11 classifier for the training
sources, using at most 1,000 fitting and 128 tuning clips per source. Both
heads see the latent mean and a Gaussian draw during fitting. Training-split
episodes select the fitting duration; manifest validation episodes are used
only for final results. Both commands compare `z=μ` to 64 Gaussian draws and
save the fitted head and `results.json`.

For VLA_touch, the 50/80/90/95% intervals are marginal **optical RGB pixel**
intervals, not pressure or whole-image intervals. The high-error quartile
breakdown identifies pixels where the deterministic model was most wrong on
validation. For classification, probability-set coverage concerns the 11
discrete source labels; compare it with the **actual selected probability
mass**, which can exceed the requested threshold when one class dominates.
The classifier also reports NLL, Brier score, and 10-bin calibration error.
Neither evaluation measures decoder weight
uncertainty; dataset identity may be revealed by cameras, scenes, or tasks.
