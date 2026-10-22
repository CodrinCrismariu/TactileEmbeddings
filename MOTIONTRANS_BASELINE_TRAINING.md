# MotionTrans sensor-disjoint baseline and direct control

Current design, 2026-10-01. This replaces VLA_touch as the active holdout.
VLA_touch shares GelSight Mini with Unit and Unit_Bimanual; its previous
holdout was not unseen hardware. MotionTrans is the only current source
with the six-channel InspireHand finger/thumb torque target. RDP numerical
gripper force remains in training: the test changes sensor, target geometry
and embodiment, rather than isolating hardware alone.

## Models and sequential execution

| Model | Encoder training data | MotionTrans decoder | Final evaluation |
| --- | --- | --- | --- |
| Fresh non-adversarial joint baseline | All 11 sources except `ftp_motiontrans`, including `ftp_vla_touch` | New 32→512→6 head fitted **after freezing** the encoder | All 1,972 MotionTrans validation clips |
| Fresh MotionTrans-only control | Only MotionTrans train episodes | 32→512→6 head trained jointly with the encoder | The same 1,972 validation clips |

Both use frozen pretrained DINOv3-B/16; all learned transformer, Gaussian
latent and decoder weights start fresh. Both initialization fields are
null. There is no CVAE, hand-pose input, classifier or gradient reversal.
Encoder inputs are images and camera availability masks, never torque or
dataset labels. The shared Gaussian latent has 32 dimensions per frame.

The launcher executes, without concurrent GPU training:

1. Joint baseline: 50 epochs, up to 8,000 clips/source/epoch.
2. Freeze the final joint encoder, select decoder fitting duration on
   episode-disjoint MotionTrans training fit/tune subsets, then refit a new
   head on all 18,613 training clips and evaluate validation once.
3. MotionTrans-only control: 50 epochs, 8,000 clips/epoch.
4. Evaluate its jointly trained decoder, then save the comparison.

Batch size is 2, accumulation 16, learning rate 5e-5, standard-normal KL
weight ramps to .001 over 10 epochs. Reconstruction objectives and model
sizes retain the established baseline settings. Cycle sampling walks
shuffled windows without repeat until exhausted, then reshuffles. Small
sources use all available clips once each epoch, not 8,000 duplicated clips.

The joint schedule has **62,991 clips and 31,497 microbatches per epoch**.
MotionTrans-only has 8,000 clips and 4,000 microbatches per epoch, covering
its 18,613-clip pool over successive epochs. Joint validation samples up to
64 clips per training source and never touches MotionTrans. Direct training
monitors all 1,972 MotionTrans validation clips each epoch; this is a fixed
50-epoch run with final `last.pt`, not validation-based early stopping.
Consequently that validation set is not a previously unmonitored test set
for the direct model. No parameters or duration are tuned on it.

## Torque evaluation

Target layout is `[8,6,1,1]`: four finger channels followed by two thumb
channels. Values are `0.5 + 0.5*tanh(raw/1000)`, the existing loader
transform. The scale is a project setting, not a verified SI calibration.
All reported errors and widths are in this normalized torque space. Do not
compare their numerical sizes directly against old optical VLA_touch MAE.

Both final evaluations use the **same head per model** for `z=mu` and 64
Gaussian draws, with fixed seeds. The frozen head trains on means and draws
paired with the same recorded target, exactly as the old protocol did.
Report deterministic/ensemble MAE and MSE, a training mean-channel-vector
baseline, CRPS, predictive standard deviation, best-of-64 clip MAE,
uncertainty/error correlation, and 50/80/90/95% marginal interval coverage
and width. Coverage errors and below/above misses expose undercoverage;
aggregate high-error quartile and per-finger/thumb summaries are saved.
Example PNGs show labelled channel tables, not invented RGB pressure maps.
Legacy `valid_pixels`, `other_pixels` and `mean_image_baseline_mae` JSON keys
remain for compatibility; here they describe scalar readings and a mean
torque vector. `mean_target_baseline_mae` is also provided.

Coverage is **empirical calibration of predictions from this decoder**;
sampling the latent does not measure decoder-weight uncertainty or ensure
every physically plausible outcome is represented. This comparison is not
zero-shot: the frozen model's new head receives MotionTrans training labels.
It also does not match total training compute or supervision between models.

## Commands and output locations

Run the two separate plumbing tests (never used as full-run initialization):

```bash
bash scripts/train_motiontrans_sequence.sh --smoke
.venv/bin/python scripts/smoke_motiontrans_evaluation.py
.venv/bin/python -m pytest -q
```

Launch the full sequence independently of the editor session:

```bash
mkdir -p processing/runs/motiontrans_baseline_sequence
nohup setsid bash scripts/train_motiontrans_sequence.sh >> processing/runs/motiontrans_baseline_sequence/nohup.log 2>&1 < /dev/null &
```

Use the same command after an interruption. An exclusive lock prevents
duplicate sequences; matching completed training/evaluation stages are
skipped, unfinished training resumes its last completed epoch. Interrupted
head fitting can reuse cached features but restarts the head fit. The
launcher stops on a failed stage rather than proceeding with partial results.

```bash
cat processing/runs/motiontrans_baseline_sequence/status.json
tail -f processing/runs/joint_all_holdout_motiontrans_fresh_8k_50ep/nohup.log
```

The joint checkpoint is
`processing/runs/joint_all_holdout_motiontrans_fresh_8k_50ep/last.pt`;
its evaluation is `evaluations/motiontrans/results.json` in that directory.
The direct checkpoint is `processing/runs/motiontrans_direct_vae32_50ep/last.pt`;
its results are `evaluation/results.json` in that directory.
The final comparison is
`processing/runs/motiontrans_baseline_sequence/comparison.json`.
Both trainers write TensorBoard events under their own `tensorboard/`
directories through the log bridge.

Standalone reproducible evaluations:

```bash
.venv/bin/python -m tactilelatent.evaluate_holdout_decoder --source ftp_motiontrans --checkpoint processing/runs/joint_all_holdout_motiontrans_fresh_8k_50ep/last.pt --output processing/runs/joint_all_holdout_motiontrans_fresh_8k_50ep/evaluations/motiontrans --samples 64
.venv/bin/python -m tactilelatent.evaluate_direct_sensor --source ftp_motiontrans --checkpoint processing/runs/motiontrans_direct_vae32_50ep/last.pt --output processing/runs/motiontrans_direct_vae32_50ep/evaluation --samples 64
```

Evaluators refuse to overwrite completed results by default and reject
incorrect source configurations or unfinished checkpoints. Feature caches
bind checkpoint hash, manifest hash, source, split and clip indices. Saved
training checkpoints include model, optimizer, scheduler and Python/NumPy/
Torch/CUDA RNG state; resumption is at completed epoch boundaries.

## Historical preservation

All old checkpoints and evaluation directories remain untouched. The
immutable original VLA_touch baseline remains at
`processing/evaluations/baselines/joint_50ep_vla_touch_v1/` and is verified
with `scripts/snapshot_joint_baseline.py --verify`. Existing evaluator
defaults still reproduce VLA_touch; current commands explicitly name
MotionTrans. This new experiment must never initialize from an old joint
model, since those models already saw MotionTrans.
