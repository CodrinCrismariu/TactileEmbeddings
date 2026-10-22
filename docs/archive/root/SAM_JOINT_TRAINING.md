> Historical record, relocated on October 21, 2026. Original path: `SAM_JOINT_TRAINING.md`.
> Results, proposals, and operational status below belong to their recorded experiment/date;
> use the [maintained project state](../../project_state.md) for the audited active workflow.

# Fresh joint SAM experiment

Started 2026-10-02. This run replaces AdamW's ordinary gradient update with
SAM around AdamW; it does not add dataset adversarial training or a CVAE.
All previous models and evaluations remain unchanged.

## Configuration

- Fresh shared transformer, 32D image-conditioned Gaussian, and source decoders.
  Both initialization checkpoint fields are null. DINOv3-B/16 stays pretrained
  and frozen, exactly as in the fresh non-adversarial reference.
- The same 11 sources, including VLA_touch, exclude MotionTrans. Feeling of
  Success remains evaluation-only and is not in the training manifest.
- 50 epochs, up to 8,000 distinct clips per source per epoch, cyclic sampling;
  small sources use their available clips. Same source order and seed 45.
- Microbatch 2, accumulation 16, learning rate 5e-5, weight decay .05,
  existing warmup/cosine schedule, tactile losses, KL schedule and camera dropout.
- SAM radius .05: an initial choice, not tuned for this task. DINO is never
  perturbed. Trainable encoder/Gaussian/decoder weights are perturbed together
  using the global first-pass accumulated gradient norm.

## Two-pass update

1. Extract frozen DINO features once per microbatch and cache detached patches.
2. Accumulate the first-pass tactile-plus-KL gradients across up to 16 batches.
3. Save original weights exactly, then perturb by `rho * gradient / norm`.
4. Zero first-pass gradients and replay those batches at perturbed weights.
   Camera masks are reused; Torch CPU/CUDA RNG states reproduce the same
   dropout masks and Gaussian noise. DINO is not rerun.
5. Clip the accumulated second-pass gradient to norm 1, restore original
   weights exactly, apply AdamW, and advance the scheduler once.

RNG state after the first pass is restored after replay. Parameters are also
restored if replay fails. The final partial accumulation uses the existing
baseline's scaling. Weight decay is applied by AdamW only, not the perturbation.

Epoch checkpoints include model, optimizer, scheduler, SAM radius/update count,
config and Python/NumPy/Torch/CUDA RNG. Resume is at completed epoch boundaries;
interrupted partial epochs are replayed, not claimed to resume mid-batch.
Periodic completed checkpoints are retained every 10 epochs; hard links avoid
copying the latest file while atomic replacement preserves older checkpoints.
Storage is checked against the 700 GB dataset+processing cap with a 5 GB reserve.

## Run and monitor

Configuration: `configs/joint_all_sam_holdout_motiontrans_fresh_8k_50ep.json`.
Run: `processing/runs/joint_all_sam_holdout_motiontrans_fresh_8k_50ep`.
The active supervisor and trainer are detached with nohup/setsid, so closing
VS Code or SSH does not stop them. Machine shutdown will stop training.

```bash
cat processing/runs/joint_all_sam_holdout_motiontrans_fresh_8k_50ep/status.json
tail -f processing/runs/joint_all_sam_holdout_motiontrans_fresh_8k_50ep/nohup.log
```

If the run has exited, resume explicitly:

```bash
nohup setsid .venv/bin/python -u scripts/launch_joint_sam.py --resume >> processing/runs/joint_all_sam_holdout_motiontrans_fresh_8k_50ep/launcher.log 2>&1 < /dev/null &
```

Do not start a second launcher while a trainer is running. Without `--resume`,
existing checkpoints are refused. TensorBoard receives source training losses,
source validation MAE, schedule, SAM diagnostics, mean latent coordinate
variance, predicted Gaussian variance and mean-latent effective rank.

```bash
.venv/bin/tensorboard --logdir processing/runs/joint_all_sam_holdout_motiontrans_fresh_8k_50ep/tensorboard --host 127.0.0.1 --port 6011
```

The original 50-epoch reference took approximately 16h55m. SAM doubles the
trainable-model forward/backward work, but cached DINO extraction is performed
once. Total runtime must be estimated from completed SAM epochs, not assumed
to be exactly twice the original.

## Verification and interpretation

The actual 11-source GPU smoke test completed one accumulated SAM update and
validation, saving a separate smoke checkpoint. All 46 tests pass, including
direction/radius, frozen parameter isolation, exact restoration, finite-gradient
guard, stochastic replay and exact optimizer/SAM/RNG continuation in a toy model.
The checkpoint verified null initialization and MotionTrans decoder exclusion.

SAM is weight-neighborhood robustness, not a guarantee of dataset invariance,
sensor transfer or calibrated latent uncertainty. Effective rank is logged to
detect representation compression. After training, the frozen evaluators can
use the new final checkpoint in separate output directories. This launcher
does not automatically run holdout evaluations or retrain a MotionTrans-only
control; those were not part of this training request.

## Concurrent intermediate MotionTrans evaluation

At the user's request, a separate evaluation was launched from the completed
epoch-3 checkpoint while training continues. The evaluator uses an immutable
hard-linked checkpoint snapshot, so later atomic replacements of `last.pt`
cannot change its encoder. Both its encoder and DINO are frozen; it trains
only a new original-size `32→512→6` head using the established episode-disjoint
tuning/refit protocol, then scores all 1,972 validation clips with 64 draws.
All 18,613 training clips are used. This is not the larger-decoder experiment.

The exact snapshot, output directory, and command are recorded at
`processing/runs/joint_all_sam_holdout_motiontrans_fresh_8k_50ep/evaluations/current_motiontrans_evaluation.json`.
The output contains `status.json`, `evaluation.log`, frozen feature caches,
and, on completion, `decoder.pt`, `results.json`, examples and
`baseline_comparison.json`. That comparison explicitly labels epoch-3 SAM
versus final epoch-50 AdamW; it cannot isolate optimizer effects at equal
training duration. Live joint training is never paused or signaled by this
evaluator, although shared GPU use can slow it temporarily.

The normal evaluator still requires the final checkpoint by default; the
explicit `--allow-intermediate` flag permits this requested early evaluation.
Checkpoint SHA-256 is checked again before result publication. The supervisor
is detached under nohup/setsid. Do not relaunch it to monitor: read its logs.

## Final MotionTrans comparison — 2026-10-03

The final epoch-50 SAM encoder has now been frozen and evaluated using the
original `32→512→6` decoder protocol, all 18,613 training clips, all 1,972
validation clips, and 64 latent draws. Manifest hash, source, episode partitions,
seed, target layout, and decoder architecture match the final non-SAM baseline.
Decoder duration selected on training-only tune episodes was 58 epochs for
SAM features versus 21 for non-SAM features; both heads were then refitted on
all training clips. The head optimizer remains AdamW for both evaluations,
isolating the encoder-training change rather than also changing head optimization.

| Metric | Non-SAM | SAM |
| --- | ---: | ---: |
| Deterministic MAE ↓ | 0.058843 | 0.058810 |
| Deterministic MSE ↓ | 0.009058 | 0.008630 |
| 64-draw ensemble MAE ↓ | 0.061797 | 0.064124 |
| CRPS ↓ | 0.048898 | 0.049648 |
| Uncertainty/error Pearson correlation ↑ | 0.4047 | 0.4308 |
| Nominal 95% marginal interval coverage | 66.48% | 65.20% |
| Mean 95% interval width | 0.12591 | 0.13466 |

Deterministic MAE is effectively tied (SAM 0.055% lower), while deterministic
MSE improves 4.72%. Ensemble MAE worsens 3.76% and CRPS worsens about 1.53%.
SAM's intervals are wider but cover slightly fewer readings. Thus this run
does not show a clear overall held-out sensor transfer or calibration benefit.
It is one run per optimizer, not evidence of statistical significance or a
claim that all SAM settings fail. All metrics remain normalized torque,
not verified SI units. Baseline artifacts remain unchanged; full SAM results,
head, examples and `non_sam_comparison.json` are saved in
`processing/runs/joint_all_sam_holdout_motiontrans_fresh_8k_50ep/evaluations/motiontrans_final`.

Reproduce the matched comparison after the two result files exist:

```bash
.venv/bin/python scripts/compare_sam_motiontrans.py
```

The comparison script checks protocol identities and refuses to overwrite its
completed comparison file by default.
