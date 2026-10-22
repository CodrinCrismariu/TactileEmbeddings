> Historical record, relocated on October 21, 2026. Original path: `ALTERNATING_ADVERSARIAL_TRAINING.md`.
> Results, proposals, and operational status below belong to their recorded experiment/date;
> use the [maintained project state](../../project_state.md) for the audited active workflow.

# Alternating dataset-adversarial training

**Historical VLA_touch-holdout experiment.** Its completed results are retained.
The current baseline has MotionTrans held out and no dataset adversary;
see [MOTIONTRANS_BASELINE_TRAINING.md](MOTIONTRANS_BASELINE_TRAINING.md).

The earlier online-GRL experiment was deliberately stopped after **15 completed
epochs**. Its last checkpoint, log, epoch-10 independent probe, and metrics remain
in `processing/runs/joint_all_grl_fresh_8k_50ep/`; it is not a final model. Its
launcher was stopped before either final evaluation could run. The original
non-adversarial baseline remains in
`processing/evaluations/baselines/joint_50ep_vla_touch_v1/`.

This experiment starts from fresh shared-transformer, Gaussian-latent, decoder,
and classifier weights. Only pretrained DINOv3 is frozen. It uses the same 11
training sources and leaves `ftp_vla_touch` out. Each source contributes up to
8,000 clips per epoch for 50 epochs.

## What changes

The two source classifiers are trained in separate updates on a recent balanced
bank of detached mean and sampled 32D latents. Five classifier steps precede
every visual-model optimizer step. One classifier persists; the other is reset
and retrained at the start of every epoch. While the visual model updates, both
classifiers are frozen and the better classifier on the current batch sends a
reversed gradient through the latent. The first epoch fills the bank and trains
the classifiers without reversal. Reversal is then calibrated to comparable
latent-gradient magnitude, ramped over five epochs, and doubled (up to 2.0)
when an independent episode-disjoint diagnostic exceeds 14% macro accuracy.

Low accuracy of either live classifier is **not** evidence of invariance. Fresh
probes at epochs 5, 10, 20, 30, 40, and 50 provide that check. The same frozen
VLA_touch decoder and independent dataset-classifier evaluators run after the
last epoch. Their output directories are separate from the baseline.

## Reproduce and monitor

```bash
.venv/bin/pytest -q
.venv/bin/python -m tactilelatent.train_joint_alternating \
  --config configs/joint_all_alternating_smoke.json
nohup bash scripts/train_joint_alternating_fresh.sh \
  > processing/runs/joint_all_alternating_fresh_8k_50ep/nohup.log 2>&1 < /dev/null &
tail -f processing/runs/joint_all_alternating_fresh_8k_50ep/nohup.log
```

TensorBoard listens on remote `127.0.0.1:6008`. In a remote VS Code session,
forward port 6008 in the **Ports** panel and open its forwarded address; local
`localhost:6008` will not work until that forwarding is configured.

For epoch-boundary resumption, run
`bash scripts/train_joint_alternating_fresh.sh --resume`. The checkpoint stores
both classifiers, both optimizers, scheduler, replay bank, reversal controller,
diagnostics, and RNG states. The final checkpoint's `model` key is compatible
with both existing frozen evaluators. Their completed results refuse overwrite
unless `--overwrite` is explicitly passed.

The run estimates empirical decoder calibration, not uncertainty in decoder
weights. Strong source suppression can trade against tactile prediction or
collapse latent directions; report both diagnostics even if identity improves.
