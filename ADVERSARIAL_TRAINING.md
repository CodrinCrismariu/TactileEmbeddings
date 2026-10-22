# Fresh joint training with dataset gradient reversal

**Historical VLA_touch-holdout design.** The current experiment is a fresh
non-adversarial baseline with MotionTrans held out and VLA_touch included.
See [MOTIONTRANS_BASELINE_TRAINING.md](MOTIONTRANS_BASELINE_TRAINING.md).

The original 50-epoch model and frozen VLA_touch/classifier results are pinned
under `processing/evaluations/baselines/joint_50ep_vla_touch_v1`. Verify the
16 copied files and SHA-256 checksums without changing them:

```bash
.venv/bin/python scripts/snapshot_joint_baseline.py
```

The new run starts **fresh**: frozen pretrained DINOv3, but randomly
initialized cross-view/space-time encoder, 32D Gaussian heads, all 11 tactile
decoders, and the 32→512→11 source classifier. Neither prior EgoTouch nor
eight-epoch joint weights are loaded. `ftp_vla_touch` stays excluded.

```bash
# One-batch-per-source smoke test (already completed)
.venv/bin/python -m tactilelatent.train_joint_adversarial \
  --config configs/joint_all_grl_smoke.json

# Full run; the launcher automatically evaluates the final checkpoint.
bash scripts/train_joint_grl_fresh.sh

# If interrupted after an epoch checkpoint:
bash scripts/train_joint_grl_fresh.sh --resume
```

The full run writes `processing/runs/joint_all_grl_fresh_8k_50ep/last.pt`,
`metrics.jsonl`, periodic `epoch_XX.pt`, and diagnostic-probe results. Its
launcher then writes the frozen VLA_touch and independent dataset-identity
evaluations under `evaluations/` in that run directory. To monitor a detached
run:

```bash
tail -f processing/runs/joint_all_grl_fresh_8k_50ep/nohup.log
```

Live training and classifier curves are also served by TensorBoard at
`http://localhost:6007` (forward port 6007 in remote VS Code).

During epoch 1, the classifier learns source labels with zero reversed
gradient. The trainer calibrates a moderate reversal coefficient from latent
gradient norms and ramps it over the next five epochs. A *fresh* classifier
probe at epochs 10/20/30/40 uses only fixed episode-disjoint subsets of the
manifest training split. If mean- or 8-draw sampled-latent macro accuracy is
above 14%, reversal strength doubles (capped at 2.0). The final comparison
reuses the original held-out VLA_touch and full 11-source evaluation protocols
with 64 draws; diagnostic probes do not touch those validation episodes.

The in-training classifier accuracy is not an independent invariance result.
Near-chance performance of the separate MLP probe is the primary objective;
latent variance/effective rank show whether that result merely reflects
collapse. Tactile losses are reported but do not limit adversarial strength.
Because the reference model used prior pretraining while this one does not,
differences cannot be attributed solely to gradient reversal.
