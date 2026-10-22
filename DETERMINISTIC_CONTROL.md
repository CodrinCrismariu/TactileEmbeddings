# Fresh deterministic 16D control

Started **October 14, 2026, 11:35 AM EDT**, at the user's request.

**Stopped by user — October 14, 2026, 1:50 PM EDT:** The training process group was terminated and verified to have no remaining live workers. The last saved checkpoint is epoch 5, including optimizer/RNG state and its viewer snapshot. The partially completed epoch 6 was not checkpointed. See [stop status](processing/runs/joint_16d_deterministic_control_v1/stop_status.json).

The previous study controller and paused trajectory worker were terminated. The saved trajectory checkpoint reaches epoch 17; completed runs and evaluation artifacts are preserved. The original training-source files were verified against the saved provenance and archived under `processing/runs/invariance_16d_v1/stopped_code_snapshot/` before implementing the new mode.

## Architecture and training

```text
Ego clips → frozen DINO → randomly initialized spatial/temporal encoder
                                    ↓
                        deterministic 16D latent
                                    ↓
                        existing sensor-specific adapter
                                    ↓
                     cached sensor AE code / numeric output
```

- **Fresh shared encoder and adapters:** epoch-0 random initialization, seed 45. No parent checkpoint is loaded. Training epochs are numbered 1–20.
- **No Gaussian sampling, learned variance branch, or KL penalty.** KL weight is zero and logged KL is exactly zero. Source code/numeric losses remain active.
- Existing frozen DINO and the pinned sensor autoencoders/cached codes are retained. No teachers are retrained.
- Existing adapter widths remain `16 → 128 → 128 → sensor code`. Numeric heads remain simple sigmoid heads.
- Relational, trajectory, and meta objectives are disabled for this control.
- The eleven training sources and MotionTrans holdout remain unchanged; native target dimensions, camera roles, and chronological frame pairing are preserved.
- Existing training settings remain batch size 2, accumulation 16, 8,000 windows/source/epoch, 25% matched-bank sampling, SAM-AdamW, learning rate 5e-5, weight decay 0.05, two warmup epochs, and cosine scheduling over the fresh 20-epoch budget.
- Ordinary training dropout and wrist-view augmentation are retained. “Deterministic latent” means no learned latent distribution or Gaussian reparameterization; it does not disable standard training augmentation/dropout.

## Configuration and live artifacts

- [Configuration](configs/joint_16d_deterministic_control.json)
- [Launch metadata](processing/runs/joint_16d_deterministic_control_v1/training.pid)
- [Initialization proof](processing/runs/joint_16d_deterministic_control_v1/initialization.json)
- [Live progress](processing/runs/joint_16d_deterministic_control_v1/status.json)
- [Training log](processing/runs/joint_16d_deterministic_control_v1/training.log)

The run is detached under `nohup`. Full optimizer/RNG checkpoints and lightweight viewer snapshots are saved after each completed epoch. The viewer service was reloaded and verified reachable on port 8080 so it can discover deterministic checkpoints when they become available.

Resume a saved epoch with:

```bash
.venv/bin/python -m tactilelatent.train_invariance \
  --config configs/joint_16d_deterministic_control.json --resume
```

Use the established CUDA library environment described in [the invariance guide](INVARIANCE_TRAINING.md).

## Compatibility and verification

Historical Gaussian checkpoints retain their original architecture by default. Deterministic checkpoints explicitly set `latent_mode="deterministic"`, `latent_objective="deterministic"`, and `stochastic=false`. The learned variance projection is removed from their parameters/state dict.

For the existing `(mu, logvar)` encoder/probe interface, the deterministic encoder returns its latent as `mu` and constant negative-infinite `logvar`, representing exactly zero sampling variance. The loss bypasses the Gaussian KL calculation and rejects a nonzero KL coefficient. Probe draws therefore equal the deterministic latent exactly. This compatibility representation does not introduce a learned Gaussian.

Tests verify no variance parameters or latent noise, finite ordinary/numeric losses, direct latent-projection gradients, deterministic checkpoint round trips, SAM replay, fresh initialization, and exact interrupted/resumed training. They also compare the refactored Gaussian encoder against the archived implementation with identical weights and inputs: outputs match exactly. The relevant regression suite passed **49 tests**, and a two-update GPU verification completed with finite loss and zero KL before launching production training.

**Uncertainty:** this run does not provide latent-derived predictive uncertainty. Repeating deterministic latent draws produces identical predictions and zero-width intervals. Evaluations label that limitation explicitly; those intervals must not be described as calibrated uncertainty. Any future uncertainty method needs a separately specified objective or calibration procedure.
