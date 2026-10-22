# Full-spatial V-JEPA deterministic joint training

Started October 15, 2026, from fresh task weights at epoch 0. The run is configured for 30 epochs, seed 45, with an actual saved epoch-0 checkpoint. MotionTrans is excluded from all source training and validation.

## Architecture

```text
Eight chronological RGB frames per available camera, stride 2
→ frozen official V-JEPA 2.1 Base encoder at 384×384
→ all 4 × 24 × 24 × 768 tokens per camera
→ per-token LayerNorm + Linear(768,256), spatial/time/camera-role positions
→ eight learned frame queries through two attention decoder blocks
→ LayerNorm → Linear(256,128) → GELU → Linear(128,32)
→ one deterministic 32D latent per image
→ existing 32 → 128 → 128 → sensor-code adapters / simple numeric heads
```

Each decoder block has four-head self-attention among the eight frame queries, cross-attention to every valid spatial/time/camera token, and a 256→1024→256 GELU feed-forward network. There is no fixed spatial or camera averaging, including in query initialization. Missing cameras are masked. Each pair member has a distinct query; tactile readings are never averaged. FreeTacMan's slot-1 scene camera is assigned the scene role without changing its file mapping. The model uses bidirectional clip context, as before.

Only task layers are optimized. The backbone and all tactile autoencoders remain frozen in evaluation mode. There is no log-variance head, sampling, KL, FGW, relational objective, retention term, meta-learning, or static episode context.

## Live inputs and unchanged tactile objective

No visual-feature cache or per-frame sensor-code cache is read or written. Source RGB and native tactile values are loaded for every batch; the V-JEPA encoder and the frozen tactile AE encoders execute online. Input RGB still receives the pretrained checkpoint's required 384-pixel resize and ImageNet normalization. Tactile images retain their existing native shapes and chronological mapping.

The only reused statistics are fixed train-only code means/stds, exported from the previous epoch-0 checkpoint into `configs/vjepa_live_teacher_normalization.json`. Their teacher and manifest hashes are checked. They do not initialize task weights and are not cached sample targets. Native tactile supervision, including all four optical datasets mentioned in the paused marker discussion, remains the existing sensor-code objective. Marker prototypes are unused; no marker-target dataset migration occurred.

SAM reuses detached features and teacher outputs within the current optimizer group in RAM. These transient tensors are released after the update and never become a dataset cache. Dropout RNG is restored during SAM replay.

## Training settings and artifacts

- SAM AdamW, peak LR **3e-4**, minimum 1e-6, two warmup epochs, weight decay .05, SAM radius .05. The peak LR is higher than the historical linear readout's 5e-5, following the readout underfitting diagnosis; this is therefore not a pure architecture-only ablation.
- Batch size 2, accumulation 16, up to 8,000 clips per source per epoch, four loader workers, wrist dropout .3.
- 31,497 batches per epoch; fixed source validation cap 64 clips/source.
- Native reconstruction MAE, supervision loss, effective rank/spread, runtime and peak GPU memory are logged. Frozen AE reconstruction provides the oracle comparison.
- Full resumable checkpoints and small viewer snapshots are saved each epoch. `last.pt` resumes from the most recent complete epoch with optimizer, scheduler, SAM and RNG state.

Run directory: `processing/runs/joint_vjepa21_spatial_live_32d_v1`.
Configuration: `configs/joint_vjepa21_spatial_live_32d.json`.
The user subsequently shortened this run to **20 completed epochs**. A detached controller, `scripts/stop_spatial_vjepa_after_epoch.py`, waits for the epoch-20 validation and all checkpoint saves to finish, then terminates only this trainer's process group. The original 30-epoch configuration and learning-rate horizon stay intact for provenance and checkpoint compatibility. `stop_request.json` and `stop_controller_status.json` record the active stopping boundary; `stop_status.json` records completion. The MotionTrans epoch-15 evaluation runs separately and continues unaffected. Do not resume beyond epoch 20 without a new instruction.

The initial production batches averaged about 0.076 seconds/batch, around 40 minutes per epoch. The provisional 30-epoch estimate is 20–24 hours, subject to full-epoch data loading and validation measurements. This uses measured live throughput, not the older cached training speed.

## Validation before launch

Twenty-two focused tests passed across spatial encoding, existing V-JEPA behavior and viewer compatibility. Tests cover distinct ordered per-frame latents, sensitivity to spatial layout despite identical arithmetic means, absent-camera masking, corrected FreeTacMan role embedding, deterministic KL exclusion, SAM updates and checkpoint/optimizer/RNG continuation equivalence.

The separate real-data preflight exercised training/SAM and native reconstruction for all eleven sources, including FreeTacMan at 6×480×640 and Unit Bimanual at 6×240×320. V-JEPA and tactile teachers received no gradients. It also measured complete accumulation-16 update groups on one-, two-, and three-camera datasets. Preflight used a copy of task weights and did not initialize production from those updates.

Evidence: `preflight.json`, `initialization.json`, `provenance.json`, `status.json`, and `nohup.log` in the run directory. The prediction viewer exposes the new model as **V-JEPA 2.1 B · deterministic · 32D · spatial attention · live**. Old model types retain their separate loaders and feature keys.
