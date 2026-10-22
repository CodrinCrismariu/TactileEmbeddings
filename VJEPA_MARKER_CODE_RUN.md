# V-JEPA joint training with cached marker codes

Launched October 16, 2026, from fresh task weights at epoch 0, seed 45, initially for 20 epochs. **The user extended the target to 30 epochs while epoch 9 was running.** The pretrained V-JEPA backbone is frozen. This is one deterministic model with **32D per image**, without KL or latent sampling.

The extension controller (`scripts/extend_vjepa_markers.py`) waits for the complete epoch-9 validation/checkpoint commit, validates the permitted duration-only migration, and resumes from that checkpoint. It retains model, optimizer, SAM, scheduler update count, and RNG state. The cosine tail is anchored at the saved learning rate and decays to the same minimum at epoch 30; prior training is unchanged. `extension_request.json`, `extension_status.json` and `extension.log` record this handover. The active process and its configuration still report the original 20-epoch budget until that boundary; the configuration becomes 30 epochs at handover. Original trainer/config/preflight files and the handover checkpoint are preserved under `before_epoch_extension/`.

The extension contract passed against the real epoch-8 checkpoint before queuing; nine focused tests passed, including LR continuity, extended-schedule resume, and rejection of changes to model/data/cache/settings. The inherited GPU preflight remains pinned. A normal resume after handover uses `preflight_extension.json` and strict updated provenance. Expected remaining time when requested: approximately 11 hours at 30.5 minutes per epoch.

Configuration: `configs/joint_vjepa21_spatial_marker_codes_32d.json`.
Run: `processing/runs/joint_vjepa21_spatial_marker_codes_32d_v1`.
Continuous log: `VJEPA_MARKER_TRAINING.log`.
Progress: the run's `status.json`; completed-epoch results: `metrics.jsonl`.

## Architecture and supervision

Eight chronological ego frames per available camera, stride 2, enter frozen V-JEPA 2.1 Base at its required 384-pixel visual resolution. All spatial tokens enter the existing two-layer attention decoder (width 256, four heads, feedforward width 1024), with eight ordered frame queries. Its MLP produces one deterministic 32D latent for each image. There is no spatial pooling or persistent visual-feature cache.

The sensor adapters remain `32 → 128 → 128 → sensor code`. Four optical image-code targets are replaced by frozen marker-displacement AE codes:

| Source | Sensor-code dimension |
|---|---:|
| FreeTacMan | 32 |
| FTP UniT | 8 |
| FTP UniT Bimanual | 16 |
| FTP VLA-Touch | 8 |

The cache is `processing/sensor_code_cache_markers_v1`, with pinned marker teachers in `processing/joint_alignment/marker_teachers_v1.json`. It contains every chronological frame, original masks and splits, verified teacher/target hashes, and training-only normalization. The other five sensor AEs retain their existing verified code caches. Two numeric sources retain direct numeric prediction. MotionTrans is excluded from representation training and source validation.

Training reads standardized frozen codes; it does not reconstruct optical RGB for these four sources. Validation uses frozen marker decoders to report native-pixel displacement and temporal-change errors. Existing native reconstruction metrics remain available for the other sources. Cached codes replace tactile-encoder forward passes; the visual backbone still runs online.

## Schedule and checkpoints

SAM AdamW: peak learning rate 0.0003, minimum 0.000001, two warmup epochs, weight decay 0.05, SAM radius 0.05. Batch size 2, accumulation 16, up to 8,000 clips per source per epoch, 64 fixed validation clips per source, four loader workers, wrist-camera dropout 0.3. There is no FGW, relational objective, retention, meta-learning, or static episode context.

The run saves the actual fresh epoch-0 checkpoint, full resumable checkpoints and viewer snapshots each epoch. The previous image-supervised V-JEPA checkpoints remain separate. The viewer identifies this model by **marker AE codes** in its name; available epochs grow as checkpoints are saved.

The real GPU preflight passed training and SAM updates on all 11 source datasets and confirmed no backbone or tactile-teacher gradients. Its temporary task updates are discarded before production initialization. Evidence is in `preflight.json`, `preflight.log`, `launch.json`, and `provenance.json` in the run directory. Earlier cache verification and the all-source input check are in the cache root.

To resume after an interruption:

```bash
bash scripts/run_vjepa_markers.sh --resume
```

The checkpoint fixes the configuration, optimizer, scheduler, SAM, RNG state, data, teacher and cache provenance. Do not change those inputs in place during this run.
