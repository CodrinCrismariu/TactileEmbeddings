> Historical record, relocated on October 21, 2026. Original path: `VJEPA_FEATURE_CACHE.md`.
> Results, proposals, and operational status below belong to their recorded experiment/date;
> use the [maintained project state](../../project_state.md) for the audited active workflow.

# Full-clip V-JEPA feature cache

Requested October 15, 2026. The deterministic/stochastic training pair was stopped during epoch 1, after approximately 11,901 of 31,497 batches. No epoch had completed, so **epoch 0 is the latest saved checkpoint**; partial epoch-1 optimization was not saved. At the user’s later request, both models resumed from epoch 0 using the completed cache, toward epoch 30.

## What is cached

Every eligible clip indexed by the current native chronological loader, across all twelve datasets and all splits, including separately partitioned MotionTrans:

- **349,081 eight-frame clips**, each sampled at stride two within a 15-frame span.
- **648,017 camera clips**. Each available camera is processed as a complete eight-frame video, not as independent images.
- Frozen official V-JEPA 2.1 ViT-B/16 at 384×384; same pinned encoder/checkpoint as the stopped runs.
- After video inference, average the 24×24 spatial patches, preserving all four temporal groups and 768 features per group.
- Save float32 `[available_camera_clips, 4, 768]` arrays. **No camera averaging, camera dropout, normalization fitted to datasets, latent projection, or tactile input** occurs during caching.
- Encoder computation uses the same bfloat16 autocast as live training; spatial averaging and cache storage are float32. No additional precision reduction is introduced by caching.
- Original tactile pixels, cached sensor codes, frame mappings, and camera roles remain unchanged.

Feature payload is **7,962,832,896 bytes (7.96 GB / 7.42 GiB)**, plus small clip/episode indexes and provenance. Only available cameras occupy disk rows; the reader reconstructs zero-filled missing slots and an availability mask for training. The complete build took **2 hours 43 minutes**. All partitions are complete and their checksums are recorded.

This covers the existing loader's clip windows, not every possible overlapping video window. Different surrounding frames define different video features. The reader rejects arbitrary starts, reversed frames, or different temporal spacing rather than incorrectly treating the cache as independent frame embeddings.

## Location and progress

`processing/vjepa21_clip_cache_v1/`

- `index.json`: complete inventory, byte estimate, source/backbone/code provenance.
- `status.json`: stage, current source/split, completed clips and camera clips, measured extraction rate and estimated remaining time.
- `nohup.log`, `launcher.pid`: detached process output and identity.
- `<source>/<split>/clips.npy`: ordered clip records (local episode, first frame, camera-row offset).
- `<source>/<split>/index.json`: episode identities, camera slots, feature shapes, mapping digest.
- `<source>/<split>/features.partial.npy`: unfinished float32 output.
- `<source>/<split>/progress.json`: durably committed prefix for resuming.
- `<source>/<split>/features.npy`, `complete.json`: finished partition and SHA-256 checksums.
- Root `complete.json`: written only after every partition passes finite-value/shape checks and the source signature is rechecked.

Progress is committed every 512 clips and at partition boundaries. Feature data is flushed and fsynced before the atomic progress update. After interruption, uncommitted rows are recomputed. Completed partitions are retained. A file lock prevents duplicate writers.

Start or resume the same cache, after ensuring its previous writer is no longer running:

```bash
bash scripts/cache_vjepa_clips.sh
```

The builder validates existing provenance and resumes automatically. It never launches training.

## Reuse for training

`VJEPAFeatureCache` provides strict reads by source, split, episode identity, and the exact ordered eight-frame indices. `CachedAlignedClips` wraps the existing aligned source dataset, returning `video_features` instead of decoding RGB, while retaining its original window index, sensor-code targets, numeric targets, native evaluation targets, and camera mask.

The paired trainer now reads `video_features` directly for both training and validation, applies the original camera dropout, and updates the two task models. No V-JEPA backbone is instantiated in this path. The cached configuration is `configs/joint_vjepa21_pair_32d_cached_v1.json`; resume with `bash scripts/resume_vjepa_cached.sh` only when the existing trainer is stopped. The stopped trainer and state are archived before migration. Startup verifies complete cache provenance, all source partition checksums, exact clip coverage, camera slots, and unchanged teacher-code eligibility. MotionTrans is excluded by the original source dataset construction. Thirteen cache/training/model tests passed, followed by 352 real cached training batches and native decoder validation over all eleven source datasets.

Keep MotionTrans partitions out of source-training dataset construction. Their frozen feature extraction is separate from fitting any representation, normalizer, or sensor head. The current builder fits no statistics on any split.

## Verification

- Ten cache/model unit tests passed, including interrupted-write recovery, exact float32 disk round trips, source/split separation, missing-camera restoration, mismatched clip rejection, nonfinite-output rejection, and native-target preservation without RGB decoding.
- Real GPU verification used two full clips each from EgoTouch (three cameras), VLA Touch (two cameras), and MotionTrans (one camera).
- Cached pooled features were **bit-for-bit identical** to the corresponding live features.
- Deterministic and stochastic model latent means had **maximum absolute difference 0.0** between cached and live input. The stochastic comparison used the Gaussian parameters, not independent random draws.
- Verification artifacts: `processing/vjepa2_1/cache_tests.log` and `processing/vjepa2_1/cache_verification.json`.
