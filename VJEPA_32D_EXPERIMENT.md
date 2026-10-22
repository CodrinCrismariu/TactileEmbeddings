# Frozen V-JEPA 2.1: deterministic and stochastic 32D joint models

**MotionTrans viewer integration:** Completed deterministic transfer heads at encoder epochs 5, 10, and 28 and Gaussian heads at epoch 28 are now registered under their original runs. The epoch selector distinguishes head seeds 45/46/47. Fit-only feature z-score normalization is preserved at inference; all exports reproduce the original head computation exactly. These are existing evaluated heads, not new fits, and the interrupted epoch-15/20/25 evaluation remains stopped. The newer spatial-attention model has no MotionTrans transfer head yet. Registration evidence: `processing/evaluations/motiontrans_vjepa21_best_v1/viewer_registration.json` and `viewer_inference_check.json`.

**Current status — October 15, 2026:** Both 30-epoch cached runs completed in 1 h 56 min. Best source validation is epoch 28 for both: deterministic 0.741732, stochastic 0.740994; DINO best source score was 0.655483. MotionTrans transfer of these frozen epoch-28 models is now complete: deterministic MAE 0.040215 ± 0.000534, stochastic 0.040216 ± 0.000223 across three fitted head seeds. Both improve approximately 8.1% over the matched DINO epoch-7 32D latent (0.043769), with paired episode-bootstrap intervals supporting improvement. The two V-JEPA latent modes are effectively tied. Frozen temporal DINO features still perform better at 0.031462. See [transfer report](processing/evaluations/motiontrans_vjepa21_best_v1/analysis.md).

Implemented and launched on October 15, 2026. Both models start at epoch 0 and train for **30 epochs**, sharing frozen video features. Source training excludes MotionTrans. This is a new architecture, not a continuation of a DINO checkpoint.

## Architecture

Eight chronological frames at stride two, with the existing scene-camera roles, enter the official V-JEPA 2.1 ViT-B/16 encoder at 384×384. It has **86,833,152 frozen parameters**. The released model family lists Base as approximately 80M.

Only the encoder is used. Its final normalized block produces 4×24×24 video tokens, each 768D. The four temporal positions correspond to frame pairs `(0,1), (2,3), (4,5), (6,7)`. Average the spatial tokens, then average available cameras using the camera mask. A shared linear projection `768 → 64`, reshaped into two ordered 32D vectors per temporal position, yields `[batch, 8, 32]`. Its two sets of weights distinguish the two frames; tactile targets are neither averaged nor interpolated. Fine temporal distinctions must be recovered from the pair's video descriptor, so this remains a limitation to evaluate.

The deterministic model has only this mean/vector projection. The stochastic model additionally projects `768 → 64` log variances, clamps them to `[-8,4]` as in the previous architecture, and samples a diagonal Gaussian during training. Its KL weight ramps linearly from zero to 0.001 over ten epochs. Evaluation and the viewer use latent means. Variance is **uncalibrated**, not a calibrated confidence estimate.

Both models use sensor-code adapters `32 → 128 → 128 → code_dim` with GELU. The one- and two-value numeric sources retain simple linear sigmoid heads. Their trainable parameter counts are **442,339 deterministic** and **491,555 stochastic**. Frozen tactile AE decoders are used for native reconstruction evaluation, not training supervision.

## Matched training and data

- One seed: 45. Fresh task parameters; common weights copied exactly between arms. Actual epoch-0 checkpoints and hashes are saved.
- Same source batches, episode/frame indices, and camera dropout masks. Camera dropout uses a separate deterministic stream, independent of Gaussian sampling.
- Eleven training sources. Existing teacher registry, train-only code statistics, cached codes, and native chronological audit are checked before startup. MotionTrans is absent from source training and validation.
- 8 frames per clip, stride 2; batch size 2, accumulation 16; up to 8,000 clips/source/epoch. This produces **62,991 clips and 31,497 microbatches per epoch**.
- Four data-loader workers. Validation uses a fixed maximum of 64 clips per source and decodes full-resolution tactile outputs in frame microbatches.
- SAM AdamW, learning rate 5e-5, minimum 1e-6, weight decay 0.05, radius 0.05; two-epoch warmup and 30-epoch cosine schedule. Wrist dropout 0.3.
- No DINO, custom spatial/temporal transformer, FGW, relational loss, retention, meta-learning, or episode context.

The data-loader RGB-size option defaults to the previous 224 pixels for existing callers. This study explicitly requests 384. Only visual inputs are resized; cached tactile codes, original tactile images, camera roles, and frame mappings are unchanged.

## Execution, artifacts, and resume

Configuration: `configs/joint_vjepa21_pair_32d_v1.json`.

Run directory: `processing/runs/joint_vjepa21_pair_32d_v1/`:

- `deterministic/`, `stochastic/`: epoch-0 and every completed epoch's full model/optimizer/scheduler/SAM checkpoint; `last.pt`, `best.pt`, viewer snapshots, initial hashes, per-source metrics, and latent statistics.
- `pair_epochs/`: per-epoch paired RNG/data-loader state and commit metadata.
- `pair_state.pt`: atomic latest paired commit, written after both arm checkpoints. Resume uses this commit so interrupted writes cannot mix epochs.
- `status.json`, `nohup.log`, `launcher.pid`: live progress and process identity.
- `provenance.json`: backbone, implementation, manifest, teachers, and code-cache provenance.

The paired trainer runs one frozen V-JEPA extraction per batch. Each arm has an independent optimizer and SAM replay, which reuses detached features and restores its original latent draws. Resume at the latest committed boundary with:

```bash
bash scripts/run_vjepa_pair.sh --resume
```

Do not launch a second copy while the existing process is running. The launcher supplies the local CUDA library path and four-thread CPU settings. Original training runs remain stopped and preserved.

The prediction viewer on port 8080 includes **V-JEPA 2.1 B · deterministic · 32D** and **V-JEPA 2.1 B · gaussian · 32D**. Epoch-0 initialization is available immediately; completed epochs appear as training progresses. Video features are cached by the entire ordered clip, camera selection, and pinned backbone, because a frame's descriptor depends on its surrounding video.

## Validation and measured compute

**34 targeted tests passed**, covering frame-pair order, camera masking, unchanged native targets, deterministic/no-KL behavior, Gaussian gradients, matched initialization, frozen-backbone behavior, exact paired optimizer/RNG resume, and both old/new viewer paths. A real GPU preflight executed forward/backward and native decoder validation for all eleven sources. All outputs were finite, the frozen backbone received no gradients, and no MotionTrans examples entered the run.

The short benchmark includes real data loading, frozen feature extraction, task forward/backward, and SAM replay/update, with 16 warmup and 32 timed batches at batch size 2:

| Pipeline | One camera | Three cameras |
|---|---:|---:|
| Existing DINO + trainable trunk, one model | 38.9 ms/batch | 59.0 ms/batch |
| Frozen V-JEPA, one task model | 35.5 ms/batch | 77.6 ms/batch |
| Frozen V-JEPA, both task models together | 32.8 ms/batch | 72.6 ms/batch |

Short timing differences between single and paired V-JEPA are measurement/cache noise; paired computation is not intrinsically cheaper. These results establish roughly comparable compute, especially because the paired experiment shares its backbone. Benchmark training peak allocation was about 0.85 GB for the V-JEPA pair versus 2.45 GB for the existing DINO pipeline; full native validation uses additional memory. Measurements exclude model startup, checkpoint writes, and full validation. The first 1,201 production batches averaged 49.6 ms/batch for both models together: about 26 minutes per epoch before validation. Allow approximately **14–16 hours total** for both 30-epoch runs, including validation and I/O variation. Refine this estimate after the first complete epoch.

Benchmark: `processing/vjepa2_1/benchmark.json`; real-data preflight: `processing/vjepa2_1/preflight.json`; tests: `processing/vjepa2_1/tests.log`.

## Pinned upstream provenance

- Official repository: https://github.com/facebookresearch/vjepa2
- Commit: `204698b45b3712590f06245fbfba32d3be539812`
- Checkpoint: https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt
- SHA-256: `848a77c33cc9e6649ed2119c9bea1e2c569bcdab9539ff3e7c02ccc2959ddf4d`
- Weights loaded strictly from `ema_encoder`, stripping official `module.backbone.` prefixes. The predictor is not instantiated.
- Additional pinned dependencies: `timm==1.0.22`, `einops==0.8.1`.
- Upstream's pinned Hub helper contains a localhost download URL; the integration uses the official public checkpoint link above directly, without modifying upstream code.

## Resume using cached clips

The cache completed in 9,804.9 seconds (2 h 43 min) and contains 349,081 clips / 648,017 camera clips. Startup checks all source train/validation feature checksums, camera roles, exact window coverage, teacher/code-cache identities, and the pinned backbone. Thirteen relevant tests and the real all-source cached training/native reconstruction check passed.

Use `bash scripts/resume_vjepa_cached.sh` after ensuring the existing process is stopped. The original trainer code and pre-migration state are archived in `processing/runs/joint_vjepa21_pair_32d_v1/provenance_revisions/before_clip_cache/`. The explicit cache migration permits only the input-path change and its audited trainer implementation; other configuration, model, data, optimizer and sampling changes are rejected. `cache_adoption.json` records the transition. Existing epoch-0 snapshots remain unchanged; subsequent epochs save the cache provenance. Live output is `cached_nohup.log`.
