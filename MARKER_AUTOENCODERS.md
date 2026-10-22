# Marker autoencoders and frozen-code joint supervision

This is the current replacement for the four image/code targets from FreeTacMan, FTP UniT, FTP UniT Bimanual and FTP VLA-Touch. The earlier direct-displacement joint-head prototype is superseded by **reconstruction autoencoders followed by frozen-code supervision**, as requested.

**Current FreeTacMan run:** the user requested starting one model immediately on processed frames. The waiting FreeTacMan two-candidate queue was handed over to a **single 16D, 100-epoch run** at `processing/runs/marker_autoencoders_freetacman_available_v1`. Its frozen snapshot contains 5,026 training episodes and all 186 validation episodes (5,212/5,625 total). Training retains the 100,000-frame budget at stride 2 and every marker in each selected frame; validation uses all 27,900 frames at stride 2. Ongoing extraction does not change this run's dataset. `selection.json` and `available_manifest.json` record the exact cutoff and included/missing episodes.

The first three source finalists remain pinned. On completion, the 16D FreeTacMan teacher is added to the common registry with explicit partial-training metadata. The continuous log remains `MARKER_AUTOENCODER_TRAINING.log`. To resume this particular run after a stop, use `.venv/bin/python scripts/train_freetacman_available_markers.py` with the same CUDA runtime environment as the other training commands. The earlier two-candidate FreeTacMan queue is no longer running.

**Additional requested 32D comparison:** `configs/marker_freetacman_available_32d.json` starts a fresh 100-epoch model at `processing/runs/marker_autoencoders_freetacman_available_32d_v1`. It copies the 16D run's exact frozen episode snapshot and uses the same seed, sampled frame budget, validation frames, architecture width, optimizer and schedule. Only the sensor-code dimension changes. It has an independent process/lock, log (`FREETACMAN_32D_TRAINING.log`) and teacher registry, so it can run alongside the 16D model without replacing that teacher. Both models appear in the reconstruction viewer once their first checkpoints exist. Resume the 32D run with `.venv/bin/python scripts/train_freetacman_available_markers.py --config configs/marker_freetacman_available_32d.json` using the same CUDA runtime environment.

```text
Tactile dot displacement + validity + reference geometry
                         ↓
              sensor-specific marker AE encoder
                         ↓
                  small sensor code
                         ↓
              sensor-specific marker AE decoder
                         ↓
                 reconstructed (dx,dy)

After AE training, freeze the teacher:

8 ego frames → frozen full-spatial V-JEPA → attention + MLP → 32D/frame
                                                           ↓
                                             128 → 128 → sensor code
                                                           ↑
                                      MSE against the frozen AE encoder's code
```

The frozen marker decoder is used to evaluate and display reconstruction. Joint training aligns the predicted sensor code with the frozen tactile encoder's output. It does not optimize RGB reconstruction for these four sources. The shared visual latent remains **32D per image** and is independent of the smaller sensor AE code.

## Small latent search

| Sensor source | Candidate sensor-code dimensions |
|---|---|
| FTP UniT | **4, 8** |
| FTP VLA-Touch | **4, 8** |
| FTP UniT Bimanual | **8, 16** |
| FreeTacMan | **16 and 32**, separate 100-epoch runs on the same processed-episode snapshot |

The first three sources used two candidates each and one seed (45). Each candidate trained for 30 screening epochs; the better validation candidate continued to 100 total epochs. FreeTacMan uses the two separately requested runs described above, without a screening stage. Selection uses held-out source episodes and never MotionTrans. Trained candidates remain selectable for reconstruction comparison.

Training retains the previous AE frame budget: stride 2, up to 100,000 training frames per source, all validation frames at that stride. Every selected frame retains all detected markers at native subpixel precision. There is no point subsampling, rasterization, resizing, optical target reconstruction, or video feature cache in marker AE training.

Plan: `configs/marker_autoencoders_v1.json`.
Controller: `python -m tactilelatent.train_marker_autoencoders`.
Run directory: `processing/runs/marker_autoencoders_v1`.
Continuous log: `MARKER_AUTOENCODER_TRAINING.log`.

The controller starts sources whose extraction is complete, then waits for FreeTacMan's remaining episodes before loading that source. It is resumable: completed pinned teachers are verified and skipped; interrupted trials reload model, optimizer, scheduler, RNG, and loader RNG state. The cosine schedule always has a 100-epoch horizon, so extending the winner does not restart its learning-rate schedule.

## Autoencoder architecture and masks

`MarkerAutoencoder` is a point-set model with 128-wide layers and eight learned attention queries. Each point encoder receives signed displacement, normalized static reference coordinates, its validity flag, and a surface embedding. Attention compresses the observed point set; an MLP produces the small sensor code. Permuting marker order leaves the code unchanged, up to floating-point summation differences.

The decoder is a coordinate-conditioned MLP that receives **only the sensor code plus reference coordinates and surface IDs**. No current displacement or encoder point features bypass the bottleneck. Variable marker counts therefore do not require inventing a universal dot index or reducing the native data to a coarse grid.

Reconstruction outputs are signed linear values. A sigmoid would be inappropriate because displacement can be negative. Invalid observations and padded points do not contribute to loss or metrics. The encoder distinguishes an unobserved dot from an observed stationary dot. A null attention token makes entirely missing frames numerically safe; unsupported frames are excluded from AE fitting and teacher statistics.

Displacement is represented in each surface's marker-spacing units and scaled by a fixed training-source RMS. The loss is masked Smooth L1 with equal supported surface/frame weighting. Reported MAE and endpoint errors are converted back to native pixels. Training uses AdamW, initial learning rate 0.001, minimum 0.00001, weight decay 0.0001, batch size 128, and no latent sampling or KL.

## Frozen teachers and joint integration

Finalists are copied to immutable hash-named teacher files. `teachers.json` records each chosen dimension, checkpoint hash, source-target snapshot, best epoch, validation errors, and training-only code mean/std. The registry is marked complete only after all four teachers finish.

The active joint configuration is `configs/joint_vjepa21_spatial_marker_codes_32d.json`. The adapter remains **32 → 128 → 128 → selected sensor-code dimension**. New model type: `joint_vjepa21_spatial_marker_codes_v1`.

The requested code cache is `processing/sensor_code_cache_markers_v1`, built by `python -u -m tactilelatent.marker_code_cache`. Progress is in its `status.json`; the continuous log is `MARKER_CODE_CACHING.log`. The pinned marker registry is `processing/joint_alignment/marker_teachers_v1.json`: **FreeTacMan 32D, UniT 8D, UniT Bimanual 16D, VLA-Touch 8D**. FreeTacMan's 32D model completed 100 epochs with best validation MAE **0.297541 px** at epoch 96, versus **0.358711 px** for the 16D model. Both remain available in the viewer.

**Cache completed:** 1,675,765 frames in 86.2 seconds. New float32 code/mask arrays occupy 203,596,565 bytes (194.16 MiB), plus metadata. Every marker source passed fresh-code verification. `training_input_check.json` confirms that the real training loader supplies the correct cached dimensions for all nine AE sources and direct targets for both numeric sources, with no tactile encoder invocation and no MotionTrans inclusion. The targeted marker and viewer suite passed 36 tests. The full GPU training preflight is still required before a joint run.

Caching encodes **every chronological frame**, including the FreeTacMan episodes processed after the AE fitting snapshot was frozen. It uses the fixed pretrained AE; it does not refit that AE on those later episodes. The cache preserves validity masks, native signed displacement, all markers, original episode splits, and VLA's chronological-to-stored mapping. Raw codes are float32; normalization statistics use valid **training frames only**. Both code and mask arrays have hashes, and fresh-encoding checks cover training/validation examples plus the repaired FreeTacMan startup. Per-episode progress allows interrupted builds to resume.

The five unchanged native sensor-code caches are verified and linked into this new root. The four old optical image-code directories are never linked or accepted for marker supervision. The two numeric sources still use direct numeric targets; MotionTrans stays excluded. No V-JEPA/ego-image features are cached.

The trainer now reads cached standardized codes and bypasses frozen AE encoding during supervision. Native targets and frozen decoders remain available for reconstruction validation. The user subsequently requested joint training: the full GPU preflight passed on all 11 sources and the fresh deterministic 32D, 20-epoch run was launched on October 16. See `VJEPA_MARKER_CODE_RUN.md` and `VJEPA_MARKER_TRAINING.log`.

`MixedMarkerTeachers` accepts cached targets for all eight chronological frames and exposes frozen decoders for displacement reconstruction. Its uncached fallback runs marker encoders in frozen FP32/evaluation mode. The four old optical RGB AE teachers are replaced. The five other AEs and the two direct numeric heads retain their supervision. MotionTrans stays excluded from representation training and teacher statistics.

The joint trainer fails if target extraction or any teacher is incomplete, if provenance changes, or if an episode is missing. All prerequisites have completed and the user-requested joint run is active. Launch/resume commands:

```bash
bash scripts/run_vjepa_markers.sh --preflight-only
bash scripts/run_vjepa_markers.sh
# Resume only after stopping, with the same configuration:
bash scripts/run_vjepa_markers.sh --resume
```

The default script now selects frozen marker-code supervision. The older direct-displacement prototype is available only by explicitly passing its older configuration; it is not the default.

## Repaired FreeTacMan startup

`Slot/Slot_164` has an actual corrupted startup on its first tactile camera: white/black and partially decoded images precede stable sensor images. The second tactile surface remains usable.

The repair requires five consecutive trackable frames with stable marker counts and at least 90% correspondence support. It takes the first frame of the first qualifying window. For this episode, reference frames are **[45, 0]** for the two surfaces. The first surface's observations before frame 45 remain invalid; the other surface's observations are retained. No chronological frames are shifted or replaced with stationary labels.

The repair is recorded in that episode's metadata with its own script hash. A detached finalizer reconciles the original extraction rejection with the verified repaired artifact when extraction ends. The viewer reads each surface's actual reference frame instead of assuming frame zero. These references are still not verified unloaded states, and the vectors are not calibrated forces.

## Viewer and verification

The **Sensor reconstruction** page discovers the marker AE checkpoints and prefers them for these four sources. Both code-size candidates are available as they train. It displays native measured/predicted vector overlays and errors in pixels. The **Ego → tactile prediction** page supports the new joint model type and frozen marker decoders once real joint checkpoints exist. Historical RGB models keep their original interfaces and remain available for comparison.

Tests cover permutation invariance, missing-point and NaN handling, signed output, bottleneck gradients, frozen checkpoint round trips, fitting a known smooth deformation, code supervision into the visual encoder, decoder reconstruction, SAM replay, and frozen teacher gradients. The existing native loader and viewer tests are also run. Real trained checkpoints are checked through the hosted reconstruction API; evidence is saved in `processing/runs/marker_autoencoders_v1/viewer_http_check.json`.

Compare finalists using native-pixel reconstruction, active-motion error, endpoint error, tracking coverage, and the no-displacement baseline. A small AE code is accepted on measured reconstruction quality; its dimension alone does not establish better invariance or transfer.
