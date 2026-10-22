> Historical record, relocated on October 21, 2026. Original path: `PROJECT_HISTORY_AND_RESULTS.md`.
> Results, proposals, and operational status below belong to their recorded experiment/date;
> use the [maintained project state](../../project_state.md) for the audited active workflow.

# TactileLatent: goals, implementation history, and results

**Snapshot:** October 12, 2026, 11:12 PM EDT (America/New_York). Training status is a point-in-time observation; consult linked live status files for later progress.

**V-JEPA training and MotionTrans evaluation completed — October 15, 2026:** Both 32D cached training arms completed epoch 30 in 1 h 56 min. Their best source validation checkpoints are both epoch 28: deterministic 0.741732 and stochastic 0.740994, versus DINO best 0.655483. MotionTrans evaluation used those frozen epoch-28 encoders, cached full-video inputs, the same three-hidden-layer 512-unit head, fit-only z-scoring, three tune-selected LRs, and three head-fit seeds as the latest DINO comparison. Mean MAE ± head-seed SD: deterministic 0.040215 ± 0.000534, stochastic 0.040216 ± 0.000223. Both improve about 8.1% over the matched DINO epoch-7 latent (0.043769); paired episode-bootstrap difference intervals exclude zero. The V-JEPA variants are tied (paired difference CI includes zero). Frozen temporal DINO features remain stronger at 0.031462. This demonstrates better transfer despite worse source fit, relative to the trained 32D DINO reference. No classifier or predictive-uncertainty calibration was evaluated. See [full transfer report](../../../processing/evaluations/motiontrans_vjepa21_best_v1/analysis.md).

**V-JEPA cached training resumed — October 15, 2026:** The full clip cache finished in 2 h 43 min (349,081 clips; approximately 7.96 GB of float32 features). Both original 32D task models resumed from the last saved epoch-0 state toward epoch 30, with their original optimizer/scheduler/RNG state, deterministic zero KL, and stochastic Gaussian sampling plus KL ramp. Training and native reconstruction validation now read cached clips, and no V-JEPA backbone is loaded. All eleven source partitions were checked against cache checksums, camera roles, exact frame windows and teacher-code provenance; MotionTrans stays held out. Thirteen tests passed, and 352 real cached training batches plus all-source native validation passed. The short benchmark measured about 4.1 training minutes/epoch before validation. See [experiment](VJEPA_32D_EXPERIMENT.md) and [live status](../../../processing/runs/joint_vjepa21_pair_32d_v1/status.json).

**V-JEPA training stopped; full-clip cache launched — October 15, 2026:** At the user’s request, both V-JEPA task models and their data-loader workers were stopped during epoch 1. Epoch 0 remains the latest saved checkpoint; partial epoch-1 updates were not saved. A detached, resumable job now extracts float32, spatially pooled video features for all 349,081 current-loader clips across all twelve datasets and splits (648,017 camera clips, approximately 7.96 GB feature payload). Whole eight-frame clips are encoded jointly; cameras remain separate. MotionTrans is cached in its own partitions without fitting statistics. Ten tests passed and live-vs-cache features/latent means matched exactly on one-, two-, and three-camera clips. Training will not restart automatically. See [cache details](VJEPA_FEATURE_CACHE.md) and historical cache status `processing/vjepa21_clip_cache_v1/status.json` (not retained locally at the October 21, 2026 audit).

**V-JEPA pair launched — October 15, 2026:** Two fresh 32D task models (deterministic and Gaussian) now train for 30 epochs with a shared frozen V-JEPA 2.1 ViT-B/16 backbone at 384 pixels. The previous DINO/spatial-temporal stack is removed for these runs. All 11 sources retain native tactile targets and cached teacher-code supervision; MotionTrans stays held out. Matched initial task weights, per-epoch snapshots, paired resume state, and viewer integration are implemented. All 34 targeted tests and the real-data/native-decoder preflight passed. Initial mixed-source throughput is about 50 ms/batch for both models together (roughly 26 minutes per epoch before validation); provisional total budget 14–16 hours. Prior representation runs remain stopped. See [experiment and reproducibility details](VJEPA_32D_EXPERIMENT.md) and [live status](../../../processing/runs/joint_vjepa21_pair_32d_v1/status.json).

**Completed October 15, 2026:** All 72 tuning fits and 24 refits finished in 34.8 minutes, including frozen extraction. Mean MotionTrans MAE over three head seeds: temporal DINO 0.031462, single-frame DINO 0.033007, reconstructed epoch-0 latent 0.046751, epoch-7 latent 0.043769, epoch-100 latent 0.048189, reconstructed epoch-0 wide trunk 0.035368, epoch-7 wide trunk 0.035325, epoch-100 wide trunk 0.035938. Epoch-100 wide features are not clearly worse than epoch-7 wide features (paired interval includes zero), while the 32D latent deteriorates clearly. This points toward a bottleneck/projection issue under this readout; head capacity/input dimension remains a comparison limitation. All representation training remains stopped. See [full results](../../../processing/evaluations/motiontrans_representation_screen_v1/analysis.md).

**Continuation stopped at epoch 22 — October 15, 2026:** All four source-training arms completed epoch 22 and its transfer evaluation; no epoch-27 continuation is scheduled. MotionTrans MAE: control 0.047140, source stopping 0.046555, KL-off retention 0.046653, combined 0.044933. Combined preserves the epoch-7 level best, but its paired 95% difference interval against control narrowly includes zero. KL-off nominal 90% coverage remains only 4.72%/7.28%. The follow-up compares frozen single-frame/temporal DINO and reconstructed epoch-0, epoch-7 and epoch-100 representations using fit-only z-scoring, three tune-selected LRs and three head seeds. Initialization reconstruction exactly reproduces the recorded 16D hash, while an independent original 32D hash is unavailable. See [protocol](FROZEN_REPRESENTATION_SCREEN.md).

**32D epoch-7 continuation study — October 14, 2026, 2:08 PM EDT:** The saved nineteen-checkpoint screen finds strong negative post-7 correlations between source supervision loss and MotionTrans transfer MAE for EgoTouch (−0.961), VLA Touch (−0.953), and ViTaMIn (−0.945), while FreeTacMan, Unit Bimanual, and OpenTouch worsen alongside transfer. These are time-confounded associations, not causal source attributions. Four matched arms are queued from original Gaussian 32D epoch-7 weights through epoch 27: control, source-validation gates, KL-off with a frozen epoch-7 mean anchor, and both. The missing parent optimizer state is handled by identical fresh optimizers across arms. Control and source stopping started epoch 8; the other two are queued. Probes use a three-hidden-layer head every five additional epochs. The new epoch-7 baseline is MAE 0.044773 with nominal 90% coverage 32.64%, so predictive uncertainty is still uncalibrated. Estimated total duration is 30–36 hours from paired GPU benchmarks. The stopped deterministic run remains stopped. See [full experiment](CONTINUATION_TRANSFER_EXPERIMENT.md), [source screen](../../../processing/continuation_32d_v1/source_screen/analysis.md), and [live queue status](../../../processing/runs/continuation_32d_v1/status.json).

**Deterministic training stopped — October 14, 2026, 1:50 PM EDT:** At the user's request, the deterministic control's process group was terminated and verified to have no remaining live workers. Epoch 5 is the latest saved checkpoint, with optimizer/RNG state and a viewer snapshot preserved. Partial epoch-6 progress was not saved. See [stop metadata](../../../processing/runs/joint_16d_deterministic_control_v1/stop_status.json).

**Fresh deterministic control — October 14, 2026, 11:35 AM EDT:** At the user's request, the previous study controller and trajectory worker were stopped, preserving saved checkpoints and completed evaluations. A fresh control was launched for 20 epochs with a deterministic 16D latent, no learned variance branch, no latent sampling, and zero KL penalty. The shared encoder and adapters are randomly initialized at epoch 0; frozen DINO and the existing sensor teachers/caches are retained. Relational, trajectory, and meta objectives are disabled. See [run documentation](DETERMINISTIC_CONTROL.md), [configuration](../../../legacy/v1/configs/joint_16d_deterministic_control.json), and [live status](../../../processing/runs/joint_16d_deterministic_control_v1/status.json). This run supplies no latent-derived uncertainty estimate.

**Deeper transfer heads — October 14, 2026:** At the user's request, all four completed epoch-20 models were evaluated with three and four hidden layers of 512 units, reusing their full-split MotionTrans caches and existing fitting settings. MAE (three/four hidden layers): control 0.054314/0.054404; relational 0.055284/0.055276; meta 0.054798/0.054655; matched meta control 0.054792/0.054965. Three hidden layers improve over two by 2.33–3.39% across the models; all four paired episode-bootstrap intervals for that increase exclude zero. Four versus three hidden layers has no clear additional benefit. Meta still shows no clear advantage over its matched control. Nominal 90% coverage remains approximately 34–37%. See [report](../../../processing/runs/invariance_16d_v1/evaluations/epoch_020_deeper_heads/analysis.md), [all-depth results](../../../processing/runs/invariance_16d_v1/evaluations/epoch_020_deeper_heads/all_depths.json), and [figure](../../../processing/runs/invariance_16d_v1/evaluations/epoch_020_deeper_heads/head_depth_comparison.png). Trajectory and its controller remain paused.

**Epoch-20 transfer evaluation — October 14, 2026:** Control, relational, meta-learning, and matched meta control completed training. The user paused trajectory matching at saved epoch 17, preserving its in-progress epoch-18 state, and requested evaluation of the others. Full-split MotionTrans MAE (shallow/deep): control 0.063107/0.056219; relational 0.062746/0.056600; meta 0.062816/0.056640; matched meta control 0.062878/0.056484. All four matched paired-episode comparison intervals include zero; neither new objective demonstrates a clear full-budget transfer gain. Nominal 90% coverage remains 28–40%. See [analysis](../../../processing/runs/invariance_16d_v1/evaluations/epoch_020/analysis.md) and [summary](../../../processing/runs/invariance_16d_v1/evaluations/epoch_020/summary.json). Trajectory and the controller remain paused, with no automatic resume.

**Training resumed — October 13, 2026, 8:17 PM EDT:** At the user's explicit request, the paused control and relational workers and queue controller were resumed with SIGCONT, preserving partial-epoch progress and optimizer/RNG state. Remaining conditions continue to follow the original two-worker queue. See [resume metadata](../../../processing/runs/invariance_16d_v1/resume_status.json) and [live status](../../../processing/runs/invariance_16d_v1/status.json).

**Paused transfer evaluation — October 13, 2026:** The user requested a training pause and quick held-out evaluation. Training workers and the queue controller are suspended with partial-epoch state retained. The matched full-split MotionTrans epoch-15 evaluation completed in about 5 minutes using shared frozen DINO extraction. Control versus relational MAE: shallow 0.062466 versus 0.062809 (+0.55%); deep 0.055551 versus 0.056115 (+1.02%). Both paired episode-bootstrap 95% intervals include zero. Nominal 90% coverage remains approximately 30–40%. See [the evaluation report](../../../processing/runs/invariance_16d_v1/evaluations/epoch_015/analysis.md) and [summary](../../../processing/runs/invariance_16d_v1/evaluations/epoch_015/summary.json). Training remains paused; no automatic resume is scheduled.

**Training launch — October 13, 2026:** At the user's request, all five implemented invariance conditions were submitted for epochs 11–20 from the same 16D non-FGW epoch-10 parent, using seed 45. The queue runs two workers concurrently: control and relational first, followed by meta, matched meta control, and trajectory. See [live status](../../../processing/runs/invariance_16d_v1/status.json). No new completed-epoch or transfer results are implied by this launch.

**Implementation update — October 13, 2026:** Optional relational distillation, frozen-tactile trajectory GW matching, and rotating-source meta-learning have been implemented and tested on the existing 16D architecture. The user removed the proposed 4D static episode context. Only bounded verification updates have run; there are no new long-run or transfer results. See [implementation, commands, and floor-audit limitations](INVARIANCE_TRAINING.md).

**Later update — October 12, 2026, 11:48 PM EDT:** At the user's request, both 16D training runs were stopped. The non-FGW run saved through epoch 16 and the visual-FGW run through epoch 14; checkpoints and optimizer/RNG states are preserved. The historical snapshot below is unchanged. See the [stop record](../../../processing/runs/joint_16d_pair_v1/stop_status.json) and the [shared/private, relational distillation, trajectory matching, and meta-learning implementation plan](SHARED_PRIVATE_META_LEARNING_PLAN.md). These new methods are planned, not implemented or trained.

This document consolidates the project discussion, local implementation, training logs, audits, and saved evaluation artifacts. It distinguishes completed experiments, superseded experiments, active work, and proposed ideas. Metrics below were read from saved artifacts; this documentation task did not launch or repeat training or evaluation. Tables link to the underlying files for complete metrics, configurations, and examples.

## 1. Main goal and criteria for success

The main goal is to learn a visual representation of tactile interaction that transfers across different sensors, robots, hands, environments, and datasets. An ego-view image sequence should produce a compact shared representation from which a small sensor-specific prediction head can recover tactile measurements. The eventual model should also report predictive uncertainty whose intervals correctly reflect observed errors.

The project has three separate scientific objectives:

1. **Useful tactile prediction:** predict changing tactile measurements from scene images, including contact-rich regions and within-episode dynamics.
2. **Generalization and sensor invariance:** preserve interaction information that supports learning a new sensor/task with a frozen visual encoder. Dataset identity should not be mistaken for shared physics, and suppressing identity must not destroy useful information.
3. **Calibrated uncertainty:** nominal prediction intervals should achieve their advertised coverage, including difficult contacts, exact-zero pressure, unfamiliar episodes, and relevant target channels.

Reconstructing a sensor well, reducing transport distance, increasing effective rank, lowering dataset-classifier accuracy, or producing visually mixed clusters does not independently demonstrate these goals. The principal current benchmark is supervised transfer to **MotionTrans**, with its encoder frozen and a newly fitted six-output head. This is adaptation using labeled MotionTrans training episodes, not zero-shot prediction.

MotionTrans is excluded from the recent shared representation training, FGW banks, barycenters, and source-code statistics. Its validation results have already influenced development, so it is a development benchmark rather than an untouched final test set. Earlier VLA Touch holdout experiments used a different source set and data representation.

## 2. Data and invariants

Twelve datasets are indexed. Eleven are sources for the current representation experiments; MotionTrans is the transfer target. The following table comes from the canonical schema-2 manifest and pinned teacher registry. Episode counts are manifest counts; usable clips additionally depend on clip length and valid targets.

| Dataset | Train episodes | Validation episodes | Native target shape | Pinned AE code |
| --- | --- | --- | --- | --- |
| EgoTouch | 1458 | 174 | 2×21×21 | 256 |
| EgoTactile | 692 | 75 | 1×1×256 | 16 |
| OpenTouch | 2552 | 406 | 1×16×16 | 32 |
| OpenGraph | 68 | 4 | 2×1×80 | 16 |
| FreeTacMan | 5439 | 186 | 6×480×640 | 256 |
| Unit | 155 | 15 | 3×240×320 | 256 |
| Unit Bimanual | 354 | 26 | 6×240×320 | 256 |
| VLA Touch | 338 | 42 | 3×240×320 | 256 |
| ViTaMIn | 680 | 67 | 6×224×224 | 256 |
| RDP | 134 | 6 | 1 value | Direct head; no AE |
| RDP Bimanual | 47 | 3 | 2 values | Direct head; no AE |
| MotionTrans (held out) | 1601 | 167 | 6 values | Direct head; no AE |

The eleven source training splits contain **11,917 episodes and 4,361,637 frames**, providing 287,814 potential nonoverlapping 15-frame-span windows before target-validity filtering. MotionTrans separately contains 1,601 training episodes and 167 validation episodes, yielding 18,613 and 1,972 evaluated clips.

Project constraints established during development:

- Preserve native tactile image dimensions and sensor channel order. No unapproved tactile resizing or cropping.
- Scene RGB is preprocessed to 224×224 for the frozen DINO backbone. This is separate from tactile target resolution.
- Preserve chronological scene/target pairing and audited camera roles.
- Keep the **common visual latent dimension** separate from the **sensor autoencoder code dimension**. A 16D or 32D shared latent can predict a 256D sensor code through an MLP.
- Use direct heads for one/two-value force outputs and the six-value MotionTrans transfer target; do not train unnecessary numeric autoencoders.
- Preserve historical runs, provenance, teacher identities, and train-only normalization statistics.
- Pressure/image outputs currently use sigmoid, following the user's explicit request to return to sigmoid.
- The current experiments use one representation-training seed and two conditions; no additional seed sweep or screening is active.

Force and torque errors are in the project's normalized reading space. MotionTrans uses `0.5 + 0.5*tanh(raw/1000)`; RDP uses `0.5 + 0.5*tanh(raw/30)`. Raw zero maps to 0.5 for these signed transformations. These scales do not establish calibrated SI units.

Evidence: [canonical manifest](../../../processing/joint_alignment/manifest.json), [data audit](../../../processing/joint_alignment/audit.json), [teacher registry](../../../processing/joint_alignment/teachers.json), [joint alignment guide](JOINT_SENSOR_ALIGNMENT.md).

## 3. Architecture evolution

### 3.1 Initial visual tactile models

Early work developed EgoTouch models using frozen DINO features, camera fusion, spatial/temporal processing, and pressure prediction. Saved variants include DINOv2, DINOv3 with a 128D latent, probabilistic 128D models, image-only VAE/CVAE variants, and a 32D pose-teacher CVAE. These established the prediction pipeline and exposed weaknesses in uncertainty and sparse-pressure reconstruction.

The established DINOv3-B/16 backbone supplies 196 patch vectors of width 768 per 224×224 image. Learned camera weighting combines available scene views; three divided space-time blocks process the eight-frame clip. Gaussian heads produce a mean and log variance per frame. Tactile targets supervise learning but are not visual-encoder inputs. Historical pose-teacher variants are separately identified in their artifacts.

Camera fusion weights whole views and combines corresponding image-grid positions; it does not establish geometric contact correspondences across cameras. Temporal attention is bidirectional. Intermediate clip positions may use later clip frames; the final position has a past-and-present interpretation.

### 3.2 Direct heterogeneous joint training

The first joint architecture used one shared visual encoder and sensor-specific direct decoders. It was extended to heterogeneous pressure grids, taxel vectors, optical tactile images, gripper readings, and torque. Ordinary training, online gradient-reversal adversarial training, alternating adversarial training, and SAM were investigated. Historical optical heads used reduced targets and must be interpreted under that older protocol.

### 3.3 Specialized tactile autoencoders

The user proposed removing the burden of learning each reconstruction decoder during visual training:

```text
Measured tactile output → sensor AE encoder → sensor AE code → sensor AE decoder

Ego images → shared visual encoder → common latent → trainable adapter
                                                    ↓
                                              sensor AE code
                                                    ↓
                                              frozen decoder
```

Sensor autoencoder codes need not have the same width as the common latent. The intervening nonlinear adapter translates a shared interaction representation into the coordinate system needed by each frozen sensor decoder.

### 3.4 Current cached-code supervision

The next change uses the **frozen tactile AE encoder** as a teacher. Tactile targets are encoded once, cached, standardized using training frames, and used to supervise the sensor adapter directly:

```text
Measured tactile output → frozen AE encoder → cached sensor code target

Ego images → frozen DINO → shared spatial/temporal encoder
                                      ↓
                              common Gaussian latent
                                      ↓
                             sensor-specific adapter
                                      ↓
                            predicted sensor AE code
```

Training minimizes standardized sensor-code MSE plus Gaussian KL; numeric sources retain direct prediction losses. The frozen decoder is used for native reconstruction validation and viewer inference, rather than every training update. Code accuracy and native reconstruction accuracy are both logged because the frozen decoder is nonlinear and imperfect.

The current image/taxel adapters have two 128-unit hidden layers with GELU: `D → 128 → 128 → sensor_code_dim`. These widths were inherited to keep the comparison focused on common latent width and FGW; they were not demonstrated optimal. Numeric heads remain simple sigmoid projections.

For the 16D pair, code256 adapters have 51,712 parameters each; OpenTouch code32 has 22,816; EgoTactile/OpenGraph code16 have 20,752. The direct one/two-output numeric heads have 17/34 parameters. Both 16D models have 44,266,260 model parameters, excluding the separate frozen DINO backbone and pinned AE teachers.

Evidence: [architecture guide](ARCHITECTURE_GUIDE.md), [joint sensor model](../../../legacy/v1/src/tactilelatent/alignment_model.py), [alignment trainer](../../../legacy/v1/src/tactilelatent/train_alignment.py), [16D configuration](../../../processing/runs/joint_16d_pair_v1/configs/no_fgw.json).

## 4. Data problems discovered and corrected

### 4.1 Unapproved optical resizing

The first standalone optical autoencoder loader resized tactile targets to 32×32. This concealed fine sensor changes and violated the intended native-resolution experiment. Those optical models were superseded and disabled/omitted from runnable native viewer choices. Their files remain as provenance. Replacement models reconstruct all stored pixels: FreeTacMan 480×640, Unit/Unit Bimanual/VLA Touch 240×320, and ViTaMIn 224×224 per sensor image.

Internal encoder downsampling is a learned architectural operation and is separate from externally resizing targets. A model can reduce internal feature maps while still reconstructing and evaluating the complete native image.

### 4.2 FreeTacMan camera roles

The scene and Surface1 streams were swapped in part of the data. A full 5,625-episode camera-role audit established episode-specific roles. `SprinkleSesame_1` through `_53` use camera1 as ego and camera3 as Surface1; Surface2 stays camera2. Other episodes retain ego camera3 and tactile cameras1/2. The viewer, corrected manifests, and canonical alignment pipeline use these assignments.

### 4.3 VLA Touch chronology and apparently unchanged tactile images

An exact native-pixel audit matched all 28,272 stored tactile frames to the original numbered `gel_N.jpg` files. **379 of 380 episodes had scrambled tactile frame order.** The upstream converter's sorting helper recognized `rgb_N.jpg` but not `gel_N.jpg`, leaving GelSight files in arbitrary listing order.

The project records a verified chronological-to-stored lookup. The corrected joint manifest, code builder, and viewer use that lookup. Historical VLA visual-to-tactile results predate this correction and cannot substantiate correct chronological transfer.

One episode genuinely contains one repeated tactile image across 46 frames; reordering cannot recover absent information. Correctly ordered cup marker movement correlated 0.997 with the authors' displacement labels, versus −0.161 in stored order. Small marker motion on a large constant background explains why some true changes remain subtle.

The audit verifies released/raw frame indexing and exact tactile pixels. It does not independently establish hardware synchronization or calibrated force units. Single-frame AE reconstruction is not invalidated by scrambled ordering, but its stride-two sampling subset differs between stored and chronological order.

### 4.4 OpenTouch storage and anatomical display

The stored 16×16 readings were faithful to the released HDF5 values: 8,874 source/loader comparisons were exact, and both historical pressure caches matched the source. The official hand layout uses **169 mapped channels** and excludes 87 unique nodes from the 256-position array.

Both viewer pages now use the official right-hand positions and mapped-channel mask. Colors display fixed-scale `1 - raw/3072` relative response; hover retains the underlying normalized reading. Generic pressure-contact metrics are hidden for this sensor because the old threshold labeled 99.44% of readings as contact and was not a useful contact definition.

**Training remains on the original 256-channel representation.** The existing OpenTouch teacher and cached codes were not silently retrained or reinterpreted. A mapped-channel training revision requires a new teacher/cache with explicit provenance. The historical AE contact F1 in the result tables is therefore a saved metric, not validated contact segmentation.

Evidence: [optical diagnosis](../../../processing/audits/OPTICAL_SIGNAL_DIAGNOSIS.md), [VLA chronology audit](../../../processing/audits/vla_touch_native_dynamics/REPORT.md), [OpenTouch audit](../../../processing/audits/opentouch_format/REPORT.md), [camera-role configuration](../../../legacy/v1/configs/freetacman_camera_roles.json).

## 5. Tactile autoencoder training and results

### 5.1 EgoTouch reconstruction ablations

The initial small/medium/large experiment was expanded to sensor-code widths 16/32/64/128. All twelve matched grid models completed 30 epochs. The best was medium/code128: validation MAE **0.016361**, contact MAE **0.04265**, contact F1 **0.8554**. Medium/code64 gave MAE **0.01809** with about half the parameters. The larger preset was not uniformly better; different spatial compression meant preset names did not form a monotonic parameter ranking.

The initial high-learning-rate small/code32 attempt converged to a near-zero predictor: MAE approximately 0.05530 and contact F1 0. Its saved run is a failed reconstruction attempt, not evidence that a 32D sensor code is intrinsically inadequate. Most successful grid models selected the final epoch, motivating longer training.

Output handling evolved through a two-output hurdle proposal and single-output clipping before returning to sigmoid at the user's request. The hurdle approach could emit exact zero but introduced an unwanted contact branch. The clipping approach trained raw linear predictions and clipped for evaluation, preserving corrective gradients outside bounds. Both were abandoned in favor of the requested single sigmoid output.

The final EgoTouch v4 search completed 28 runs: 24 screens, six extended configurations within those screens, and four additional seed repetitions. Screening used 60 epochs, finalists 200 total epochs. Sensor codes were 64/128/256; architecture, learning rate, batch size, weight decay, contact weighting, and MAE weighting were explored. Selection used `(overall MAE + contact MAE + high-pressure MAE)/3`, with 200,000 training frames per epoch and all 80,987 stride-two validation frames.

The selected **wide64/code256** model (`[64,128]` channels) completed 200 epochs and selected epoch 198:

- Overall MAE: **0.00354457**.
- Contact MAE: **0.00784668**.
- High-pressure MAE: **0.01138487**.
- Contact F1: **0.969707**.
- Balanced selection score: **0.00759204**.
- Total parameters: **2,515,330**.

Across seeds 45/46/47, wide64/code256's balanced score mean was 0.00772211, sample SD 0.00022138. Wide48/code256's mean was 0.00908339, SD 0.00038902. These are validation-selected reconstruction results, not untouched-test results or evidence of calibrated uncertainty. The grid and v4 use different validation budgets/selection criteria, so their improvement is not a controlled attribution to one changed factor.

Sigmoid still has negligible exact-zero recall for the selected model (approximately 0.000736% in the saved report). Returning to sigmoid resolved the output-mode preference, **not** the exact-zero representation/calibration problem.

### 5.2 Native sensor-specific decoders

The other eight reconstruction sources completed native-resolution training. The final reduced search tried exactly two candidates per source for 30 screening epochs and promoted one overall winner to 100 total epochs. Extra seeds and unnecessary numeric AEs were removed. Wider optical decoders were selected under sensor-specific names such as `large_code256`; the `wide48_*`/`wide64_*` names belong to EgoTouch.

| Sensor | Selected model | Code | Epochs trained | Best epoch | MAE | Contact MAE¹ | Contact F1¹ | PSNR dB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EgoTouch | wide64_code256 | 256 | 200 | 198 | 0.003545 | 0.007847 | 0.9697 | — |
| EgoTactile | medium_code16 | 16 | 100 | 93 | 0.002752 | 0.039447 | 0.7992 | — |
| OpenTouch | large_code32 | 32 | 100 | 99 | 0.015582 | 0.015001 | 0.9976 | — |
| OpenGraph | compact128_code16 | 16 | 100 | 100 | 0.013336 | 0.064702 | 0.7619 | — |
| FreeTacMan | large_code256 | 256 | 100 | 100 | 0.013905 | — | — | 30.64 |
| Unit | large_code256 | 256 | 100 | 100 | 0.005484 | — | — | 40.95 |
| Unit Bimanual | large_code256 | 256 | 100 | 97 | 0.007324 | — | — | 36.98 |
| VLA Touch | large_code256 | 256 | 100 | 99 | 0.007331 | — | — | 38.87 |
| ViTaMIn | large_code256 | 256 | 100 | 95 | 0.005864 | — | — | 40.50 |

¹ Pressure/glove metrics use the saved source convention. OpenTouch contact statistics are not validated contact segmentation; see Section 4.4.

MAE is source-normalized and must not be ranked across different sensor types as if it measured a common physical quantity. Optical PSNR measures image reconstruction, not contact/force calibration. Many finalists select epochs near 100, so a global optimum or fully saturated training budget is not established.

The selected teachers are pinned independently of changing viewer/search selections. Exact architecture/code provenance is in the teacher registry. These AEs reconstruct measured tactile data; their scores do not measure prediction from ego images.

Full grid and search tables, including screened candidates, are in Appendix A. Evidence: [initial EgoTouch analysis](../../../processing/runs/egotouch_ae_sensor_code_grid_v1/ANALYSIS.md), [v4 search](../../../processing/runs/egotouch_ae_search_v4/comparison.csv), [native search](../../../processing/runs/sensor_autoencoders_native_v2/comparison.csv), [selected native models](../../../processing/runs/sensor_autoencoders_native_v2/best_by_source.json).

## 6. Caching and training efficiency

All nine selected AE teachers have completed target-code caches. The cache represents **4,893,180 frames** and approximately **3.99 GiB of float32 codes**, plus validity flags and metadata. It includes indexed splits for reconstruction/inference use, but only training frames contribute to statistics and only designated source training splits enter representation training.

Each entry refers to an exact source, episode, and chronological frame. Teacher, loader, source snapshot, manifest, and builder hashes protect compatibility. Cache generation runs frozen tactile encoders; it does not run DINO. Native images remain in their compressed source media, avoiding a multi-terabyte decoded image cache.

Cached supervision avoids repeated tactile AE encoding and frozen-decoder passes during visual updates. **DINO scene features are still computed online** in the current runs. Precomputing DINO features was discussed as a possible speed optimization, but a completed all-source DINO feature cache has not been established by this snapshot. No measured percentage of current epoch time attributable to DINO was recovered for this report.

The previous single-run joint rate was approximately 24.6–26 minutes/epoch. Early measurements of the concurrent 16D pair were approximately 42.4 minutes/epoch for the control and 47.4 for FGW. These rates vary with contention and evaluation jobs; concurrency shares one RTX 5090 and does not double throughput. The 16D bottleneck leaves the expensive 768-wide spatial/temporal layers intact.

The current visual FGW refresh typically takes approximately 77 seconds, mostly extracting bank features; the saved epoch-10 transport/mixing computation took approximately 1.29 seconds. A code-only pilot benchmark processed eleven 256-clip teacher batches in approximately 0.55 seconds. This distinguishes transport cost from video/feature extraction cost.

Evidence: [cache directory](../../../processing/sensor_code_cache_v1), [cache implementation](../../../legacy/v1/src/tactilelatent/sensor_code_cache.py), [pilot benchmark](../../../processing/runs/fgw_study_v1/benchmark.json), [16D training guide](TRAINING_16D_PAIR.md).

## 7. Historical transfer, invariance, and SAM results

### 7.1 Original direct-output MotionTrans benchmark

These historical runs used direct source-output training, preceding cached-code supervision and canonical native corrections. They evaluate the same 1,972 MotionTrans validation clips. Head duration is selected using disjoint training episodes, followed by refitting on all MotionTrans training episodes.

| Model | Deterministic MAE | Ensemble MAE | CRPS | Nominal 95% coverage | Evidence |
| --- | --- | --- | --- | --- | --- |
| Training-mean target baseline | 0.086392 | — | — | — | — |
| Ordinary frozen encoder, original head | 0.058843 | 0.061797 | 0.048898 | 66.48% | [artifact](../../../processing/runs/joint_all_holdout_motiontrans_fresh_8k_50ep/evaluations/motiontrans/results.json) |
| Ordinary frozen encoder, larger head | 0.049438 | 0.055319 | 0.040542 | 69.29% | [artifact](../../../processing/runs/joint_all_holdout_motiontrans_fresh_8k_50ep/evaluations/motiontrans_large_decoder/results.json) |
| SAM frozen encoder, original head | 0.058810 | 0.064124 | 0.049648 | 65.20% | [artifact](../../../processing/runs/joint_all_sam_holdout_motiontrans_fresh_8k_50ep/evaluations/motiontrans_final/results.json) |
| Direct MotionTrans specialist | 0.026056 | 0.025744 | 0.023833 | 51.66% | [artifact](../../../processing/runs/motiontrans_direct_vae32_50ep/evaluation/results.json) |

The larger frozen head was `32 → 1024 → 1024 → 512 → 6` (1,611,270 parameters), compared with the original `32 → 512 → 6` (19,974). Its lower error showed that the original probe underused the representation. Direct MotionTrans training remained substantially more accurate. SAM alone produced almost unchanged deterministic MAE while worsening ensemble MAE/CRPS in this comparison. Accuracy and interval coverage did not improve together.

### 7.2 Dataset identity and adversarial experiments

| Encoder | Overall accuracy | Balanced/macro accuracy | MC balanced/macro accuracy | Evidence |
| --- | --- | --- | --- | --- |
| Historical VLA-holdout ordinary encoder | 95.03% | 97.10% | 97.06% | [artifact](../../../processing/evaluations/dataset_identity_probe/results.json) |
| Historical VLA-holdout alternating adversarial | 28.86% | 21.99% | 27.05% | [artifact](../../../processing/runs/joint_all_alternating_fresh_8k_50ep/evaluations/dataset_identity/results.json) |
| Historical MotionTrans-holdout SAM encoder | 92.50% | 95.55% | 95.61% | [artifact](../../../processing/runs/joint_all_sam_holdout_motiontrans_fresh_8k_50ep/evaluations/dataset_identity/results.json) |
| 32D visual FGW, epoch 12 | 97.39% | 98.80% | 98.82% | [artifact](../../../processing/runs/fgw_two_placements_v1/fgw_visual_partial_seed45/evaluations/dataset_identity_epoch_012/results.json) |

High classifier accuracy shows that dataset identity remains accessible; it does not prove that no transferable subspace exists. Chance balanced accuracy is approximately 9.09% for eleven classes. Different source sets, sampling, and historical representations prevent treating every row as a matched comparison.

Historical VLA transfer yielded MAE **0.01032348** with a frozen non-adversarial encoder, versus **0.01031665** for direct VLA training. Alternating adversarial training lowered identity accuracy substantially but increased VLA MAE to **0.03434784**, close to the approximately 0.03718 constant baseline. Its effective rank fell to approximately **1.67** and nominal 95% coverage was only **10.79%**. The online GRL run stopped at 15 completed epochs and is not a completed 50-epoch comparator.

These VLA scores used historical 32×32 targets and the subsequently discovered scrambled chronology. They describe that historical protocol and cannot establish native, correctly synchronized VLA performance. The result remains a warning that erasing source identity can sacrifice prediction; corrected experiments are necessary for current physical claims.

### 7.3 Feeling of Success

Frozen visual latents were evaluated on grasp-success prediction using two real pre-lift RGB snapshots and object-disjoint partitions. The saved baseline/SAM test assignments match: 1,421 trials across sixteen test objects.

| 64-draw prediction | Accuracy | Balanced accuracy | AUROC | Brier | Failure recall | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| Constant training prevalence | 66.64% | 50.00% | 0.5000 | 0.223125 | 0.00% | — |
| Non-SAM | 66.64% | 50.16% | 0.4949 | 0.226962 | 0.63% | [artifact](../../../processing/runs/joint_all_holdout_motiontrans_fresh_8k_50ep/evaluations/feeling_of_success/metrics.json) |
| SAM | 66.08% | 50.68% | 0.5852 | 0.220444 | 4.43% | [artifact](../../../processing/runs/joint_all_sam_holdout_motiontrans_fresh_8k_50ep/evaluations/feeling_of_success/metrics.json) |

The non-SAM model did not establish useful discrimination. SAM improved ranking and Brier score modestly but detected only 21 of 474 failures at the default threshold. This is weak practical failure detection despite an AUROC gain. Statistical variation across test objects was not reported. No subsequent result for the new cached-code/FGW models was found.

## 8. Cached-code 32D training and checkpoint transfer

The native cached-code model trained on all eleven sources with MotionTrans excluded. It uses SAM-AdamW, Gaussian KL, frozen teachers, and sensor adapters. Training originally targeted 50 epochs, was extended toward 200, then explicitly stopped at **100 completed epochs**. The 200-epoch configuration is historical schedule information, not evidence of 200 completed epochs.

Two MotionTrans transfer heads were evaluated at epochs 10,20,…,100:

- Shallow: `32 → 512 → 6`.
- Deep: `32 → 512 → 512 → 6`.

| Encoder epoch | Logged effective rank | Shallow MAE | Deep MAE | Deep nominal 90% coverage |
| --- | --- | --- | --- | --- |
| 10 | 16.044 | 0.052103 | 0.048094 | 41.17% |
| 20 | 19.633 | 0.054840 | 0.049557 | 32.72% |
| 30 | 21.473 | 0.056413 | 0.050998 | 29.88% |
| 40 | 22.261 | 0.056721 | 0.051795 | 25.25% |
| 50 | 22.637 | 0.057101 | 0.050763 | 26.17% |
| 60 | 22.933 | 0.056819 | 0.050123 | 25.82% |
| 70 | 23.246 | 0.057090 | 0.050695 | 26.66% |
| 80 | 23.498 | 0.057758 | 0.051307 | 25.01% |
| 90 | 23.756 | 0.057807 | 0.051882 | 21.54% |
| 100 | 23.924 | 0.057987 | 0.051210 | 24.49% |

Later checkpoints did not improve transfer, despite continued source training. Within the sampled 10–100 range, higher logged effective rank correlated with worse MAE: Pearson approximately +0.9933 for the shallow head and +0.8722 for the deep head. These are descriptive serial-checkpoint correlations, not independent interventions.

The first ten epochs were subsequently evaluated to locate the early transfer optimum:

| Encoder epoch | Logged effective rank | Shallow MAE | Deep MAE | Deep nominal 90% coverage |
| --- | --- | --- | --- | --- |
| 1 | 8.143 | 0.053254 | 0.050404 | 47.31% |
| 2 | 8.840 | 0.051793 | 0.048538 | 33.85% |
| 3 | 10.112 | 0.052781 | 0.049241 | 29.08% |
| 4 | 11.489 | 0.051802 | 0.046814 | 31.72% |
| 5 | 12.612 | 0.052012 | 0.047480 | 37.30% |
| 6 | 13.430 | 0.052010 | 0.047844 | 39.88% |
| 7 | 14.192 | 0.052071 | 0.046085 | 42.42% |
| 8 | 14.859 | 0.052676 | 0.047069 | 39.66% |
| 9 | 15.446 | 0.054915 | 0.048062 | 38.50% |
| 10 | 16.044 | 0.052103 | 0.048094 | 41.17% |

The best shallow result was epoch 2, MAE **0.05179333**, rank **8.84**. The best deep result was epoch 7, MAE **0.04608450**, rank **14.19**. Epoch 7 was approximately 4.18% better than deep epoch 10. Its paired episode-bootstrap difference against epoch 10 was approximately −0.002010 with 95% interval [−0.003520, −0.000549]. Differences versus nearby epochs 4 and 8 were not resolved by the reported intervals.

Across epochs 1–10, rank/MAE Pearson correlation was **+0.1754 shallow** and **−0.6223 deep**. Thus rank increased during useful early learning and also increased during less useful late training. There is no universal conclusion that lower rank produces better transfer, nor evidence that rank 14.19 should be imposed as a training target.

The best early deep checkpoint still had only approximately **42.42% nominal 90% coverage**. Its better point prediction did not solve uncertainty calibration.

The checkpoint choices use MotionTrans validation and are exploratory development selections. Evidence: [10–100 sweep](../../../processing/runs/joint_alignment_holdout_motiontrans/evaluations/motiontrans_checkpoint_sweep/summary.json), [early sweep](../../../processing/runs/joint_alignment_holdout_motiontrans/evaluations/motiontrans_early_rank_sweep/summary.json), [early interpretation](../../../processing/runs/joint_alignment_holdout_motiontrans/evaluations/motiontrans_early_rank_sweep/analysis.md).

### Source reconstruction monitoring

The following mean-latent native MAEs use the trainer's 64 validation clips per source. They are source diagnostics rather than full validation/test evaluations. Optical outputs retain native resolution. OpenTouch retains the historical 256-channel training mask. The two 16D columns are matched; comparisons to 32D also change schedule and sampling.

| Source | Frozen AE MAE at these clips | 32D epoch 10 | 32D epoch 100 | 16D control epoch 10 | 16D FGW epoch 10 |
|---|---|---|---|---|---|
| EgoTouch | 0.002974 | 0.041227 | 0.037927 | 0.042530 | 0.042239 |
| EgoTactile | 0.002692 | 0.003751 | 0.003799 | 0.003621 | 0.003762 |
| OpenTouch | 0.016431 | 0.043265 | 0.044887 | 0.043295 | 0.043296 |
| OpenGraph | 0.012365 | 0.025062 | 0.024828 | 0.026210 | 0.026608 |
| FreeTacMan | 0.011645 | 0.088830 | 0.085275 | 0.075111 | 0.063516 |
| Unit | 0.005081 | 0.010351 | 0.009668 | 0.010311 | 0.010392 |
| Unit Bimanual | 0.007577 | 0.020231 | 0.022858 | 0.020335 | 0.020230 |
| VLA Touch | 0.007404 | 0.011370 | 0.010423 | 0.011471 | 0.011942 |
| ViTaMIn | 0.006141 | 0.019399 | 0.013752 | 0.021739 | 0.021058 |
| RDP | — | 0.006432 | 0.005242 | 0.007463 | 0.007526 |
| RDP Bimanual | — | 0.013161 | 0.009616 | 0.010038 | 0.017965 |

Sources: [32D training metrics](../../../processing/runs/joint_alignment_holdout_motiontrans/metrics.jsonl), [16D control metrics](../../../processing/runs/joint_16d_pair_v1/no_fgw/metrics.jsonl), [16D FGW metrics](../../../processing/runs/joint_16d_pair_v1/visual_fgw/metrics.jsonl). The frozen-AE score is a reference reconstruction error, not a guaranteed lower bound for every predicted code.

## 9. FGW alignment: implementation, experiments, and interpretation

### 9.1 Motivation and implementation

FGW was introduced to align unpaired interaction representations using both their coordinates and internal relational geometry. The starting reference is [Gong, Yue and Xu, AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/download/33849/36004). The analogy to ICP is iterative correspondence estimation followed by representation updates, with a shared barycenter replacing a rigid point-cloud reference.

The implementation uses POT transport solvers and a coordinate barycenter update. It is an alternating approximation, not a guarantee of globally optimizing the nonconvex FGW barycenter objective. `structure_weight=0` means feature-only OT and `1` means pure GW.

The current visual mixer samples 256 episode-balanced states per source, each representing the final valid frame of an eight-frame clip. It computes transport to 256 support points for five update rounds, with structure weight 0.5 and transported mass 0.75. Plans are detached. Transport push/pull operations normalize by actual received mass. Unmatched states retain ordinary prediction supervision and receive no consistency loss.

Stochastic mixing chooses a supported source contribution independently for each latent coordinate. The consistency penalty is mean squared error per dimension, weight 0.1, ramped over two epochs. A source-level spread penalty, weight 0.01, activates below half its initial mean coordinate variance. Banks refresh each epoch; targets and masks persist through SAM replay. Ordinary inference requires no transport solver.

Implemented tests cover transport mass/normalization, constant signals, unequal banks, synthetic correspondence/ambiguity scenarios, missing overlap/outliers, finite gradients, held-out exclusion, provenance, and canonical native pairing. Implementation checks do not verify the physical correctness of learned correspondences.

### 9.2 Two placements

The original large multi-seed/variant study was stopped after the user requested a smaller first test. Only two placements were pursued with seed 45:

1. **Visual:** mix the existing 32D visual latent means during joint training.
2. **Tactile-first:** project frozen sensor AE codes into a common 32D teacher representation, train reverse adapters plus FGW, freeze/cache that teacher, and supervise the visual encoder against its common codes as well as original sensor targets.

Both visual runs continued from the original non-FGW epoch-10 checkpoint. “Start at epoch 10” means load that trained model, optimizer, RNG, and continuation contract; its next update belongs to epoch 11. It does not mean skip ten epochs of learning. The tactile teacher completed 20 reconstruction warmup plus 50 FGW epochs and its common-code cache completed.

The visual placement completed epoch 30. The tactile-first visual run completed epoch 21 before the user stopped the classifier and training. The latest 32D visual epoch 30 was not evaluated on MotionTrans in the saved results; its only completed transfer/classifier evaluation is epoch 12. Tactile-first was evaluated at epoch 20. The stopped classifier has no final result.

| Placement | Encoder epoch | Head | MAE | Nominal 90% coverage | Evidence |
| --- | --- | --- | --- | --- | --- |
| Visual FGW | 12 | 32-512-6 | 0.053761 | 36.82% | [artifact](../../../processing/runs/fgw_two_placements_v1/fgw_visual_partial_seed45/evaluations/motiontrans_epoch_012_shallow/results.json) |
| Visual FGW | 12 | 32-512-512-6 | 0.047260 | 41.37% | [artifact](../../../processing/runs/fgw_two_placements_v1/fgw_visual_partial_seed45/evaluations/motiontrans_epoch_012_deep/results.json) |
| Tactile-first FGW | 20 | 32-512-6 | 0.061793 | 28.63% | [artifact](../../../processing/runs/fgw_two_placements_v1/fgw_tactile_first_partial_seed45/evaluations/motiontrans_epoch_020_shallow/results.json) |
| Tactile-first FGW | 20 | 32-512-512-6 | 0.054518 | 37.16% | [artifact](../../../processing/runs/fgw_two_placements_v1/fgw_tactile_first_partial_seed45/evaluations/motiontrans_epoch_020_deep/results.json) |

The visual deep head at epoch 12 was 1.74% better than the non-FGW epoch-10 parent, but 2.55% worse than the stronger non-FGW epoch-7 checkpoint. That comparison lacks a matched continuation control. Tactile-first epoch 20 was worse than the parent by approximately 18.60% shallow and 13.36% deep. These results do not establish a robust FGW transfer benefit.

The epoch-12 visual dataset classifier retained **98.80% balanced accuracy**; strict whole-latent sensor invariance was not achieved. Final logged effective ranks were approximately 18.73 for visual epoch 30 and 17.65 for tactile-first epoch 21. Rank alone is not a success criterion.

### 9.3 Why the current attraction may be insufficient

FGW consistency does pull selected latent means toward shared transported/mixed targets. It does not establish that matched points describe equivalent contact, force, or motion. The transport geometry in the visual placement is constructed from visual latents, not directly from measured tactile trajectories.

The saved epoch-10 16D diagnostics show:

- Zero reported solver failures, making numerical failure an unsupported explanation for the missing gain.
- Zero transport row entropy, indicating effectively hard assignments rather than calibrated correspondence uncertainty.
- Overall training latent variance 7.11 control versus 3.47 FGW, while effective rank increases 11.99 → 12.99. The latent changes substantially without a demonstrated transfer gain.
- Source-bank variances 167–730 times their initialization references. The spread penalty is inactive in all 129 logged epoch-10 alignment batches; its small initialization reference is a weak safeguard for preserving subsequently learned variation.
- Uneven objective scales. In sampled RDP alignment batches, weighted consistency averaged approximately 0.01267 versus 0.00551 task loss; in EgoTouch approximately 0.01030 versus 1.00931. These sampled loss magnitudes are not gradient-norm measurements and are not full-epoch averages.

The new 16D FGW condition begins aligning before its shared encoder has learned tactile structure. That may reinforce uninformative early matches. However, the older 32D placement already used a trained parent, so warmup alone is not a demonstrated solution.

Additional hypotheses are appearance-driven matching, arbitrary coordinate combinations, weak/incorrect overlap assumptions, stale epoch-wide targets, and loss competition. They require diagnostics or controlled experiments; they are not proven causes. Low transport cost and hard correspondences do not establish physical equivalence.

Evidence: [mixer](../../../legacy/v1/src/tactilelatent/fgw_mixer.py), [objective](../../../legacy/v1/src/tactilelatent/fgw_objective.py), [equation guide](FGW_ALIGNMENT_AND_MIXING.md), [stopped pilot](../../../processing/runs/fgw_two_placements_v1/status.json), [epoch-10 transport diagnostics](../../../processing/runs/joint_16d_pair_v1/visual_fgw/alignment/epoch_010.json).

## 10. Current matched 16D experiment

Two fresh models train concurrently toward epoch 30: plain cached-code supervision and visual stochastic partial FGW. Both start from verified identical seed-45 weights; the old 32D Gaussian bottleneck is not loaded into the 16D models.

Both retain width 768, eight attention heads, two cross-view layers, three temporal layers, two 128-unit adapter hidden layers, frozen DINO/AE teachers, direct numeric heads, and native tactile outputs. Training uses batch 2, accumulation 16, SAM-AdamW with rho 0.05, learning rate 5e-5, two warmup epochs, cosine floor 1e-6, weight decay 0.05, dropout 0.1, wrist dropout 0.3, and Gaussian KL ramp to 0.001 over ten epochs.

Both use matched source scheduling and 25% bank-clip replacement. The plain run has no transport/consistency/spread objective. Refresh restores Torch RNG so FGW bank feature extraction does not independently change dropout/latent-sampling draws. Initial weight SHA256 is recorded and checked by the controller.

This is a matched FGW-versus-control comparison. It is **not** a clean 16D-versus-historical-32D comparison: the historical run used a different LR horizon and ordinary sampling. For a source hitting the 8,000-clip cap, approximately 6,000 ordinary clips and 2,000 repeated draws from at most 256 bank clips are used per epoch. Small sources contribute fewer batches; the cap does not equalize contributions.

### Completed epoch-10 MotionTrans evaluation

The four requested head evaluations completed while representation training continued. Protocol: full MotionTrans training split, seed 45, episode-disjoint fit/tune, 100 head-epoch cap, patience ten, refit on all training episodes, and 64 latent draws. There are 16,736 fit clips, 1,877 tune clips, and 1,972 validation clips from 167 episodes.

| Condition | Head | MAE | Ensemble MAE | CRPS | 90% coverage | Refit epochs | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Non-FGW | 16-512-6 | 0.06039082 | 0.060674 | 0.053619 | 32.32% | 70 | [artifact](../../../processing/runs/joint_16d_pair_v1/no_fgw/evaluations/motiontrans_epoch_010_shallow/results.json) |
| Non-FGW | 16-512-512-6 | 0.05348656 | 0.053766 | 0.045314 | 40.69% | 9 | [artifact](../../../processing/runs/joint_16d_pair_v1/no_fgw/evaluations/motiontrans_epoch_010_deep/results.json) |
| FGW | 16-512-6 | 0.06157224 | 0.061760 | 0.055737 | 27.11% | 73 | [artifact](../../../processing/runs/joint_16d_pair_v1/visual_fgw/evaluations/motiontrans_epoch_010_shallow/results.json) |
| FGW | 16-512-512-6 | 0.05475087 | 0.054883 | 0.046421 | 39.45% | 21 | [artifact](../../../processing/runs/joint_16d_pair_v1/visual_fgw/evaluations/motiontrans_epoch_010_deep/results.json) |

Paired episode bootstrap, 10,000 replicates, FGW minus control:

- Shallow MAE difference **+0.00118142**, 95% interval **[−0.00081875, +0.00317700]**.
- Deep MAE difference **+0.00126432**, 95% interval **[−0.00086468, +0.00343639]**.

Both intervals include zero. The current evidence is no demonstrated gain at epoch 10; it does not establish general harm from FGW. FGW point MAE is slightly higher for every target channel. The deep head improves prediction for both models. Nominal 90% coverage remains only 27–41%.

Epoch-10 training effective rank is 11.9933 control versus 12.9897 FGW; mean predicted Gaussian variance is 0.02770 versus 0.01694. These two conditions do not identify a causal rank/transfer relationship.

| Run | Process state at snapshot | Completed epochs | Target epochs | Latest logged rank | Source validation score¹ |
| --- | --- | --- | --- | --- | --- |
| no_fgw | Running | 15 | 30 | 12.8382 | 0.648094 |
| visual_fgw | Running | 13 | 30 | 13.3497 | 0.657300 |

¹ Equal-source mean supervision loss (AE-code MSE or numeric loss); not a common physical MAE or MotionTrans score. Training continues beyond this static document snapshot. Only epoch 10 has completed transfer evaluations for this pair.

Evidence: [16D evaluation interpretation](../../../processing/runs/joint_16d_pair_v1/evaluations/epoch_010/analysis.md), [paired intervals and channel errors](../../../processing/runs/joint_16d_pair_v1/evaluations/epoch_010/paired_episode_comparisons.json), [comparison plot](../../../processing/runs/joint_16d_pair_v1/evaluations/epoch_010/comparison.png), [controller status](../../../processing/runs/joint_16d_pair_v1/status.json).

## 11. Effective rank: definition and limitations

For Gaussian means collected during a training epoch, compute covariance eigenvalues `λ_i ≥ 0`, normalize `p_i = λ_i / sum(λ)`, and report:

```text
effective_rank = exp(-sum_i p_i * log(p_i))
```

This summarizes how evenly variance is distributed across directions. A nominal 32D latent can concentrate its variation in fewer directions. Effective rank is not the number of nonzero neurons, a direct physical-dimension count, or a sensor-invariance score.

The logged global rank pools sources, times, and batches while model weights change during the epoch. It is not a fixed-checkpoint MotionTrans covariance estimate. Per-source bank rank is a different diagnostic. Variance magnitude must be considered alongside rank: a small cloud can have a high effective rank. The current implementation's numerical zero-covariance convention should not be interpreted as evidence of a physical dimension.

Early and late checkpoint correlations differ in sign. The results support monitoring rank and representation evolution, not imposing lower rank as a universal transfer objective.

## 12. Uncertainty: what is implemented and what remains unresolved

The visual encoder produces a diagonal Gaussian. Evaluation samples it 64 times and reports ensemble mean error, expected sample error, CRPS, predictive standard deviation, interval coverage/width, PIT diagnostics, channel metrics, and selected error-region diagnostics.

The image-only sampled reconstruction objective plus standard-normal KL does not directly optimize calibrated output probabilities. It lacks a conventional target-informed posterior at inference. Latent variability can be suppressed or transformed unpredictably by adapters/decoders; it does not automatically capture sensor noise, representation error, AE reconstruction error, epistemic uncertainty, or domain shift.

Historical EgoTouch uncertainty comparisons on the saved unseen protocol:

| Model, all views | Ensemble MAE | CRPS |
|---|---:|---:|
| Image-only 32D VAE | 0.05597060 | 0.05363902 |
| Pose-teacher 32D CVAE | 0.05711763 | 0.04576302 |

The CVAE case diagnosis found only **16.33% nominal 95% coverage**. Approximately 64.3% of evaluated pressure cells were exactly zero; the zero/near-zero group had zero central-interval coverage. Sigmoid outputs generally remain positive and do not model an explicit zero mass. Small numerical exact zeros in saturated floating-point outputs are not a calibrated zero model.

The current MotionTrans transfer intervals likewise substantially undercover. Good CRPS/error rankings and a positive uncertainty/error correlation do not replace coverage validation. Best-of-64 is an oracle diagnostic, not a deployable predictor. FGW stochastic coordinate mixing is augmentation and is not a calibrated uncertainty estimator.

The initial broad FGW study described separate episode-level conformal calibration, but that broad study was stopped. **No completed calibration result for the current finalists is reported.** A subsequent calibration experiment should reserve disjoint calibration episodes, evaluate channel/time dependence, and state exchangeability assumptions and the coverage unit. Alignment and calibration changes should be evaluated separately.

Evidence: [historical VAE](../../../processing/evaluations/latent32_vae_vs_cvae_uncertainty/vae.json), [historical CVAE](../../../processing/evaluations/latent32_vae_vs_cvae_uncertainty/cvae.json), [zero/contact diagnosis](../../../processing/evaluations/latent32_vae_vs_cvae_uncertainty/cvae_case_diagnosis.json), [current evaluation](../../../processing/runs/joint_16d_pair_v1/evaluations/epoch_010/summary.json).

## 13. Viewer and experiment tooling completed

The viewer has two pages:

1. **Sensor reconstruction:** select dataset/sensor, split, episode, and trained AE; scrub/play frames; compare measured tactile data with AE reconstruction and error. Pressure/taxel/optical/numeric sensors have appropriate displays, native originals, common scales, hover values, and optional amplified temporal change.
2. **Ego → tactile prediction:** select joint models and existing saved epochs, including completed checkpoints from ongoing runs; compare two models on the same target. Cached-code models reconstruct through their pinned frozen decoders. Held-out datasets require a fitted transfer head.

Model discovery was corrected to restore available architectures, expose sensor-specific architecture details, include saved transfer heads, and remove smoke runs, invalid/unrunnable choices, obsolete numeric AEs, and superseded reduced optical models. EgoTouch's `wide*` names do not imply matching names exist for other sensors.

Remote access was moved to port 8080, with VS Code forwarding/SSH/browser-proxy support. The user service is `tactilelatent-viewer.service`. This report records implemented viewer capabilities, not a fresh browser connectivity test.

The visual viewer exposes ego-only/all-available camera selection and actual input indices. ViTaMIn is wrist-only. Its displayed target frame is at position three of a bidirectional eight-frame clip, so it can use future clip frames. It is not a streaming-causal evaluation. Historical 32×32 optical prediction heads are omitted from native-image comparison; saved numerical historical evaluations remain documented here.

Training/evaluation tooling includes resumable supervisors, process locks, atomic checkpoints, per-epoch viewer snapshots, exact checkpoint copies, code/data/teacher hashes, feature-cache reuse across transfer heads, episode-disjoint selection, paired episode bootstrap, channel/dynamics metrics, and exportable plots. Successful smoke/regression checks establish implementation behavior, not physical invariance or hardware synchronization.

Evidence: [viewer guide](EGOTOUCH_AUTOENCODER_VIEWER.md), [viewer launcher](../../../legacy/v1/scripts/view_sensor_autoencoders.py), [checkpoint evaluator](../../../legacy/v1/scripts/evaluate_alignment_checkpoint_sweep.py), [matched pair evaluator](../../../legacy/v1/scripts/evaluate_16d_pair.py).

## 14. Ideas discussed but not implemented

These are future research proposals, not active objectives or achieved results:

- **Tactile trajectory matching:** use within-clip distances between cached tactile codes, temporal ordering, absolute levels, and changes to match interaction evolution rather than one latent frame. Match descriptors may be normalized while native targets remain unchanged. Similar trajectories still do not guarantee physical equivalence.
- **Prediction-preserving correspondence:** assess whether a matched latent from another source predicts the original sensor code through its adapter. Use slowly updated teacher scoring, temporal-change checks, and constant-baseline safeguards. Adapter incompatibility can reject good matches, so confidence gating needs validation.
- **Future tactile prediction:** predict codes/change beyond the encoder's observed window, with several horizons and a persistence baseline. This can favor dynamics over static scene shortcuts; absent actions can make the future ambiguous.
- **Static episode context:** retain a 16D interaction latent and condition adapters on a small, fixed episode vector inferred from an initial ego-only prefix. A proposed 4D context would add capacity beyond the common 16D path. It cannot infer invisible sensor calibration reliably and must not use future frames or episode-ID lookup. Test transfer from the interaction latent alone and audit shortcut leakage.
- **Meta-learning for new heads:** repeatedly fit temporary source-shaped heads on support episodes and update the encoder using disjoint query episodes, differentiating through adaptation or an appropriate differentiable solver. This resembles head-adaptation meta-learning more directly than exact classical MLDG. Rotate sources, vary support size, retain ordinary supervision, and keep MotionTrans excluded. Rotating sources simulate adaptation; they are not globally unseen throughout training.
- **Selective shared/private information:** do not require scalar sensors to represent all tactile-image spatial detail. Shared/private branches and confidence-weighted partial overlap are hypotheses to test with transfer and reconstruction preservation.
- **Broader episode coverage:** measure unique clips/episodes and task/object/session diversity under capped training and repeated-bank sampling. The smallest sources have only 47–68 training episodes, and some validation sets only 3–4; correlated frames do not replace independent environments.
- **Matched baselines and calibration:** compare DINO-only features, frozen/fine-tuned/direct target training, support-label budgets, and calibration with identical procedures. Avoid conflating latent dimension, LR schedule, sampling, and alignment changes.

References motivating these proposals: [Gromov dynamic time warping](https://proceedings.mlr.press/v130/cohen21a/cohen21a.pdf), [Domain Separation Networks](https://research.google/pubs/domain-separation-networks/), [MLDG](https://ojs.aaai.org/index.php/AAAI/article/view/11596), [ANIL](https://arxiv.org/abs/1909.09157), [V-JEPA 2](https://arxiv.org/abs/2506.09985). None establishes that its adaptation to this project will improve MotionTrans.

## 15. Conclusions supported by current evidence

1. Specialized tactile reconstruction is feasible and high quality for several sensors; native code teachers/caches and visual comparison tooling are implemented.
2. Frozen visual representations contain useful information for supervised MotionTrans adaptation, but the best saved frozen transfer result still trails the direct target specialist. Protocol differences limit comparisons across generations.
3. More representation-training epochs do not necessarily improve transfer. The best early cached-code deep head came from epoch 7, not the 100-epoch checkpoint.
4. Effective rank is associated with different outcomes at different stages and is not an identified causal lever.
5. Aggressive dataset-identity suppression can destroy useful prediction. Current FGW does not establish strict sensor invariance or a robust transfer benefit.
6. The matched 16D epoch-10 comparison shows slightly worse FGW point error, with paired intervals including zero. Its 30-epoch training is ongoing at this snapshot.
7. Predictive uncertainty is substantially undercalibrated. Returning to sigmoid and adding stochastic FGW did not solve this objective.
8. Data corrections are central: historical resized optical targets, camera-role errors, and VLA chronology must remain explicit when interpreting old scores.

The next scientific advance should be judged by preserved useful tactile information, better adaptation to unfamiliar sensors, and separately validated uncertainty—not solely by making latent distributions look more alike.

## Appendix A. Complete retained autoencoder comparison tables

The tables below reproduce compact metrics from the retained comparison artifacts. Each row's scores belong to its selected checkpoint and documented budget. Screens and extended finalists have different completed budgets. Full files contain optimizer, contact/high-pressure, zero-output, timing, and parameter details. Superseded runs are explicitly labeled and are not current native-teacher candidates.

### Initial matched EgoTouch grid: all 12 runs

Source: [complete comparison CSV](../../../processing/runs/egotouch_ae_sensor_code_grid_v1/comparison.csv).

| Sensor | Model/preset | Code | Seed | Epochs | Best epoch | Score | MAE | Contact MAE | Contact F1 | High-pressure MAE | PSNR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EgoTouch | small | 16 | 45 | 30 | 30 | — | 0.026732 | 0.068449 | 0.7746 | 0.109797 | — |
| EgoTouch | medium | 16 | 45 | 30 | 30 | — | 0.025234 | 0.062767 | 0.7841 | 0.092899 | — |
| EgoTouch | large | 16 | 45 | 30 | 27 | — | 0.024691 | 0.064038 | 0.7865 | 0.093881 | — |
| EgoTouch | small | 32 | 45 | 30 | 30 | — | 0.023782 | 0.060287 | 0.7987 | 0.089436 | — |
| EgoTouch | medium | 32 | 45 | 30 | 30 | — | 0.020427 | 0.052065 | 0.8219 | 0.073548 | — |
| EgoTouch | large | 32 | 45 | 30 | 30 | — | 0.020776 | 0.052164 | 0.8185 | 0.073232 | — |
| EgoTouch | small | 64 | 45 | 30 | 30 | — | 0.020668 | 0.052942 | 0.8207 | 0.077560 | — |
| EgoTouch | medium | 64 | 45 | 30 | 30 | — | 0.018088 | 0.045341 | 0.8390 | 0.061711 | — |
| EgoTouch | large | 64 | 45 | 30 | 30 | — | 0.019974 | 0.051503 | 0.8273 | 0.072785 | — |
| EgoTouch | small | 128 | 45 | 30 | 30 | — | 0.020425 | 0.054326 | 0.8257 | 0.081386 | — |
| EgoTouch | medium | 128 | 45 | 30 | 30 | — | 0.016363 | 0.042646 | 0.8554 | 0.060235 | — |
| EgoTouch | large | 128 | 45 | 30 | 29 | — | 0.019584 | 0.049985 | 0.8291 | 0.070378 | — |

### Completed EgoTouch sigmoid v4: all 28 runs

Source: [complete comparison CSV](../../../processing/runs/egotouch_ae_search_v4/comparison.csv).

| Sensor | Model/preset | Code | Seed | Epochs | Best epoch | Score | MAE | Contact MAE | Contact F1 | High-pressure MAE | PSNR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EgoTouch | wide64_code256 | 256 | 45 | 200 | 198 | 0.007592 | 0.003545 | 0.007847 | 0.9697 | 0.011385 | — |
| EgoTouch | wide64_code256_seed46 | 256 | 46 | 200 | 197 | 0.007597 | 0.003591 | 0.007911 | 0.9689 | 0.011288 | — |
| EgoTouch | wide64_code256_seed47 | 256 | 47 | 200 | 197 | 0.007978 | 0.003765 | 0.008368 | 0.9684 | 0.011799 | — |
| EgoTouch | wide48_code256_seed46 | 256 | 46 | 200 | 198 | 0.008681 | 0.004031 | 0.009080 | 0.9663 | 0.012932 | — |
| EgoTouch | wide48_code256_seed47 | 256 | 47 | 200 | 197 | 0.009112 | 0.004287 | 0.009645 | 0.9627 | 0.013404 | — |
| EgoTouch | wide48_code256 | 256 | 45 | 200 | 196 | 0.009457 | 0.004286 | 0.009951 | 0.9637 | 0.014135 | — |
| EgoTouch | wide64_code128 | 128 | 45 | 200 | 200 | 0.012889 | 0.005881 | 0.014543 | 0.9496 | 0.018242 | — |
| EgoTouch | batch64_code128 | 128 | 45 | 200 | 199 | 0.013583 | 0.006268 | 0.015143 | 0.9457 | 0.019338 | — |
| EgoTouch | lr3e4_code128 | 128 | 45 | 60 | 55 | 0.019818 | 0.008680 | 0.021763 | 0.9261 | 0.029010 | — |
| EgoTouch | lr2e4_code128 | 128 | 45 | 60 | 55 | 0.019884 | 0.008726 | 0.022133 | 0.9250 | 0.028794 | — |
| EgoTouch | wide48_code128 | 128 | 45 | 60 | 55 | 0.020237 | 0.009017 | 0.022463 | 0.9218 | 0.029231 | — |
| EgoTouch | medium_code256 | 256 | 45 | 60 | 58 | 0.020776 | 0.009328 | 0.022712 | 0.9168 | 0.030288 | — |
| EgoTouch | mae025_code128 | 128 | 45 | 60 | 58 | 0.022455 | 0.010171 | 0.024586 | 0.9089 | 0.032609 | — |
| EgoTouch | wide64_code64 | 64 | 45 | 200 | 196 | 0.022628 | 0.010608 | 0.026640 | 0.9045 | 0.030636 | — |
| EgoTouch | wd0_code128 | 128 | 45 | 60 | 60 | 0.022691 | 0.010156 | 0.024772 | 0.9101 | 0.033146 | — |
| EgoTouch | wd1e3_code128 | 128 | 45 | 60 | 60 | 0.022746 | 0.010134 | 0.024878 | 0.9104 | 0.033226 | — |
| EgoTouch | medium_code128 | 128 | 45 | 60 | 60 | 0.022845 | 0.010096 | 0.024794 | 0.9109 | 0.033646 | — |
| EgoTouch | contact5_code128 | 128 | 45 | 60 | 56 | 0.023195 | 0.011620 | 0.025407 | 0.8900 | 0.032558 | — |
| EgoTouch | wide48_code64 | 64 | 45 | 200 | 196 | 0.023337 | 0.010924 | 0.027273 | 0.9013 | 0.031812 | — |
| EgoTouch | mae100_code128 | 128 | 45 | 60 | 55 | 0.023550 | 0.010630 | 0.026168 | 0.9055 | 0.033850 | — |
| EgoTouch | contact1_code128 | 128 | 45 | 60 | 60 | 0.024919 | 0.009218 | 0.027121 | 0.9290 | 0.038419 | — |
| EgoTouch | batch256_code128 | 128 | 45 | 60 | 58 | 0.026376 | 0.012059 | 0.028912 | 0.8893 | 0.038156 | — |
| EgoTouch | deep_code256 | 256 | 45 | 60 | 57 | 0.026521 | 0.012190 | 0.029717 | 0.8909 | 0.037656 | — |
| EgoTouch | deep_code128 | 128 | 45 | 60 | 58 | 0.029556 | 0.013205 | 0.032891 | 0.8811 | 0.042572 | — |
| EgoTouch | medium_code64 | 64 | 45 | 60 | 58 | 0.029674 | 0.013673 | 0.033466 | 0.8766 | 0.041883 | — |
| EgoTouch | lr5e5_code128 | 128 | 45 | 60 | 58 | 0.030580 | 0.013933 | 0.034138 | 0.8729 | 0.043669 | — |
| EgoTouch | deep_code64 | 64 | 45 | 60 | 60 | 0.031951 | 0.014918 | 0.036250 | 0.8664 | 0.044686 | — |
| EgoTouch | lr3e5_code128 | 128 | 45 | 60 | 58 | 0.038709 | 0.017033 | 0.042754 | 0.8488 | 0.056341 | — |

### Completed native sensor suite: all 16 candidates

Source: [complete comparison CSV](../../../processing/runs/sensor_autoencoders_native_v2/comparison.csv).

| Sensor | Model/preset | Code | Seed | Epochs | Best epoch | Score | MAE | Contact MAE | Contact F1 | High-pressure MAE | PSNR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EgoTactile | medium_code16 | 16 | 45 | 100 | 93 | 0.031734 | 0.002752 | 0.039447 | 0.7992 | — | — |
| EgoTactile | medium_code32 | 32 | 45 | 30 | 30 | 0.047311 | 0.003506 | 0.050229 | 0.7593 | — | — |
| FreeTacMan | large_code256 | 256 | 45 | 100 | 100 | 0.013905 | 0.013905 | — | — | — | 30.64 |
| FreeTacMan | medium_code128 | 128 | 45 | 30 | 29 | 0.017120 | 0.017120 | — | — | — | 28.76 |
| Unit | large_code256 | 256 | 45 | 100 | 100 | 0.005484 | 0.005484 | — | — | — | 40.95 |
| Unit | medium_code128 | 128 | 45 | 30 | 30 | 0.006820 | 0.006820 | — | — | — | 38.94 |
| Unit Bimanual | large_code256 | 256 | 45 | 100 | 97 | 0.007324 | 0.007324 | — | — | — | 36.98 |
| Unit Bimanual | medium_code128 | 128 | 45 | 30 | 30 | 0.008930 | 0.008930 | — | — | — | 35.26 |
| ViTaMIn | large_code256 | 256 | 45 | 100 | 95 | 0.005864 | 0.005864 | — | — | — | 40.50 |
| ViTaMIn | medium_code128 | 128 | 45 | 30 | 28 | 0.007811 | 0.007811 | — | — | — | 37.80 |
| VLA Touch | large_code256 | 256 | 45 | 100 | 99 | 0.007331 | 0.007331 | — | — | — | 38.87 |
| VLA Touch | medium_code128 | 128 | 45 | 30 | 29 | 0.008903 | 0.008903 | — | — | — | 36.88 |
| OpenGraph | compact128_code16 | 16 | 45 | 100 | 100 | 0.056220 | 0.013336 | 0.064702 | 0.7619 | — | — |
| OpenGraph | compact128_code8 | 8 | 45 | 100 | 95 | 0.072436 | 0.016602 | 0.079866 | 0.7182 | — | — |
| OpenTouch | large_code32 | 32 | 45 | 100 | 99 | 0.014833 | 0.015582 | 0.015001 | 0.9976 | — | — |
| OpenTouch | medium_code16 | 16 | 45 | 30 | 30 | 0.021861 | 0.023483 | 0.022076 | 0.9965 | — | — |

### Superseded sensor v1: retained partial results

Source: [complete comparison CSV](../../../processing/runs/sensor_autoencoders_v1/comparison.csv).

| Sensor | Model/preset | Code | Seed | Epochs | Best epoch | Score | MAE | Contact MAE | Contact F1 | High-pressure MAE | PSNR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EgoTactile | medium_code64 | 64 | 45 | 3 | 3 | 0.111330 | 0.005456 | 0.074330 | 0.6500 | — | — |
| MotionTrans (held out) | medium_code2 | 2 | 45 | 3 | 3 | 0.022418 | 0.022418 | — | — | — | — |
| RDP | medium_code1 | 1 | 45 | 3 | 3 | 0.004582 | 0.004582 | — | — | — | — |
| RDP Bimanual | medium_code2 | 2 | 45 | 3 | 3 | 0.010370 | 0.010370 | — | — | — | — |
| Unit | medium_code128 | 128 | 45 | 3 | 3 | 0.026742 | 0.026742 | — | — | — | 27.45 |
| Unit Bimanual | medium_code128 | 128 | 45 | 3 | 3 | 0.026305 | 0.026305 | — | — | — | 26.25 |
| ViTaMIn | medium_code128 | 128 | 45 | 3 | 3 | 0.027804 | 0.027804 | — | — | — | 27.31 |
| VLA Touch | medium_code128 | 128 | 45 | 3 | 3 | 0.042402 | 0.042402 | — | — | — | 23.39 |
| OpenGraph | medium_code32 | 32 | 45 | 3 | 1 | 0.212348 | 0.053148 | 0.169955 | 0.3532 | — | — |
| OpenTouch | medium_code64 | 64 | 45 | 3 | 3 | 0.032356 | 0.036401 | 0.032637 | 0.9956 | — | — |

### Other retained EgoTouch attempts and short diagnostics

These include abandoned hurdle/clamp trials, short LR diagnostics, and the initial failed run. Best-epoch labels are copied from these AE reports; row counts show saved metric records, not an inferred planned budget.

| Run | Metric records | Reported best epoch | MAE | Contact MAE | Contact F1 | Exact-zero recall | False-zero rate | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| egotouch_ae_clamp_check | 3 | 3 | 0.028197 | 0.060061 | 0.7687 | 38.06% | 3.57% | [artifact](../../../processing/runs/egotouch_ae_clamp_check/best_metrics.json) |
| egotouch_ae_conv_small_z32 | 13 | 4 | 0.055300 | 0.229335 | 0.0000 | — | — | [artifact](../../../processing/runs/egotouch_ae_conv_small_z32/best_metrics.json) |
| egotouch_ae_hurdle_check | 3 | 3 | 0.032941 | 0.077006 | 0.7534 | 89.10% | 24.93% | [artifact](../../../processing/runs/egotouch_ae_hurdle_check/best_metrics.json) |
| egotouch_ae_search_v2/trials/medium_code64 | 2 | 2 | 0.035692 | 0.088085 | 0.7248 | 89.87% | 31.68% | [artifact](../../../processing/runs/egotouch_ae_search_v2/trials/medium_code64/best_metrics.json) |
| egotouch_ae_search_v3/trials/medium_code64 | 22 | 20 | 0.017213 | 0.036796 | 0.8552 | 36.07% | 2.43% | [artifact](../../../processing/runs/egotouch_ae_search_v3/trials/medium_code64/best_metrics.json) |
| egotouch_ae_sensor_code_lr_check | 3 | 3 | 0.045623 | 0.111653 | 0.6363 | — | — | [artifact](../../../processing/runs/egotouch_ae_sensor_code_lr_check/best_metrics.json) |

## Appendix B. Saved historical EgoTouch visual training results

These are the latest logged validation results of the retained non-smoke EgoTouch visual runs, not a matched ablation or untouched-test evaluation. Logged epoch indices are zero-based for these historical visual trainers; the table converts them to completed epoch counts. Different run lengths, objectives, and validation protocols limit comparisons.

| Run | Completed epochs | Ego validation MAE | All-view validation MAE | Zero baseline | Contact F1 | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| egotouch_dinov2_long50 | 50 | 0.066250 | — | 0.050999 | 0.4497 | [log](../../../processing/runs/egotouch_dinov2_long50/metrics.jsonl) |
| egotouch_dinov3_latent128 | 13 | 0.043930 | 0.043380 | 0.055686 | 0.6749 | [log](../../../processing/runs/egotouch_dinov3_latent128/metrics.jsonl) |
| egotouch_dinov3_probabilistic_latent128 | 19 | 0.039557 | 0.039274 | 0.055686 | 0.7046 | [log](../../../processing/runs/egotouch_dinov3_probabilistic_latent128/metrics.jsonl) |

The selected unseen probabilistic evaluations and case diagnoses are separately linked in Section 12 and Appendix C. The earliest DINOv2 run's final monitored error was worse than its zero baseline; later DINOv3 variants improved their own monitored pressure prediction. This does not isolate backbone choice alone.

## Appendix C. Index of saved non-smoke prediction/classifier evaluations

This index includes retained evaluation `results.json` artifacts under the main run/evaluation roots, excluding duplicate baseline snapshots and archived copies. It includes superseded/historical protocols and explicitly identifies the evaluated head and completed encoder epoch. The 32D and 16D sweeps above are included for completeness. Full JSON links provide all available channel, dynamics, uncertainty, baseline, and fit/refit metrics. A missing value means that metric is not present in that report, not a zero score. Smoke tests are implementation checks and are not scientific results.

| Evaluation | Completed encoder epoch | Head/protocol | Deterministic MAE | 95% coverage | Identity macro accuracy | Full report |
| --- | --- | --- | --- | --- | --- | --- |
| dataset_identity_probe | 50 | 32-512-11 | — | — | 97.10% | [artifact](../../../processing/evaluations/dataset_identity_probe/results.json) |
| egotouch_epoch_latest / test_seen / ego | 12 | Historical EgoTouch prediction | 0.038933 | — | — | [artifact](../../../processing/evaluations/egotouch_epoch_latest/results.json) |
| egotouch_epoch_latest / test_seen / all | 12 | Historical EgoTouch prediction | 0.038327 | — | — | [artifact](../../../processing/evaluations/egotouch_epoch_latest/results.json) |
| egotouch_epoch_latest / test_unseen / ego | 12 | Historical EgoTouch prediction | 0.054078 | — | — | [artifact](../../../processing/evaluations/egotouch_epoch_latest/results.json) |
| egotouch_epoch_latest / test_unseen / all | 12 | Historical EgoTouch prediction | 0.053146 | — | — | [artifact](../../../processing/evaluations/egotouch_epoch_latest/results.json) |
| egotouch_probabilistic_epoch19_unseen / test_unseen / ego | 19 | Historical EgoTouch prediction | 0.054252 | — | — | [artifact](../../../processing/evaluations/egotouch_probabilistic_epoch19_unseen/results.json) |
| egotouch_probabilistic_epoch19_unseen / test_unseen / all | 19 | Historical EgoTouch prediction | 0.053569 | — | — | [artifact](../../../processing/evaluations/egotouch_probabilistic_epoch19_unseen/results.json) |
| vla_touch_frozen_decoder | 50 | 32-512-3072 | 0.010323 | 55.47% | — | [artifact](../../../processing/evaluations/vla_touch_frozen_decoder/results.json) |
| fgw_two_placements_v1/fgw_tactile_first_partial_seed45 / motiontrans_epoch_020_deep | 20 | 32-512-512-6 | 0.054518 | 42.56% | — | [artifact](../../../processing/runs/fgw_two_placements_v1/fgw_tactile_first_partial_seed45/evaluations/motiontrans_epoch_020_deep/results.json) |
| fgw_two_placements_v1/fgw_tactile_first_partial_seed45 / motiontrans_epoch_020_shallow | 20 | 32-512-6 | 0.061793 | 32.99% | — | [artifact](../../../processing/runs/fgw_two_placements_v1/fgw_tactile_first_partial_seed45/evaluations/motiontrans_epoch_020_shallow/results.json) |
| fgw_two_placements_v1/fgw_visual_partial_seed45 / dataset_identity_epoch_012 | 12 | 32-512-11 | — | — | 98.80% | [artifact](../../../processing/runs/fgw_two_placements_v1/fgw_visual_partial_seed45/evaluations/dataset_identity_epoch_012/results.json) |
| fgw_two_placements_v1/fgw_visual_partial_seed45 / motiontrans_epoch_012_deep | 12 | 32-512-512-6 | 0.047260 | 47.15% | — | [artifact](../../../processing/runs/fgw_two_placements_v1/fgw_visual_partial_seed45/evaluations/motiontrans_epoch_012_deep/results.json) |
| fgw_two_placements_v1/fgw_visual_partial_seed45 / motiontrans_epoch_012_shallow | 12 | 32-512-6 | 0.053761 | 42.18% | — | [artifact](../../../processing/runs/fgw_two_placements_v1/fgw_visual_partial_seed45/evaluations/motiontrans_epoch_012_shallow/results.json) |
| joint_16d_pair_v1/no_fgw / motiontrans_epoch_010_deep | 10 | 16-512-512-6 | 0.053487 | 46.43% | — | [artifact](../../../processing/runs/joint_16d_pair_v1/no_fgw/evaluations/motiontrans_epoch_010_deep/results.json) |
| joint_16d_pair_v1/no_fgw / motiontrans_epoch_010_shallow | 10 | 16-512-6 | 0.060391 | 37.13% | — | [artifact](../../../processing/runs/joint_16d_pair_v1/no_fgw/evaluations/motiontrans_epoch_010_shallow/results.json) |
| joint_16d_pair_v1/visual_fgw / motiontrans_epoch_010_deep | 10 | 16-512-512-6 | 0.054751 | 45.23% | — | [artifact](../../../processing/runs/joint_16d_pair_v1/visual_fgw/evaluations/motiontrans_epoch_010_deep/results.json) |
| joint_16d_pair_v1/visual_fgw / motiontrans_epoch_010_shallow | 10 | 16-512-6 | 0.061572 | 31.09% | — | [artifact](../../../processing/runs/joint_16d_pair_v1/visual_fgw/evaluations/motiontrans_epoch_010_shallow/results.json) |
| joint_alignment_holdout_motiontrans / motiontrans_epoch_001_deep | 1 | 32-512-512-6 | 0.050404 | 53.49% | — | [artifact](../../../processing/runs/joint_alignment_holdout_motiontrans/evaluations/motiontrans_epoch_001_deep/results.json) |
| joint_alignment_holdout_motiontrans / motiontrans_epoch_001_shallow | 1 | 32-512-6 | 0.053254 | 56.30% | — | [artifact](../../../processing/runs/joint_alignment_holdout_motiontrans/evaluations/motiontrans_epoch_001_shallow/results.json) |
| joint_alignment_holdout_motiontrans / motiontrans_epoch_002_deep | 2 | 32-512-512-6 | 0.048538 | 38.76% | — | [artifact](../../../processing/runs/joint_alignment_holdout_motiontrans/evaluations/motiontrans_epoch_002_deep/results.json) |
| joint_alignment_holdout_motiontrans / motiontrans_epoch_002_shallow | 2 | 32-512-6 | 0.051793 | 41.20% | — | [artifact](../../../processing/runs/joint_alignment_holdout_motiontrans/evaluations/motiontrans_epoch_002_shallow/results.json) |
| joint_alignment_holdout_motiontrans / motiontrans_epoch_003_deep | 3 | 32-512-512-6 | 0.049241 | 33.51% | — | [artifact](../../../processing/runs/joint_alignment_holdout_motiontrans/evaluations/motiontrans_epoch_003_deep/results.json) |
| joint_alignment_holdout_motiontrans / motiontrans_epoch_003_shallow | 3 | 32-512-6 | 0.052781 | 39.57% | — | [artifact](../../../processing/runs/joint_alignment_holdout_motiontrans/evaluations/motiontrans_epoch_003_shallow/results.json) |
| joint_alignment_holdout_motiontrans / motiontrans_epoch_004_deep | 4 | 32-512-512-6 | 0.046814 | 36.36% | — | [artifact](../../../processing/runs/joint_alignment_holdout_motiontrans/evaluations/motiontrans_epoch_004_deep/results.json) |
| joint_alignment_holdout_motiontrans / motiontrans_epoch_004_shallow | 4 | 32-512-6 | 0.051802 | 40.08% | — | [artifact](../../../processing/runs/joint_alignment_holdout_motiontrans/evaluations/motiontrans_epoch_004_shallow/results.json) |
| joint_alignment_holdout_motiontrans / motiontrans_epoch_005_deep | 5 | 32-512-512-6 | 0.047480 | 42.86% | — | [artifact](../../../processing/runs/joint_alignment_holdout_motiontrans/evaluations/motiontrans_epoch_005_deep/results.json) |
| joint_alignment_holdout_motiontrans / motiontrans_epoch_005_shallow | 5 | 32-512-6 | 0.052012 | 41.25% | — | [artifact](../../../processing/runs/joint_alignment_holdout_motiontrans/evaluations/motiontrans_epoch_005_shallow/results.json) |
| joint_alignment_holdout_motiontrans / motiontrans_epoch_006_deep | 6 | 32-512-512-6 | 0.047844 | 45.39% | — | [artifact](../../../processing/runs/joint_alignment_holdout_motiontrans/evaluations/motiontrans_epoch_006_deep/results.json) |
| joint_alignment_holdout_motiontrans / motiontrans_epoch_006_shallow | 6 | 32-512-6 | 0.052010 | 44.91% | — | [artifact](../../../processing/runs/joint_alignment_holdout_motiontrans/evaluations/motiontrans_epoch_006_shallow/results.json) |
| joint_alignment_holdout_motiontrans / motiontrans_epoch_007_deep | 7 | 32-512-512-6 | 0.046085 | 48.21% | — | [artifact](../../../processing/runs/joint_alignment_holdout_motiontrans/evaluations/motiontrans_epoch_007_deep/results.json) |
| joint_alignment_holdout_motiontrans / motiontrans_epoch_007_shallow | 7 | 32-512-6 | 0.052071 | 43.43% | — | [artifact](../../../processing/runs/joint_alignment_holdout_motiontrans/evaluations/motiontrans_epoch_007_shallow/results.json) |
| joint_alignment_holdout_motiontrans / motiontrans_epoch_008_deep | 8 | 32-512-512-6 | 0.047069 | 44.99% | — | [artifact](../../../processing/runs/joint_alignment_holdout_motiontrans/evaluations/motiontrans_epoch_008_deep/results.json) |
| joint_alignment_holdout_motiontrans / motiontrans_epoch_008_shallow | 8 | 32-512-6 | 0.052676 | 40.95% | — | [artifact](../../../processing/runs/joint_alignment_holdout_motiontrans/evaluations/motiontrans_epoch_008_shallow/results.json) |
| joint_alignment_holdout_motiontrans / motiontrans_epoch_009_deep | 9 | 32-512-512-6 | 0.048062 | 43.91% | — | [artifact](../../../processing/runs/joint_alignment_holdout_motiontrans/evaluations/motiontrans_epoch_009_deep/results.json) |
| joint_alignment_holdout_motiontrans / motiontrans_epoch_009_shallow | 9 | 32-512-6 | 0.054915 | 42.52% | — | [artifact](../../../processing/runs/joint_alignment_holdout_motiontrans/evaluations/motiontrans_epoch_009_shallow/results.json) |
| joint_alignment_holdout_motiontrans / motiontrans_epoch_010_deep | 10 | 32-512-512-6 | 0.048094 | 46.84% | — | [artifact](../../../processing/runs/joint_alignment_holdout_motiontrans/evaluations/motiontrans_epoch_010_deep/results.json) |
| joint_alignment_holdout_motiontrans / motiontrans_epoch_010_shallow | 10 | 32-512-6 | 0.052103 | 46.73% | — | [artifact](../../../processing/runs/joint_alignment_holdout_motiontrans/evaluations/motiontrans_epoch_010_shallow/results.json) |
| joint_alignment_holdout_motiontrans / motiontrans_epoch_020_deep | 20 | 32-512-512-6 | 0.049557 | 37.62% | — | [artifact](../../../processing/runs/joint_alignment_holdout_motiontrans/evaluations/motiontrans_epoch_020_deep/results.json) |
| joint_alignment_holdout_motiontrans / motiontrans_epoch_020_shallow | 20 | 32-512-6 | 0.054840 | 36.26% | — | [artifact](../../../processing/runs/joint_alignment_holdout_motiontrans/evaluations/motiontrans_epoch_020_shallow/results.json) |
| joint_alignment_holdout_motiontrans / motiontrans_epoch_030_deep | 30 | 32-512-512-6 | 0.050998 | 34.56% | — | [artifact](../../../processing/runs/joint_alignment_holdout_motiontrans/evaluations/motiontrans_epoch_030_deep/results.json) |
| joint_alignment_holdout_motiontrans / motiontrans_epoch_030_shallow | 30 | 32-512-6 | 0.056413 | 32.03% | — | [artifact](../../../processing/runs/joint_alignment_holdout_motiontrans/evaluations/motiontrans_epoch_030_shallow/results.json) |
| joint_alignment_holdout_motiontrans / motiontrans_epoch_040_deep | 40 | 32-512-512-6 | 0.051795 | 29.40% | — | [artifact](../../../processing/runs/joint_alignment_holdout_motiontrans/evaluations/motiontrans_epoch_040_deep/results.json) |
| joint_alignment_holdout_motiontrans / motiontrans_epoch_040_shallow | 40 | 32-512-6 | 0.056721 | 29.24% | — | [artifact](../../../processing/runs/joint_alignment_holdout_motiontrans/evaluations/motiontrans_epoch_040_shallow/results.json) |
| joint_alignment_holdout_motiontrans / motiontrans_epoch_050_deep | 50 | 32-512-512-6 | 0.050763 | 30.19% | — | [artifact](../../../processing/runs/joint_alignment_holdout_motiontrans/evaluations/motiontrans_epoch_050_deep/results.json) |
| joint_alignment_holdout_motiontrans / motiontrans_epoch_050_shallow | 50 | 32-512-6 | 0.057101 | 28.03% | — | [artifact](../../../processing/runs/joint_alignment_holdout_motiontrans/evaluations/motiontrans_epoch_050_shallow/results.json) |
| joint_alignment_holdout_motiontrans / motiontrans_epoch_060_deep | 60 | 32-512-512-6 | 0.050123 | 29.89% | — | [artifact](../../../processing/runs/joint_alignment_holdout_motiontrans/evaluations/motiontrans_epoch_060_deep/results.json) |
| joint_alignment_holdout_motiontrans / motiontrans_epoch_060_shallow | 60 | 32-512-6 | 0.056819 | 25.44% | — | [artifact](../../../processing/runs/joint_alignment_holdout_motiontrans/evaluations/motiontrans_epoch_060_shallow/results.json) |
| joint_alignment_holdout_motiontrans / motiontrans_epoch_070_deep | 70 | 32-512-512-6 | 0.050695 | 30.88% | — | [artifact](../../../processing/runs/joint_alignment_holdout_motiontrans/evaluations/motiontrans_epoch_070_deep/results.json) |
| joint_alignment_holdout_motiontrans / motiontrans_epoch_070_shallow | 70 | 32-512-6 | 0.057090 | 26.46% | — | [artifact](../../../processing/runs/joint_alignment_holdout_motiontrans/evaluations/motiontrans_epoch_070_shallow/results.json) |
| joint_alignment_holdout_motiontrans / motiontrans_epoch_080_deep | 80 | 32-512-512-6 | 0.051307 | 28.96% | — | [artifact](../../../processing/runs/joint_alignment_holdout_motiontrans/evaluations/motiontrans_epoch_080_deep/results.json) |
| joint_alignment_holdout_motiontrans / motiontrans_epoch_080_shallow | 80 | 32-512-6 | 0.057758 | 24.21% | — | [artifact](../../../processing/runs/joint_alignment_holdout_motiontrans/evaluations/motiontrans_epoch_080_shallow/results.json) |
| joint_alignment_holdout_motiontrans / motiontrans_epoch_090_deep | 90 | 32-512-512-6 | 0.051882 | 25.05% | — | [artifact](../../../processing/runs/joint_alignment_holdout_motiontrans/evaluations/motiontrans_epoch_090_deep/results.json) |
| joint_alignment_holdout_motiontrans / motiontrans_epoch_090_shallow | 90 | 32-512-6 | 0.057807 | 24.40% | — | [artifact](../../../processing/runs/joint_alignment_holdout_motiontrans/evaluations/motiontrans_epoch_090_shallow/results.json) |
| joint_alignment_holdout_motiontrans / motiontrans_epoch_100_deep | 100 | 32-512-512-6 | 0.051210 | 28.31% | — | [artifact](../../../processing/runs/joint_alignment_holdout_motiontrans/evaluations/motiontrans_epoch_100_deep/results.json) |
| joint_alignment_holdout_motiontrans / motiontrans_epoch_100_shallow | 100 | 32-512-6 | 0.057987 | 23.56% | — | [artifact](../../../processing/runs/joint_alignment_holdout_motiontrans/evaluations/motiontrans_epoch_100_shallow/results.json) |
| joint_all_alternating_fresh_8k_50ep / dataset_identity | 50 | 32-512-11 | — | — | 21.99% | [artifact](../../../processing/runs/joint_all_alternating_fresh_8k_50ep/evaluations/dataset_identity/results.json) |
| joint_all_alternating_fresh_8k_50ep / vla_touch | 50 | 32-512-3072 | 0.034348 | 10.79% | — | [artifact](../../../processing/runs/joint_all_alternating_fresh_8k_50ep/evaluations/vla_touch/results.json) |
| joint_all_holdout_motiontrans_fresh_8k_50ep / motiontrans | 50 | 32-512-6 | 0.058843 | 66.48% | — | [artifact](../../../processing/runs/joint_all_holdout_motiontrans_fresh_8k_50ep/evaluations/motiontrans/results.json) |
| joint_all_holdout_motiontrans_fresh_8k_50ep / motiontrans_large_decoder | 50 | 32-1024-1024-512-6 | 0.049438 | 69.29% | — | [artifact](../../../processing/runs/joint_all_holdout_motiontrans_fresh_8k_50ep/evaluations/motiontrans_large_decoder/results.json) |
| joint_all_sam_holdout_motiontrans_fresh_8k_50ep / dataset_identity | 50 | 32-512-11 | — | — | 95.55% | [artifact](../../../processing/runs/joint_all_sam_holdout_motiontrans_fresh_8k_50ep/evaluations/dataset_identity/results.json) |
| joint_all_sam_holdout_motiontrans_fresh_8k_50ep / motiontrans_epoch_003_1790974301642035852 | 3 | 32-512-6 | 0.063710 | 48.22% | — | [artifact](../../../processing/runs/joint_all_sam_holdout_motiontrans_fresh_8k_50ep/evaluations/motiontrans_epoch_003_1790974301642035852/results.json) |
| joint_all_sam_holdout_motiontrans_fresh_8k_50ep / motiontrans_final | 50 | 32-512-6 | 0.058810 | 65.20% | — | [artifact](../../../processing/runs/joint_all_sam_holdout_motiontrans_fresh_8k_50ep/evaluations/motiontrans_final/results.json) |
| motiontrans_direct_vae32_50ep / results.json | 50 | 32-512-6 | 0.026056 | 51.66% | — | [artifact](../../../processing/runs/motiontrans_direct_vae32_50ep/evaluation/results.json) |
| vla_touch_direct_vae32_50ep / results.json | 50 | 32-512-3072 | 0.010317 | 74.78% | — | [artifact](../../../processing/runs/vla_touch_direct_vae32_50ep/evaluation/results.json) |

Other historical uncertainty reports with different schemas:

- [CVAE versus pose VAE: CVAE](../../../processing/evaluations/cvae_vs_pose_vae_uncertainty/cvae.json).
- [CVAE versus pose VAE: pose VAE](../../../processing/evaluations/cvae_vs_pose_vae_uncertainty/pose_vae.json).
- [32D VAE unseen comparison](../../../processing/evaluations/latent32_vae_vs_cvae_uncertainty/vae.json).
- [32D CVAE unseen comparison](../../../processing/evaluations/latent32_vae_vs_cvae_uncertainty/cvae.json).
- [CVAE case diagnosis](../../../processing/evaluations/latent32_vae_vs_cvae_uncertainty/cvae_case_diagnosis.json).

## Appendix D. Run state and supporting documents

| Experiment/artifact | State at snapshot | Interpretation |
| --- | --- | --- |
| Original EgoTouch visual variants | Historical partial/completed runs; logs retained | See Appendix B |
| Direct-output ordinary/SAM joint baselines | 50 completed epochs each | Historical output/data protocol |
| Online GRL | Stopped after 15 epochs | No final automatic evaluation |
| Alternating adversarial | 50 completed epochs | Historical VLA target protocol |
| EgoTouch AE v2/v3 | Stopped/superseded by sigmoid v4 | Stale screening status files remain |
| EgoTouch AE v4 | Complete: 28 trials | Three-seed repetition only for two AE finalists |
| Sensor AE v1 | Superseded | Reduced optical targets; numeric AEs excluded |
| Native sensor AE v2 | Complete: 16 candidates / 8 sources | One finalist/source at 100 epochs |
| Sensor-code cache | Complete: 9 AE sources | 4,893,180 frames |
| Cached-code non-FGW 32D | Stopped at 100 completed epochs | Both 1–10 and 10–100 transfer sweeps complete |
| Broad FGW study | Stopped for smaller two-placement test | Full variant/seed/calibration matrix not executed |
| 32D visual FGW | Completed epoch 30; controller stopped | Completed evaluation at epoch 12 only |
| 32D tactile-first FGW | Stopped after epoch 21 | Epoch-20 transfer complete; classifier stopped |
| Matched 16D control | Running; latest completed epoch 15 | Target 30; epoch-10 transfer complete |
| Matched 16D FGW | Running; latest completed epoch 13 | Target 30; epoch-10 transfer complete |
| Trajectory matching / episode context / meta-learning | Discussed, not implemented | No result claims |

Archived or stale `status.json` fields do not imply a process is active. The active pair's controller/worker processes were checked for this snapshot; stopped historical searches are identified from the later session instructions and preserved stop/migration artifacts.

Supporting documents: [earlier project analysis](PROJECT_ANALYSIS.md), [training guide](TRAINING_GUIDE.md), [historical joint training](JOINT_ALL_TRAINING.md), [SAM](SAM_JOINT_TRAINING.md), [adversarial training](ADVERSARIAL_TRAINING.md), [alternating adversarial training](ALTERNATING_ADVERSARIAL_TRAINING.md), [MotionTrans baseline](MOTIONTRANS_BASELINE_TRAINING.md), [large transfer decoder](MOTIONTRANS_LARGE_DECODER.md), [sensor pretraining](SENSOR_AUTOENCODER_PRETRAINING.md), [FGW study history](FGW_STUDY.md), [FGW equations](FGW_ALIGNMENT_AND_MIXING.md), [16D pair](TRAINING_16D_PAIR.md).

Several older guides describe runs as active or planned beyond their eventual stopping point. This consolidated snapshot takes precedence for reported completion status; linked configuration files retain the original experiment history.
