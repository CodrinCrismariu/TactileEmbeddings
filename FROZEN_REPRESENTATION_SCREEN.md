# Frozen representation screen and epoch-22 stopping decision

**Completed October 15, 2026:** All 72 tuning fits and 24 refits finished in 34.8 minutes, including frozen extraction. Mean MotionTrans MAE over three head seeds: temporal DINO 0.031462, single-frame DINO 0.033007, reconstructed epoch-0 latent 0.046751, epoch-7 latent 0.043769, epoch-100 latent 0.048189, reconstructed epoch-0 wide trunk 0.035368, epoch-7 wide trunk 0.035325, epoch-100 wide trunk 0.035938. Epoch-100 wide features are not clearly worse than epoch-7 wide features (paired interval includes zero), while the 32D latent deteriorates clearly. This points toward a bottleneck/projection issue under this readout; head capacity/input dimension remains a comparison limitation. All representation training remains stopped. See [full results](processing/evaluations/motiontrans_representation_screen_v1/analysis.md).

The four continuation arms stopped after their completed epoch-22 MotionTrans evaluations on October 15, 2026. No continuation to epoch 27 is scheduled. The existing model and optimizer checkpoints remain preserved.

## Epoch-22 results

| Arm | Epoch 12 MAE | Epoch 17 MAE | Epoch 22 MAE | Epoch 22 nominal 90% coverage |
|---|---:|---:|---:|---:|
| Control | 0.044714 | 0.045679 | 0.047140 | 32.64% |
| Source stopping | 0.045406 | 0.046104 | 0.046555 | 44.94% |
| KL-off retention | 0.043810 | 0.045839 | 0.046653 | 4.72% |
| Combined | 0.044493 | 0.045369 | 0.044933 | 7.28% |

The original epoch-7 three-hidden-layer readout scored 0.044773. Combined is almost flat relative to that point (+0.36%), whereas the control deteriorates by 5.29%. Its difference from control at epoch 22 is −0.002207, with paired episode-bootstrap 95% CI [−0.004485, +0.000011]. This is suggestive, but the interval still includes zero and the comparison uses one representation-training seed. KL-off uncertainty remains severely overconfident. The user's stop at 22 is honored despite the nominal combined result.

## Representations in the revised screen

| Representation | Dimensions | Source training |
|---|---:|---|
| DINO patch/camera mean per frame | 768 | Frozen pretrained backbone only |
| DINO current frame + clip mean + frame difference | 2304 | Same eight input frames and available camera set |
| Reconstructed epoch-0 latent mean | 32 | No source optimization; audited initialization reconstruction |
| Original epoch-7 latent mean | 32 | Seven epochs; matched reference |
| Original epoch-100 latent mean | 32 | One hundred epochs; bottleneck reference |
| Reconstructed epoch-0 trunk output | 768 | No source optimization; audited initialization reconstruction |
| Original epoch-7 trunk output | 768 | Seven epochs |
| Original epoch-100 trunk output | 768 | One hundred epochs |

The two trained 32D references help distinguish a failure of the bottleneck/projection from a loss extending into the trunk. There is no new encoder or backbone training.

## Inputs, pooling and temporal context

All conditions receive the same canonical eight-frame clips, native target readings, and available camera slots. DINO is averaged over spatial patch tokens and valid cameras per frame. Its single-frame baseline preserves the original frame/target correspondence. Its temporal baseline concatenates each frame vector, the mean of all eight frame vectors, and the difference from its preceding frame. The first difference is zero; it reads no frame outside the clip. This version is bidirectional through its clip mean, consistent with the trained temporal trunk's access to the whole clip.

Trunk features are exactly `visual.encode_context`, the 768D vector feeding the mean and log-variance projections. Both DINO and the trunk use arithmetic spatial mean pooling. The trunk's learned spatial/temporal processing, normalization, image_pool transformation, and cross-view attention remain part of its computation. DINO does not borrow camera weights from the trained model. Thus learned camera fusion remains an architectural difference rather than an additional trained component inserted into the raw-DINO baseline. No tactile target is pooled, resized, reordered, or swapped.

## Historical initialization audit

No initial checkpoint or initial-weight hash was saved for the original 32D run. The saved seed-45 initial hash belongs to the later 16D pair; the continuation initialization hashes describe their epoch-7 parent weights.

The original Gaussian alignment constructor was located in an archive and its file hash exactly matches the original 32D checkpoint's recorded model-code hash. Replaying seed 45, constructing DINO, and then constructing the archived Gaussian model **exactly reproduced the recorded 16D full-model hash**:

`89b6b4bfc87e32977246967c7cf304927bde977c618f6f3938d3d0630f467b19`

The identical pre-model RNG state and archived constructor were then used for the 32D architecture. Its shared trunk tensors are exactly equal to the verified 16D initial trunk; latent heads and adapters have their appropriate 32D shapes. The reconstructed 32D model hash is:

`f43818138e450fb6b7783a4838336a0d3f5134e551db85eb498b593928356814`

This is stronger than an arbitrary fresh draw, but an original 32D checkpoint/hash is unavailable for independent verification of the full historical state. Results are explicitly labeled **reconstructed epoch 0**, not a recovered saved checkpoint. See `initialization_audit.json` and `reconstructed_epoch000.pt` in the screen output.

## Standardization, head tuning and seeds

Only MotionTrans readout heads are fitted. Every representation uses `input_dim → 512 → 512 → 512 → 6`, with the same GELU/dropout, optimizer type, target regression loss and episode partitions. Different input dimensions necessarily change the first-layer parameter count, which is reported.

For each representation:

1. Fix the episode-level fit/tune partition using split seed 45: 16,736 fit clips and 1,877 tune clips, drawn from 18,613 training clips. The 1,972 validation clips belong to 167 separate episodes.
2. Compute per-coordinate mean/std **only from fit episodes**. Apply that z-score, with std floored at 1e-6, to fit, tune, full-training refit, and validation. Keep the same statistics at refit; tune/validation features do not contribute statistics.
3. Fit readout heads at learning rates **0.0003, 0.001, 0.003** for head seeds **45, 46, 47**. Use at most 100 epochs, patience 10, AdamW weight decay 0.0001. All feature vectors are deterministic means; no latent noise is added in any condition.
4. Select one LR per representation using the mean best **tune loss across the three seeds**. Each selected seed retains its own tune-selected duration. Validation is not accessed for LR/duration selection.
5. Refit all three selected heads on every training episode, using the fit-only z-score and selected durations. Then evaluate on validation.

This is nine small tuning fits and three refits per representation, or **96 head fits total** over eight representations. It is not a multi-seed encoder-training study. Trial/refit results are individually resumable. The matched epoch-7 reference is re-evaluated under this protocol; earlier MAE 0.044773 used a different head-fitting protocol and should not be substituted as the new reference.

Report mean MAE and sample standard deviation over the three head seeds, individual seed metrics and durations, selected LR, per-channel errors and within-episode channel correlations. A 5,000-draw episode bootstrap averages per-episode absolute errors across the three heads, then resamples matched episodes. Its confidence interval captures episode variation conditional on those fitted heads; head-fit SD is reported separately. This is a point-prediction screen with no uncertainty-distribution or calibration claim.

## Cache and execution

The old caches contain only 32D means/log-variances, so they cannot recover the wider DINO/context tensors. One shared frozen extraction pass runs DINO once per input batch and all three frozen trunks. It caches the pooled vectors, temporal summary and latent means, with canonical window indices, episode IDs, targets, masks, checkpoint hashes and code hashes. The eight float32 feature arrays require about **3.36 GiB**, plus small target metadata and the reconstructed initial checkpoint. Full DINO patch banks are not stored.

Artifacts:

- `processing/runs/continuation_32d_v1/stop_status.json`: all four stopped at 22.
- `processing/runs/continuation_32d_v1/evaluations/epoch_022/analysis.md`: completed continuation results.
- `processing/runs/continuation_32d_v1/finish_status.json`: evaluation-chain status.
- `processing/evaluations/motiontrans_representation_screen_v1/status.json`: current screen stage.
- Screen output `summary.json` and `analysis.md`: aggregate results and paired comparisons.
- Per-representation `probes/NAME/tuning_trials/`, `normalization.pt`, `seed_*/head.pt`, and `seed_*/results.json`.

Implementation is in `scripts/finish_continuation_and_probe.py`, `scripts/evaluate_representation_screen.py`, and `src/tactilelatent/representation_probe.py`. Twenty-one targeted checks passed for pooling and temporal summaries, target/order preservation, fit-only standardization, head fitting, held-out episode separation, tune-based LR selection, seed aggregation, cached features, and existing transfer compatibility.

## Interpretation

DINO or an untrained trunk matching epoch 7 establishes no measured benefit from source tactile representation training under this target/readout protocol. It does not prove that the trained model contains zero additional transferable information or that the result holds for every future target.

If epoch-100 wide features transfer well while its 32D latent does not, the compression/projection becomes a stronger suspect. If wide and narrow features both degrade, the change extends into the trunk. The DINO temporal-versus-single-frame comparison measures how much this cheap temporal summary alone adds. The reconstructed epoch-0 wide output checks whether random spatial/temporal mixing changes the readout before any source optimization.

MotionTrans validation has informed experimental design and remains a development benchmark. No fresh final-test claim is made. Source coverage changes, ego-video self-supervision, and DINO fine-tuning are possible follow-ups; none is launched automatically.
