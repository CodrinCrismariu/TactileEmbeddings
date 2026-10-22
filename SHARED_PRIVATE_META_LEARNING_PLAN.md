# Shared/private tactile representation: implementation plan

**Superseding implementation — October 13, 2026:** The user explicitly removed the 4D static episode-context proposal. Relational distillation, teacher-based trajectory GW matching, and rotating-source meta-learning are now implemented as optional objectives on the existing 16D architecture. No private branch is added. See [the current implementation and commands](INVARIANCE_TRAINING.md). The design below is retained as historical rationale; its shared/private architecture and “not implemented” status no longer describe the current code.

**Status:** design and code investigation completed; the architecture and objectives below are not yet implemented or trained. This is a staged, single-seed experiment, not a new sweep.

The user requested stopping the current training and investigating rotating-source meta-learning, with relational distillation and tactile-based trajectory matching tried first. Both 16D runs were stopped on **October 12, 2026, at 11:48 PM EDT**. The non-FGW run saved through epoch 16; the visual-FGW run saved through epoch 14. Checkpoints, optimizer states, and RNG states are preserved. See the [stop record](processing/runs/joint_16d_pair_v1/stop_status.json) and [project history and results](PROJECT_HISTORY_AND_RESULTS.md).

## 1. Objective and experiment order

The representation succeeds if a new sensor's prediction head can learn from few labeled episodes while the shared encoder remains frozen. Source reconstruction is a preservation check; improving it is not the main objective. Dataset-classifier accuracy and effective rank are diagnostics, not optimization targets or selection rules.

Use this sequence:

1. Implement the shared/private architecture and a matched control.
2. Add within-clip relational distillation to the shared branch.
3. If shared latents track tactile dynamics better without damaging transfer, test cross-sensor trajectory matching using frozen tactile teachers.
4. If the earlier stages help, add rotating-source, head-adaptation meta-learning.

Keep one seed, the existing 16D shared latent, the audited native targets, pinned frozen sensor AEs, chronological pairing, and existing train-only cached-code statistics. Exclude MotionTrans from representation training, trajectory banks, normalization fitting, task sampling, and hyperparameter selection. MotionTrans is an established development transfer benchmark, not a pristine final test set.

## 2. Shared/private architecture

Proposed first version:

```text
Current ego clip → frozen DINO → trainable spatial/temporal trunk
                                      ↓
                           shared Gaussian latent, 16D/frame
                                      ↓
                        [shared z ; private episode context]
                                      ↓
                         existing sensor adapter, expanded
                                      ↓
                      sensor AE code / direct numeric output

Fixed ego prefix of this episode → frozen DINO → context projector
                                                     ↓
                                   private static context, 4D/episode

Relational loss, trajectory matching, and new-sensor heads use shared z only.
Frozen native-resolution decoders remain available for reconstruction evaluation.
```

**The proposed 4D private width is an initial engineering choice, not an established optimum.** It keeps the earlier static episode-context idea distinct from a second unrestricted per-frame latent. A time-varying private branch could carry the whole tactile interaction and defeat the purpose of the shared branch; it is not the default here.

Compute the context from the first valid eight-frame ego clip in the episode using the existing chronological stride. Pool valid frozen DINO features over its frames and available cameras, then use a small source-specific `768 → 32 → 4` GELU projector. Broadcast this context to all prediction frames from that episode. Do not give the context encoder tactile measurements, learned episode-ID embeddings, or future/full-episode frames. The initial prefix may contain contact, so static context is not guaranteed to represent appearance alone. This is an offline reconstruction experiment; a streaming version would need an explicit startup policy.

Cache the fixed prefix's frozen DINO features, not its trainable 4D output. This avoids stale context after parameter updates and avoids repeatedly decoding the same prefix video. Cache keys must include the episode, exact frame indices, camera mapping, backbone weights, and preprocessing provenance. Missing prefix cameras use the same masks as ordinary ego input.

Retain sensor-code adapters with hidden widths `(128,128)`, expanding their input from 16 to 20. Keep direct numeric heads simple. Drop the complete private vector with proposed probability 0.25 during source training so the shared route remains useful. Apply identical private dropout and sampling in the control. A low-dimensional static branch and dropout limit information bypass; neither guarantees semantic disentanglement.

For source reconstruction, use shared plus private. For MotionTrans and temporary meta heads, export only the 16D shared mean/log variance. No private vector is fitted from MotionTrans labels. Preserve the Gaussian sampling and KL objective; this architecture does not solve the existing uncertainty calibration failures.

### Initialization and fair comparison

Use the saved non-FGW **epoch-10 16D checkpoint** as the common starting point for the two first experiments. It has an existing transfer evaluation and is earlier than the newly stopped epoch-16 checkpoint. This is a proposed warm-start comparison, not training from scratch.

Copy the shared trunk and Gaussian heads exactly. In each expanded adapter's first linear layer, copy the old 16 input columns and initialize the four private columns to zero. Copy the bias and remaining layers. Apply the equivalent migration to numeric heads. This preserves the old model's initial predictions, even with a nonzero private context. Check this numerically in evaluation mode.

Use a new model type and explicit migration command. Start a new optimizer for both conditions; do not treat an expanded adapter as an ordinary exact optimizer resume. Preserve the parent checkpoint hash and old runs. Keep all subsequent schedule, sample order, augmentation, optimizer/SAM settings, and KL settings matched between conditions. Record parent epoch and continuation epoch separately.

The existing non-FGW run is useful historical context but is not the matched control for an architecture with added private capacity.

## 3. Option 1: relational distillation

Use all valid timesteps of each existing eight-frame clip. Let `c_t` be the frozen, train-standardized sensor AE code and `mu_t` the predicted shared latent mean. For direct numeric sources, use valid numeric target vectors with train-only scaling; do not invent numeric autoencoders.

For valid off-diagonal pairs:

```text
Dc[i,j] = ||c_i - c_j||₂
Dz[i,j] = ||mu_i - mu_j||₂
qc      = RMS(Dc over valid off-diagonal pairs)
qz      = RMS(Dz over valid off-diagonal pairs)

Lrel = mean((Dz / max(qz, epsilon_z)
             - Dc / max(qc, tau_sensor))²)
```

Specify RMS normalization explicitly; do not accidentally sum losses over a variable number of valid frames or code dimensions. Exclude diagonal entries, missing targets, and clips with fewer than two valid timesteps. For partially observed numeric outputs, form distances over a consistent valid channel set; skip clips without one. Compute in float32 even under mixed precision.

**Near-static clips:** a floor avoids numerical blowup but does not by itself distinguish noise from real small changes. Initially gate the relational term off when tactile trajectory magnitude is below the estimated sensor floor; retain the ordinary code/numeric loss. Log how often this happens for every source. Do not interpret a stationary tactile signal as no contact. Do not force normalized visual distances to match zero for constant clips: normalization can make that loss insensitive to latent scale.

### Estimate floors in the correct units

An AE's native reconstruction MAE cannot directly serve as a distance threshold in standardized code space. Estimate a train-only code-space consistency proxy on a small episode-balanced sample:

```text
c        = E(y)
c_round  = E(D(c))
r        = standardize(c_round) - standardize(c)
```

Use RMS residual magnitude across the sample to set a documented initial floor in code-distance units, and inspect sensitivity on source-only diagnostics before a long run. This round-trip residual measures teacher consistency, **not calibrated measurement noise**. In particular, systematic reconstruction error and sensor noise are different. Store the sampled episodes, estimator, statistics, and teacher hashes in a separate artifact.

For numeric sources, there is no round-trip AE estimate. Use a separately identified train-only robust temporal-difference proxy and inspect it for suppression of real slow changes. If the proxy cannot be justified, retain those sources' original prediction loss and omit their relational term in the first test, recording that choice explicitly.

Optical targets stay at native resolution during the one-time round-trip audit. Ordinary relational training reads cached codes; no decoder pass is added to every training step.

### Training and diagnostics

Proposed starting loss: existing prediction loss + existing KL + `0.1 × Lrel`, ramping the relational coefficient over two continuation epochs. Treat 0.1 as a pilot value; inspect actual shared-trunk gradient contributions before committing the run. Keep the control's relational coefficient zero.

Normalization removes scale information and admits independent rotations/translations across sources. Consequently this objective can improve tactile dynamics without aligning cross-sensor coordinates. Monitor shared variance, within-source effective rank, native reconstruction, and shared-only transfer. If collapse appears, use a train-only warm-parent variance reference for a spread floor, applied equally to both conditions; do not reuse the old FGW floor measured at random initialization or require every source to fill all 16 dimensions.

## 4. Option 2: tactile-based trajectory matching

Construct episode-balanced clip banks from frozen tactile codes, not visual latents. Describe each clip with its normalized temporal distance matrix, validity mask, and the magnitude/reliability diagnostics from Option 1. Different sensor code widths are compatible because matching compares internal relations rather than code coordinates.

Prefer a temporally constrained, smoothed Gromov-DTW implementation for the first chronological matcher. Plain GW can permute or reverse time; it does not inherently enforce temporal order. Gromov-DTW permits monotone timing differences between trajectories in incomparable spaces. See [Cohen et al., AISTATS 2021](https://proceedings.mlr.press/v130/cohen21a.html).

Implementation sequence:

1. Validate synthetic trajectories with different widths, speeds, missing overlap, reversed order, and symmetric ambiguities.
2. Build small banks from source training episodes; compare a bounded number of candidate cross-source clip pairs instead of all dataset pairs.
3. Compute soft temporal couplings from the frozen tactile descriptors. Hold them fixed during the corresponding visual update; persist their provenance.
4. Normalize coupling rows by the mass they receive when transporting latent targets. Unsupported rows receive no alignment loss.
5. Pull the matched **shared** latent trajectories toward detached transported targets, symmetrically. Preserve ordinary source supervision and Option 1.
6. Inspect correspondence stability under resampling/perturbations and temporal consistency. Reject poor-fit and ambiguous pairs rather than forcing universal matching.

Entropy is controlled partly by the solver temperature; it is not a calibrated probability that a pair is physically correct. Use teacher-only fit and stability diagnostics to define a bounded reliability weight, validate it on synthetic correspondences, and report it as a heuristic. Never derive matching confidence from the visual latents being aligned.

Distance trajectories also have genuine ambiguities: two unrelated interactions can have the same temporal distance pattern. Fixed teachers remove the circular visual matching problem, but do not establish physical equivalence. Keep a rejection option and report accepted matching coverage by source.

## 5. Option 3: rotating-source head-adaptation meta-learning

Use head-only inner adaptation in the style of [ANIL, Raghu et al., ICLR 2020](https://arxiv.org/abs/1909.09157). The encoder is fixed during each inner fit and updated by the outer query loss. This differs from simply training an ordinary head on cached, detached latents and from the original [MLDG domain-generalization objective](https://ojs.aaai.org/index.php/AAAI/article/view/11596).

### One outer task, exactly

1. Rotate deterministically through the 11 source datasets, designating one source `s` as pseudo-held-out for the task. No MotionTrans task is allowed.
2. Sample support and query from disjoint **whole episodes** of `s`, with object/session grouping where metadata supports it. Proposed pilot: four support episodes and four query episodes, one eight-frame clip per episode. Do not randomly split neighboring frames into support/query.
3. Compute shared Gaussian parameters for support and query with gradients retained to the shared encoder. Neither set gets a private input for this task.
4. Initialize a fresh temporary head `16 → 512 → K_s`; do not copy the persistent source adapter. Use linear outputs for standardized AE codes, sigmoid for the existing bounded numeric targets. A common sigmoid probe builder is incorrect for standardized codes.
5. Fit only temporary head parameters for five differentiable SGD steps on support. Choose inner learning rate using source-only pilot tasks. For the first gradient check use deterministic means; in training preserve the existing mean/sample supervision convention with reproducible draws.
6. Evaluate the adapted head on query episodes. Backpropagate query loss **through all five support updates** into the shared encoder, without applying inner updates to the encoder. Keep the support gradient graph (`create_graph=True` or an equivalent functional implementation).
7. Combine this query objective with the established source training objective sampled from other sources. The selected source's persistent adapter and private projector are not used by this task. Average query errors per valid target element, then balance tasks equally by source rather than by frame count or output dimension.
8. Take one outer update; discard the temporary head. Rotate the pseudo-held-out source next time. Log before/after adaptation support and query losses and the encoder gradient norm.

For task-local few-shot realism, fit any additional target normalizer on support alone and apply it unchanged to query. Existing teacher/cache train statistics remain fixed and must be disclosed as source-training information. The frozen AE teacher is assumed available for each source; the method is not meta-learning a brand-new optical decoder from four episodes.

The held-out source is excluded **for that task**, not throughout the experiment: it has been seen in other tasks and supervised updates. The purpose is to learn a representation that supports rapid new-head fitting. MotionTrans remains the globally excluded representation-training dataset.

Source-specific static private branches receive ordinary reconstruction gradients; the meta head never reads their values. The shared trunk can still contain domain identity. Success means better adaptation, not a guarantee of statistical independence from sensor identity.

### Compute and replay

Do not re-run DINO or the shared trunk for every inner head update. Compute support/query features once per task; reuse them with the graph intact. Five inner steps then cost five small-head updates plus their backward dependencies, not five full encoder training steps. Memory for the retained encoder graph and second-order derivatives still needs a measured benchmark.

The current `frozen_probe.fit_head()` optimizes detached cached latents with a mutating optimizer and is appropriate for evaluation, not this meta-training loop. Implement functional fast head weights separately. Do not silently detach support features or use first-order gradients while claiming the exact unrolled objective.

If SAM remains enabled, replay the complete task with identical episodes, temporary-head initialization, dropout masks, and latent draws under the perturbed outer weights. Recompute adaptation; reusing the old adapted head would optimize a different objective. Benchmark this cost before choosing the run budget. Any optimizer simplification requires an equally changed control.

## 6. Concrete code changes

| File/component | Required change |
| --- | --- |
| `src/tactilelatent/model.py` | Preserve the existing `encode()` contract; expose intermediate context only if needed without changing old checkpoint behavior. |
| New shared/private model module | Add prefix-context projectors, expanded source heads, explicit shared-only encoding, new model type, and tested checkpoint migration. |
| Dataset/collator and prefix cache | Return stable episode IDs and prefix references; preserve masks, native targets, camera roles, and chronological indices. |
| New relational-loss module | Masked pairwise geometry, source floor artifact, static-clip gating, and per-source diagnostics. |
| New trajectory-matching module | Frozen teacher descriptors, temporal couplings, rejection/reliability diagnostics, and detached transport targets. |
| New meta-task sampler and trainer | Source rotation, disjoint episode support/query, functional differentiable head fitting, outer objective, and SAM replay. |
| `src/tactilelatent/frozen_probe.py` | Recognize the new model and export shared latents only; retain the old evaluation protocol for existing checkpoints. |
| `src/tactilelatent/joint_viewer.py` | Load the new model type and required ego-prefix context for source reconstruction; preserve old model/epoch selection. |
| Checkpoint-sweep preflight | Accept the new model type explicitly and verify shared width and representation-training holdout. |

Keep frozen teacher IDs, cache hashes, floor statistics, episode/task split manifests, prefix provenance, and shared/private dimensions in checkpoints/run metadata. Save task RNG state for exact continuation. Never reinterpret an old checkpoint as the new architecture based only on its filename.

## 7. Verification before any long run

- Check migrated source predictions match the parent before training, and old checkpoints still load.
- Verify relational masks, constant/near-constant clips, unequal valid lengths, finite gradients, and numeric/code units. Test known rotations/scales to demonstrate intended relational invariances and ambiguities.
- Verify static context is identical for all clips of an episode at fixed weights, uses only the recorded ego prefix, and cannot enter shared-only probe inputs.
- Finite-difference a tiny two-step meta task to verify the outer encoder gradient, including its path through support adaptation. A nonzero query gradient alone does not prove this path is correct.
- Verify support/query episode and grouping disjointness, and no query labels enter inner updates or normalization fits.
- Verify numeric head activation versus unbounded code output, source-balanced loss reduction, and that temporary heads are freshly initialized and discarded.
- Verify MotionTrans exclusion from every training artifact and teacher-based matcher; preserve native shapes, canonical camera roles, and frame mappings.
- Verify stochastic/SAM replay and resumable task order. Exercise viewer and frozen-probe loading for both old and new model types.

## 8. Small first experiment and decision gates

First run only **two conditions**, one seed: shared/private control and shared/private + relational distillation. Use the same migrated epoch-10 parent, matched sampling, and ten continuation epochs each. Save every epoch. No multi-seed matrix and no automatic launch of later stages.

Before scheduling, benchmark ordinary and relational steps with real source batches and the prefix cache. Record GPU memory, cache bytes, clips/second, and the extra round-trip-audit cost. Relational distance computation is only an 8×8 matrix per clip; prefix I/O and model execution are likely more consequential, but measure rather than promise a speedup. Existing pair-run timings are not a reliable meta-learning ETA.

For comparison, report source-validation relational error, reconstruction preservation, within-episode dynamics, and shared-latent spread. Evaluate both finished conditions using the existing frozen MotionTrans protocol and both `16 → 512 → 6` and `16 → 512 → 512 → 6` heads, with paired episode-bootstrap intervals. Add episode-level 10%, 25%, and 100% transfer-training subsets when assessing few-shot claims, using fixed nested subsets across conditions.

Better relational fit alone is not grounds for launching the expensive meta-learning stage. Use source-only pseudo-held-out adaptation diagnostics for method selection and report MotionTrans as the established development benchmark. Reject changes that destroy useful source reconstruction; do not require source reconstruction improvement. Preserve uncertainty evaluation unchanged and report its current coverage limitations separately.

If Option 1 fails to preserve tactile information in the shared branch, investigate private bypass, floor gating, temporal sampling, and loss gradients before adding cross-source matching. If Options 1–2 help, implement and benchmark one meta-learning continuation against an equally budgeted matched control. Meta-learning remains a follow-up, as requested.
