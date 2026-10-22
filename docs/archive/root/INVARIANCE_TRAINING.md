> Historical record, relocated on October 21, 2026. Original path: `INVARIANCE_TRAINING.md`.
> Results, proposals, and operational status below belong to their recorded experiment/date;
> use the [maintained project state](../../project_state.md) for the audited active workflow.

# Invariance objectives on the existing 16D model

Implemented October 13, 2026. **No static episode context, private latent branch, or decoder change is included.** The latest user instruction supersedes the shared/private architecture in the earlier [design document](SHARED_PRIVATE_META_LEARNING_PLAN.md).

**Deeper-head evaluation — October 14, 2026:** All four completed epoch-20 encoders were evaluated with three/four hidden layers of width 512 using the existing cached MotionTrans latents. The evaluator now supports explicit `--heads shallow deep deeper deepest`; its default remains the original shallow/deep pair. The new reports are separate from the earlier pair. See [results and interpretation](../../../processing/runs/invariance_16d_v1/evaluations/epoch_020_deeper_heads/analysis.md). Trajectory remains paused.

**Epoch-20 evaluation — October 14, 2026:** Control, relational, meta-learning, and matched meta control completed epoch 20 successfully. At the user's request, the remaining trajectory worker and its controller were paused with SIGSTOP, retaining the in-progress epoch-18 matching state and the saved epoch-17 checkpoint. All four completed models are evaluated on the full MotionTrans split with both standard heads. Frozen DINO extraction is shared across all four. See [evaluation status](../../../processing/runs/invariance_16d_v1/evaluations/epoch_020/status.json). The previous epoch-15 pause metadata is archived as [pause_status_epoch15.json](../../../processing/runs/invariance_16d_v1/pause_status_epoch15.json); current [pause metadata](../../../processing/runs/invariance_16d_v1/pause_status.json) refers to trajectory only. No automatic resume is scheduled.

**Launch update — October 13, 2026:** The user subsequently requested all conditions. All five passed preflight and were submitted to a detached, resumable queue, with control and relational starting first and a limit of two simultaneous workers. Meta, matched meta control, and trajectory follow automatically. See the [live queue status](../../../processing/runs/invariance_16d_v1/status.json) and [frozen run plan](../../../processing/runs/invariance_16d_v1/plan.json). The implementation-only observations below predate this launch.

**Pause and evaluation — October 13, 2026:** At the user's request, the controller and both active worker process groups were paused with SIGSTOP. Control has saved through epoch 17 and relational through epoch 15; current partial-epoch state remains in memory. The other three runs remain queued. A matched MotionTrans evaluation uses epoch 15 for both models, both existing transfer heads, and the full established train/validation split. Shared frozen DINO extraction writes the original probe-cache format, with a parity test. See [pause metadata](../../../processing/runs/invariance_16d_v1/pause_status.json) and [evaluation status](../../../processing/runs/invariance_16d_v1/evaluations/epoch_015/status.json). No automatic training resume is scheduled.

**Resume — October 13, 2026, 8:17 PM EDT:** The user explicitly requested resuming training. Both worker process groups and the controller received SIGCONT, continuing from their retained partial-epoch and optimizer/RNG state. The existing two-worker queue remains in effect. See [resume metadata](../../../processing/runs/invariance_16d_v1/resume_status.json) and [live status](../../../processing/runs/invariance_16d_v1/status.json).

The inference path remains:

```text
Ego images → frozen DINO → spatial/temporal encoder → 16D Gaussian latent
                                                        ↓
                                      existing sensor-specific adapter
                                                        ↓
                                      cached AE code / numeric output
                                                        ↓
                                  frozen native decoder when applicable
```

The implementation adds optional training objectives. Existing viewer checkpoints and MotionTrans/classifier probes still receive the same model type and shared latent. No transport solver or temporary meta head is required for ordinary inference.

## Implemented options

| Configuration | Objective |
| --- | --- |
| [Control](../../../legacy/v1/configs/invariance_16d_control.json) | Existing code/numeric supervision and Gaussian KL |
| [Relational](../../../legacy/v1/configs/invariance_16d_relational.json) | Control + normalized within-clip tactile/latent distance matching |
| [Trajectory](../../../legacy/v1/configs/invariance_16d_trajectory.json) | Relational + frozen-tactile cross-source trajectory GW matching |
| [Meta](../../../legacy/v1/configs/invariance_16d_meta.json) | Relational + rotating-source differentiable head adaptation |
| [Meta control](../../../legacy/v1/configs/invariance_16d_meta_control.json) | Same tasks, excluded-source ordinary batches, and compute as meta, with zero query-loss weight |

These are **individually selectable configurations, not a queued sweep**. Start by comparing control and relational. Trajectory and meta are implemented follow-ups; no conditional training launcher automatically starts them. The supplied meta condition does not also enable trajectory loss, so its contribution can be measured separately. Both can be enabled explicitly in one configuration later.

All configs use one seed (45), the same pinned non-FGW 16D epoch-10 weights, fresh matched optimizers, and ten continuation epochs, numbered 11–20. The original completed/stopped runs remain untouched. No adapter migration is necessary because the architecture is unchanged. Exact resume is supported at saved epoch boundaries, including optimizer, scheduler, RNG, and meta-task position.

## Relational loss and floor audit

For each informative eight-frame clip, compare off-diagonal Euclidean distance matrices of standardized frozen tactile codes and shared latent means. Normalize each matrix by its off-diagonal RMS. Average within each clip, then equally across informative clips. Missing frames and diagonal entries are excluded. Teacher targets are detached, and distance arithmetic is float32.

The initial coefficient is 0.1, ramped over two continuation epochs. Clips below a per-source code-distance floor keep ordinary supervision but receive no relational term. Numeric sources retain their existing losses; no numeric AE or unvalidated numeric noise estimator was introduced.

The [floor audit](../../../processing/invariance/relational_floors.json) uses 32 training episodes per AE source and up to four sampled frames per episode:

```text
raw code c = existing cached E(y)
round-trip code = E(D(c))
residual = (round-trip code - c) / training code standard deviation
distance floor = sqrt(2) × RMS residual vector norm
```

The floor is a teacher-consistency proxy, **not a calibrated sensor-noise estimate**. Outputs are decoded at native resolution. The existing teacher caches are not modified. The artifact records sample episode/frame IDs, cache signatures, builder hash, estimator, and a temporal-clip gating diagnostic.

Initial informative fractions in the audit were approximately:

| Source | Clips above the floor |
| --- | ---: |
| EgoTouch | 3.1% |
| EgoTactile | 44.5% |
| OpenGraph | 37.5% |
| OpenTouch | 100% |
| FreeTacMan | 53.9% |
| Unit | 44.5% |
| Unit Bimanual | 34.4% |
| VLA Touch | 54.7% |
| ViTaMIn | 10.9% |

**The EgoTouch floor is very conservative.** Inspect this before a long relational run. The explicit `floor_scale` setting defaults to 1.0 and supports a source-only sensitivity experiment without modifying the original audit. Both control and treatment use the same floor for validation diagnostics. No lower floor has silently been substituted.

Normalized geometry does not align independent sensor coordinate rotations/translations, and does not prevent scale collapse on its own. Log reconstruction, relational coverage, latent variance/effective rank, and transfer; do not infer invariance from relational loss alone.

## Teacher-based trajectory matching

The matcher uses POT **0.9.6.post1**, already pinned in the project's `alignment` extra. It compares frozen tactile temporal distance matrices with entropic GW; visual latents do not select correspondences. See the [POT solver documentation](https://pythonot.github.io/gen_modules/ot.gromov.html#ot.gromov.entropic_gromov_wasserstein).

This implementation is **GW with a chronological rejection check, not Gromov-DTW**. Plain GW does not enforce monotone alignment. Here a match is rejected if its expected matched time reverses or fails to progress, its fit is poor, or its coupling is too ambiguous. This does not provide a full monotone-path guarantee. Synthetic reversed and symmetric trajectories are tested.

Each epoch samples episode-balanced source banks using the same sampler as the control. The trajectory condition examines up to 32 bank clips per source and four candidate partners per informative clip. Numeric sources are omitted until their temporal floor is justified. Codes of different dimensions can be compared through their internal distance matrices.

The coupling is also solved at a 10% higher entropy temperature to measure sensitivity. Final marginal errors and convergence are checked; intermediate solver warnings are recorded. If the initial 1,000-iteration Sinkhorn budget fails the final checks, a bounded 10,000-iteration retry keeps the same entropy setting and tolerances. Fit, row entropy, and temperature stability produce a bounded **heuristic reliability weight**, not a match probability. Poor or failed matches receive no trajectory loss.

After matching, the current visual model supplies detached bank means. Couplings transport these into symmetric targets with row-mass normalization; unsupported rows do not contribute. Correspondences, episode/frame provenance, diagnostics, and transported targets are saved under each run's `trajectory/` directory. SAM replay retains those targets. The next epoch refreshes bank sampling and visual targets.

## Rotating-source meta-learning

Every 16 outer optimizer updates, rotate through the eleven source datasets. The selected source's ordinary supervised batches are excluded from that accumulation group. Its persistent adapter is not used in the meta objective.

1. Sample four support and four query episodes, with one clip per episode. The episode sets are disjoint.
2. Encode their images with frozen DINO once. Keep the shared encoder's latent computation differentiable.
3. Initialize a fresh `16 → 512 → K_source` head. Its output is linear for standardized AE codes and sigmoid for bounded numeric targets.
4. Take five functional SGD steps on support targets, initial inner learning rate 0.01. Only temporary head weights change inside these steps.
5. Evaluate query error after adaptation and differentiate it **through all five support updates** into the shared encoder.
6. Combine query loss, weighted initially at 0.1, with the ordinary/relational gradients from other sources in that accumulation group. Discard the temporary head after the outer update.

This is head-only inner adaptation in the spirit of [ANIL](https://arxiv.org/abs/1909.09157). It is not ordinary probe fitting on detached latents. Training uses both latent means and Gaussian draws. The temporary MLP has no dropout; the shared encoder's existing dropout and stochastic latents remain active.

SAM replays the same support/query data, head initialization, shared-encoder dropout, and latent draws under perturbed outer weights, recomputing the inner adaptation. The replay does not reuse stale adapted weights.

Episode groups can be enforced with `meta.group_field`; requesting an unavailable field fails explicitly. The current manifest lacks a consistent object/session grouping field, so supplied configs guarantee episode separation, not object-disjoint tasks. No task-local statistics are fitted on query data. Existing teacher normalization remains fixed from source training; no additional target normalizer is introduced.

MotionTrans is excluded from source tasks, floor estimation, trajectory banks, and representation updates. A pseudo-held-out source is unseen only within its task; it has been seen in other training steps. The globally held-out transfer evaluation is still MotionTrans, with its already documented development-set limitations.

## Commands

Use the existing environment. On this machine CUDA needs the installed compatible library directory:

```bash
export LD_LIBRARY_PATH="$PWD/processing/runtime/nvidia-580.173.02/root/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
```

Prepare or verify the resumable floor audit:

```bash
.venv/bin/python scripts/prepare_relational_floors.py
```

Read-only preflight, including the parent checkpoint and all teacher/data/cache contracts:

```bash
.venv/bin/python scripts/train_invariance.py --config configs/invariance_16d_relational.json --preflight
```

Bounded GPU verification, which writes a separate benchmark report and **no production checkpoint**:

```bash
.venv/bin/python scripts/train_invariance.py --config configs/invariance_16d_relational.json --benchmark-updates 2
.venv/bin/python scripts/train_invariance.py --config configs/invariance_16d_meta.json --benchmark-updates 1
```

When starting the first comparison, launch each condition explicitly:

```bash
.venv/bin/python scripts/train_invariance.py --config configs/invariance_16d_control.json
.venv/bin/python scripts/train_invariance.py --config configs/invariance_16d_relational.json
```

Use `--resume` with the exact original config to continue a saved run. Config/code/data changes fail the resume contract. A changed experiment needs a new output directory and explicit parent. No long training job was launched as part of implementation.

## Verification and limits

Tests cover detached teachers, masked/constant/near-static trajectories, rotation and scale invariance, numeric versus code head activations, source rotation, disjoint episodes/groups, held-out exclusions, constant-signal transport with unequal sizes, ambiguous/reversed matches, solver failures, finite-difference meta-gradients through support adaptation, SAM RNG replay, and epoch-boundary continuation.

The combined trainer test uses the actual Gaussian spatial/temporal model, relational loss, meta loss, SAM, and checkpoint serialization. Interrupted/resumed training matches uninterrupted final weights exactly. Existing alignment, viewer, and frozen-probe tests are included in regression verification.

Real-data bounded GPU checks completed for relational and meta paths. The relational check used two outer updates; the meta check used one update containing one four-support/four-query task. These checks establish executable gradients and replay, **not transfer quality or a reliable full-run ETA**. Benchmark JSON records measured time and peak allocated GPU memory. Native resolution remains unchanged.

A small real-data trajectory check (up to two bank clips per source, one candidate partner each) completed with six candidate pairs and zero solver failures after the bounded-retry fix. All six pairs were rejected by the quality checks. The full trajectory condition therefore still needs a source-only correspondence pilot; do not mistake a successfully running solver for evidence of useful alignment. Accepted-match transport gradients are covered by synthetic tests.

Uncertainty calibration is unchanged and unresolved. Neither relational geometry, heuristic transport reliability, nor meta-learning establishes calibrated predictive intervals.
