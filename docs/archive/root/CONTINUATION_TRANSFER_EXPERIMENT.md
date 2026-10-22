> Historical record, relocated on October 21, 2026. Original path: `CONTINUATION_TRANSFER_EXPERIMENT.md`.
> Results, proposals, and operational status below belong to their recorded experiment/date;
> use the [maintained project state](../../project_state.md) for the audited active workflow.

# Epoch-7 continuation experiment

**October 15 update:** All four arms stopped after completed epoch-22 evaluations at the user's request. The former epoch-27 schedule below is historical. The follow-up is a frozen representation screen with temporal DINO, fit-only z-scoring, three tune-selected learning rates and three head seeds. See [revised protocol](FROZEN_REPRESENTATION_SCREEN.md).

This experiment tests whether MotionTrans transfer can be retained while the source prediction tasks continue training. It uses the original **32D Gaussian model**, not the recent 16D deterministic run. All four arms start with the exact original epoch-7 encoder and adapter weights, seed 45, and train for twenty additional epochs, finishing at epoch 27.

## Cheap source screen

The saved transfer evaluations cover nineteen distinct epochs: 1–10, then 20, 30, …, 100. The screen correlates each source's validation supervision loss and native reconstruction MAE with MotionTrans transfer MAE. It evaluates both existing head depths, early and late windows, and descriptive time-adjusted and finite-difference correlations.

The main post-7 ranking uses the two-hidden-layer head that produced the original epoch-7 transfer optimum:

| Source | Pearson correlation, epochs ≥7 | Source supervision loss change, 7→100 |
|---|---:|---:|
| EgoTouch | −0.961 | −8.0% |
| VLA Touch | −0.953 | −26.9% |
| ViTaMIn | −0.945 | −30.0% |
| Unit | −0.911 | −4.6% |
| RDP | −0.881 | −22.0% |
| RDP Bimanual | −0.840 | −17.8% |
| EgoTactile | −0.630 | −2.6% |
| OpenGraph | −0.247 | +0.4% |
| FreeTacMan | +0.720 | +8.9% |
| Unit Bimanual | +0.864 | +15.5% |
| OpenTouch | +0.880 | +14.0% |

Both axes measure error. Negative correlation means source prediction improves while transfer worsens. Positive correlation here includes sources whose own validation also deteriorates. These are two different candidate mechanisms: specialization despite source improvement, and source overfitting. Neither proves that a particular source caused the transfer decline. All checkpoints belong to one evolving multi-source encoder and are confounded with time. The residual correlations are sensitivity checks, not causal evidence.

Native reconstruction MAE supports the negative pattern for EgoTouch (−0.925), VLA Touch (−0.951), and ViTaMIn (−0.937). However, FreeTacMan's native-MAE correlation is only +0.021, despite its +0.720 supervision-loss correlation; OpenGraph's native-MAE correlation is −0.725 rather than its supervision-loss correlation −0.247. Stopping by standardized code loss therefore does not establish that native reconstruction has overfit. Both metrics remain in the reports so this distinction can be assessed experimentally.

Full artifacts: [screen report](../../../processing/continuation_32d_v1/source_screen/analysis.md), [CSV](../../../processing/continuation_32d_v1/source_screen/results.csv), [JSON](../../../processing/continuation_32d_v1/source_screen/results.json). Reproduce with `.venv/bin/python scripts/screen_source_transfer.py`.

## Four matched arms

| Arm | Source task/KL gradients into trunk | KL | Retention |
|---|---|---|---|
| `control` | All eleven sources throughout | Historical ramp to 0.001 | None |
| `source_stopping` | Permanent per-source validation gates | Same ramp for active sources | None |
| `retention` | All eleven sources throughout | Off | Frozen epoch-7 mean anchor |
| `combined` | Same source gates | Off | Same anchor, including stopped sources |

There are no extra seeds, architecture searches, FGW losses, or static/private episode codes in this experiment. The reconstruction teachers, sensor code dimensions, adapter sizes, native tactile targets and chronological frame/camera mappings are unchanged. RDP and RDP Bimanual retain their direct numeric heads.

### Source stopping

Track each source's running minimum validation **supervision loss**. If its loss exceeds that minimum by more than **1% for three consecutive validations**, permanently detach its task and KL paths from the shared encoder. A flat plateau alone does not trigger stopping. The source's adapter continues training, and other sources continue updating the shared encoder.

Initialize this state using only source-validation history through epoch 7. The history already stops FreeTacMan after epoch 4 and OpenGraph after epoch 7, so their task gradients are detached from the first continuation epoch. These dates describe historical rule triggers; no earlier checkpoint is restored. OpenTouch starts with one deterioration validation and may stop later if the condition persists. Future best-source epochs from the old 100-epoch run never enter the online stopping rule.

Source stopping does not freeze the representation for that source: gradients from other sources can still change its encoder outputs. The combined arm also permits the retention objective to update the trunk on stopped-source clips; only their supervised task/KL paths are detached.

### KL-off retention

Freeze a copy of the epoch-7 **visual encoder**, including its mean projection. On the same sampled source clip, frozen DINO features, and wrist/view mask, compute its latent mean with teacher dropout disabled. Penalize the current encoder's valid-frame mean drift:

```text
L_anchor = mean_valid_frames,coordinates[(mu_current − mu_epoch7)^2] / 5.5915864468
L_total  = L_source + lambda_anchor * L_anchor
```

The divisor is the epoch-7 mean coordinate variance measured during **source training**, not MotionTrans or validation statistics. `lambda_anchor` ramps from 0.05 in the first continuation epoch to **0.1** thereafter. This is a fixed first-test setting, not a tuned optimum. Log the normalized unweighted drift as `train[source].retention`.

The student retains its original training dropout, so anchor error need not be exactly zero at initialization even with identical weights. The teacher uses evaluation mode and never receives gradients. The same detached teacher targets, masks, dropout/sampling RNG states, and source-gate decisions are reused in SAM's second pass.

KL-off does **not** make the latent deterministic: training still samples the original Gaussian latent, and the reconstruction objective still trains the variance projection. The anchor constrains means only. Existing Monte Carlo intervals remain uncalibrated; no improved coverage is assumed.

## Matching and checkpoint details

The epoch-7 snapshot contains model weights but no optimizer/RNG state. All arms use the same **fresh AdamW + SAM optimizer**, fresh identical RNG initialization, and a common twenty-epoch learning-rate schedule: two warmup epochs, cosine decay from 5e-5 to 1e-6, weight decay 0.05, SAM radius 0.05. Consequently, the new control is a matched continuation baseline rather than an exact replay of original epochs 8–27.

Preserve the original chronological dataset and cycle sampler, starting at absolute epoch index 7: up to 8,000 clips per source per epoch, batch size 2, accumulation 16, 64 validation clips per source, eight frames per clip, wrist dropout 0.3. No new 25% alignment-bank sampling is introduced. Each arm therefore sees the same sampled clips, batch ordering, and training RNG schedule. The source-stopping decisions may differ between stopping-only and combined because their own validation trajectories differ.

Fresh optimizer initialization and weight equality are recorded in each arm's `initialization.json`; the queue verifies matching initial weight hashes. Every completed epoch gets an ordinary inference snapshot that uses the existing model format. Full optimizer, scheduler, SAM, RNG, and source-stop states are saved at probe boundaries for exact resume. The frozen anchor is rebuilt from the pinned parent checkpoint on restart, never from the latest student.

## Training and transfer schedule

1. Evaluate epoch 7 with the **32→512→512→512→6** transfer head. Reuse its original cached MotionTrans means/logvars so no baseline DINO extraction is needed.
2. Train control and stopping-only to epoch 12, then retention and combined to epoch 12, with two simultaneous GPU workers.
3. Exit workers cleanly at that checkpoint boundary. Share one frozen DINO extraction across all four encoders for MotionTrans and evaluate all four with the same head/protocol.
4. Repeat for epochs **17, 22, and 27**. Resume full optimizer and RNG states between boundaries; the learning-rate schedule never restarts at a probe boundary.
5. Keep all four arms on their fixed twenty-epoch budgets. The probes do not automatically pick a winner, modify hyperparameters, or stop a training arm.

The new epoch-7 three-hidden-layer baseline completed before training began: **MotionTrans MAE 0.04477278**, selected head duration five epochs, nominal 90% coverage **32.64%**. This is better than the old epoch-7 two-layer head's MAE 0.04608450, but the coverage remains poor. It is a head-depth/protocol reference, not a gain from either continuation intervention.

MotionTrans probes use all 18,613 training clips and 1,972 validation clips, the same training-episode fit/tune split, seed 45, at most 100 head-training epochs with patience 10, and 64 predictive draws. Head duration is selected on training-episode tuning and refit on the full training split. Reports include native normalized torque MAE/MSE, per-channel errors, episode errors/dynamics, uncertainty coverage and width, and 5,000 paired episode-bootstrap comparisons against both the matched control and the epoch-7 baseline. The three-hidden-layer baseline is measured afresh, so its numbers are comparable to these probes rather than mixed with the older two-layer results.

MotionTrans validation has informed the screen and experimental design. This is explicitly a **development experiment**, not evaluation on a pristine untouched test set. No MotionTrans episode, target, teacher, validation statistic, or anchor enters representation training.

## Runtime and operation

Two simultaneous 24-update GPU benchmarks measured **1.33 seconds/update** for the control and **1.45 seconds/update** for retention. With 1,969 optimizer updates per epoch, these project to approximately **44 and 47 minutes per epoch per worker**, excluding source validation. This is a short warmup-sensitive benchmark; source stopping may reduce encoder work, and full-epoch throughput will replace this estimate.

Across all four twenty-epoch arms, expect roughly **30–36 hours** on the current single RTX 5090 with two workers, including source validation, checkpoint writing, and periodic full transfer probes. This estimate is provisional. The measured retention benchmark added about 9% update time in this paired test; it does not imply a universal 9% overhead.

The detached queue runs under `nohup` at `processing/runs/continuation_32d_v1`. Its `status.json`, `nohup.log`, `logs/*.log`, per-arm `status.json`, and `evaluations/epoch_*/summary.json` record progress and results. The plan and configurations are pinned by SHA256. No old run was overwritten. The previously stopped deterministic run remains stopped.

To inspect progress:

```bash
cat processing/runs/continuation_32d_v1/status.json
tail -n 5 processing/runs/continuation_32d_v1/logs/control.log
```

To resume the queue after stopping it, run the same command with the recorded NVIDIA library path (never start a second controller while the first is alive):

```bash
LD_LIBRARY_PATH="$PWD/processing/runtime/nvidia-580.173.02/root/usr/lib/x86_64-linux-gnu" \
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
.venv/bin/python -u scripts/run_continuation_study.py run --parallel 2
```

Implementation: `src/tactilelatent/continuation_objectives.py`, `src/tactilelatent/train_continuation.py`, `scripts/run_continuation_study.py`, `scripts/evaluate_continuation_holdout.py`. Twenty targeted checks passed for source/anchor gradients, missing targets, held-out exclusion, SAM replay/weight preservation, cache signatures, and transfer-head/snapshot compatibility. The benchmarks also ran actual source data and the pinned teacher caches without altering native tactile data.
