# Research question, evidence and limits

The project asks whether heterogeneous human/robot video–touch datasets can teach a shared **visually predicted tactile representation** that transfers to a new sensor rather than encoding dataset identity. The [original proposal](proposal/main.pdf), dated August 24, 2026, motivates reconstruction, dataset probing, physical structure, temporal context and eventual uncertainty-aware use. It is a proposal: none of those claims follows automatically from implementing a shared latent.

The current [architecture](architecture.md) uses frozen V-JEPA video features, trainable cross-attention, a deterministic 32D latent and specialized frozen tactile teachers. The [dated state snapshot](project_state.md) identifies the actual running configuration and distinguishes it from the old DINO/Gaussian pipeline described in archived reports.

## What has been implemented

- Chronological multi-camera video/target pairing, native tactile targets, explicit masks and source-specific normalization/provenance.
- Image/pressure/taxel AEs and marker-set AEs; frozen teacher-code caches; direct heads for small numerical targets.
- DINO spatial/temporal models, Gaussian/deterministic bottlenecks, SAM, classifier/adversarial experiments, FGW placements, relational supervision, trajectory matching and rotating held-out-source meta-learning variants.
- Pooled and full-spatial V-JEPA architectures, all-frame versus current-frame prediction, 32D/128D comparisons, scene-RGB compression and extra source integration.
- Frozen MotionTrans transfer heads, dataset probes, effective-rank diagnostics and historical uncertainty metrics.
- Reconstruction/prediction/mirror/attention/training-history visualization.

Implementation availability does not imply each experiment was completed successfully. The broad multi-seed FGW matrix was superseded by smaller tests; trajectory work was paused; the static episode-context proposal was explicitly excluded. Source histories and stopped/failed attempts remain in the [archive](archive/README.md) and under `processing/`.

## Selected completed evidence

These are **separate protocols**, not a single leaderboard. Numbers are in the project's existing normalized target coordinates, not independently verified physical torque units. Readouts, sample counts and stochastic evaluation differ between experiments.

| Question / matched comparison | Finding | Evidence |
|---|---|---|
| Does longer DINO code-supervised training improve transfer? | In the original deep-head sweep, epoch 7 MAE 0.0460845 versus epoch 100 0.05121: later source fitting did not improve this transfer. | [Historical training/results, §§8–11](archive/root/PROJECT_HISTORY_AND_RESULTS.md) |
| Does a matched means-only frozen screen favor the trained bottleneck? | Temporal DINO 2304D: 0.031462; DINO epoch-7 32D: 0.043769; epoch-100 32D: 0.048189. Wide trunk epoch 7/100: 0.035325/0.035938; their paired difference interval includes zero. | [Matched screen](../processing/evaluations/motiontrans_representation_screen_v1/analysis.md), [protocol](archive/root/FROZEN_REPRESENTATION_SCREEN.md) |
| Did pooled V-JEPA improve over the trained 32D DINO reference? | Source-selected epoch-28 deterministic/stochastic: 0.040215/0.040216, about 8.1% better than matched DINO epoch 7; the two latent modes were indistinguishable. Temporal raw DINO remained stronger. | [V-JEPA transfer](../processing/evaluations/motiontrans_vjepa21_best_v1/analysis.md) |
| Did current-frame 128D beat 32D? | At epoch 30, matched current-frame MAE was 0.047950 versus 0.047972; the paired difference interval includes zero. A refit of the previous model at 0.047654 also showed no clear difference. | [Current-frame comparison](../processing/evaluations/motiontrans_vjepa21_current_epoch030_v1/REPORT.md), [matched previous baseline](../processing/evaluations/motiontrans_vjepa21_current_vs_previous_epoch030_v1/REPORT.md) |
| Did FGW establish better transfer? | No robust demonstrated benefit in completed tests; the 16D epoch-10 paired differences include zero. Tactile-first was worse than its parent under the tested readout. | [FGW implementation/study](archive/root/FGW_STUDY.md), [history §9–10](archive/root/PROJECT_HISTORY_AND_RESULTS.md) |
| Did relational/meta-learning solve transfer? | Completed epoch-20 comparisons did not establish a clear full-budget gain; deeper readout heads helped several conditions. | [Invariance experiments](archive/root/INVARIANCE_TRAINING.md), [deeper-head evaluation](../processing/runs/invariance_16d_v1/evaluations/epoch_020_deeper_heads/analysis.md) |
| Did retention/source stopping stop late decline? | Epoch-22 combined continuation 0.044933 versus control 0.047140; the paired difference interval narrowly includes zero. | [Continuation protocol/results](archive/root/CONTINUATION_TRANSFER_EXPERIMENT.md) |
| Are mirror teachers prepared? | All five enabled mirror sources have ready supervision; OpenGraph/Unit shared-hand AEs attain near-identical paired reconstruction error while passing absolute quality checks. | [Current mirror guide](BIMANUAL_MIRRORING.md), [v4 quality report](../processing/bimanual_mirror_v4/REPORT.md) |

The raw-DINO representations are wider and have larger first readout layers. Three head-fit seeds estimate readout variation, not three independent source-training runs. The epoch-0 DINO reference was reconstructed and matched a recorded 16D initialization hash, but no independently saved original 32D epoch-0 hash/checkpoint exists. These qualifications belong with the comparison.

Current-frame evaluation contributes **one target per clip**; earlier all-frame evaluation often contributed eight. Do not compare their MAEs as if all inputs, targets and fit counts were matched. The current 17-source run is still in progress at the [observation](project_state.md); no new transfer success is claimed.

## Invariance is operational, not visual mixing

A useful latent should let a new sensor's readout learn from limited data while preserving source information. Dataset-classifier accuracy, plots, FGW objective and effective rank are diagnostic. Lower dataset accuracy can result from collapse; higher rank can accompany useful early learning and harmful late specialization. Serial checkpoint correlations do not show that changing rank causes transfer to improve or worsen.

MotionTrans was excluded from representation fitting, but its validation results have repeatedly guided design. It is therefore a development benchmark, not a pristine final test. A strong future claim needs a clearly separated target evaluation and controlled comparisons. No new follow-up is launched automatically by this document.

## Calibration and data-quality limits

Historical Gaussian latents and Monte Carlo reconstructions substantially undercovered nominal intervals; some reported 90% coverage values were only tens of percent or lower. Sigmoid outputs do not model a calibrated point mass at exact zero. FGW stochastic mixing is augmentation, not predictive uncertainty. Attention weight concentration is not calibrated confidence. The active deterministic latent supplies no probabilistic uncertainty output.

Sensor-native values are heterogeneous: marker displacement is relative to saved geometry; released numerical zeros may include upstream taring/drift correction; FEEL contact estimates and HRDex counts are not universally calibrated SI force. [Fixed zero references](FIXED_SENSOR_ZERO_REFERENCES.md) document what is known and unresolved. [OpenGraph's mirror policy](BIMANUAL_MIRRORING.md) is user-requested displayed-grid geometry, not independently calibrated wiring.

Preserve these qualifications when citing historical findings. Complete records live in the [archive](archive/README.md), original `processing/` reports/metrics, pinned configurations and checkpoint provenance; filenames alone do not determine scientific authority.
