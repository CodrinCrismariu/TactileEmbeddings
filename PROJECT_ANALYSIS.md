Analysis dated October 5, 2026. Based on the local proposal, implementation, configurations, training logs, and saved evaluation artifacts. The existing test suite was run: **47 passed in 10.75 seconds**. Training and full dataset evaluations were not rerun, and checkpoint hashes were not independently recomputed during this review.

**Assessment.** TactileLatent implements a substantial heterogeneous tactile-supervision pipeline and demonstrates useful supervised transfer from frozen visual latents. The results do not yet establish a dataset-invariant physical representation, an advantage over direct target training, or calibrated tactile uncertainty. Its strongest scientific result is the distinction between successful reconstruction, transferable information, and invariance: these are measurably different properties.

The proposal in [main.pdf](main.pdf) asks whether visual observations can support a common tactile representation across incompatible sensors. The current implementation takes eight RGB frames at stride two, with up to three camera slots, through frozen DINOv3-B/16 patch features. Learned camera weighting and three divided space-time blocks produce a 32-dimensional diagonal Gaussian per frame. Source-specific MLPs decode pressure arrays, optical tactile RGB, gripper force, or torque. Tactile measurements supervise training but do not enter the visual encoder. The source identifier selects the decoder; it is not an explicit encoder feature.

The camera gate weights whole views rather than matching contact patches across cameras. Corresponding image-grid positions are summed despite different viewpoints. Temporal attention is bidirectional, so only the final clip position has a strictly past-and-present interpretation. These are architectural limitations to investigate, not demonstrated explanations of the measured errors. See [model.py](src/tactilelatent/model.py) and [joint_model.py](src/tactilelatent/joint_model.py).

The current joint baseline trains on 11 of 12 indexed sources, excludes MotionTrans, and starts learned weights fresh while retaining pretrained frozen DINO. A new MotionTrans decoder is fitted after freezing the encoder. This is supervised adaptation, not zero-shot prediction. The final SAM run also completed 50 epochs; README language describing it as active is stale. Earlier VLA_touch holdouts and adversarial experiments have different training-source sets and must remain separate comparisons.

**MotionTrans results.** These scores use the same 1,972 validation clips. Errors are in normalized torque space, not verified physical units. Frozen heads receive all 18,613 target training clips after training-only episode-disjoint duration selection.

| Model | Deterministic MAE ↓ | Ensemble MAE ↓ | CRPS ↓ | Nominal 95% coverage |
| --- | ---: | ---: | ---: | ---: |
| Training mean target | 0.08639 | — | — | — |
| Frozen joint encoder, original head | 0.05884 | 0.06180 | 0.04890 | 66.48% |
| Frozen joint encoder, larger head | 0.04944 | 0.05532 | 0.04054 | 69.29% |
| Frozen SAM encoder, original head | 0.05881 | 0.06412 | 0.04965 | 65.20% |
| Direct MotionTrans training | 0.02606 | 0.02574 | 0.02383 | 51.66% |

The original frozen representation reduces MAE by 31.9% relative to the constant mean. Increasing the decoder from 19,974 to 1,611,270 parameters reduces MAE a further 16.0%, so the original probe underestimates what a more expressive readout can extract. Nevertheless, even the larger head has approximately 1.90 times the direct model's error. SAM changes deterministic MAE by only 0.055%, improves MSE by 4.72%, and worsens ensemble MAE and CRPS. This single-seed comparison does not demonstrate an overall MotionTrans benefit from SAM.

The direct model is substantially more accurate but has much narrower, under-covering intervals. Thus point accuracy and calibration must be assessed separately. For the original frozen head, nominal 95% coverage falls to 11.71% in its highest-error quartile. The larger head improves its own highest-error-quartile coverage to 35.34%; those quartiles are model-specific subsets. Best-of-64 is an oracle diagnostic and cannot be reported as deployable accuracy.

Sources: [baseline/direct comparison](processing/runs/motiontrans_baseline_sequence/comparison.json), [larger head](processing/runs/joint_all_holdout_motiontrans_fresh_8k_50ep/evaluations/motiontrans_large_decoder/results.json), and [final SAM](processing/runs/joint_all_sam_holdout_motiontrans_fresh_8k_50ep/evaluations/motiontrans_final/results.json).

**Invariance and adversarial training.** On the earlier VLA_touch protocol, the non-adversarial frozen decoder achieved MAE 0.010323, essentially matching direct training at 0.010317. The independent source probe still recovered dataset identity with 97.10% macro accuracy, versus 9.09% chance. Good transfer therefore coexists with extensive source information.

Alternating adversarial training lowered deterministic source-probe macro accuracy to 21.99%, but VLA_touch MAE rose to 0.03435, close to the constant baseline of 0.03718. Nominal 95% coverage fell to 10.79%. Its final logged effective rank was approximately 1.67. Together, these observations are consistent with severe representation compression that sacrifices useful prediction, rather than successful preservation of transferable physics. The earlier online GRL experiment stopped at 15 epochs and is not a completed comparator.

The newer SAM source probe achieves 95.55% deterministic macro accuracy, confirming that dataset identity remains readily available. It uses a different source set and explicitly balanced probe fitting, so its percentage should not be directly interpreted as an improvement over the historical 97.10% result. High source identifiability rejects strict invariance of the whole latent, but does not prove that no shared physical subspace exists.

Sources: [historical identity probe](processing/evaluations/dataset_identity_probe/results.json), [adversarial transfer](processing/runs/joint_all_alternating_fresh_8k_50ep/evaluations/vla_touch/results.json), [adversarial identity probe](processing/runs/joint_all_alternating_fresh_8k_50ep/evaluations/dataset_identity/results.json), and [SAM identity probe](processing/runs/joint_all_sam_holdout_motiontrans_fresh_8k_50ep/evaluations/dataset_identity/results.json).

**Grasp-success transfer.** Feeling of Success uses two real pre-lift RGB snapshots and holds out entire objects. Baseline and SAM saved split assignments are identical: 1,421 test trials across 16 objects.

| 64-draw prediction | Accuracy | Balanced accuracy | AUROC | Brier ↓ | Failure recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Constant training prevalence | 66.64% | 50.00% | 0.5000 | 0.22313 | 0% |
| Non-SAM latent | 66.64% | 50.16% | 0.4949 | 0.22696 | 0.63% |
| SAM latent | 66.08% | 50.68% | 0.5852 | 0.22044 | 4.43% |

The non-SAM result does not demonstrate useful discrimination. SAM shows modest ranking and probabilistic-score improvements, but detects only 21 of 474 failures at the default decision threshold. This is a more nuanced outcome than saying SAM never helps: it helps ranking on this task in this run, while leaving practical classification weak. Statistical uncertainty over the 16 test objects is unreported. A different threshold must be selected on tune objects, not retrospectively on these test outcomes.

Sources: [non-SAM success metrics](processing/runs/joint_all_holdout_motiontrans_fresh_8k_50ep/evaluations/feeling_of_success/metrics.json) and [SAM success metrics](processing/runs/joint_all_sam_holdout_motiontrans_fresh_8k_50ep/evaluations/feeling_of_success/metrics.json). The latter results are missing from the existing benchmark narrative.

**Uncertainty objective.** The current image-only Gaussian bottleneck is VAE-style, but its sampled reconstruction loss plus standard-normal KL is not a conventional conditional ELBO with a target-informed posterior. It does not directly optimize calibrated output probabilities. A nonzero latent variance can coexist with tiny decoder-output variance, and stochastic predictions need not improve MAE.

Historical EgoTouch experiments reinforce this issue. The saved 32D image-only VAE has all-view unseen ensemble MAE 0.05597 and CRPS 0.05364. The historical pose-teacher CVAE has MAE 0.05712 and better CRPS 0.04576, yet its case diagnosis reports only 16.33% nominal 95% coverage. Approximately 64.3% of evaluated pressure cells are exactly zero, with zero coverage in that group. Strictly positive sigmoid draws cannot represent an exact zero mass in their central sample intervals. A contact/no-contact component deserves investigation alongside pressure regression. These historical variants are not controlled comparisons to the current joint runs, and their old execution paths are not all supported by the active code.

**Training and engineering assessment.** Episode-disjoint head selection, fresh refitting, object-disjoint success evaluation, cache/checkpoint identities, explicit holdout checks, and preserved controls are valuable safeguards. The passing tests verify implementation behavior, not scientific validity or raw synchronization across every dataset.

The joint baseline's monitoring MAE improves on 10 of 11 sources between its first and final epochs. OpenTouch remains effectively flat (0.04524 to 0.04559). FreeTacMan training loss falls substantially while validation remains much higher; SAM's final FreeTacMan MAE is 0.08588 versus 0.07173 for ordinary training. These warrant per-source investigation. Monitoring uses only 64 clips/source, and training loss includes different terms from validation MAE, so these observations are diagnostic rather than definitive overfitting claims. The cap of 8,000 clips/source also does not equalize source contributions: small datasets contribute fewer batches. Native targets have different distributions and loss scales despite normalization.

**Most informative next experiments, in order.**

1. Fit matched target heads on frozen DINO-only features, and on an untrained shared encoder where useful, using the existing train/tune/validation protocol. Without these controls, the benefit attributable to tactile pretraining remains unidentified.
2. Compare frozen transfer, direct training, and pretrained-encoder fine-tuning across target-label budgets, with matched head choices. Current full-label comparisons cannot establish sample efficiency; encoder pretraining compute also differs substantially.
3. Repeat key runs across seeds and calculate paired uncertainty intervals by resampling episodes or objects. Frames and sensor readings within an episode are not independent experimental units.
4. Run single-frame and temporal-shuffle ablations, plus matched deterministic and stochastic bottlenecks. The proposal's temporal-benefit hypothesis remains untested by the reviewed results.
5. Evaluate shared physical properties and retrieval where labels can be harmonized. Suppressing all domain identity can remove embodiment or task information relevant to physical prediction; shared and source-specific latent components are a hypothesis worth testing, not an established solution.
6. Treat probabilistic calibration as its own objective and evaluation problem. Compare explicit output-distribution or quantile models and tune-only calibration, respecting exact-zero pressure targets and episode dependence.

Against the proposal's four hypotheses: superior joint transfer is not established; reconstruction alone failing to produce invariance is supported, but adversarial improvement is not; temporal benefit remains untested; and shared physical structure remains unproven, with weak grasp-success evidence. The existing work supports a rigorous empirical study of when heterogeneous tactile supervision transfers and when invariance pressure destroys useful information.
