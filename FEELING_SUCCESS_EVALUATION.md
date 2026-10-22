# Feeling of Success: frozen joint-baseline evaluation

This benchmark tests whether the frozen tactile latent supports **binary
grasp-success prediction on new objects**. It is an evaluation-only companion
to MotionTrans, not a joint training source. It does not reproduce the original
paper's multimodal input or split, and its accuracy must not be compared as if
the protocols were identical.

## Model and inputs

Checkpoint: `processing/runs/joint_all_holdout_motiontrans_fresh_8k_50ep/last.pt`.
This is the final fresh, non-adversarial 50-epoch joint baseline, trained on
11 sources including VLA_touch, excluding MotionTrans and Feeling of Success.
Both DINOv3 and the complete shared transformer/Gaussian encoder are frozen
and kept in evaluation mode. No existing tactile decoder is trained or used.

Each trial supplies only `kinectA_rgb_before` and `kinectA_rgb_during`, resized
with the project's bicubic 224×224 image preprocessing. These are two actual
event snapshots, not an eight-frame video or repeated synthetic timesteps.
Only main-camera slot 0 is active. External Kinect scene RGB and the two-frame
duration differ from the training domains; this is part of the transfer test.
The bidirectional temporal encoder can see both pre-lift images, but NEVER
the outcome-revealing `kinectA_rgb_after`. The during-grasp output gives one
32D mean and log-variance per trial. Labels, object names and tactile images
are never encoder inputs.

The new head is `32 → 512 → GELU → dropout(0.1) → 2 logits` (no output sigmoid).
Cross entropy trains it equally on means and random latent draws. AdamW uses
learning rate 0.001, weight decay 0.0001, and batch size 256.

## Splits and selection

The full release has 9,296 distinct trial IDs and 105 object names. Its 38
downloaded shards passed integrity checks. The published headline count
(9,269) differs; the manifest records this discrepancy without dropping trials.

Sorted object names are permuted once with NumPy seed 45: 16 objects become
test, 16 tune, and 73 fit. Trial counts need not follow the object proportions.
The same object cannot cross partitions. Labels do not determine assignment.
Training duration is selected using tune cross entropy, at most 100 epochs
with patience 10. A freshly initialized head is then trained on fit+tune for
the selected duration. Final evaluation uses the untouched test objects.
Trial IDs and assignments are saved in `splits.json`.

## Metrics and interpretation

The **same head** predicts from `z=mu` and from the average probabilities of
64 independent Gaussian draws. A constant-probability baseline uses only the
fit+tune success prevalence. Reported metrics include accuracy, balanced
accuracy, object-macro accuracy, confusion matrices, success precision/recall/F1,
AUROC, average precision, cross entropy/NLL, conventional binary Brier score,
10-bin confidence ECE, per-object results, and entropy-based risk/coverage.

Uncertainty diagnostics include success-probability sample spread, entropy's
ability to rank classification errors (AUROC), and 50/80/90/95% top-probability
class-set coverage and size. With two classes, high coverage can simply mean
returning BOTH classes. Class-set coverage minus selected probability mass is
also reported: nominal levels alone are insufficient for these discrete sets.
Sampling the latent does not measure decoder-weight uncertainty, guarantee
all plausible outcomes, or make a calibrated success predictor automatically.

## Run and inspect

```bash
.venv/bin/python -u -m tactilelatent.evaluate_feeling_success
```

The default output is
`processing/runs/joint_all_holdout_motiontrans_fresh_8k_50ep/evaluations/feeling_of_success`.
Completed heads/results cannot be overwritten: choose a new `--output` for a
new experiment. Feature caches are tied to checkpoint and manifest SHA-256,
trial order, and temporal adapter. The checkpoint hash is checked again at
completion. Files include config, splits, cached latents, duration-selection
history, refitted head, per-trial predictions, and final metrics.

```bash
tail -f processing/runs/joint_all_holdout_motiontrans_fresh_8k_50ep/evaluations/feeling_of_success/evaluation.log
.venv/bin/python -m pytest tests/test_feeling_success_evaluation.py tests/test_feeling_success_download.py -q
```

Acquisition retains lossless scene RGB and labels, not raw GelSight streams.
Datasets plus processing were 678.4 GB before this small evaluation, under
the existing 700 GB cap. The source archive identity and compact checksums
are recorded in the evaluation-only manifest.

## Completed result — 2026-10-02

All 9,296 trials were encoded. Partitions contain 5,504 fit trials on 73 objects,
2,371 tune trials on 16 objects, and 1,421 test trials on 16 objects. Tuning
selected seven head epochs; refitting used all 7,875 fit+tune trials. There are
947 successes and 474 failures in the final test set.

| Metric | Mean latent | 64-draw ensemble | Constant training-prevalence baseline |
| --- | ---: | ---: | ---: |
| Accuracy | 66.57% | 66.64% | 66.64% |
| Balanced accuracy | 50.00% | 50.16% | 50.00% |
| AUROC | 0.4950 | 0.4949 | 0.5000 |
| Average precision | 0.6415 | 0.6407 | 0.6664 |
| Binary Brier, lower is better | 0.22636 | 0.22696 | 0.22313 |
| NLL, lower is better | 0.64572 | 0.64696 | 0.63848 |
| Confidence ECE | 6.24% | 6.18% | 2.87% |
| Object-macro accuracy | 60.39% | 60.45% | — |

The ensemble identifies just 3 of 474 failures; it predicts success for 1,415
of 1,421 trials. Its apparent 66.64% accuracy reflects class imbalance, not
successful discrimination. Under this fixed protocol, the frozen latent plus
selected head does not demonstrate useful grasp-success transfer.

Mean standard deviation of sampled success probability is 0.07183 (7.18
percentage points). Predictive-entropy error AUROC is 0.49485 and sample-spread
error AUROC is 0.41415: higher uncertainty does not reliably identify errors.
The 80/90/95% class sets achieve 100% coverage only by including BOTH classes
for every trial; they do not provide selective high-confidence predictions.

These results do not isolate the cause or prove that all information is absent
from the encoder. External scene-camera input, only two snapshots versus the
training clip length, visual ambiguity in grasp success, and the particular
probe/selection protocol remain possible limitations. No new adapter or
hyperparameters were selected using these test results.

Full results: `processing/runs/joint_all_holdout_motiontrans_fresh_8k_50ep/evaluations/feeling_of_success/metrics.json`.
Sample variability is measured by `latent_sampling.mean_probability_std`;
no unmeasured generic-helper disagreement field is reported.
