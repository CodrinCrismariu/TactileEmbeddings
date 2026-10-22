# MotionTrans transfer with a larger frozen-encoder decoder

The encoder is the same completed, non-adversarial 11-source joint baseline
with MotionTrans excluded. No encoder or DINO weights are updated. Only a
new MotionTrans regression decoder is trained, on the identical frozen latents.

The original decoder is `32 → 512 → 6` (19,974 parameters). This experiment
uses `32 → 1024 → 1024 → 512 → 6` (1,611,270 parameters). Every hidden layer
uses GELU and dropout 0.1; the output uses sigmoid for normalized torque.
Default evaluator behavior and original head state-dict keys remain unchanged.

Loss, AdamW settings, episode-disjoint duration tuning (up to 100 epochs,
patience 10), refitting on all training clips, seed 45, validation clips and
64 latent draws remain the baseline protocol. The diagnostic question is
whether a more expressive readout extracts more of the information in the
same 32D Gaussian latent. Increased capacity can also overfit; duration is
selected on training-only tune episodes, not validation.

```bash
.venv/bin/python -u -m tactilelatent.evaluate_holdout_decoder \
  --source ftp_motiontrans \
  --checkpoint processing/runs/joint_all_holdout_motiontrans_fresh_8k_50ep/last.pt \
  --output processing/runs/joint_all_holdout_motiontrans_fresh_8k_50ep/evaluations/motiontrans_large_decoder \
  --cache-dir processing/runs/joint_all_holdout_motiontrans_fresh_8k_50ep/evaluations/motiontrans \
  --hidden-dims 1024 1024 512 --samples 64
```

Existing cached train/validation features are signature-checked against the
checkpoint hash, manifest hash, source, split, trial indices and target mode.
They are reused without copying. The new decoder, configuration, selection
history, metrics and examples are saved in a separate directory; existing
baseline results are not overwritten. The launched job uses nohup and setsid.

```bash
tail -f processing/runs/joint_all_holdout_motiontrans_fresh_8k_50ep/evaluations/motiontrans_large_decoder/evaluation.log
```

Errors and interval widths are normalized torque, not verified SI units.
Coverage describes empirical predictions from the trained decoder, not
decoder-weight uncertainty. Best-of-64 uses the true target to choose a
sample and is an oracle diagnostic, not achievable prediction accuracy.

## Completed result — 2026-10-02

Both decoders use identical checkpoint and manifest hashes, 16,736 fit clips,
1,877 episode-disjoint tune clips, and all 1,972 validation clips. Tuning
selected 18 epochs for the larger head (21 for the original). The selected
larger head was refitted on all 18,613 training clips. Full suite: 42 tests pass.

| Metric | Original 32→512→6 | Larger 32→1024→1024→512→6 |
| --- | ---: | ---: |
| Deterministic MAE ↓ | 0.05884 | 0.04944 |
| 64-draw ensemble MAE ↓ | 0.06180 | 0.05532 |
| CRPS ↓ | 0.04890 | 0.04054 |
| Sample spread/error Pearson correlation ↑ | 0.405 | 0.551 |
| Nominal 95% marginal interval coverage | 66.48% | 69.29% |
| Mean 95% interval width | 0.12591 | 0.13037 |
| Highest-error-quartile 95% coverage | 11.71% | 35.34% |
| Oracle best-of-64 clip MAE ↓ | 0.04991 | 0.04031 |

Deterministic error falls by 16.0%, showing that the small decoder was partly
limiting this transfer test. The larger decoder remains worse than the directly
trained MotionTrans model (MAE 0.02606). The larger decoder extracts more useful
information, but this single comparison does not isolate every remaining
limitation or establish statistical significance.

Uncertainty ranking and interval coverage improve, but 69.3% coverage remains
far below nominal 95%. Ensemble averaging still worsens MAE relative to z=mu.
High-error quartiles are defined separately from each model's errors, so their
coverage numbers do not refer to an identical fixed subset of readings.

Results and trained decoder:
`processing/runs/joint_all_holdout_motiontrans_fresh_8k_50ep/evaluations/motiontrans_large_decoder/results.json`
and `decoder.pt`. The original baseline remains unchanged.
