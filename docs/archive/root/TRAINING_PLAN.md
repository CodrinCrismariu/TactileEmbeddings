> Historical record, relocated on October 21, 2026. Original path: `TRAINING_PLAN.md`.
> Results, proposals, and operational status below belong to their recorded experiment/date;
> use the [maintained project state](../../project_state.md) for the audited active workflow.

# EgoTouch image-only VAE research plan

The active code implements one 32D Gaussian-latent predictor for EgoTouch.
`README.md` describes its exact data flow and sizes. No training is currently
running; the historical 32D VAE checkpoint is retained for evaluation and
optional weight initialization.

1. Verify the EgoTouch split and pressure normalization, and record exact
   train/validation/test clip counts and missing episodes.
2. Evaluate the existing 32D checkpoint on seen and unseen tests under
   ego-only and all-view inputs. Record point and distribution metrics with
   the same sampling count and seeds.
3. Train the simplified image-only VAE in a new output directory. The config
   currently warm-starts model weights from the existing 32D VAE but resets
   optimizer and scheduling; report it as a continuation experiment, not an
   independent run from random initialization.
4. Add a matched deterministic/no-bottleneck control before attributing
   differences from TouchAnything to the 32D latent. DINOv3-B/16 is also a
   deliberate backbone change from the DINOv2 paper reference.
5. Measure whether uncertainty is useful: pressure-space variance versus
   error, CRPS, empirical interval coverage and width, PIT, and performance
   separately on contact and noncontact regions. If it is too narrow, test
   calibrated sampling or an output-noise model as separate ablations.
6. A six-source data contract, manifest, and source-specific-decoder VAE
   trainer are now prepared separately (see `JOINT_DATASETS.md`). Do not
   interpret a shared encoder alone as dataset invariance. Resolve the
   upstream EgoTouch split gap and source-license restrictions before making
   broader reproducibility or release claims.

Important protocol caveat: the three temporal attention blocks are
bidirectional within each 8-frame clip. If strictly online per-frame
predictions are needed, use a causal mask and retrain; simply changing the
evaluation code would not make the trained checkpoint causal.
