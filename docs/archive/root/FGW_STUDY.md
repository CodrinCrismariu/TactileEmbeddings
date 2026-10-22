> Historical record, relocated on October 21, 2026. Original path: `FGW_STUDY.md`.
> Results, proposals, and operational status below belong to their recorded experiment/date;
> use the [maintained project state](../../project_state.md) for the audited active workflow.

# Self-supervised FGW study

The study compares mixing the visual 32D mean latents with first learning a
common tactile teacher from the existing cached sensor codes. It excludes
MotionTrans from both alignment stages. No human labels or cross-dataset pairs
are supplied. Native tactile targets and canonical chronological frame pairing
are preserved.

For the equations, worked transport/mixing example, both architectures,
gradient flow and checkpoint initialization, see
[FGW_ALIGNMENT_AND_MIXING.md](FGW_ALIGNMENT_AND_MIXING.md).

## Current first test: two placements only

The active experiment is `processing/runs/fgw_two_placements_v1`. The earlier
full sweep has been stopped and its completed checkpoints preserved.

This first test uses seed 45, stochastic partial FGW (mass 0.75), and exactly
two joint models: visual alignment and tactile-first alignment. Both start
from the existing epoch-10 parent and now train through epoch 30, following
the requested extension on October 12, 2026: 20 additional epochs per model,
40 joint epochs in total. There are no extra seeds, no control runs, no variant
screening, no finalist selection, and no automatic extension. The active
visual run resumes after its next completed checkpoint with its optimizer,
RNG, FGW state and learning rate preserved; its cosine decay is extended to
epoch 30. The queued tactile-first run uses the full epoch-30 schedule.

MotionTrans transfer is the primary result. Improvements in source dataset
reconstruction are not expected or required; those metrics are diagnostic.

Tactile-first requires one code-only teacher: 20 reconstruction warmup epochs
and 50 FGW epochs. It does not retrain the sensor autoencoders or decode
images. After both joint models finish, four small MotionTrans transfer heads
are fitted: the previously requested shallow and deep heads for each model,
using the full transfer-training split. Label-budget and calibration sweeps
are deferred. Comparison with the old parent is exploratory; extra training
is not isolated by a matched new control.

```bash
.venv/bin/python -m tactilelatent.fgw_pilot prepare
nohup .venv/bin/python -u -m tactilelatent.fgw_pilot run > processing/runs/fgw_two_placements_v1/controller.log 2>&1 < /dev/null &
```

Use the NVIDIA/BLAS environment variables below before running. Rerunning the
pilot resumes incomplete stages. `status.json` identifies its active stage;
`summary.json` records native reconstruction and transfer results. The pilot
also holds the old full-sweep lock to prevent an accidental concurrent restart.
The extension adds approximately 9–10 GPU-hours to the original budget.
`extension_status.json` records the checkpoint handover. The
full-ablation procedure below is retained for reference and is not active.

## Run and resume

Install the optional solver dependency with `.venv/bin/python -m pip install -e '.[alignment]'`.
Run from the project root. On this machine the matching NVIDIA userspace
libraries must currently be set before starting Python:

```bash
export LD_LIBRARY_PATH="$PWD/processing/runtime/nvidia-580.173.02/root/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
.venv/bin/python -m tactilelatent.fgw_study prepare
.venv/bin/python -m tactilelatent.fgw_study benchmark
nohup .venv/bin/python -u -m tactilelatent.fgw_study run > processing/runs/fgw_study_v1/controller.log 2>&1 < /dev/null &
```

Rerun the last command after an interruption. A controller lock prevents two
studies from owning the GPU concurrently. Completed jobs are reused and
incomplete trainers resume their last complete epoch. Config, code and teacher
fingerprints are checked; a software change requires an explicitly new study,
not resuming a mixture of implementations.

`processing/runs/fgw_study_v1/status.json` names the active worker and log.
`study.json` specifies the matrix; `configs/` contains reviewable trainer
configurations. Tactile-first visual configs are materialized after their
frozen teacher and common-code cache exist, with the exact checkpoint hash.

The runner screens five variants at both placements for seeds 45, 46 and 47.
Seed 45 forks the existing epoch-10 encoder; the other seeds get independent
10-epoch parents. Every visual condition continues from its matched parent.
Screening stops at completed epoch 20 while keeping a cosine horizon through
epoch 40, so finalist resumption does not reset its learning rate.

Teacher training shares a reconstruction warmup checkpoint per seed (20
epochs), then forks five 50-epoch conditions. Its loader reads existing codes
and native numeric values without loading scene images or tactile decoders.
Common-code caches contain source train/validation frames only. They retain
the common coordinates: their per-source stored statistics are diagnostic and
are deliberately not used to renormalize supervision independently by source.

## Mixer details

The source and support relational matrices use Euclidean distances. The feature
cost is squared Euclidean distance in the current common coordinates. One
distance scale is fixed at the first active epoch rather than renormalizing
away a collapsing representation at every solve.

`structure_weight=0` gives feature-only OT; `1` gives GW. The defaults are 256
episode-balanced states per source, 256 support points, five alternating
rounds, FGW weight 0.5 and consistency weight 0.1 ramped over two epochs.
Each state is the last valid frame in a canonical eight-frame clip. A quarter
of normal source clip budgets is replaced with bank clips, identically in all
conditions. Targets and masks are generated once per refresh and reused by SAM.

The coordinate update is the OTM alternating barycentric approximation, not a
globally optimal FGW solution. Conditional-gradient transport is nonconvex.
The deterministic variant uses the coordinate barycenter; stochastic variants
select exactly one supported source contribution per latent coordinate.

All push/pull operations divide by actual received mass. Partial matching uses
mass 0.75 and leaves untransported states without consistency supervision.
POT 0.9.6.post1 shares the partial outer-iteration cap with its inner EMD
solver, so it uses a separate 10,000-iteration limit and eight dummy points.
Mass-one partial matching uses the equivalent balanced solver. Invalid plans,
solver warnings and nonfinite losses stop a job rather than becoming targets.

## Evaluation and artifacts

Finalists are selected before MotionTrans is opened: rank equal-source native
MAE ratios, and reject any mixer worsening a source by more than 10% against
its placement-matched control in any seed. If no mixer passes, keep the
control and record that the placement failed its gate.

The original control and eligible placement winners continue to epoch 40.
The runner evaluates screening and finalist checkpoints using both 512-wide
heads, 10/25/100% nested training-episode budgets, and 64 latent draws. Heads
select duration using training-episode fit/tune partitions, then refit on the
selected training episodes. Validation episodes never train a head.

`summary.json` includes seed variation, episode-bootstrap MAE intervals,
paired differences against the original control, channel errors, dynamic
correlations and uncertainty coverage. `summary.csv` and exported PNG/PDF
plots summarize transfer. Run `.venv/bin/python -m tactilelatent.fgw_study report`
to refresh a partial report.

The existing viewer discovers regular model and per-epoch inference
checkpoints automatically. Transport banks and tactile teachers are separate
training artifacts; no solver is needed for visual inference. Validation
checks live outside the viewer's production run roots.

Lower transport cost or prettier latent plots do not establish physical
equivalence. Source sensors measure different quantities, and geometry can
have ambiguous correspondences. Reconstruction preservation and held-out
transfer are the scientific acceptance criteria. Previously inspected
MotionTrans validation is a development benchmark. Latent-sampling intervals
remain uncalibrated: stochastic mixing is augmentation, not a new uncertainty
estimator. After the point benchmark, the runner performs a separate calibration
experiment on the finalists. It retrains each head using a deterministic
70/15/15 episode split for fit/tune/calibration, and excludes calibration
episodes from refitting. The score is the maximum standardized residual over
an entire episode, across time and all six channels. Finite-sample conformal
quantiles produce 90% and 95% joint episode intervals. Their guarantee requires
exchangeable calibration/evaluation episodes and does not imply OOD coverage.
These heads and results live in `calibration/`, separate from the viewer and
the full-label point benchmark.
