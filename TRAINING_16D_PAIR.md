# Matched 16D latent experiment

Requested on October 12, 2026. Two models train concurrently from fresh,
identical seed-45 weights to epoch 30:

| Run | Alignment | Common latent |
|---|---|---:|
| `processing/runs/joint_16d_pair_v1/no_fgw` | Original sensor-code + numeric supervision and Gaussian KL | 16 |
| `processing/runs/joint_16d_pair_v1/visual_fgw` | Same objectives plus visual stochastic partial FGW and spread penalty | 16 |

The shared DINO/spatial/temporal architecture, width 768, Gaussian heads,
128→128 sensor adapters, frozen sensor AE codes/decoders, and numeric sigmoid
heads are preserved. Sensor-specific code dimensions remain 16–256; reducing
the common latent does not reduce sensor output size or resolution. All eleven
source datasets participate; MotionTrans is excluded from representation
training, alignment banks, and statistics.

Both runs use batch size 2, accumulation 16, SAM-AdamW (rho 0.05), learning
rate 5e-5 with two warmup epochs and cosine decay to 1e-6, weight decay 0.05,
dropout 0.1, wrist dropout 0.3, and Gaussian KL ramp to 0.001 over ten epochs.
Each source retains its existing maximum 8,000 training windows per epoch and
64 source-validation windows. Source validation remains diagnostic; there is
no source-validation early stopping or sweep selection.

For a matched comparison, both use the same source schedule and 25% bank
clip replacement, with episode-balanced banks of at most 256 states per
source. The plain model receives no transport, consistency or spread loss.
The FGW refresh restores the global Torch RNG afterward so bank feature
extraction does not change training dropout or latent-sampling draws. Both
initial weight hashes are checked by the controller and saved in
`matched_initialization.json`. This plain run differs from the historical
32D run in its explicit matched sampling protocol as well as its 16D latent.

FGW retains the previous visual settings: transported mass 0.75, structure
weight 0.5, 256 barycenter support points, five update rounds, consistency
weight 0.1 ramped over two epochs, and spread weight 0.01. Training alternates
detached epoch-wise correspondence refreshes and ordinary SAM updates;
transport targets persist across SAM replay. No tactile-first teacher is
trained for this pair.

The models start at epoch 1 without loading the old 32D bottleneck or its
optimizer. Checkpoints, effective rank, native reconstruction diagnostics,
and per-epoch viewer snapshots are saved for each run. Transfer and classifier
evaluations are not automatically scheduled by this launcher.

## Operation

The active nohup controller is `scripts/train_16d_pair.py run`. Its root
`status.json` lists both worker PIDs and last completed epochs;
`controller.pid` identifies the supervisor. Logs are `logs/no_fgw.log` and
`logs/visual_fgw.log`. The controller fails both jobs if either exits with an
error or the verified initial weights differ. Completed weights are retained.

With the matching NVIDIA userspace libraries and BLAS thread limits set,
rerun `.venv/bin/python scripts/train_16d_pair.py run` to resume this pair
after a requested stop. Do not call `prepare` over an existing experiment.
The controller lock prevents duplicate launches.

Two concurrent jobs share one GPU; concurrency does not double throughput.
The previous single-run rate was roughly 25–26 minutes per epoch. Sixty total
epoch-equivalents suggest about 26 GPU-hours before contention and overhead;
replace this estimate with the new pair's measured complete-epoch throughput.
