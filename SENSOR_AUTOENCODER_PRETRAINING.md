Native sensor autoencoder pretraining
====================================

Train tactile-only autoencoders for eight reconstruction datasets, while reusing
the completed EgoTouch searches. RDP (one force), RDP Bimanual (two forces), and
MotionTrans (six torque values) are excluded from autoencoder pretraining; these
small numeric outputs should use direct prediction heads in joint training.
Optical inputs and outputs retain every
stored pixel: no resizing or cropping is performed. The earlier v1 optical runs
used unapproved 32×32 targets and are superseded. Their checkpoints are disabled
in the viewer. Full-resolution replacement runs are in sensor_autoencoders_native_v2.
No scene RGB, DINO, shared visual encoder, KL or uncertainty head is
trained here. Sensor code sizes remain independent of the shared visual latent.

The user's selected budget is 30 screening epochs and 100 total epochs for
finalists. The active search uses exactly two candidates per dataset, in
configs/sensor_autoencoder_search_native_two_reconstruction.json. Both screen for
30 epochs; exactly one overall validation winner advances to 100 total epochs,
irrespective of code dimension. No extra seed runs or LR/loss variants are scheduled.
Promotion uses each candidate's first 30 epochs, even if a previous plan already
extended one candidate. Completed historical checkpoints remain preserved.
Initial models run three bootstrap epochs
first so every sensor can appear in the viewer before full searches complete;
these checkpoints continue into screening with optimizer/RNG state intact.

| Source | Reconstruction | Architecture | Code dimensions |
| --- | --- | --- | --- |
| EgoTouch | Bilateral [2,21,21] pressure | Existing trained conv models | Existing grid/search |
| EgoTactile | Right-hand [1,1,256] stored taxels | MLP [256,128] | 16,32 |
| OpenTouch | Right-hand [1,16,16] pressure | Conv [32,64] / [64,128] | 16,32 |
| OpenGraph | Bilateral [2,1,80] glove vectors | MLP [128] | 8,16 |
| FreeTacMan | Two [6,480,640] optical surfaces | Medium / large conv | 128,256 |
| Unit Bimanual | Bilateral [6,240,320] optical grippers | Medium / large conv | 128,256 |
| Unit / VLA Touch | Right [3,240,320] optical gripper | Medium / large conv | 128,256 |
| ViTaMIn | Two right-gripper [6,224,224] optical images | Medium / large conv | 128,256 |
| RDP / RDP Bimanual / MotionTrans | 1 / 2 / 6 numeric values | Direct prediction; no AE search | Not applicable |

The optical candidates are medium/code128 ([16,32,64,128,128]) and
large/code256 ([32,64,128,256,256]). Encoder stride operates on internal learned features;
the input, output, and reconstruction loss cover the full recorded image.
OpenGraph compares compact128/code8 and compact128/code16; EgoTactile compares
medium/code16 and medium/code32; OpenTouch compares medium/code16 and large/code32.
Historical numeric autoencoder checkpoints and affine
decoder support remain preserved, but no force/torque autoencoders are scheduled
or considered for promotion by the active search.
All decoders use a single sigmoid output. Their final shapes match the sensor.
Invalid cells are masked at input and excluded from loss and evaluation.

AdamW starts at 1e-4 for the active pressure/glove/optical sensors.
The previous exhaustive plan's LR/loss variants are excluded. The loss
is squared error plus .5 absolute error. Pressure/glove readings above .1 have
weight 3; optical and signed measurements have no pressure contact weighting.
ReduceLROnPlateau halves LR with patience 6 and minimum 1e-6. Selection uses
balanced overall/contact/high-pressure MAE for pressure and MAE otherwise.
Optical reports include PSNR. Scores are compared within each dataset only.
Force/torque retain the existing tanh normalization: raw zero is .5, not zero.

Data caching and launch
-----------------------

Numeric sensors use float32 memory-mapped stride-two train/validation caches,
retaining invalid cells as NaN. Optical sensors decode full-resolution frames
directly from their existing compressed videos or tar/Zarr arrays. This avoids
creating a multi-terabyte decoded optical cache. Episode-local batches reuse
decoding within each batch. No reduced optical cache is consumed or built.
Each epoch samples up to 100,000 training frames; validation uses all cached
validation frames. Both train and validation retain their existing episode splits.
Test targets are not cached or used for selection. Cache building resumes at
episode boundaries. Manifest and target-loader hashes protect cache compatibility;
source/plan hashes protect search resumes. Audited underlying media must remain
unchanged. Optical batch sizes are 4 for FreeTacMan and 8 for the FTP sensors;
CUDA computation uses bfloat16, while targets and error metrics remain float32.

FreeTacMan camera roles are assigned per episode. SprinkleSesame_1 through _53
swap ego camera1 with Surface1 camera3; Surface2 remains camera2. All other
episodes retain ego camera3 and tactile cameras1/2. The full 5,625-episode camera
audit is in processing/audits/freetacman_camera_roles. FreeTacMan trials use
processing/joint_all/manifest_sensor_roles_corrected_v2.json; other sources keep
their unchanged input manifests. The indexer uses configs/freetacman_camera_roles.json
for future manifest generation. Historical FreeTacMan validation scores from the
uncorrected manifest require reevaluation.

```bash
mkdir -p processing/runs/sensor_autoencoders_native_v2
nohup setsid .venv/bin/python -u scripts/train_sensor_autoencoders.py \
  --plan configs/sensor_autoencoder_search_native_two_reconstruction.json \
  --output processing/runs/sensor_autoencoders_native_v2 \
  > processing/runs/sensor_autoencoders_native_v2/nohup.log 2>&1 < /dev/null &
```

The active suite has 16 screened candidates, eight single finalists and no seed
repetitions: 1,040 planned epochs across eight sources (2×30 + 70 per source).
The previous 3,680-, 4,990- and 7,990-epoch plans are preserved for provenance.
Existing starter
checkpoints excluded by the new plan remain available, but are filtered out of
search comparison and finalist selection. Previously completed work beyond
the new budget is historical work and is not rerun or erased.

The initial four-candidate handover completed after the FreeTacMan bootstrap;
four_candidate_migration.json records that historical transition. A subsequent
user-requested removal of the three force/torque sources is recorded in
removed_numeric_autoencoders.json, with old plan, provenance, status and comparisons
archived under provenance_revisions/before_removing_force_torque_autoencoders.
The further reduction to two candidates is recorded in two_candidate_migration.json,
with old plan, provenance, coordinator code and promotion decisions archived under
provenance_revisions/before_two_candidates. Promotion now uses one overall winner
and matching screening budgets, and seed repetitions are disabled.
Existing retained-source trial parameters and checkpoint directories are unchanged.
The active nohup coordinator uses the reconstruction-only plan. Relaunching skips
completed work and resumes unfinished checkpoints. status.json records cache,
bootstrap, screening, finalist and repetition stages. comparison.csv and
best_by_source.json update after completed runs. Frozen per-source finalist and
repeat files preserve promotion decisions. seed_stability.json is written when
the suite completes. With one seed, sample_std is null; no repeated-seed stability
claim is made. The exact resolved plan is stored in plan.json.

For one source or another budget, use a fresh output:

```bash
.venv/bin/python scripts/train_sensor_autoencoders.py --sources ftp_unit \
  --screen-epochs 30 --final-epochs 100 --output processing/runs/ftp_unit_ae_search
```

The dedicated trainer also accepts explicit channels, hidden widths, learning
rate, batch size and code size; see scripts/pretrain_sensor_autoencoder.py --help.
Optical standalone training requires --native-optical and refuses reduced targets.
The active plan is configs/sensor_autoencoder_search_native_two_reconstruction.json. Copy and
edit it before a new search, then pass --plan YOUR_PLAN.json with a fresh output.
For an affine decoder, pass --architecture mlp --hidden-dims with no values.
Existing nonempty-width MLP and conv checkpoints load without parameter changes.

The design uses EgoTouch v4 ablations as a prior for detailed reconstructions:
at a matched first-60-epoch budget, wide64/code256 MAE was .006845 versus .008249
for code128 and .012311 for code64; deep/code256 was .012190. This supports
width/code capacity for richer targets, not large codes for one or two numbers.
Performance on a different sensor still requires validation. EgoTouch's completed
wide64/code256 winner and repetitions are preserved without new training.

Viewer and decoder reuse
------------------------

```bash
.venv/bin/python scripts/view_sensor_autoencoders.py --port 8080
```

The dataset dropdown filters episodes and checkpoints to the selected sensor.
Pressure grids show heatmaps; taxel vectors show stored-index bar charts; optical
sensors show RGB images with mean-channel error maps; force and torque show
bounded values plus numerical inverse-transform comparisons in source reading
units. The viewer does not assert SI calibration. Models refresh every 30 seconds
or via Refresh trained models. Original optical panels show full-resolution,
pixel-exact source RGB. The optional source inspection also shows amplified
absolute change from episode frame zero, which may already contain contact.
Reduced-resolution v1 optical checkpoints remain listed but cannot be selected.
Current best checkpoints reload when training
updates them. An untrained source shows original data with blank reconstruction
and an explicit awaiting-model message. Legacy EgoTouch models remain selectable.
See EGOTOUCH_AUTOENCODER_VIEWER.md for remote access instructions.

Each best.pt contains model_config, full model, standalone encoder/decoder state
dicts, source config, and validation metrics. Later visual training can reuse it:

```python
saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
ae = make_autoencoder(saved["model_config"])
ae.load_state_dict(saved["model"])
ae.eval().requires_grad_(False)
with torch.no_grad():
    teacher_code = ae.encode(target, valid)
sensor_code = adapter(shared_latent)  # [B,T,K], from a trainable multilayer adapter
shape = saved["model_config"].get("target_shape", [2, 21, 21])
prediction = ae.decode(sensor_code.flatten(0, 1)).reshape(B, T, *shape)
# Keep autograd enabled through the frozen decoder to train the adapter/encoder.
```

Pretraining MotionTrans uses its training tactile readings to learn a target
sensor decoder. A subsequent transfer experiment must disclose that target-sensor
pretraining; it would not be a strict experiment with no target-sensor data.
This suite does not modify existing joint encoder runs or establish invariance
or calibrated predictive uncertainty by itself.
