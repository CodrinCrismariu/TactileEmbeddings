Sensor autoencoder viewer
=========================

From the project root:

```bash
.venv/bin/python scripts/view_egotouch_autoencoder.py
```

Open http://127.0.0.1:8080 in a browser. The sensor reconstruction page runs on CPU
with no additional dependencies. Select a dataset/sensor, split and video, choose an architecture/checkpoint, and
scrub the frame slider or use Play. The recorded camera frame, measured pressure,
reconstructed pressure, and absolute errors update together. Hover over a grid
cell for numerical values. Pressure maps share a scale; the error scale can be
adjusted separately to expose small discrepancies. Gray cells are invalid and
excluded from the frame metrics. Frame indices start at zero.

The dropdown includes the latest EgoTouch v4 search, its initial sensor-code
grid, the original EgoTouch conv/MLP preset runs, and full-resolution sensor
models under processing/runs/sensor_autoencoders_native_v2.
Obsolete force/torque autoencoders, smoke runs, and reduced-resolution optical
models are omitted. Only runnable models belonging to the selected dataset are
offered; each page lists datasets with a compatible reconstruction model. Optical sensors use
RGB images, taxel vectors use stored-index charts, and force/torque include raw
reading comparisons. Models refresh every 30 seconds or via the Refresh button.
Optical originals retain their stored pixels, with no resize or crop. Enable
"Inspect source resolution and temporal changes" for an amplified RGB difference
from episode frame zero. This reference may already contain contact. Browser
layout fits images to the screen without changing their source pixel arrays.
FreeTacMan uses audited per-episode camera roles; the selected ego and sensor video
filenames are visible in the controls. In SprinkleSesame_1 through _53, ego is
camera1 and Surface1 is camera3. Surface2 remains camera2. Other episodes keep
their original camera assignments.
VLA Touch tactile frames use an exact-pixel lookup against the original numbered
GelSight JPEGs to correct scrambled upstream frame order in 379 of 380 episodes.
The remaining episode is genuinely constant. The controls label corrected
chronology; reconstruction and temporal differences use the corrected frame
and reference. Original source arrays and active decoder training are preserved.
See processing/audits/vla_touch_native_dynamics/REPORT.md for evidence and the
separate corrections still needed before VLA joint visual-to-touch training.
See SENSOR_AUTOENCODER_PRETRAINING.md for the complete training plan and targets.
The latest search's best validation score is selected first. Architecture details,
sensor-code dimension, best epoch, and validation MAE appear below the dropdown.
The 10 `wide48_*` and `wide64_*` checkpoints belong to EgoTouch. Direct links can
select a dataset and autoencoder by name, for example
`http://localhost:8080/?page=sensor&dataset=egotouch&model=wide64_code256`.
Other sensors use sensor-specific names: FreeTacMan, Unit, Unit Bimanual,
VLA Touch, and ViTaMIn have `large_code256` models; OpenTouch has `large_code32`.
The dropdown shows actual convolutional channels or MLP hidden widths alongside
the code dimension. EgoTactile and OpenGraph retain their trained MLP candidates;
there are no separate saved `wide*` runs for these two datasets.
Two models and two pressure episodes are cached in memory; checkpoint weights
are loaded only when selected. Video frames are decoded on demand. Playback
is intended for inspection and can run slower than the original video; the time
readout uses the source video's frame rate.

To choose different model directories or a port:

```bash
.venv/bin/python scripts/view_egotouch_autoencoder.py \
  --checkpoint-roots processing/runs/egotouch_ae_search_v4/trials \
  --port 8081
```

To leave the viewer running after the terminal closes:

```bash
nohup setsid .venv/bin/python -u scripts/view_egotouch_autoencoder.py \
  > processing/runs/egotouch_autoencoder_viewer.log 2>&1 < /dev/null &
```

The current port 8080 server is managed by the user service
`tactilelatent-viewer.service`, with restart on failure, so it is independent of
the tool session that launched it. Use `systemctl --user status
tactilelatent-viewer.service` to check it and `systemctl --user restart
tactilelatent-viewer.service` after Python changes. Its log is
`processing/joint_alignment/viewer/server.log`.

The autoencoder receives measured pressure, not RGB. This viewer checks tactile
reconstruction and provides video context. Browsing test episodes is qualitative
inspection; no test evaluation or training is launched.

Access from another computer over SSH
------------------------------------

Run this in a terminal on your own computer, not in the server's SSH terminal:

```bash
ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:8080:127.0.0.1:8080 codrincrismariu@130.64.84.163
```

Leave the terminal open, then visit http://127.0.0.1:8080 in your local browser.
Use the SSH host alias you normally connect with instead of the IP if necessary.
If local port 8080 is occupied, change the first 8080 to 8081 and browse
http://127.0.0.1:8081. In VS Code Remote SSH, you can instead forward port 8080
from the Ports panel and open the forwarded address. No server firewall changes
are needed for either method.

From VS Code's connected integrated terminal, you can also ask the editor to
resolve and open the remote viewer directly:

```bash
code --openExternal http://127.0.0.1:8080
```

The viewer also supports browser IDE proxy prefixes such as /proxy/8080/:
its API requests stay under the same prefix as the page.

Visual-model comparison page
----------------------------

Click **Ego → tactile prediction**, or open
http://127.0.0.1:8080/?page=joint. This page predicts tactile outputs from scene
images, without feeding recorded tactile data into the visual model. It supports
joint baseline, SAM, adversarial, and sensor-code alignment checkpoints, as well
as the numeric direct controls. The new alignment model passes predicted sensor
codes through its pinned frozen decoder at native resolution.

Choose a dataset, episode, visual model A, and saved epoch. Optionally choose
model B and its epoch for simultaneous comparison. Both panels show the same
recorded target, their respective predictions, errors, and frame metrics; they
share the display scales. The preview camera selector only changes the video.
**Model input cameras** defaults to ego only and also offers all available scene
cameras. ViTaMIn has only a wrist scene camera, so ego-only input is disabled.

Inference uses the Gaussian mean and eight temporal frames at stride two,
with the selected frame at position three. Clip boundaries repeat the nearest
valid frame. Bidirectional attention can see future frames in the clip; this
viewer does not perform streaming-causal inference. The exact input indices,
camera slots, loaded epoch, and response time are displayed. Original tactile
targets are never resized. Historical 32x32 optical heads are omitted from
native-image comparison. Held-out sensors require a trained head;
the viewer does not invent a head for them.

Models and saved epochs refresh every 30 seconds or on **Refresh trained models**.
An ongoing run becomes selectable after its first complete checkpoint is saved.
**Latest saved** reloads when the trainer atomically replaces `last.pt`. Epoch
options list only existing snapshots; missing historical epochs cannot be
recovered. Smoke runs, invalid checkpoints, and models without saved weights are
omitted. The full new alignment experiment appears after its first valid
checkpoint is saved. Dataset splits and camera choices include available data only.

MotionTrans transfer evaluations appear as separate `MotionTrans transfer`
models on the visual prediction page. They use the frozen shared encoder and
the separately fitted six-output decoder saved by the held-out evaluation.
SAM has matched encoder/decoder pairs for encoder epochs 3 and 50; the non-SAM
holdout also offers its evaluated decoder sizes. Each selector entry shows the
encoder epoch and decoder widths. The viewer validates the encoder SHA256 saved
with each decoder and omits missing, changed, or incompatible pairs. Transfer
heads appear only for MotionTrans, and only evaluated pairs can be selected.

The transfer preview is deterministic: `decoder(mu)` for the selected frame.
It is not the evaluation's Monte Carlo ensemble mean or an uncertainty interval.
The shared encoder was frozen during decoder fitting; the decoder was trained
on MotionTrans training labels, so this is supervised transfer to a held-out
sensor. Both modules are frozen during viewing. Model A/B can compare epoch 3
against epoch 50 with their respective fitted decoders, without retraining.

Future joint, SAM, adversarial, and alignment trainer executions save immutable
inference weights after each completed epoch under `viewer_epochs/epoch_NNN.pt`.
These omit optimizers and RNG state and cannot resume training; use `last.pt`
for resumption. Typical full-size weights use about 0.2 GiB per epoch, or roughly
10 GiB across 50 epochs. Set `save_viewer_epochs: false` in a training config to
retain only its existing periodic/full checkpoints.

The server loads DINO only on the first visual prediction. It keeps two visual
models, 64 recent frames of DINO features, and eight predictions in bounded
memory caches. CUDA is selected when available; CPU inference is supported but
slower. To customize discovery or device:

```bash
.venv/bin/python scripts/view_egotouch_autoencoder.py \
  --joint-roots processing/runs processing/joint_alignment \
  --joint-device auto --port 8080
```

The browser check covers actual SAM epoch 50 versus epoch 10, numeric output
tables, native optical alignment decoding, camera availability, and both pages.

OpenTouch uses the official right-hand channel coordinates on both pages,
including model A/B comparisons. The bundled mapping is pinned to upstream
commit `6d1fa38f19d034da431f7a81ef376ad09381ca89`: 169 mapped channels and 87
unique excluded nodes. Row/column indices, orientation, and coordinate aspect
ratio are preserved, with one dot per channel and no interpolation or resizing
of sensor data. Hover shows the exact channel ID and normalized stored values.

Original and reconstruction colors show `1 - raw / 3072` as relative touch
response, using the same fixed scale across frames and models. This is not a
force calibration or the upstream MANO renderer's per-frame normalization.
Error metrics use the mapped channels only; generic pressure contact metrics
are hidden for OpenTouch. Existing autoencoder inputs, teachers, caches, and
training losses remain unchanged. The mapping source is linked in the viewer.
