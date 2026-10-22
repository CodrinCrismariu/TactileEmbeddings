# Baihu, EgoPressure and FEEL integration

Live status: [processing/data_expansion_v1/STATUS.md](processing/data_expansion_v1/STATUS.md).
Machine-readable status and append-only progress/deletion logs are beside it.

The `tactilelatent-expansion-downloads`, `tactilelatent-expansion-baihu`,
`tactilelatent-expansion-teachers`, and `tactilelatent-expansion-monitor` user
services continue independently of this terminal. Acquisition is pinned to
the revisions in `data_expansion.py`; staged downloads resume through the Hub.
The existing project-local NVIDIA 580.173.02 runtime is applied only to these
processes, matching the loaded kernel driver without a host installation.

## Stored data and deletion

Baihu has 3,013 canonical episodes / 2,564,143 frames. Available scene cameras
become 384×384 H.264 CRF18 videos with GOP 16 and unchanged frame counts.
PIKA lacks a head view; its camera mask reflects that. Four tactile surfaces
retain their order: left-left, left-right, right-left, right-right.

Marker extraction reads the native 240×240 tactile streams sequentially. The
temporal centroid tracker preserves accumulated displacement from the first
trackable reference; its 0.4-pitch association gate applies per frame, not to
total deformation. Missing observations remain NaN/false. Recovery uses the
local displacement of supported neighboring tracks; quality is a tracking
heuristic, not predictive uncertainty. The pilot uses two episodes per task.
Episodes with any surface below 90% average coverage retain their source
media and block promotion; the pipeline reports them rather than dropping them.

The initial reference-gated pilot revealed motion beyond its gate. Its exact
original tactile bytes are reacquired from the pinned mirror and verified
against their original checksums before replacing those initial pilot tracks.
No production teacher is trained on that preliminary representation.

Before removal, each episode must pass full RGB decoding/count checks,
marker artifact reopening, a disposable four-surface AE training update, a
code save/reload check, and geometry-only viewer rendering. Accepted receipts
pin every replacement and original checksum. Removal is file-by-file and
journaled before and after unlinking. Changed originals are retained.

After acceptance, original scene RGB, tactile streams, bimanual previews,
depth and IR are deleted. Only compressed scene RGB, marker vectors/masks,
reference geometry, metadata and joints remain. No tactile reference image
is retained. Viewer panels show measured/reconstructed markers on a sensor
grid. Top-level preview assets are removed after all canonical episodes pass.

EgoPressure keeps the original pressure/annotation Parquet and sequence
configurations; only ego RGB is resized/compressed and its original JPEG shard
retired. Every row is joined by the explicit `frame` identifier, including the
release's one-based numbering. Hand UV maps remain 224×224. Their magnitude is
restored by multiplying by `pressure_map_range[1]`. Negative interpolation
artifacts and absent annotations are masked, never converted to zero. True
zero observations remain supervised; no positive-pressure filtering is used.
The source's raw 105×185 sensor grid is retained unchanged for verification.

Participants 001–018 define training, 019–021 validation. A fixed sample of
training-only positive pressure defines scale `s` (99th percentile). The
transform is `y = p / (p + s)`, inverse `p = s*y/(1-y)`; no pressure amplitude
is clipped. Native values and calibration remain available. A full expanded
float32 UV-image cache is deliberately avoided.

FEEL Kitchen/P12 retains twelve sensor channels, left sensor0…5 then right
sensor0…5. Explicit frame IDs are matched to RGB. Empty/NaN channels remain
missing. These released processed readings are not presented as calibrated
physical force. The pilot is not admitted to source training.

## Teachers and future joint training

After a source is completely verified, train one teacher with seed 45 for 30
epochs: a four-surface, width-128, eight-query marker-set AE with 32D code for
Baihu; a 16/32/64/128-channel convolutional AE with 64D code for EgoPressure.
The pressure decoder uses sigmoid. Training uses up to 100,000 frames per
epoch; a fixed validation sample spans every validation episode. Report
reconstruction, zero baselines, temporal-change errors and pressure-count MAE.
CUDA is required for production teachers; no CPU fallback is automatic.

Best validation checkpoints are frozen and encoded chronologically into
per-episode float32 codes with explicit validity. Normalization statistics use
supported training frames only. Publication waits for BOTH sources, completed
teachers and checked code indexes. It writes a new manifest, registry and
`configs/joint_vjepa21_spatial_expanded_32d.json`. Historical manifests, teacher
checkpoints and caches are not rewritten.

The new joint entrypoint composes the original ten-source data/cache preparation
with the two new sources, retaining the current-frame / seven-history RGB
architecture and the existing SAM loop. It keeps MotionTrans held out and
FreeTacMan excluded. It does not create a visual-feature cache. Joint training
is NOT launched by acquisition or teacher training.

Once `ready.json` exists, explicitly invoke:

```bash
.venv/bin/python scripts/train_expanded_vjepa.py --config configs/joint_vjepa21_spatial_expanded_32d.json --preflight-only
.venv/bin/python scripts/train_expanded_vjepa.py --config configs/joint_vjepa21_spatial_expanded_32d.json
```

The sensor reconstruction and marker-tracking pages discover accepted Baihu
episodes while extraction progresses. EgoPressure appears after its fixed
normalization is published; teacher checkpoints appear as training saves them.
Original viewer episode/model IDs are retained within the running viewer.

## Verification

`tests/test_data_expansion.py` covers native pressure magnitude, explicit-ID
pairing, missing/zero distinctions, reversible normalization, four-surface
gradients and old two-surface defaults, cumulative shear, deletion guards,
failed-extraction retention, and artifact-only consumers after real deletion
of disposable test videos. Existing viewer/transfer checks remain applicable.
