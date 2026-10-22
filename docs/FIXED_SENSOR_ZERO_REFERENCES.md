# Fixed sensor zeros across episodes

Update: TacQuad and T-Dex were subsequently removed by the user on 2026-10-21.
Their measurements below describe the historical audit only; their data and
references are deleted and they have no viewer entries. References for the
remaining sources are unchanged. See `processing/retired_datasets/` for deletion
ledgers and former calibration metadata.

FEEL display update: its default is now one shared linear 0–2.5 detail range,
chosen from training positive values. Rare larger readings are explicitly flagged,
and full training ranges remain selectable. This changes only plotting; the saved
zero offsets and target values below remain unchanged. See
[FEEL display details](ACQUIRED_DATASETS_VIEWER.md).

The new viewer uses one frozen reference for each sensor layout, applied identically
to every training and validation episode. It also defaults to a fixed training-derived
display range, so changing episodes cannot magnify small signals by changing the scale.

**A common reference has been implemented; a reliable physical no-contact calibration
has not been established for every source.** HRDexDB's unloaded readings change between
recordings, and TacQuad DuraGel's neutral appearance changes substantially. Subtracting
one constant cannot remove those differences. We retain them and show the limitation,
rather than erasing weak interactions with an unapproved threshold.

## Results by sensor

| Source | Fixed reference shared across episodes | Evidence and limitation |
|---|---|---|
| T-Rex | Released numeric zero; 60 offsets of 0 | The current upstream release already tares each episode using its first 15 clean frames. We add no tare. Those original offsets are absent, so this is a common released coordinate convention, not a recovered common pre-tare physical zero. |
| FEEL Kitchen | Released processed zero; 12 offsets of 0 | Upstream piecewise drift correction produces contact-estimation targets, not calibrated force. Missing channels remain missing. |
| FEEL Lab | Released processed zero; 5 offsets of 0 | Five right-hand channels, verified against official Lab preprocessing and previews. A different native layout from Kitchen, requiring its own fixed range. No extra episode or participant correction is fitted. |
| T-Dex | One 720-component offset, shape 15 × 16 × 3 | Median across three visually screened open-hand recordings. Offsets range from −5.775 to +7.535 native counts. This is a provisional estimate with limited calibration coverage, not certified physical calibration. |
| HRDexDB Allegro V5 | One four-component median offset: `[0, 0, 0, 0]` | Visually screened unloaded training snapshots still reach **424 counts**. One offset cannot make all recordings zero. |
| HRDexDB Inspire F1 | One fixed zero for normal/tangential force channels | Screened unloaded training forces reach **413 counts**. All eight direction coordinates are retained unchanged, including native −1 values; directions are not force measurements. |
| TacQuad GelSight Mini | One native 320 × 240 RGB neutral texture | Object-balanced median of training pre-contact candidates, shared across all subsets and episodes. Useful as an image-change reference; not pressure calibration. |
| TacQuad DIGIT | One native 640 × 480 RGB neutral texture | Same method. Lighting and neutral-texture differences remain; an image residual is not automatically contact. |
| TacQuad DuraGel | One diagnostic native 640 × 480 RGB template | **Fails the contact-separation check.** Kept explicitly labelled as unreliable for no-contact calibration; not promoted as a contact detector. |
| TacQuad Tac3D | Native force, torque and displacement nominal zero retained | No independent own-stream unloaded intervals are available. Optical annotation times are not borrowed. Absolute `pos` arrays contain geometry; zero position does not represent no contact. |

HRDexDB Inspire DFTP's published tactile arrays are empty. It has no available tactile
target to calibrate; missing data is not filled with zeros.

For previously integrated sources: EgoPressure's released pressure value 0 already maps
to normalized 0. Baihu's zero displacement means its saved reference marker geometry;
that geometry is not independently certified as unloaded. Existing teachers and codes
are unchanged.

## How the references were estimated

Numeric transformation is `referenced = raw − fixed_offset`, applied per native channel.
It preserves signed values, native shape, and missing masks. No per-episode, per-frame,
or per-participant offset lookup is used at inference. No deadband, clamping, or minimum
subtraction is applied. Consequently a noisy unloaded measurement need not be exactly 0.

HRDexDB uses the exact screened snapshots saved in the candidate montage pages:
39 training episodes per morphology fit the median and empirical unloaded envelope;
11 validation episodes per morphology check them. These are visual screens for external
object contact, not certified mechanical calibration recordings. Pose, preload, sensor
drift, or other sensor effects can remain in a free-air reading. The empirical envelope
does not isolate electronic noise, establish SI gains, or constitute a confidence interval.

For Allegro, 4.55% of validation snapshot components lie outside the training envelope;
for Inspire force components, 2.84% do. Their largest absolute validation unloaded
residuals are 153 and 135 native counts respectively. Direction components are excluded
from this calculation. An observation inside the envelope is **not** automatically
labelled no contact.

T-Dex uses demonstrations 6, 8 and 10, at tactile frame indices 70, 77 and 84. All nine
RGB frames were inspected and show an open hand above the objects. The offset is the
median of the three recording medians, giving equal recording weight. The largest
absolute residual among these nine observations is 23.865 counts. There is no independent
visually certified T-Dex calibration validation set yet. Other recordings start while
holding objects, so their first frames were deliberately excluded.

Only training episodes fit numeric ranges. Each compatible quantity shares the maximum
absolute referenced training value across its channels: separate force/moment ranges
for T-Rex, separate XELA axes, and separate normal/tangential/direction ranges for Inspire.
The same saved ranges apply to validation episodes. Values exceeding a fixed range remain
in the exact numeric table and are marked in the plot; they are not clipped in the data.

## TacQuad optical checks

Templates use each optical stream's **own** annotated contact interval. Candidate frame 0
must precede contact by at least 10 frames and have a verified zero-based native filename
origin. This is an annotation-based candidate rule; it is not a guarantee of zero force.
No cross-sensor timing assumption is made.

Objects are split by `SHA256("45:" + folder) mod 5`: zero is validation, the rest training.
The same object cannot fit a template in one subset and validate it in another. Median
images are computed per object and then across training objects at native resolution.

| Sensor | Training objects | Validation objects | Validation pre-contact MAE | Validation contact MAE | Contact difference exceeds its own pre-contact difference |
|---|---:|---:|---:|---:|---:|
| GelSight Mini | 109 | 28 | 2.340 | 5.647 | 32/32 (100%) |
| DIGIT | 121 | 30 | 3.586 | 7.305 | 32/34 (94.1%) |
| DuraGel | 77 | 19 | 14.983 | 19.193 | 12/19 (63.2%) |

MAE columns report medians over observations in native 0–255 RGB counts. Duplicate object
appearances account for different object and observation counts. The simple separation
check is descriptive, not a fitted contact detector. Its training-only acceptance rule
requires a larger contact residual in at least 90% of pairs; DuraGel fails in training
(59.0%) as well as validation. No threshold is tuned on validation.

The diagnostic view shows mean absolute RGB difference on a fixed 0–64 count color scale,
at native pixel dimensions. Black means no difference from the saved template; it does
not guarantee no contact. The signed full-resolution residual is available in the Python
helper. Raw image downloads remain byte-identical.

## Viewer and training use

In Sensor reconstruction, select a newly acquired numeric source:

- **Zero reference → Fixed sensor reference · all episodes** is the default.
- **Sensor display scale → Fixed training range · all episodes** is the default.
- Amber bands show the screened training unloaded envelope where available.
- Select **Unchanged released readings** to compare coordinates. The exact table shows
  raw values, displayed values, and the saved offset side by side.

On TacQuad benchmark, choose **Difference from fixed sensor background · all episodes**
for an optical tactile stream. The DuraGel failure is labelled explicitly. Independent
timelines, scene-camera mappings, raw downloads, historical decoders and checkpoint
selection are preserved.

No joint run, teacher retraining, or code-cache rewrite was launched. New data training
can explicitly opt into `FixedReferences.transform(source, values, valid, channels,
revision)` and pin the registry hash. It does not accept an episode identity or fit any
statistics. Applying a new target convention to a teacher requires a separately versioned
training/cache setup; historical targets must not change beneath existing checkpoints.

## Artifacts and verification

- Registry: `processing/zero_reference_v1/references.json`, with a verified content hash.
- Exact selected frames and raw target/RGB hashes: `reviewed_numeric_frames.json`.
- Optical candidates, object splits, member hashes, and contact-frame paths:
  `optical_candidates.json`.
- Native neutral templates: `neutral_*.npz`, checksummed in the registry.
- Statistics and omitted optical candidates: `audit.json`.
- Live browser evidence: `browser_verification.json`, screenshots, and `browser.log`.
- Builder: `scripts/build_zero_references.py`; all source inputs remain immutable.

The builder verified target checksums for all 5,619 acquired numeric recordings before
using their training statistics. MotionTrans and FreeTacMan do not participate. Forty
focused reference/acquisition/mirroring tests passed, followed by browser checks of two
episodes for every numeric source, both FEEL layouts, and all three optical previews.
The browser confirmed unchanged raw values and identical reference metadata across
episodes of the same sensor layout. All 12 storage-provenance file hashes remain intact.

Rebuild explicitly with:

```bash
.venv/bin/python -u scripts/build_zero_references.py
.venv/bin/python -m pytest -q tests/test_zero_reference.py tests/test_acquisition_viewer.py tests/test_bimanual_mirror.py
.venv/bin/python -u processing/zero_reference_v1/browser_audit.py
```

The reviewed calibration list is explicit. Merely adding new episodes does not trigger
any background refit. A deliberate rebuild can change the reference version and hash.

## Primary-source provenance

- [T-Rex release](https://huggingface.co/datasets/zekaiwang/trex_dataset/blob/8b478400a9167d59ede0fdbfea616cb5e9791885/README.md): upstream median taring and blackout fills.
- [FEEL release](https://huggingface.co/datasets/edessa/feel/blob/0f8179649b50afcfd684c87784c67787871db361/README.md) and its `preprocess.py`: processed contact targets, drift correction, and missing-channel policy. Its separately referenced `zero_correct.py` is not provided in the release inspected here.
- [T-Dex implementation](https://github.com/irmakguzey/tactile-dexterity/tree/4e0af42f8de1fad243657619763c6f73f4c0d512): native XELA loader and axis order. Acquisition metadata and retained target hashes pin the selected local pilot; its manifest has no dataset revision string.
- [HRDexDB release](https://huggingface.co/datasets/HRDexDB/HRDexDB/tree/98b8f8ea2e361a3c7df38b5d8f3f8395913773f2): native tactile payloads and explicit timestamp mappings. No independently documented unloaded tactile calibration was found.
- [TacQuad release](https://huggingface.co/datasets/xxuan01/TacQuad/tree/7b46e7836f6688788eb52e6988d563ea334f5dd0): native archives and contact interval CSVs.
