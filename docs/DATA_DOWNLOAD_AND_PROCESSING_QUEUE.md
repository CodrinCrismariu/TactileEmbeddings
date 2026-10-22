# Approved download and processing queue

Live status: [processing/data_queue_v2/STATUS.md](../processing/data_queue_v2/STATUS.md).
Active configuration: [configs/data_queue_active.json](../legacy/v1/configs/data_queue_active.json).
The original v2 configuration remains frozen for historical acquisition receipts.

The queue acquires and processes these sources with independent workers. It does not start a new
joint model training run or change historical manifests, checkpoints, or viewer
entries. Existing EgoPressure acquisition, Baihu extraction, and their teacher
jobs continue independently.

| Order | Source | Scope | Retained allocation |
|---|---|---|---:|
| 1 | T-Rex | Full selected release: three scene cameras and native 60D wrench estimates; no tactile video | 100 GB |
| 2 | FEEL | Expand P12 to all participants with final aligned CSVs; reuse existing P12 video | 12 GB |
| 3 | HRDexDB | Up to 50 episodes per tactile-enabled robot embodiment, balanced across objects; one fixed scene camera | 18 GB |

TacQuad and T-Dex were removed by the user on 2026-10-21. Their dataset files,
previews and references were deleted. They are absent from the active queue,
and the normal launcher rejects requests to acquire either source.

All sizes are decimal GB. The 130 GB active allocation is a ceiling, not an assurance
that every release will fit its projection. A source pauses if its retained
allocation is exceeded. The queue also enforces a 700 GB combined
`datasets/` + `processing/` cap with 30 GB reserved for future experiments, a
100 GiB physical free-space reserve, and at most 65 GB of raw staging. Raw
FEEL archives temporarily exceed the size of their retained compressed videos;
they count against the project cap and physical reserve while present.

## Common processing

1. Acquire a pinned release. HF files get revision, byte length, upstream etag,
   SHA256 and receipt checks; partial transfers resume with HTTP Range. T-Dex
   uses a pinned Drive file ID, archive size and member-index digest; each
   selected member is CRC-checked and gets a local SHA256 receipt. A Drive file
   ID alone is not treated as an immutable content version.
2. Preserve native tactile arrays, signed values, original channel identities,
   timestamps, explicit frame mappings and validity masks. Measured zero stays
   valid. Missing data never becomes a zero observation.
3. Convert **scene RGB only** to 384×384 H.264 CRF18, preserving one encoded
   frame per input frame. Decode the replacement, check its frame count and
   chronological timestamps, and check sample fidelity before publishing it.
   Default chroma sampling is YUV 4:2:0. If it exceeds the fixed 8/255 sample
   RGB MAE limit, retry the complete video in YUV 4:4:4 at the same CRF18 and
   resolution. The limit is unchanged; failure of both formats retains the
   original. Receipts record the actual format and retry diagnostics.
4. Delete only the corresponding verified raw RGB staging file. Keep a per-file
   deletion ledger with hashes and replacement identity. Native tactile assets
   are never resized or lossily compressed by this queue.
5. Publish complete episodes incrementally in a separate source manifest.
   Failed episodes do not enter that manifest; their raw staging data remains
   available for investigation. Final processing computes train-only reversible
   normalization statistics and decodes an eight-frame history for each available
   source schema/split group.

The numeric reader in `acquisition_data.py` returns native values and masks,
eight-frame RGB histories and current-frame targets. It uses explicit camera
frame maps and rejects histories that cross episode boundaries or move backward
through a camera stream. This is a separate acquisition contract; publishing a
manifest does **not** automatically add an unreviewed source to existing training.

## Source-specific details

### T-Rex

Download the released Parquet targets and metadata, then interleave head,
left-wrist and right-wrist video conversion. Episodes may cross numeric Parquet
shards; reconstruction joins by episode, frame and global index. Shared video
shards stay shared, with verified per-episode frame offsets.

Keep all 60 channels in the released order: left/right, five fingers, then
Fx/Fy/Fz/Mx/My/Mz. These are sensor-derived wrench **estimates**. The release
includes taring and previous-frame blackout fills without an exposed per-frame
imputation mask. Finite values are not presented as proof of independent physical
measurements or calibrated sensor accuracy.

### FEEL

Read final processed sensor columns dynamically: the inspected Lab CSV has five
channels, while Kitchen has six per hand. Keep these schemas distinct. Retain
processed missing values as NaN plus masks and preserve raw CSVs for provenance.
Join on explicit `frame_idx`; natural-sort JPEG indices before conversion. Split
by participant, including across environments. Do not interpret processed contact
values as calibrated force.

The pinned release documents corrupt JPEGs in Lab/P21. Such a participant must
remain excluded if it cannot pass frame decoding; its archive is retained and
the quality report makes that exclusion explicit. Participants without final
aligned CSVs are recorded without downloading unusable RGB.

### T-Dex (removed; historical processing notes)

Removed at the user's request on 2026-10-21. Its pilot data and caches have been
deleted, and it is excluded from the active queue. The notes below describe the
previous acquisition; they are not instructions for a scheduled job.

Read selected members directly from the 31.44 GB play-data ZIP over bounded HTTP
Ranges. Download neither the full archive nor depth arrays. Preserve native XELA
shape `[15,16,3]`, sensor/taxel/axis order, and all signed values. Convert the
official video-metadata timestamps from milliseconds to seconds and pair each
RGB observation to its nearest tactile timestamp within a cadence-based tolerance.
Targets outside sensor support or across long gaps remain masked; no force
interpolation is introduced. Keep the original HDF5 sensor stream. Equal camera
timestamps are preserved and both ambiguous frames are masked; no timestamps or
video frames are reordered. Identical repeated tactile packets are allowed,
while conflicting packets at the same timestamp cannot supervise a target.
An episode with a missing prefix is usable if it contains a complete supported
eight-frame history. Report coverage rather than rejecting the whole episode
solely because its total coverage is below 90%.

The sampled AVI contains one more frame than the metadata timestamp list. Retain
all RGB frames, use the official preprocessor's index mapping for annotated
frames, and leave the extra trailing RGB frame without an invented target. Frame
count differences beyond this audited zero/one-frame tail block the episode.

### HRDexDB

Use the catalog's actual asset availability and a deterministic object-balanced
pilot selection. All embodiments of an object share a split. The catalog's
`train` label denotes the full catalog and is not reused as a benchmark split.
Camera 22641005 is labeled `fixed_scene`, never called ego or wrist.

Allegro's sampled native tactile stream has four channels. Inspire F1 stores
named normal/tangential-force and direction fields in object arrays. Those are
read through a restricted NumPy metadata unpickler, then saved as ordinary numeric
arrays. Preserve the official mapping `video index = frame_id - 1`; callback
timestamp order can differ from frame-ID order. Reject nonchronological camera
histories rather than silently pairing by row position. A partially covered
video retains its supported histories; out-of-range frame IDs stay missing.

The sampled Inspire DFTP tactile array is empty. Keep and audit its separate
force-feedback file, but do not automatically substitute that stream for tactile
data or guess how to handle its 65535 values. Unsupported episodes remain
excluded with a quality report. The October 20 audit checked all 50 selected
DFTP recordings: their native tactile arrays are empty. Their separate six-DOF
force feedback remains archived for audit and is not relabeled as tactile data.
Completed availability audits use `complete_with_exclusions`; unknown quality
failures remain `blocked_quality`. This does not prevent verified embodiments
from being processed.

### Removed source: TacQuad

Removed because the user found its interactions insufficiently dynamic and
sensor visibility unsuitable. No TacQuad data or live viewer entry is retained.
The deletion ledger and former manifest checksums are under
`processing/retired_datasets/tacquad/`.

## Operation and outputs

The persistent user service is `tactilelatent-dataset-queue-v2.service`.

```bash
# Combined readable/JSON status
.venv/bin/python scripts/run_dataset_queue.py status

# Run/resume the whole queue; only one queue owner is allowed
.venv/bin/python -u scripts/run_dataset_queue.py run

# Explicitly retry a source after resolving a quality issue
.venv/bin/python -u scripts/run_dataset_queue.py source hrdexdb

# Service log (source logs are also retained below)
journalctl --user -u tactilelatent-dataset-queue-v2.service
```

Each `processing/data_queue_v2/<source>/` contains `status.json`,
`progress.jsonl`, `worker.log`, download receipts, implementation hashes,
`manifest.json`, `quality.json`, and applicable `targets/`, `rgb/`,
`normalization.json`, `acceptance.json` and `deletions.jsonl`.

Network failures retry independently with bounded exponential backoff, including
metadata/DNS requests. A source-local staging/budget pause does not block other
datasets. Global project/free-space pauses defer new launches. On resume, verify
and retire leftover original RGB archives even if their compressed replacements
already have receipts; changed hashes or frame mappings retain the originals.
A quality block keeps verified episodes available, skips the affected source on
automatic passes, and lets unrelated sources continue. Correct the underlying
schema/data issue before explicitly retrying it.

MotionTrans is excluded from acquisition statistics and training manifests.
FreeTacMan remains excluded. No full V-JEPA feature cache is generated.
