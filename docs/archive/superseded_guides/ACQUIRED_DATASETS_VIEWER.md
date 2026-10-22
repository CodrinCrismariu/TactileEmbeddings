> Historical record, relocated on October 21, 2026. Original path: `docs/ACQUIRED_DATASETS_VIEWER.md`.
> Results, proposals, and operational status below belong to their recorded experiment/date;
> use the [maintained project state](../../project_state.md) for the audited active workflow.

# Browsing newly acquired datasets

Open the existing viewer on port **8080**, select **Sensor reconstruction**, then
choose **FEEL**, **T-Rex**, **HRDexDB · Allegro V5**, or
**HRDexDB · Inspire F1**. These sources currently provide **measured values only**;
the model selector states that no trained decoder exists and is disabled. This
integration does not start autoencoder or joint training.

The same episode selector, split filter, video slider and camera selector work
for these sources. Completed, accepted T-Rex episodes appear as its manifest
grows; use **Refresh datasets / models** to discover additional recordings.
The automatic catalog refresh also discovers datasets and new checkpoints.

## Native numerical targets

The numerical view provides signed bars, channel groups, exact native readings,
and a selected-channel trace over ±150 observations. Gray bars and gaps denote
missing observations. A measured zero remains a valid zero. Target values are
never normalized, clipped, resized, or rewritten.

**T-Rex defaults to fixed training-derived ranges across both hands and all fingers:** one range
for force components and a separate range for moment components. The selected
channel trace uses that same range. Numerical range labels are shown above the
bars. Thus small idle-hand fluctuations remain small beside active-hand contact.

**HRDexDB also defaults to fixed training-derived ranges.** Allegro V5's four native tactile
channels share one range. Inspire F1 has separate normal-force,
tangential-force, and direction ranges, with each range shared across all eight
finger/palm groups. Direction is not plotted on a force scale. Its native values,
including −1, are preserved rather than reinterpreted as negative force.

The new default uses frozen sensor references and fixed training-derived ranges
across all episodes. Raw readings remain available. See
[fixed-reference findings and limitations](../../FIXED_SENSOR_ZERO_REFERENCES.md).

**HRDexDB zero is the released numeric value 0, not an independently verified
no-contact baseline.** The screened training median offsets were all zero. Some Inspire F1
channels start nonzero; that does not establish whether they have an offset,
mechanical preload, or actual contact. The first frame or episode minimum must
not be automatically used as a physical zero. A baseline correction would need
documented calibration or a verified unloaded interval, and direction values
need separate handling. Shared plotting ranges correct visual amplitude
comparisons in native units; they do not establish a no-contact zero or SI gains.

The optional **Per channel · magnifies small fluctuations** mode explicitly
enlarges each channel's own episode range for inspection; its bars cannot be
compared across channels. The optional shared-episode ranges cover every frame,
including short peaks; per-channel episode ranges use a bounded sample. Fixed
training ranges remain the default for all newly acquired numerical sources.
No scale is fitted separately to the currently displayed frame.

**FEEL defaults to a fixed 0–2.5 detail view**, shared across every channel,
both hands, Lab/Kitchen, and all episodes. The released processed values are
nonnegative, so zero sits at the bottom of its bars and traces. This replaces a
default range dominated by rare peaks of 11.94–16.25, which made ordinary changes
look almost flat. The common upper limit rounds the training-positive 99.9th
percentile (2.363636) up to 2.5; validation does not fit it. This display policy is
separate from the saved zero references, which remain unchanged.

Values above 2.5 display an orange triangle and their actual number; trace peaks
outside the visible range are marked too. **Full training range** shows their full
heights. Current values appear above each FEEL bar, and the exact table retains
the full displayed precision. The optional **Shared quantity · this episode**
mode uses one common range across both hands and includes every frame's peaks.
Neither mode changes the sensor values or labels contact. FEEL's supplied values
are processed contact-estimation signals, not independently calibrated force.
Audit evidence is in `processing/audits/feel_display_v1/` and the fixed policy is
`configs/feel_display_v1.json`.

| Source | Native targets | Display grouping |
| --- | --- | --- |
| FEEL Kitchen / Lab | 12 / 5 processed contact-estimation values | Kitchen: six indices per hand; Lab: five right-hand indices |
| T-Rex | 60 fingertip wrench estimates | Ten fingers, six released components per finger |
| HRDexDB Allegro V5 | Four tactile values | Released channel indices; anatomical correspondence unverified |
| HRDexDB Inspire F1 | 24 tactile values | Eight released anatomical groups, three components each |

Taxel-index bars do not assert a physical surface layout. Values with unverified
physical calibration are labelled accordingly. In particular, FEEL's processed
values are not labelled as calibrated SI force. Inspire F1 force and direction
components retain their native values and separate quantity-specific display ranges.

Scene cameras keep their actual roles. HRDexDB is labelled **Fixed
scene camera**. T-Rex exposes ego, left wrist and right wrist where available.
T-Rex frame offsets address the correct episode segment within shared video
shards. HRDexDB's explicit RGB frame map is honored: a missing mapping displays
an unavailable-video message without substituting a neighboring frame.
The displayed timestamp comes from the target artifact, rather than assuming
all sources share one frame rate.

FEEL hand ownership is selected per episode, never inferred from the first array
position. The official Lab preprocessing and project previews identify its five
channels as **right hand**, including validation Lab_P11. The viewer labels them
`right_sensor0` through `right_sensor4`; the left hand is explicitly not recorded.
The native `released_hand_sensor*` identifiers remain available for provenance and
fixed-reference matching. Kitchen retains its explicit left/right file mappings.
See [the ownership audit](../../FEEL_HAND_OWNERSHIP.md).

Unavailable source data remains excluded: FEEL's unreadable Kitchen/P05 CSV and
missing Lab/P20–P21 CSVs; HRDexDB InspireDFTP's empty released tactile arrays.
No force-feedback array is silently substituted for a missing tactile target.

## Removed sources: TacQuad and T-Dex

TacQuad was removed at the user's request on 2026-10-21. Its dataset files,
indexes, image previews and fixed optical templates were deleted. It no longer
appears in the dataset selector or navigation; old browser/API addresses return
HTTP 410. The active download queue excludes it. The removal freed 68.2 GB
(63.5 GiB); the deletion ledger is `processing/retired_datasets/tacquad/removal.json`.

T-Dex was also removed on 2026-10-21. Its 24-episode pilot, verification copy,
sensor/RGB files, previews and cached reference were deleted, freeing about
317 MB. It is absent from the viewer and active queue; normal acquisition
requests for it are rejected. The ledger is `processing/retired_datasets/tdex/removal.json`.

## Implementation and checks

`AcquisitionViewer` extends the existing expansion viewer; the acquisition
namespace begins at episode ID 1,000,000. Existing episode IDs and historical
model entries remain available. Target reads verify the published artifact
SHA-256 and use bounded in-memory caches. New sources do not appear as runnable
joint prediction models or mirror mappings.

Focused tests cover native signed values, measured zeros versus missing masks,
XELA tensor ordering, separate HRDexDB morphologies, episode-offset bounds,
missing RGB frame maps, chronological timestamps, catalog refresh, proxied
routes, unchanged 16-bit TacQuad images, and independent archive streams.
Live browser screenshots and acceptance records are under
`processing/audits/acquisition_viewer_v1/`.
Removal checks cover direct/proxy HTTP 410 responses, absent dataset/navigation
entries, and preservation of the other acquisitions.
