# Acquired datasets: targets, reconstruction and visualization

The viewer on port **8080** supports Baihu, EgoPressure, FEEL Lab/Kitchen, HRDexDB Allegro V5/Inspire F1 and T-Rex alongside the original sources. This October 21, 2026 guide supersedes the [earlier measured-only integration snapshot](archive/superseded_guides/ACQUIRED_DATASETS_VIEWER.md). Available trained reconstruction and joint models are selected separately; a source with four/five scalar values does not need an autoencoder.

| Source | Native target | Reconstruction / joint supervision |
|---|---|---|
| Baihu | Four-surface signed marker displacement | 32D frozen marker AE; original/mirror teacher |
| EgoPressure | One active-hand 224×224 UV pressure map | 64D pressure AE; task-only episodes |
| FEEL Lab | Five processed right-hand contact estimates | Direct numeric prediction, no AE |
| FEEL Kitchen | Twelve processed left/right contact estimates | 6D numeric AE |
| HRDexDB Allegro V5 | Four native tactile values | Direct numeric prediction, no AE |
| HRDexDB Inspire F1 | Eight groups × normal/tangential/direction = 24 values | 12D numeric AE |
| T-Rex | Ten fingertips × six wrench components = 60 values | 16D numeric AE |

**Sensor reconstruction** shows measurements and available AE outputs; direct-only sources expose measured values without inventing a decoder. **Ego → tactile prediction** shows a selected visual model's prediction, model/epoch comparison and attention for available cameras. The current model is labeled as 17 sources plus mirroring. **Training monitor** shows source losses and reconstruction histories. Mirror quality checking is available only for supported mappings, including Baihu; unresolved FEEL/T-Rex geometry is not silently guessed.

## Numerical readings, zeros and scales

Native arrays, signs, masks and chronological mappings are preserved. A measured zero is valid; missing observations are gray/gaps rather than fabricated zero. Default display ranges are fixed using training data, so changing episodes cannot magnify idle fluctuations to match active contact. Exact values remain available even beyond the displayed range.

- **T-Rex:** one shared force range and one shared moment range across both hands/fingers. Video addressing includes the episode's offset and length within shared shards. Upstream already tares each episode using its first 15 clean frames; this project does not recover unprovided pre-tare offsets or apply a new episode-specific tare.
- **HRDexDB:** Allegro channels share one range. Inspire normal, tangential and direction have separate common ranges across eight groups. Native direction values, including `-1`, are preserved; their physical period/sentinel interpretation is unresolved. Fixed scene cameras keep their role and explicit RGB mappings; missing frame maps are not replaced with neighboring images.
- **FEEL:** the default shared 0–2.5 detail range comes from training positive values and is the same across hands/episodes. Out-of-range values receive markers and exact numbers; full training ranges remain selectable. These are processed contact estimates, not independently calibrated SI force. Lab's five channels are **right hand**, including validation Lab_P11; Kitchen uses explicit left/right CSV mappings.

[Fixed sensor zero references](FIXED_SENSOR_ZERO_REFERENCES.md) distinguish released zero from certified no-contact calibration. HRDex unloaded readings can remain nonzero despite a zero fixed offset. Neither the first frame nor each episode's minimum should automatically define physical zero. Optional episode/per-channel magnification is labeled and changes the display only.

## Pressure and marker views

EgoPressure is a UV atlas, not a projected image of five fingertips. UV chart seams can split one physical contact into several flat spots. The [3D hand view](EGOPRESSURE_VISUALIZATION.md) uses installed MANO UV meshes and recorded pose, with measured/reconstructed/error textures on identical geometry. The AE is not credited with predicting the hand pose. Missing pose falls back to the atlas. [Calibration-routine episodes are excluded](EGOPRESSURE_DATA_SELECTION.md) from current training, validation and statistics; only native hand-relative UV targets are predicted, not raw pad grids.

Baihu/Unit marker views use retained reference coordinates, surface IDs, native spacing, validity and signed displacement. Tracking failures stay missing. Deleted tactile source images are not needed for teacher reconstruction or this display. Marker geometry is not an independently certified unloaded state.

## Availability, evidence and exclusions

The 17-source integration receipts are linked in [project state](project_state.md). Extra numeric AEs were trained in `processing/acquisition_teachers_v1/`; direct-head exclusions for FEEL Lab/Allegro are intentional. Historical reconstruction models remain distinct from the current visual model.

Incomplete released data remains excluded: FEEL unreadable Kitchen/P05 CSV and missing Lab/P20–P21 CSVs, plus HRDexDB InspireDFTP empty tactile arrays. No force-feedback substitute is silently relabeled as tactile data. T-Rex accepted episodes and frame mappings come from its acquisition manifest, not guessed stream positions.

FreeTacMan, TacQuad and T-Dex were removed/excluded at the user's request; do not restart their downloads through old configs. Removal ledgers remain under `processing/retired_datasets/`. The active queue is `configs/data_queue_active.json`; acquisition provenance retains older frozen configurations.

Historical visual audits live under `processing/audits/acquisition_viewer_v1/`, `feel_display_v1/`, `egopressure_surface_view_v2/` and the mirror v4 verification. Those records establish what was checked then, not that every newly added frame has been independently inspected. See [FEEL hand ownership](FEEL_HAND_OWNERSHIP.md), [mirroring](BIMANUAL_MIRRORING.md), [dashboard](TRAINING_DASHBOARD.md) and the [script catalog](script_catalog.md).
