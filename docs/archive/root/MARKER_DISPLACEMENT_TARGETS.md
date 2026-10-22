> Historical record, relocated on October 21, 2026. Original path: `MARKER_DISPLACEMENT_TARGETS.md`.
> Results, proposals, and operational status below belong to their recorded experiment/date;
> use the [maintained project state](../../project_state.md) for the audited active workflow.

# Optical tactile targets: tracked dot displacement

**Update:** the user subsequently requested autoencoders for these displacement targets before joint training. The current small-code AE search and frozen-code architecture are described in [MARKER_AUTOENCODERS.md](MARKER_AUTOENCODERS.md). The direct joint-displacement head described below is an earlier prototype; the default joint launcher now uses marker AE code supervision. Corrupted startup surfaces can use a verified first stable trackable reference instead of frame zero; `Slot/Slot_164` uses references [45, 0] with earlier missing observations masked.

## Change and rationale

The new target path replaces optical RGB reconstruction/code prediction for **FreeTacMan, FTP UniT, FTP UniT Bimanual, and FTP VLA-Touch** with signed marker displacement. It retains native images for extraction, auditing, and historical models. It does not resize tactile images, change scene cameras, or alter the canonical manifest. The currently running image-trained model still stops at epoch 20; it is not silently switched to a different objective.

VLA-Touch's [official preprocessing](https://github.com/clear-nus/vla-touch/blob/master/VLA/data/franka_data/3_gelsight_data.py) explicitly tracks a 7×9 grid and saves 63×2 displacement arrays. [GelSight's tracker](https://github.com/GelSight/tracking) is another direct precedent. [UniT's official method](https://github.com/ZhengtongXu/UniT) instead learns tactile image representations; it should not be described as a marker-only method.

Dot motion measures surface deformation and shear. Optical RGB also contains shading/contact/texture information, so this change intentionally selects a subset of the available tactile information. Improved transfer remains an experimental question.

| Source | Previous native RGB values/frame | Pilot marker vectors/frame |
|---|---:|---:|
| FTP UniT | 230,400 | 63 × (dx,dy) |
| FTP UniT Bimanual | 460,800 | 126 × (dx,dy), two separate surfaces |
| FTP VLA-Touch | 230,400 | 63 × (dx,dy) |
| FreeTacMan | 1,843,200 | 593–678 total points across two surfaces in the pilot; variable by episode |

The sensor adapter for each changed dataset is a point-query MLP. It receives the 32D visual latent, the dot's normalized reference location `(x,y)`, and a learned eight-value surface embedding. Its layers are **42 → 128 → 128 → 2**, with GELU between hidden layers. The final two outputs are signed, unbounded displacement in marker-spacing units. The same head weights process every dot of a given dataset. Reference location and surface ID are static sensor geometry, not current tactile measurements.

## Exact target convention

For episode-local marker i at chronological frame t:

`d_i(t) = detected_position_i(t) - reference_position_i(frame 0)`

- Native subpixel coordinates, positive x right and positive y down.
- Frame zero is an episode reference, **not a verified unloaded calibration**. The targets describe changes relative to that state and cannot recover load already present at the beginning.
- Targets are displacement relative to the reference, not just adjacent-frame velocity. A sustained deformation remains a nonzero target after motion stops.
- Global translation of the dot field is retained because it may represent real shear.
- Left/right tactile arrays and FreeTacMan surface1/surface2 retain their stored order.
- VLA chronological frames use the existing verified chronological-to-stored permutation. The extractor calls the canonical native-image loader and records both indices.
- No MotionTrans images, targets, or statistics participate in extraction or normalization.

## Extraction and validity

`src/tactilelatent/marker_targets.py` detects local dark blobs at native resolution, estimates contrast-weighted subpixel centers, and merges fragments separated by much less than the robust dot pitch. The merge fixes a split-dot detection observed in the first bimanual pilot.

Each entire episode is processed sequentially. Tracking is never restarted at training clip boundaries. Hungarian assignment uses previous positions, reference anchoring, area checks, dummy unmatched assignments, and an ambiguity gate. Reference displacement beyond 0.4 times the median dot pitch is conservatively masked; it is not clipped into a smaller target. This can exclude strongly deformed dots at peak contact and must be monitored in the coverage reports.

An invalid/lost observation is **NaN plus a false validity mask**, not zero displacement. Untrackable reference images are recorded as rejected episodes. Production training fails if any required episode is missing or rejected; it does not silently drop episodes or revert those sources to RGB.

Per-episode `tracks.npz` contains reference coordinates, surface IDs, spacing, full displacement trajectories, validity, a tracking-quality heuristic, and chronological/stored frame indices. `complete.json` pins the episode metadata, extractor/loader hashes, target hash, coverage, and motion statistics. The quality heuristic is not calibrated predictive uncertainty.

Files are written separately under `processing/marker_displacement_v2/targets`. Original datasets, native images, existing AE checkpoints, and older training results are retained. These are derived tactile targets; no V-JEPA spatial-token or visual-feature cache is created.

## Joint learning

`src/tactilelatent/train_vjepa_markers.py` uses the full-spatial live V-JEPA encoder and deterministic **32D per frame** attention bottleneck. The four optical sources use the new displacement heads and bypass their RGB loaders and AE teachers during training. The other five AE-supervised sources and two numeric sources retain their existing supervision.

The displacement loss is masked Smooth L1, averaged equally over supported surfaces and frames. Padding and missing points contribute no loss or gradients. For scale comparability, both predictions and targets are divided by a fixed **training-split-only per-source RMS displacement** in marker-spacing units. That scale is estimated after target extraction, frozen in the model configuration/checkpoint, and never fitted on validation. Pixel errors are reconstructed using each surface's native dot spacing.

SAM replays the same features, masks, targets, and dropout RNG. Only the current optimizer group's tensors are retained in RAM. Frozen V-JEPA and the five remaining tactile teachers stay in evaluation mode and acquire no gradients. Production checkpoints include optimizer, scheduler, SAM, RNG, teacher, target, and data provenance.

New configuration: `configs/joint_vjepa21_spatial_markers_32d.json`. It is prepared for a fresh seed-45, deterministic 32D, 20-epoch run with the current spatial-attention settings. No new production training has been launched by this change. Production preflight must run on the completed full target extraction before launch:

```bash
# Full extraction is already running in a detached CPU process.
.venv/bin/python scripts/prepare_marker_targets.py --workers 2

# After extraction is complete and index.json reports no missing/rejected episodes:
bash scripts/run_vjepa_markers.sh --preflight-only
bash scripts/run_vjepa_markers.sh
# Resume only after a stopped run, retaining the exact configuration/provenance:
bash scripts/run_vjepa_markers.sh --resume
```

The full extractor is resumable: it verifies and skips completed compatible episode artifacts. Its log is `MARKER_TARGET_EXTRACTION.log` at the project root, and its current state is `processing/marker_displacement_v2/targets/status.json`.

## Viewer and validation

The hosted viewer has a **Marker tracking** page at `/markers`. Select a dataset, a completed episode, and any frame. It shows the scene frame and native tactile surfaces with measured arrows. Completed episodes appear as extraction progresses; the pilot supplies initial examples for all four datasets. Cyan arrows are displayed ×4 for visibility; this gain never changes training targets or metrics. White circles show reference centers and gray crosses indicate missing tracks.

New marker-model checkpoints are supported by the existing Ego → tactile prediction viewer. Measured arrows appear on the current native sensor image; predicted arrows appear on the reference image. Metrics show **marker MAE in native pixels**, and hover displays exact signed vectors. The error map uses endpoint error divided by marker spacing. Older RGB models retain their original rendering and metrics. No untrained placeholder model is added to the model picker.

The 12-episode revised pilot covered 2,545 chronological frames with no extraction failure. Reference counts were 63 per UniT/VLA surface and variable for FreeTacMan. Average pilot tracking coverage was roughly 97% for FreeTacMan and 99–100% for the other sources. This is detection/tracking coverage, not independently measured ground-truth tracking accuracy. Native contact sheets are in `processing/marker_displacement_v2/audit/`.

A real-data preflight performed a forward/backward/SAM update for every one of the 11 training sources, using only a disposable model. The four optical sources used marker targets; the other sources retained native supervision. Frozen-backbone and teacher gradients were checked. Evidence: `processing/runs/joint_vjepa21_spatial_markers_32d_v1/pilot_preflight.json`.

Tests cover signed/subpixel motion, persistence of global shear, missing-point masking and recovery, out-of-gate rejection, variable marker-count padding, equal weighting of surfaces, NaN-safe loss gradients, chronological VLA indices, bypass of tactile RGB loading, gradients into the spatial bottleneck, and checkpoint round trips. Existing viewer tests also run.

## Evaluation after training

Report per-source marker-coordinate MAE, endpoint error in pixels, temporal-change error, tracking coverage, and a zero-displacement baseline. Retain native reconstruction errors for the remaining sensors. Compare the representation on the same frozen-encoder MotionTrans transfer protocol and head used previously.

The overall source selection score now contains four different objectives and **cannot be compared numerically with the old RGB/code-supervised aggregate score**. Marker reconstruction quality alone also does not establish better cross-sensor transfer or calibrated uncertainty.
