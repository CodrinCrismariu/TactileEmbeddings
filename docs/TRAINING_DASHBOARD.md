# Training monitor

Open **Training monitor** in the viewer navigation, or use `?page=training` on the existing viewer address. `/training` is also supported. The page uses the same port and SSH forwarding as the reconstruction viewer.

## What is available

- Joint/prediction models, sensor autoencoders and tactile-representation teachers with saved scalar histories. Smoke runs are excluded. Histories remain viewable if model weights were pruned.
- Select up to eight models, search by run name/dataset/code dimension, and filter by model type.
- All per-dataset curves, an equal-weight source overview where the logs permit it, and a single-dataset filter.
- Objective/supervision losses, recorded reconstruction metrics, KL, contact errors, and separate original/mirrored validation curves.
- Five-second browser refresh, model progress, hover values, EMA smoothing, log scale, drag-to-zoom, double-click reset, raw epoch CSV and PNG exports.

Epochs are displayed from one for historical zero-indexed logs. Continuation epoch numbers retain their original checkpoint numbering. Scalar values are not rescaled, missing/nonfinite measurements are not replaced with zeros, and validation MAE is never relabeled as objective loss. When older logs contain only validation MAE, the primary view shows it separately with its own labeled scale.

An aggregate source curve is an equal-weight mean over sources that reported that metric at that epoch. A pre-existing logged overall curve retains the logger's original weighting. Loss and target definitions can differ between experiments; selecting a metric does not make different target representations directly comparable.

## Live measurements

The viewer samples active runs' `status.json` files every two seconds. These samples are individual reported training-batch losses. They do not include every batch, and earlier batch history cannot be reconstructed from epoch-only logs. Collection continues while the viewer service runs, including when the browser or SSH session is closed. Journals live under `processing/viewer_training_history_v1/`.

Training/validation epoch curves update when the trainer appends a completed record to `metrics.jsonl`. Validation is not rerun by this dashboard. No model/backbone is loaded for monitoring, and training configuration, weights, caches and pinned data-loader code are unchanged.

## Implementation

- `src/tactilelatent/visualization/training_dashboard.py`: discovery, scalar normalization, safe history endpoints, process-status checks, and live journal collector.
- `src/tactilelatent/training_dashboard.html` and `training_dashboard.js`: local interactive page with no external chart/CDN dependency.
- `src/tactilelatent/visualization/server.py`: composes the current viewer with the dashboard handler. The historical viewer is isolated behind `/legacy/`.
- `GET /api/training/catalog`: available histories and current status.
- `GET /api/training/history?runs=<id>,<id>`: whitelisted histories and collected live samples. Filesystem paths are not accepted as IDs.

Roots include `processing/runs`, expansion sensor teachers, acquisition sensor teachers, and versioned bilateral teacher folders. Discovery refreshes periodically; the manual refresh button discovers new runs immediately.

## Verification

Twelve tests cover legacy/current/mirror formats, all seventeen current sources, incomplete JSONL writes, missing and zero values, epoch numbering, live sampling/deduplication, stale PIDs, path rejection, proxy routes and existing-route fallback.

Live browser verification discovered 150 non-smoke histories and verified all seventeen current source panels, model comparison, filtering, hover, smoothing, log scale, zoom, exports, separate mirrored validation, historical validation-MAE fallback and automatic collection of new batch samples. See `processing/audits/training_dashboard_v1/browser_verification.json` and the screenshots in that directory. Existing sensor/mirror/joint pages expose the new navigation link. Joint training remained active throughout deployment, and pinned storage/acquisition implementation hashes remained unchanged.
