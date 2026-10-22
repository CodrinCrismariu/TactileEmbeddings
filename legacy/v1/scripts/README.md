# Supported command entry points

Run commands from the repository root using `.venv/bin/python`. The complete
[script catalog](../docs/script_catalog.md) covers every pre-existing script,
including historical defaults, mutation hazards and uncertain classifications.

The following examples were checked against their source and service definitions.
The refactor does not launch training, download data, rebuild caches, or restart
services. If a service already runs the command, use that service rather than
starting a second copy.

| Workflow | Command | Significant effect |
|---|---|---|
| Viewer, attention overlay and live training curves | `.venv/bin/python scripts/view_egotouch_autoencoder.py --manifest processing/joint_alignment/manifest.json --host 127.0.0.1 --port 8080` | Starts HTTP viewer; inference may use the GPU. Existing service binds 0.0.0.0:8080. |
| Current integrated joint training | `.venv/bin/python scripts/run_all_sources_joint.py --config configs/joint_vjepa21_all_sources_32d_mirror_v2.json` | Preflight then starts/resumes deterministic 32D, 17-source GPU training. MotionTrans is held out. |
| Same trainer, preflight only | `.venv/bin/python scripts/train_all_sources_vjepa.py --config configs/joint_vjepa21_all_sources_32d_mirror_v2.json --preflight-only` | Performs production input/provenance checks and writes preflight metadata; not a read-only dry run. |
| Dataset queue status | `.venv/bin/python scripts/run_dataset_queue.py status` | Reads queue and updates aggregate status reports. |
| Approved acquisition queue | `.venv/bin/python scripts/run_dataset_queue.py run` | Downloads and processes selected datasets; changes storage. |
| New-source tactile AEs | `.venv/bin/python scripts/train_acquisition_autoencoders.py --config configs/acquisition_autoencoders_v1.json` | Fits/reuses configured tactile teachers on GPU; no visual backbone. |
| Mirrored tactile teachers | `.venv/bin/python scripts/prepare_all_mirrored_teachers.py` | Fits missing decoders and writes original/mirrored tactile codes. No joint run. |
| Mirror/code integration verification | `.venv/bin/python scripts/verify_all_mirrored_teachers.py` | Re-encodes selected targets, verifies codes and updates preparation receipts. No joint optimizer. |

The old `view_sensor_autoencoders.py` launcher starts the basic viewer without the
current acquisition, mirror and training-dashboard installation hooks. Use
`view_egotouch_autoencoder.py` despite its historical name.

Do not infer safety from `audit`, `verify`, `prepare`, `status`, or `smoke` in a
filename. Some checks perform GPU updates, write caches, change accepted metadata,
control services or compact/delete original archives. Inspect the catalog before
using an unfamiliar script. Seven files execute work before any CLI parser; do not
run a bulk `--help` or import sweep.

Historical bodies are indexed in [archive/README.md](archive/README.md). Where
provenance pins, services or runtime path assumptions require it, historical files
remain at their exact old paths. Those exceptions preserve reproducibility rather
than naming a second current pipeline.
