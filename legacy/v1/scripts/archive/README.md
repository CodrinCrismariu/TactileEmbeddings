# Historical script bodies

These preserved implementations support older experiments and diagnostics. They
are not default instructions for the active 17-source joint experiment. The full
[script catalog](../../docs/script_catalog.md) records every historical command,
including those kept at their original paths for compatibility.

| Historical body | Original command retained | Purpose |
|---|---|---|
| [compare_sam_motiontrans.py](compare_sam_motiontrans.py) | `.venv/bin/python scripts/compare_sam_motiontrans.py` | Summarize matched final-epoch DINO SAM/non-SAM MotionTrans probes. |
| [audit_saved_visual_video.py](audit_saved_visual_video.py) | `.venv/bin/python scripts/audit_saved_visual_video.py` | Original bounded H.264 storage/decode pilot on loader outputs. |
| [audit_visual_video_quality.py](audit_visual_video_quality.py) | `.venv/bin/python scripts/audit_visual_video_quality.py` | Original RGB reconstruction/lossless-video size pilot. |
| [download_egotouch_worker.sh](download_egotouch_worker.sh) | `bash download_egotouch_worker.sh` | Historical alternative EgoTouch HDF5 download worker; can download large data. |

The three Python bodies are unchanged copies of their original implementations.
Thin old-path wrappers execute them in the original module global namespace, with
the original `__file__`, `__name__`, arguments and sibling-module search path. No
model, loss, split, target ordering or result calculation changed. New documentation
uses the archived source links, while old shell commands and imports stay valid.

The shell worker's archived `cd` accounts for its new location; a root compatibility
shim forwards the old command and all arguments. Its download revision was not
pinned historically. Archiving does not authorize rerunning it or reacquiring
previously removed datasets/assets.

Other historical scripts deliberately remain in `scripts/`: explicit source hashes,
self-hashing via `__file__`, active or historical service commands, tests that import
literal paths, sibling imports and root-relative lookup assumptions can make a move
invalidate old workflows. They are listed as historical in the catalog instead of
being silently rewritten or declared disposable. Active service entrypoints remain
byte-for-byte unchanged. Never edit saved provenance to force moved code to pass.
