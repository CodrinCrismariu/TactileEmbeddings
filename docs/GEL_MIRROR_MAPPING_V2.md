# User-reviewed gel mirroring

On October 20, 2026, the user confirmed that the mirror previews looked correct
and selected horizontal reflection plus **exchange jaw sides** for Baihu.
This is recorded visual verification, not an independent mounting calibration.

The active viewer policy is `configs/bimanual_mirror_policy_v2.json`:

| Source | Source surface → destination surface |
| --- | --- |
| Unit Bimanual | `[1, 0]` |
| Baihu | `[3, 2, 1, 0]` in left-left, left-right, right-left, right-right order |

Every surface uses reference `(u,v) → (1-u,v)` and signed displacement
`(dx,dy) → (-dx,dy)`. The affine translation applies only to reference points.
Marker identities, native spacing, validity, missing observations, and frame
order stay attached. Both RGB wrists exchange slots and all RGB views reflect.
The Baihu mapping applies to the current D-WHEEL, PIKA and QINGLOONG mounts;
native examples from all three platforms are included in verification.

## Preparation and compatibility

Run `scripts/prepare_gel_mirror.py`. It uses the repaired normalized Baihu
teacher and the current Unit Bimanual marker teacher. It audits matched
original/reflected validation frames and reports native-pixel errors, including
each Baihu platform. If mirrored MAE is over 10% worse than original MAE, it
fine-tunes a separate teacher for 10 epochs, seed 45, with half original and
half reflected examples, AdamW at 1e-4 and cosine decay to 1e-6. Architectures,
sensor-code sizes, splits, and reconstruction losses are preserved.

An adapted teacher must preserve original MAE within 10% and improve reflected
MAE by at least 10%. Rejected teachers do not produce promoted code caches.
Accepted teachers encode **both** native target variants freshly; sensor-code
coordinates are never flipped or permuted. Code statistics use training data
only with equal variant weighting.

New receipts, teachers, paired codes and progress live under
`processing/bimanual_mirror_v2/`. Historical v1 policies, builders, accepted
teachers, and caches remain unchanged. The new preparation explicitly references
the original policy for unchanged EgoTouch/RDP artifacts and checks that their
source mappings match; it does not relabel their provenance or retrain them.

The mirror page uses the confirmed mappings immediately. It distinguishes a
verified mapping from completed teacher/code preparation. Accepted adapted
teachers appear alongside historical models. OpenGraph remains a review-only
hypothesis because the user confirmation concerned gel sensors.

The expanded 32D current-frame V-JEPA mirror configuration uses the four
verified bilateral sources at 50% clip probability. Its loader blocks any
unfinished teacher/code preparation. No production joint-training run starts
as part of applying these mappings.
