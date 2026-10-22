# EgoPressure task-only training and validation

Calibration routine recordings are excluded by episode identity from both
training and validation. This also applies to the sensor reconstruction viewer,
autoencoder samplers, joint clip construction, and code normalization statistics.
It does not remove camera calibration metadata used by ordinary task episodes.

| Split | Excluded calibration episodes | Remaining task episodes |
|---|---:|---:|
| Train, participants 001–018 | 36 | 1,116 |
| Validation, participants 019–021 | 6 | 186 |
| Total | 42 | 1,302 |

The removed recordings contain 22,992 frames. The task selection retains
524,289 frames at their original chronological indices, native 224×224 UV
resolution, and original pressure magnitudes/missing-value masks. Acquisition
files and historical manifests remain available for provenance.

The balanced v2 teacher's saved sampling index and all 42 calibration code
artifacts were inspected: **zero sampleable frames and zero supported cached
frames**. These recordings had no valid UV supervision, so they had already
contributed no samples to the balanced teacher's optimization, validation, or
valid-code statistics. Retraining is unnecessary for this exclusion. Validation
was rerun on the task-only selection and still passes the zero-collapse gates.

The active task cache reuses byte-identical codes for retained episodes and
refits statistics from supported training frames only. Fresh teacher encodings
were checked against original producer batches on both splits. The fixed
teacher's pressure normalization is preserved, as is its checkpoint identity.
The original teacher/cache metrics have not been relabeled as a new training run.

Artifacts:

- `processing/data_expansion_v1/egopressure_task_manifest_v1.json`
- `processing/data_expansion_v1/codes/egopressure_task_selection_v1/index.json`
- `processing/data_expansion_v1/egopressure_task_selection_v1/quality.json`
- `processing/data_expansion_v1/egopressure_task_selection_v1/calibration_support_audit.json`

The active source override points to this selected code index. Future expanded
joint publication checks its exact selected row hash and excludes calibration
codes, including from fitted statistics. A historical index containing those
episodes cannot be used directly by the current joint sampler.

Rebuild the selection and its validation report with:

```bash
.venv/bin/python scripts/prepare_egopressure_task_selection.py
```

Future balanced AE runs default to separate `egopressure_balanced_tasks_v1`
teacher and code folders. Their samplers and input statistics use task-only
episodes, and their provenance prevents resuming an older run under a changed
episode selection.
