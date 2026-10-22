# Package organization and compatibility

The maintained navigation starts with five namespaces:

| Namespace | Responsibilities |
|---|---|
| [`datasets`](../src/tactilelatent/datasets/README.md) | Acquisition, native targets, chronological clips, geometry, mirroring, and caches |
| [`models`](../src/tactilelatent/models/README.md) | Frozen visual encoders, trainable attention readers, adapters, and sensor autoencoders |
| [`training`](../src/tactilelatent/training/README.md) | Objectives, optimization, teacher preparation, and current/historical experiment trainers |
| [`evaluation`](../src/tactilelatent/evaluation/README.md) | Frozen transfer probes, representation diagnostics, uncertainty evaluation, and audits |
| [`visualization`](../src/tactilelatent/visualization/README.md) | Reconstruction, prediction, attention, mirror inspection, and scalar monitoring |

Each guide maps every module assigned to that responsibility. These namespaces
are lazy: importing a namespace alone does not load PyTorch, decode video, start
training, download data, or launch the viewer. A module's presence here does not
make an old experiment the recommended current workflow. Use the maintained
[project entry point](../README.md) and workflow documentation to choose commands.

## Why most implementation files remain flat

Caches and historical experiments record exact source-file hashes, sometimes
using absolute paths and sometimes sibling paths relative to `__file__`. The
current joint run also compares its complete provenance dictionary when resumed.
Changing a source's whitespace or moving it can therefore invalidate a valid
artifact. Those checks must remain strict.

The organization adds **124 canonical module paths** while preserving **122
original flat implementation files byte for byte**. For example:

```python
from tactilelatent.models.marker_code_model import MarkerCodeVJEPA
from tactilelatent.datasets.acquisition_joint import AcquisitionClips
from tactilelatent.training.acquisition_training import install
from tactilelatent.evaluation.representation_probe import fit_probe
```

For those preserved implementations, the canonical module and its historical
flat name refer to the **same Python module object**. Private helpers, global
registries, runtime patch installation, class identities, and monkeypatches are
shared. The implementation's original `__file__`, `__module__`, and sibling asset
lookups remain unchanged. These aliases are not copies of the implementation.
Old imports and old `python -m tactilelatent.<module>` commands remain available.
Canonical `python -m tactilelatent.<namespace>.<module>` wrappers call the same
`main()` for modules that originally had an executable `__main__` guard.

Hash-pinned originals are an explicit compatibility exception to physical
organization, not disposable legacy duplicates. Do not overwrite their recorded
hashes, bypass a check, or rebuild expensive caches to make a cleanup pass.

The audit found matching hashes for **81 source modules in 131 JSON records**
across the local `processing/` tree. Those records include storage migration,
dataset acquisition, original/mirrored code preparation, running experiments,
and historical audits. This is a JSON-record inventory, not a claim that every
possible hash embedded only inside a checkpoint was enumerated. Preserving all
122 untouched originals also protects sources absent from that inventory.

Notable protected groups include:

- Storage migration: `storage`, `aligned_frames`, `sensor_data`, `media`,
  `joint_data`, `native_optical`, `marker_storage`, several viewer implementations,
  and `train_vjepa_markers`.
- Acquisitions: `acquisition_queue`, `acquisition_sources`,
  `acquisition_archives`, `acquisition_pilots`, `acquisition_data`,
  `acquisition_quality`, and `data_expansion`.
- The active 17-source run: V-JEPA, attention, task adapter, frozen teacher,
  marker, mirroring, current-frame selection, native-code, optimizer, and trainer
  implementations recorded in its `provenance.json`.
- Mirrored teacher acceptance: bilateral model/preparation, gel preparation,
  and balanced teacher training.
- FEEL hand ownership: even the small `feel_hand_layout.py` helper is pinned in
  the historical handedness audit, so it remains physically unchanged.

The refactor work directory contains `package_audit/path_manifest.json`,
`immutable_hashes.json`, and `historical_source_pins.json`. These are preparation
and recovery evidence. Maintained source code does not import them or require
anything under `ai_work/`.

## Actual implementation moves

Two demonstrated-safe, unpinned implementations were moved:

| Historical name | Canonical implementation | Behavior |
|---|---|---|
| `tactilelatent.losses` | [`training/losses.py`](../src/tactilelatent/training/losses.py) | Tensor functions are unchanged; the old path is a module-object alias |
| `tactilelatent.training_dashboard` | [`visualization/training_dashboard.py`](../src/tactilelatent/visualization/training_dashboard.py) | The old path is a module-object alias; project-root discovery accounts for the added directory |

The dashboard's HTML and JavaScript move alongside its implementation. HTTP
routes and existing viewer entry points are unchanged. All root viewer HTML,
JavaScript, and JSON assets are included by package-data rules; this also fixes
several assets that were present in an editable source checkout but omitted
from the previous wheel configuration.

Checkpoint model architectures, parameter names, state dictionaries, losses,
normalization, source splits, camera roles, seeds, sampling schedules, and
evaluation protocols are unchanged. Historical serialized class import paths
continue to resolve through the old modules. No dataset, checkpoint, cache,
manifest, provenance receipt, or run configuration is rewritten by this work.

## Safe command and test validation

Inspect a command's entry point before invoking `--help`. Some historical tools
have no argument parser and execute their work even when given that flag.
Import checks, synthetic model/checkpoint tests, and locally bound HTTP tests
are sufficient for this organization; full training, downloads, migrations,
and cache regeneration are not cleanup checks.

Run tests with an explicit staged `PYTHONPATH`, CUDA disabled, one BLAS thread,
and a separate temporary output root. Never install the staged tree over the
active editable installation during verification. Tests requiring retired
FreeTacMan imagery or absent production manifests are recorded as unavailable
resource checks rather than repaired by copying or mutating the datasets.
The focused `test_package_organization.py` checks module identity, lazy imports,
preserved tensor/checkpoint behavior, resource packaging, and canonical CLI
dispatch without launching training or downloads.

The new broad import audit also exposed two pre-existing historical failures:
`prepare_pilot` and `expand_pilot` import a removed `train.check_storage` helper.
Both the original and organized paths raise the same `ImportError`. They are
recorded as strict expected failures, with an additional explicit old/new error
comparison; they are not advertised as runnable workflows or silently repaired
by removing their storage protection. All other 122 historical module imports
resolve in the tested environment.
