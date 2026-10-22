# Native dataset adapters

All sources expose the contracts in `../contracts.py`. `sources/definitions.py`
records source metadata and registration; `sources/native.py` contains native
readers, synchronization and channel/geometry conventions. `sources/video.py`
loads only visual inputs at the backbone resolution. `codecs/` holds explicit
reversible supervision normalization. Native tactile grids/images are not resized.

To add an existing representation: register a source definition/adapter, prepare
its episode metadata and code/direct-supervision specification, then select it in
`configs/datasets.json`. The registry derives its head. Do not add source branches
to the shared model or training loop. A new sensor representation may require a
codec/renderer/teacher in the appropriate maintained package.

Training uses the common `JointDataset`, one source per batch and one collator.
Only variable marker counts are padded, preserving signed vectors, reference
coordinates, surface identity, spacing and validity. Mirroring transforms native
targets and camera roles; code coordinates are never manually mirrored.
