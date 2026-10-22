# Historical artifacts and current code

The maintained package is a physical research core. It does not import a flat
collection of compatibility modules or monkeypatch historical trainers.
Historical source, commands, configurations and tests are preserved byte-for-byte
under `legacy/v1/`. Old training launchers are records, not supported commands.
Datasets, processing outputs, frozen assets and model checkpoints stay in place.

## Checkpoints and initialization

The extracted current-frame model retains historical state-dict names and exact
seed-45 constructor RNG consumption, including formerly discarded layers.
Actual saved-checkpoint outputs, all current teacher families and a complete SAM
update are compared against archived implementations in isolated CPU tests.

Current training writes `schema_version: 2` checkpoints containing full model,
optimizer, scheduler, SAM, loader/random states and artifact/implementation
provenance. Only these full checkpoints resume current training. Viewer epoch
snapshots are explicitly inference-only. Historical checkpoints remain accessible
through a separate reader and retain their own teacher choices and normalizations.

The deterministic current-frame V-JEPA family also supports an **explicit verified
import** into a new, empty run:

```bash
tl train --import-checkpoint PATH_TO_PARENT/last.pt --import-only
tl train --resume PATH_TO_NEW_RUN
```

This checks the parent configuration, model/head order and statistics, source
artifacts, teacher and mirror bindings, backbone, optimizer slots, learning-rate
position, SAM updates and every saved RNG state. It records the parent checkpoint
hash and original metadata in `checkpoint_import.json`, retains an immutable
`imported_checkpoint.pt` anchor and copies completed parent metrics. Original
files are read only. The imported `last.pt` has schema 2 and then uses the ordinary
strict resume path. Incomplete, inference-only, unrelated architectures and
changed scientific settings are rejected; a historical launcher is never run.
All 30 configured epochs remain the original budget: importing four completed
epochs continues at epoch 5 with the saved learning rate and 26 epochs remaining.

## Importing existing sensor codes

`tl data cache` verifies existing producer receipts, source/episode identities,
teacher digests, code dimensions/statistics, and mirrored policy identities before
writing separate import manifests at the versioned `datasets.json:cache_imports`
location (currently `processing/core_v2_cache_imports_standards_v1/`).
The producer digests are checked against preserved archived bytes. New consumer
files are recorded separately. Existing arrays, original receipts and their
recorded hashes are never rewritten to match the new code.

Raw extraction records and cached-code producer records describe different
stages. A raw historical extractor whose matching source bytes cannot be found
must be reported as an unavailable historical producer; a later code-cache check
does not prove that extractor was independently reproduced.

## Historical viewer

`/legacy/` explicitly proxies an isolated worker using archived Python code and
original resource conventions. The current process imports only maintained code.
Small static links from `legacy/v1/` to existing `datasets/`, `processing/`,
`mano_models/` and `.venv/` preserve resource lookup without copying these assets.
These links are not training shims. Historical model entries, attention maps,
wide autoencoders and hand-mesh resources remain available there.

Operational logs and historical shortcuts move under `logs/`; shortcut targets
continue to point at original `processing/` files. The root downloader is archived
without a forwarding script.
