# Research-core verification — October 22, 2026

These checks used the isolated prepared source tree. The latest user instruction
authorizes stopping training immediately and deploying, replacing both the earlier
30-epoch completion gate and the subsequent next-saved-epoch request. Four epochs
were committed and epoch 5 was unfinished at authorization. The migration must
verify the stopped worker and committed epoch-4 checkpoint before source cutover;
unfinished epoch-5 work is discarded. Verification itself did not modify live
training, checkpoints, source datasets or scientific cache arrays.

The original run remains an intentionally stopped 4-of-30-epoch experiment;
its original configuration, metrics, scientific hashes and checkpoint evidence
are preserved without rewriting their recorded budget. The schema-2 trainer
rejects historical checkpoints for resume, while historical inference and
evaluation remain available. No new training is launched by this migration.

## Model, optimization and resume

The actual epoch-3 checkpoint loaded strictly. Five source heads under ego-only
and all-camera masks produce bitwise-identical outputs against archived code.
All 13 frozen teacher families match for encoding/decoding/reconstruction.
The default seed-45 constructor matches all 166 initial state tensors, parameter
iteration order and RNG state. The current model is an actual standalone
implementation with no DINO-era inheritance chain.

A complete SAM AdamW update, including dropout replay, gradients, optimizer and
RNG state, matches the archived implementation. Full v2 checkpoint save/resume
reproduces the next update and scheduler/random states exactly. Inference-only,
legacy and changed-provenance resume inputs are rejected. Scientific source pins
exclude unrelated viewer/evaluation changes.

## Native datasets and extension

All 17 training sources and held-out MotionTrans load through the registry.
Thirty-four real training/validation samples compare **330 tensor fields bitwise**
against archived loaders: RGB, native targets, masks, codes, chronology and marker
geometry. All five mirrors and frame-zero preview histories were checked.

The complete ordered **748,784 training** and **148,201 validation** window lists
match archived reconstruction hashes. Full sampled schedules match at epochs
0, 1, 4, 10 and 29: 48,930 batches / 97,857 clips per training epoch.
A synthetic additional registered source passes registry construction, batching,
model head creation and objective/backpropagation without model/trainer edits.

Strict tests reject changed teachers, statistics, policies, rows, cached payloads
and consumer receipts. Importing caches writes only separate metadata manifests;
old receipts and recorded hashes are not rewritten and arrays are not regenerated.
Full-payload verification is explicit and was tested with small corruption
fixtures; a new complete traversal of every production code array was not run.

Baihu's older raw extractor/tracker hashes have no matching source bytes among
28 candidates and available backups. Import manifests explicitly report that raw
producer history as unavailable. This is separate from the verified later code
producers, accepted teacher hashes, episode identities and cached-artifact checks.
The refactor does not claim to have independently reproduced raw extraction.

## Viewer, tools and packaging

Native sensor reconstruction renders for all 17 sources using real artifacts.
Current controls have CPU browser checks with no JavaScript errors; attention
capture preserves predictions and masks missing cameras. Predictions use the
selected checkpoint's frozen teacher, not a later source teacher.

The isolated historical worker verifies 469 original source/config/script
relocation records. Actual HTTP checks pass for its catalog (58 models / 9,328
episodes), joint catalog, mirrored page, attention/pressure scripts and MANO mesh
assets. IDE-proxy prefixes and trailing-slash redirects are tested.

The final maintained suite is **99 passed**. Separate migration-helper tests are
**22 passed**, covering partial writes, rollback, tampering, protected paths,
worker gates and completion evidence. The offline wheel includes current HTML,
JavaScript and CSS resources plus operational tools, without private workspaces.
Help/discovery are lazy and side-effect-free. A temporary audit runs entirely
from its dated task workspace. Maintained code imports no private workspace files.

All 268 prior source files, 140 scripts, 61 configurations and 63 tests are
preserved byte-for-byte under `legacy/v1/`. The staged root has no logs, download
logs or downloader shim. Log links retain their original processing destinations.

## Evidence and limits

Evidence is under `ai_work/refactor/20261022_000723/`:
`model_verification.json`, `viewer_*verification.json`,
`work/research_audit/{datasets_verification,sample_parity,window_order_parity,sampling_parity}.json`,
and `scripts_audit/independent_cutover_review.json`.
The dated scratch workspace contains `audits/final_core_suite.xml`, command logs,
wheel and isolated test outputs.

An untouched archival baseline recorded 540 passes, 21 unavailable production
resource checks, three skips and two expected failures. These old tests remain
archived; their counts are not presented as the new maintained suite.
No frozen-backbone GPU benchmark, production optimization, new transfer fit,
training search, download or scientific experiment was run during this refactor.
Post-cutover CPU checks, cache-import receipts and current/historical HTTP health
checks run only after the immediate stop and saved checkpoint are verified and the source tree
has been installed. The limited root-log/downloader cleanup already applied is
separate from that pending core cutover. Read the migration status/journal for
actual deployment, rather than treating these staged checks as deployment proof.


The first live application passed all 87 then-current tests but rolled back when
cache import encountered an inherited policy path under the old `configs/`. The
resolver now uses the verified historical relocation manifest for those exact
source/config identities. A subprocess simulation made the original source,
script and configuration paths unavailable; all 18 source registrations and all
15 existing cache imports passed with unchanged scientific receipts and arrays.
The revised maintained suite has 99 passing tests. The authorized-stop gate has
32 isolated tests, including the actual SIGTERM exit convention.


## Completed live deployment

The retry completed successfully after the user-authorized immediate stop.
All 99 maintained tests passed in the installed tree, all 15 existing sensor-cache
imports were published, and both current and historical viewer catalogs returned
healthy responses. Training remains inactive with four completed epochs preserved.
The migration did not regenerate scientific arrays or restart training.
