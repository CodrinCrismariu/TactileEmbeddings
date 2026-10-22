# Organization verification

Observed October 21, 2026. This report records isolated checks performed before
deployment from `ai_work/refactor/20261021_231339/project/`. Live code and services
were unchanged during those checks. This verifies compatibility, not improved
model performance. Deployment and service-restoration evidence is recorded
separately in that refactor directory.

## Baseline and staged tests

Both runs used the existing interpreter, explicit isolated-source `PYTHONPATH`,
CUDA disabled, offline model settings, bounded CPU threads, and separate pytest
temporary/cache directories. Each process asserted `tactilelatent.__file__`
resolved inside the intended isolated tree, rather than the live editable install.

| Run | Passed | Failed | Setup errors | Skipped | Expected failures |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original source, no local resource fixtures | 353 | 31 | 35 | 3 | 0 |
| Original source, small policy fixtures and empty storage directories | 398 | 21 | 0 | 3 | 0 |
| Final staged source, identical resource fixtures | 540 | 21 | 0 | 3 | 2 |

The final comparison has **no new failures or errors**. The same 21 production
integration tests fail in both trees because `processing/joint_all/manifest.json`
and its native data are deliberately absent. They cover two joint-manifest
checks, five optical native-target checks, twelve source-loader comparisons,
native batch ordering, and retired FreeTacMan camera-role data. These checks
were reported, not silently deselected or repaired by copying large assets.

Four immutable policy-evidence files, totaling about 17 KiB, were copied equally
into both trees. Empty `datasets/` and `processing/` directories let synthetic
checkpoint tests perform their normal storage checks. No production directory
was symlinked into the test tree. The fixture manifest records all copied hashes.

The two expected failures are new compatibility checks exposing pre-existing
historical import errors in `prepare_pilot` and `expand_pilot`: both import the
removed `train.check_storage` name. Separate assertions confirm old and canonical
paths fail identically. Those commands are documented as blocked historical
converters, not advertised as supported current workflows.

The final suite adds 144 test cases: 142 pass, and the two historical failures
are strict expected failures. Coverage includes old/canonical module identity,
private monkeypatch visibility, lazy namespace imports, inspected CLI dispatch,
moved loss values/gradients, dashboard routes/install behavior, resource rules,
and archived command compatibility. An initial new-test assertion incorrectly
passed a string to `torch.testing.assert_close`; that fixture was corrected and
the full final suite was rerun.

Evidence under the refactor directory:

- `test_outputs/baseline.xml` and `baseline_run.json`.
- `test_outputs/baseline_fixture.xml`, `baseline_fixture_run.json`, and
  `policy_resource_fixtures.json`.
- `test_outputs/staged_final.xml`, `staged_final.log`, and `staged_final_run.json`.
- `test_outputs/suite_comparison.json`, including all unavailable checks by name.

## Saved checkpoint and scientific equivalence

Independent CPU processes loaded the actual completed epoch-3 snapshot:
`processing/runs/joint_vjepa21_all_sources32d_mirror_v2/viewer_epochs/epoch_003.pt`.
Its stored epoch is 2 because checkpoint metadata is zero-based. SHA-256:
`f91f59bc476e1079984b3e4651768756d09bd78d04741ee64e6088233a76e2b4`.

Both source trees strictly loaded its state dictionary. Seeded full spatial
tokens of shape `[1, 4, 3, 576, 768]` exercised EgoTouch, Baihu, EgoPressure,
T-Rex and FEEL Lab heads, with ego-only and all-camera masks. The actual frozen
Baihu and EgoPressure teachers also encoded, decoded, and reconstructed
synthetic native-contract inputs. **All 36 compared tensors are bitwise
identical; maximum absolute difference is 0.0.**

This checks the trainable reader, adapters/direct heads and selected teachers.
It does not rerun frozen V-JEPA or claim an end-to-end GPU/video benchmark.
The unchanged backbone implementation and existing tests cover its masking,
frozen status and temporal layout. No production optimizer update was executed.

Detailed evidence: `test_outputs/actual_checkpoint_equivalence.json` and the
two small output tensor files beside it. The checkpoint/teacher inputs were
read-only and their hashes remained unchanged.

## Provenance, imports, scripts and package resources

- All **122 retained flat implementations** match their original bytes. All
  **81 modules identified in 131 historical JSON provenance records** remain
  unchanged. This inventory does not claim to enumerate hashes found only in
  arbitrary binary checkpoints; retaining the other originals protects those
  unenumerated dependencies too.
- All **124 canonical module paths** are mapped to one responsibility. Old and
  new imports share the same module object, including global patches/private
  helpers. Historical `__file__` and class paths remain intact where pinned.
- Existing tests still reject stale source/mirror policies, incompatible cache
  implementation/model/config changes, invalid held-out participation, changed
  normalization/statistics, and wrong native shapes. Validation was not bypassed
  and no recorded receipt hashes were replaced.
- The script inventory covers **135 original commands**. **131 remain byte
  identical**; four retain explicit compatibility entry points to archived
  bodies. Every active and identified hash-pinned command remains unchanged.
- Three inspected old/canonical `python -m ... --help` pairs produce identical
  output without loading data or starting a server. Other commands were audited
  statically; unfamiliar scripts were not bulk-executed.
- Archived Python comparison commands were exercised on inert local result
  fixtures. The shell downloader wrapper was tested with a fake downloader,
  including failure propagation. No network/download command was executed.
- All 118 staged Python command files parse; all 21 shell command files pass
  `bash -n`.
- An offline wheel builds successfully and contains all **nine** HTML/JS/JSON
  runtime assets and all canonical modules. This corrects asset omissions from
  the old wheel configuration while retaining existing editable-checkout paths.
- Maintained source, scripts, and configurations contain no dependency on
  `ai_work/scratch/`, staged copies, or protected backups. Preparation helpers
  and verification outputs remain outside the production implementation.
- All 61 configuration JSON files parse and remain byte-identical. Fourteen
  ignore-rule checks confirm local work/data/model files are excluded while
  canonical code, archived research, the proposal, and `ai_work/README.md` remain
  visible to Git. Ignore rules do not enforce agent behavior; `AGENTS.md` does.

Audit inventories are under `package_audit/` and `scripts_audit/`. The original
proposal PDF is byte-identical after relocation; its SHA-256 is
`f8253dd7ee169000acad145b3f385d688ad65de1cee9504f99b38937de95155f`.
Documentation checks resolve maintained and archived links; references to large
local artifacts are checked against their unchanged live locations, without
making those locations writable test fixtures.

## Remaining limits and deployment checks

The full suite returns a nonzero status because the 21 unavailable production
checks remain visible. A dataset-complete end-to-end run, GPU replay, viewer
reconstruction against every production episode, and a browser audit of the
restarted service remain outside this cleanup's isolated validation. No
scientific hyperparameters, dataset formats, source membership, transfer
protocols or results were changed.

Git already existed at the audit. It was not initialized, committed, reset,
or published. Datasets, checkpoints, caches, research receipts and existing
service definitions were not rewritten during isolated preparation. The user
subsequently authorized pausing training for deployment. A disposable systemd
unit verified that cgroup freeze halts its heartbeat and thaw resumes it;
`test_outputs/freeze_mechanism.json` records this supported pause check.

[Migration and recovery](refactor_migration.md) gives the exact deployment gates,
service interruption, compatibility checks and rollback procedure. The pause
receipt records the actual process, checkpoint, approval, and affected services;
the deployment journal and runtime receipt record application and restoration.
