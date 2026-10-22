"""Immutable cache receipts can outlive absolute source/config locations."""

import json
import stat

import pytest

from tactilelatent.dataset_management.artifacts import ArtifactPaths, file_hash


@pytest.fixture
def relocation(tmp_path):
    root = tmp_path / "project"
    metadata = tmp_path / "staged"
    original = root / "configs/bimanual_mirror_policy_v2.json"
    archived = metadata / "legacy/v1/configs/bimanual_mirror_policy_v2.json"
    archived.parent.mkdir(parents=True)
    archived.write_text('{"original_policy":true}\n')
    manifest = metadata / "legacy/v1/artifact_relocations.json"
    document = {
        "schema": "historical-viewer-relocations-v1",
        "original_root": str(root),
        "files": {
            str(original): {
                "archived_path": "configs/bimanual_mirror_policy_v2.json",
                "sha256": file_hash(archived),
                "mode": stat.S_IMODE(archived.stat().st_mode),
            }
        },
    }
    manifest.write_text(json.dumps(document))
    return ArtifactPaths(root, metadata_root=metadata), original, archived, manifest, document


def test_missing_original_absolute_policy_uses_exact_verified_archive(relocation):
    paths, original, archived, manifest, document = relocation
    before = manifest.read_bytes()
    assert not original.exists()
    assert paths.resolve(original) == archived
    assert paths.resolve(original, historical=True) == archived
    assert manifest.read_bytes() == before


def test_current_collision_is_overridden_only_for_explicit_historical_reads(relocation):
    paths, original, archived, manifest, document = relocation
    original.parent.mkdir(parents=True)
    original.write_text('{"current_policy":true}\n')
    assert paths.resolve(original) == original
    assert paths.resolve(original, historical=True) == archived
    assert paths.resolve("configs/bimanual_mirror_policy_v2.json") == original
    assert paths.resolve("configs/bimanual_mirror_policy_v2.json", historical=True) == archived
    assert paths.check(original, file_hash(archived)) == archived
    assert paths.producer(original) == archived
    current = original.with_name("new_current.json")
    current.write_text("{}")
    assert paths.resolve(current) == current
    with pytest.raises(FileNotFoundError, match="No verified historical relocation"):
        paths.resolve(current, historical=True)


def test_unrecorded_archive_filename_is_not_sufficient(relocation):
    paths, original, archived, manifest, document = relocation
    document["files"] = {}
    manifest.write_text(json.dumps(document))
    with pytest.raises(FileNotFoundError, match="No verified historical relocation"):
        paths.resolve(original)
    manifest.unlink()
    with pytest.raises(FileNotFoundError, match="No historical relocation manifest"):
        paths.resolve(original)


@pytest.mark.parametrize("change", ["schema", "root", "files", "entry", "path", "hash"])
def test_malformed_or_escaping_relocation_is_rejected(relocation, change):
    paths, original, archived, manifest, document = relocation
    if change == "schema":
        document["schema"] = "unknown"
    elif change == "root":
        document["original_root"] = "/different/project"
    elif change == "files":
        document["files"] = []
    elif change == "entry":
        document["files"][str(original)] = []
    elif change == "path":
        document["files"][str(original)]["archived_path"] = "../outside.json"
    elif change == "hash":
        document["files"][str(original)]["sha256"] = "invalid"
    manifest.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="Invalid historical relocation"):
        paths.resolve(original)


def test_missing_or_changed_archived_bytes_reject_without_manifest_rewrite(relocation):
    paths, original, archived, manifest, document = relocation
    before = manifest.read_bytes()
    archived.write_text("different")
    with pytest.raises(ValueError, match="hash mismatch"):
        paths.resolve(original)
    assert manifest.read_bytes() == before
    archived.unlink()
    with pytest.raises(FileNotFoundError, match="Missing immutable historical file"):
        paths.resolve(original)


def test_data_paths_and_explicit_relative_archive_paths_are_not_redirected(relocation):
    paths, original, archived, manifest, document = relocation
    data = paths.root / "processing/codes.npy"
    assert paths.resolve(data, historical=True) == data
    assert paths.resolve("legacy/v1/configs/bimanual_mirror_policy_v2.json") == archived


def test_absolute_producer_collision_cannot_fall_back_to_unrecorded_basename(relocation):
    paths, original, archived, manifest, document = relocation
    unrecorded = paths.root / "src/tactilelatent/unrecorded.py"
    fallback = paths.archived_producers / "unrecorded.py"
    fallback.parent.mkdir(parents=True)
    fallback.write_text("same_name_is_not_evidence = True\n")
    with pytest.raises(FileNotFoundError, match="No verified historical relocation"):
        paths.producer(unrecorded)
