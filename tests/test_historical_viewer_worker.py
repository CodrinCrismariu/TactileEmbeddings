"""Explicit historical read compatibility never changes the core process."""

import builtins
import hashlib
import io
import json
import os
import stat

import pytest

from tactilelatent.visualization.historical_worker import VerifiedRelocations


def fixture(tmp_path):
    original = tmp_path / "original"
    archive = tmp_path / "archive"
    old = archive / "configs/old.json"
    old.parent.mkdir(parents=True)
    old.write_text('{"value":"historical"}')
    current = original / "configs/old.json"
    current.parent.mkdir(parents=True)
    current.write_text('{"value":"current"}')
    manifest = archive / "artifact_relocations.json"
    document = dict(
        schema="historical-viewer-relocations-v1",
        original_root=str(original),
        files={
            str(current): dict(
                archived_path="configs/old.json",
                sha256=hashlib.sha256(old.read_bytes()).hexdigest(),
                mode=stat.S_IMODE(old.stat().st_mode),
            ),
        },
    )
    manifest.write_text(json.dumps(document))
    return original, archive, old, current, manifest


def test_old_absolute_collision_reads_verified_archive_bytes_and_restores_process(tmp_path):
    original, archive, old, current, manifest = fixture(tmp_path)
    functions = builtins.open, io.open, os.stat, os.lstat
    guard = VerifiedRelocations(manifest, archive)
    assert current.read_text() == '{"value":"current"}'
    with guard:
        assert current.read_text() == '{"value":"historical"}'
        assert os.stat(current).st_size == len('{"value":"historical"}')
        with builtins.open(current) as stream:
            assert json.load(stream)["value"] == "historical"
        with pytest.raises(PermissionError, match="read-only"):
            current.write_text("overwrite")
        with pytest.raises(PermissionError, match="read-only"):
            old.write_text("overwrite")
        assert (original / "configs").is_dir()
    assert functions == (builtins.open, io.open, os.stat, os.lstat)
    assert current.read_text() == '{"value":"current"}'


def test_changed_archived_bytes_block_worker_instead_of_rewriting_expected_hash(tmp_path):
    original, archive, old, current, manifest = fixture(tmp_path)
    saved = manifest.read_bytes()
    old.write_text("changed source")
    with pytest.raises(ValueError, match="hash mismatch"):
        VerifiedRelocations(manifest, archive)
    assert manifest.read_bytes() == saved


def test_unrecorded_original_files_are_not_read_from_new_source_tree(tmp_path):
    original, archive, old, current, manifest = fixture(tmp_path)
    unrecorded = original / "configs/new.json"
    unrecorded.write_text("new core data")
    with VerifiedRelocations(manifest, archive):
        with pytest.raises(FileNotFoundError, match="No verified historical relocation"):
            unrecorded.read_text()
    assert unrecorded.read_text() == "new core data"


def test_relocation_manifest_cannot_escape_archive_or_original_roots(tmp_path):
    original, archive, old, current, manifest = fixture(tmp_path)
    document = json.loads(manifest.read_text())
    document["files"][str(current)]["archived_path"] = "../outside.py"
    manifest.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="escapes"):
        VerifiedRelocations(manifest, archive)


def test_relocation_permissions_are_verified(tmp_path):
    original, archive, old, current, manifest = fixture(tmp_path)
    document = json.loads(manifest.read_text())
    document["files"][str(current)]["mode"] ^= stat.S_IWUSR
    manifest.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="permissions"):
        VerifiedRelocations(manifest, archive)
