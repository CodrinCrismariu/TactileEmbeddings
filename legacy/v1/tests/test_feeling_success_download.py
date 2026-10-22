import importlib.util
from pathlib import Path
from types import SimpleNamespace
import zlib

import h5py
import numpy as np
import pytest


spec = importlib.util.spec_from_file_location("feeling_download", Path(__file__).resolve().parents[1] / "scripts/download_feeling_success.py")
download = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download)


def test_compaction_is_lossless_preserves_labels_and_discards_tactile(tmp_path):
    source, target = tmp_path / "source.h5", tmp_path / "compact.h5"
    rgb = np.arange(90, dtype=np.uint8).reshape(5, 6, 3)
    with h5py.File(source, "w") as f:
        for index in range(3):
            g = f.create_group(f"data/i{index}")
            g.attrs["object_name"] = np.bytes_(f"object-{index}")
            g.attrs["is_gripping"] = index % 2
            for field in download.RGB_FIELDS:
                g.create_dataset(field, data=rgb)
            g.create_dataset("gelsightA_before", data=rgb)
    sizes = []
    report = download.compact_member(source, target, "h5/source.h5", budget_check=sizes.append)
    assert len(report["trials"]) == 3 and len(report["objects"]) == 3
    assert report["trials"][1]["success"] is True
    assert sizes and sizes[0] >= 810
    with h5py.File(target, "r") as f:
        assert f.attrs["role"] == "evaluation_only"
        assert "kinectA_rgb_after" in f.attrs["outcome_only_fields"]
        for index in range(3):
            assert "gelsightA_before" not in f[f"i{index}"]
            for field in download.RGB_FIELDS:
                np.testing.assert_array_equal(f[f"i{index}/{field}"][...], rgb)


def test_budget_refuses_cap_overflow(monkeypatch):
    monkeypatch.setattr(download, "usage", lambda: download.CAP - download.RESERVE - 10)
    monkeypatch.setattr(download.shutil, "disk_usage", lambda _: SimpleNamespace(free=100_000_000_000))
    download.check_budget(10)
    with pytest.raises(RuntimeError, match="Storage cap"):
        download.check_budget(11)


def test_member_crc_is_checked(tmp_path, monkeypatch):
    monkeypatch.setattr(download, "check_budget", lambda *_: 0)
    body = b"small test HDF5 payload" * 100
    compressor = zlib.compressobj(wbits=-15)
    compressed = tmp_path / "member.part"
    compressed.write_bytes(compressor.compress(body) + compressor.flush())
    entry = SimpleNamespace(compress_type=8, file_size=len(body), CRC=zlib.crc32(body))
    destination = tmp_path / "raw.h5"
    download.inflate_member(entry, compressed, destination)
    assert destination.read_bytes() == body
    entry.CRC += 1
    with pytest.raises(ValueError, match="CRC"):
        download.inflate_member(entry, compressed, destination)
