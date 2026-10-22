"""Archived command compatibility; only inert fixture outputs are written.

The image pilots are imported after source inspection but their main functions
are never called. The comparison and shell download worker run only against
isolated fake results and a fake Hugging Face executable.
"""
from pathlib import Path
import ast
import json
import os
import runpy
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
ARCHIVED_PYTHON = (
    "compare_sam_motiontrans.py",
    "audit_saved_visual_video.py",
    "audit_visual_video_quality.py",
)


@pytest.mark.parametrize("name", ARCHIVED_PYTHON)
def test_archive_import_preserves_original_module_context(name):
    wrapper = ROOT / "scripts" / name
    body = ROOT / "scripts/archive" / name
    original_argv = sys.argv[:]
    imported = runpy.run_path(str(wrapper), run_name="compatibility_probe")
    archived = runpy.run_path(str(body), run_name="compatibility_probe")
    assert sys.argv == original_argv
    assert imported["__file__"] == str(wrapper)
    assert imported["__name__"] == "compatibility_probe"
    assert set(k for k in imported if not k.startswith("_")) == set(
        k for k in archived if not k.startswith("_")
    )
    assert imported["main"].__code__.co_code == archived["main"].__code__.co_code
    assert imported["main"].__code__.co_consts == archived["main"].__code__.co_consts
    assert imported["main"].__code__.co_filename == str(wrapper)
    assert imported["main"].__globals__ is imported["round_trip"].__globals__ if "round_trip" in imported else True


def _probe_result(multiplier):
    metrics = {
        key: 0.1 * multiplier
        for key in (
            "deterministic_mae", "ensemble_mean_mae", "deterministic_mse",
            "ensemble_mean_mse", "best_of_samples_clip_mae",
        )
    }
    metrics.update(sample_count=64, validation_clips=12)
    metrics["regions"] = {
        "all": {
            "crps": 0.02 * multiplier,
            "mean_predictive_std": 0.01,
            "std_error_pearson": 0.1,
            "intervals": {level: {"empirical_coverage": float(level)}
                          for level in ("0.5", "0.8", "0.9", "0.95")},
        }
    }
    return {
        "checkpoint_epoch": 49, "manifest_sha256": "inert-manifest",
        "source": "ftp_motiontrans", "head": [512], "seed": 45,
        "fit_clips": 10, "tune_clips": 3, "train_clips": 13,
        "target_shape": [6], "validation": metrics,
        "checkpoint_sha256": "inert-checkpoint", "selected_epochs": 7,
    }


def test_old_cli_and_archived_body_produce_identical_reports(tmp_path):
    parent = tmp_path / "processing/runs"
    sam = parent / "joint_all_sam_holdout_motiontrans_fresh_8k_50ep/evaluations/motiontrans_final"
    baseline = parent / "joint_all_holdout_motiontrans_fresh_8k_50ep/evaluations/motiontrans"
    for path, factor in ((sam, 0.9), (baseline, 1.0)):
        path.mkdir(parents=True)
        (path / "results.json").write_text(json.dumps(_probe_result(factor)))
    output = sam / "non_sam_comparison.json"
    body = ROOT / "scripts/archive/compare_sam_motiontrans.py"
    wrapper = ROOT / "scripts/compare_sam_motiontrans.py"
    archived = subprocess.run([sys.executable, str(body), "retained-positional-argument"],
                              cwd=tmp_path, capture_output=True, text=True, check=True)
    expected = output.read_bytes()
    output.unlink()
    compatible = subprocess.run([sys.executable, str(wrapper), "retained-positional-argument"],
                                cwd=tmp_path, capture_output=True, text=True, check=True)
    assert output.read_bytes() == expected
    assert compatible.stdout == archived.stdout
    assert json.loads(expected)["metrics"]["deterministic_mae"]["sam_relative_change_percent"] == pytest.approx(-10)
    repeat = subprocess.run([sys.executable, str(wrapper)], cwd=tmp_path,
                            capture_output=True, text=True)
    assert repeat.returncode != 0
    assert "FileExistsError" in repeat.stderr
    assert output.read_bytes() == expected


@pytest.mark.parametrize("download_exit", [0, 23])
def test_root_download_command_uses_only_fixture_downloader(tmp_path, download_exit):
    archive_dir = tmp_path / "scripts/archive"
    archive_dir.mkdir(parents=True)
    shutil.copy2(ROOT / "download_egotouch_worker.sh", tmp_path)
    shutil.copy2(ROOT / "scripts/archive/download_egotouch_worker.sh", archive_dir)
    binary = tmp_path / ".venv/bin/hf"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" > \"$TACTILE_TEST_ARGS\"\nexit \"$TACTILE_TEST_EXIT\"\n")
    binary.chmod(0o755)
    (tmp_path / "download_logs").mkdir()
    (tmp_path / "processing").mkdir()
    target = tmp_path / "datasets/zhenyuxie-zhzh__EgoTouch_hdf5"
    target.mkdir(parents=True)
    reservation = tmp_path / "datasets/.reservations/zhenyuxie-zhzh__EgoTouch_hdf5.bytes"
    reservation.parent.mkdir()
    reservation.write_text("fixture-budget")
    args_file = tmp_path / "fake_hf_args.txt"
    env = dict(os.environ, TACTILE_TEST_ARGS=str(args_file), TACTILE_TEST_EXIT=str(download_exit))
    result = subprocess.run(["bash", str(tmp_path / "download_egotouch_worker.sh"), "ignored-old-argument"],
                            cwd="/", env=env, capture_output=True, text=True)
    assert result.returncode == download_exit
    assert args_file.read_text().splitlines() == [
        "download", "zhenyuxie-zhzh/EgoTouch_hdf5", "--repo-type", "dataset",
        "--local-dir", "datasets/zhenyuxie-zhzh__EgoTouch_hdf5",
    ]
    assert not reservation.exists()
    assert ("DONE repo=" in result.stdout) == (download_exit == 0)


def test_archived_bodies_have_no_location_or_source_hash_dependency():
    for name in ARCHIVED_PYTHON:
        source = (ROOT / "scripts/archive" / name).read_text()
        ast.parse(source)
        assert "__file__" not in source
        assert "sys.path" not in source
