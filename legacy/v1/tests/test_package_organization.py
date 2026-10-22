"""Namespace compatibility must preserve scientific and operational behavior.

All inputs are synthetic. These checks do not access production processing
artifacts, create a backbone, or perform a training/download workflow.
"""
import importlib
from importlib import resources
import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
NAMESPACES = ("datasets", "models", "training", "evaluation", "visualization")
MODULES = [(group, name) for group in NAMESPACES
           for name in importlib.import_module("tactilelatent." + group).__all__]
HISTORICAL_IMPORT_FAILURES = {"prepare_pilot", "expand_pilot"}
IMPORT_CASES = [pytest.param(group, name, marks=pytest.mark.xfail(
                    strict=True, raises=ImportError,
                    reason="Pre-existing historical converter imports removed train.check_storage"))
                if name in HISTORICAL_IMPORT_FAILURES else (group, name)
                for group, name in MODULES]


def isolated_python(arguments, *, cwd=ROOT):
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), CUDA_VISIBLE_DEVICES="",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    return subprocess.run([sys.executable, *arguments], cwd=cwd, env=env,
                          text=True, capture_output=True, timeout=30)


@pytest.mark.parametrize("group,name", IMPORT_CASES)
def test_old_and_canonical_imports_share_exact_module_object(group, name, monkeypatch):
    old = importlib.import_module("tactilelatent." + name)
    canonical = importlib.import_module(f"tactilelatent.{group}.{name}")
    assert canonical is old
    assert sys.modules["tactilelatent." + name] is canonical
    assert sys.modules[f"tactilelatent.{group}.{name}"] is old
    # Shared module identity is required by existing runtime installation and
    # private helper monkeypatching; a from-import-star wrapper is insufficient.
    sentinel = object()
    monkeypatch.setattr(old, "_refactor_identity_probe", sentinel, raising=False)
    assert canonical._refactor_identity_probe is sentinel


@pytest.mark.parametrize("name", sorted(HISTORICAL_IMPORT_FAILURES))
def test_known_historical_import_failures_are_identical_not_repaired(name):
    # Preserve and explicitly document these baseline failures. Do not add a
    # storage-check bypass to a historical data converter during organization.
    with pytest.raises(ImportError, match="check_storage") as before:
        importlib.import_module("tactilelatent." + name)
    with pytest.raises(ImportError, match="check_storage") as after:
        importlib.import_module("tactilelatent.datasets." + name)
    assert str(before.value) == str(after.value)


def test_every_historical_module_has_one_responsibility():
    names = [name for _, name in MODULES]
    assert len(names) == len(set(names)) == 124
    assert set(names) == {p.stem for p in (ROOT / "src/tactilelatent").glob("*.py")
                          if p.stem != "__init__"}


def test_namespace_imports_are_lazy_and_resolve_staged_source(tmp_path):
    code = '''import importlib,json,sys
import tactilelatent
for name in ('datasets','models','training','evaluation','visualization'):
    importlib.import_module('tactilelatent.'+name)
print(json.dumps(dict(source=tactilelatent.__file__,torch='torch' in sys.modules,
                     av='av' in sys.modules,ot='ot' in sys.modules)))
'''
    result = isolated_python(["-c", code], cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert Path(report["source"]).resolve() == ROOT / "src/tactilelatent/__init__.py"
    assert not report["torch"] and not report["av"] and not report["ot"]


@pytest.mark.parametrize("old,canonical", [
    ("tactilelatent.train_vjepa_markers", "tactilelatent.training.train_vjepa_markers"),
    ("tactilelatent.evaluate_holdout_decoder", "tactilelatent.evaluation.evaluate_holdout_decoder"),
    ("tactilelatent.autoencoder_viewer", "tactilelatent.visualization.autoencoder_viewer"),
])
def test_audited_help_commands_preserve_arguments_and_defaults(old, canonical, tmp_path):
    # These three main functions parse argv before loading configs, writing
    # output, instantiating backbones, or starting a server. Do not expand this
    # list to unfamiliar commands without inspecting their main functions.
    before = isolated_python(["-m", old, "--help"], cwd=tmp_path)
    after = isolated_python(["-m", canonical, "--help"], cwd=tmp_path)
    assert before.returncode == after.returncode == 0, (before.stderr, after.stderr)
    assert before.stdout == after.stdout
    assert not list(tmp_path.iterdir())


def test_pinned_modules_retain_original_file_and_private_helpers():
    from tactilelatent.datasets import aligned_frames
    from tactilelatent.training import train_vjepa_markers
    from tactilelatent.visualization import autoencoder_viewer
    for module in (aligned_frames, train_vjepa_markers, autoencoder_viewer):
        assert Path(module.__file__).parent == ROOT / "src/tactilelatent"
    assert aligned_frames.file_hash.__module__ == "tactilelatent.aligned_frames"
    assert train_vjepa_markers.check_epoch_extension.__module__ == "tactilelatent.train_vjepa_markers"


def test_canonical_checkpoint_load_preserves_state_keys_and_exact_outputs(tmp_path):
    from tactilelatent.marker_code_model import MarkerCodeVJEPA as Old
    from tactilelatent.models.marker_code_model import MarkerCodeVJEPA as Canonical
    torch.set_num_threads(1)
    config = dict(sensor_code_dims={"ftp_unit": 3}, numeric_shapes={},
                  statistics={"ftp_unit": dict(mean=[0.] * 3, std=[1.] * 3)},
                  latent_dim=32, input_dim=12, width=16, heads=4, layers=2,
                  feedforward=32, grid_size=2, dropout=0., prediction_mode="current_frame")
    torch.manual_seed(45)
    original = Old(**config).eval()
    features = torch.randn(1, 4, 3, 4, 12)
    camera_mask = torch.ones(1, 8, 3, dtype=torch.bool)
    path = tmp_path / "checkpoint.pt"
    torch.save(dict(model=original.state_dict(), model_config=config), path)
    state = torch.load(path, map_location="cpu", weights_only=False)
    restored = Canonical(**state["model_config"]).eval()
    restored.load_state_dict(state["model"], strict=True)
    assert list(original.state_dict()) == list(restored.state_dict())
    old_output = original(features, camera_mask, "ftp_unit", sample_latent=False)
    new_output = restored(features, camera_mask, "ftp_unit", sample_latent=False)
    for key in old_output:
        if torch.is_tensor(old_output[key]):
            torch.testing.assert_close(old_output[key], new_output[key], atol=0, rtol=0)
        else:
            assert old_output[key] == new_output[key]


def test_moved_loss_functions_keep_private_identity_values_and_gradients():
    from tactilelatent import losses as old
    from tactilelatent.training import losses
    assert old is losses
    assert Path(losses.__file__).parent.name == "training"
    # One valid cell makes TV zero. The objective is x² + .5|x| here.
    prediction = torch.tensor([[[[[.7]]]]], requires_grad=True)
    output = dict(pressure=prediction, mu=torch.zeros(1, 1, 2), logvar=torch.zeros(1, 1, 2))
    loss = losses.objective(output, torch.full_like(prediction, .2),
                            torch.ones_like(prediction, dtype=torch.bool),
                            contact_weight=1., tv_weight=0., beta=0.)
    assert loss.item() == pytest.approx(.5)
    loss.backward()
    assert prediction.grad.item() == pytest.approx(1.5)
    assert old.objective is losses.objective


def test_dashboard_move_preserves_project_root_and_paired_assets(monkeypatch):
    from tactilelatent import training_dashboard as old
    from tactilelatent.visualization import training_dashboard
    assert old is training_dashboard
    assert Path(training_dashboard.__file__).parent == ROOT / "src/tactilelatent/visualization"
    dashboard = training_dashboard.TrainingDashboard()
    assert dashboard.project == ROOT
    assert dashboard.thread is None
    assert dashboard.journal == ROOT / "processing/viewer_training_history_v1"
    for name in ("training_dashboard.html", "training_dashboard.js"):
        assert resources.files("tactilelatent.visualization").joinpath(name).is_file()


def test_moved_dashboard_installs_into_the_same_legacy_viewer_module(monkeypatch):
    from tactilelatent import mirror_viewer as legacy
    from tactilelatent.visualization import mirror_viewer, training_dashboard
    assert mirror_viewer is legacy
    previous = lambda viewer, joint=None: None
    monkeypatch.setattr(legacy, "make_mirror_handler", previous)
    training_dashboard.install()
    installed = legacy.make_mirror_handler
    assert installed is mirror_viewer.make_mirror_handler
    assert installed is not previous and installed._training_dashboard
    training_dashboard.install()
    assert legacy.make_mirror_handler is installed


def test_package_data_contains_every_viewer_asset():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    rules = config["tool"]["setuptools"]["package-data"]
    for package in ("tactilelatent", "tactilelatent.visualization"):
        root = resources.files(package)
        for asset in root.iterdir():
            if asset.suffix in (".html", ".js", ".json"):
                assert any(Path(asset.name).match(pattern) for pattern in rules[package]), asset
