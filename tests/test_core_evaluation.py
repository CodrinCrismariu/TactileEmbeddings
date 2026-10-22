"""Small CPU probes and synthetic uniform clips; no production transfer is run."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tactilelatent.contracts import ClipSample, SensorTarget, Supervision, VisualClip
from tactilelatent.evaluation.probes import (
    episode_partition,
    feature_fingerprint,
    fit_probe_sweep,
    fit_statistics,
    score,
    validate_features,
)
from tactilelatent.evaluation.transfer import (
    checkpoint_paths,
    evaluate_experiment,
    fit_dataset_probe,
    load_probe,
    predict_probe,
    verify_checkpoint,
)


def episode(kind, number):
    for trial in range(10000):
        name = f"{kind}-{number}-{trial}"
        if episode_partition(name) == kind:
            return name
    raise AssertionError("No synthetic episode partition")


def feature_table(validation=False):
    generator = torch.Generator().manual_seed(3 if validation else 2)
    names = (
        [f"val-{i}" for i in range(4)]
        if validation
        else [episode("fit", 0), episode("fit", 1), episode("tune", 0)]
    )
    episodes = [name for name in names for _ in range(4)]
    x = torch.randn(len(episodes), 4, generator=generator)
    y = torch.sigmoid(x[:, :2])
    return dict(
        values=x,
        target=y,
        valid=torch.ones_like(y, dtype=torch.bool),
        episode_id=episodes,
        frame_indices=torch.tensor(list(range(4)) * len(names)),
    )


def config():
    return dict(
        head_seeds=[45],
        learning_rates=[0.01, 0.03],
        hidden_dims=[8],
        epochs=3,
        patience=2,
        batch_size=16,
        weight_decay=0.0001,
        bootstrap_replicates=32,
        split_seed=45,
        tune_fraction=0.1,
    )


def test_fit_only_statistics_exclude_tuning_and_validation():
    train = feature_table()
    ids = [i for i, e in enumerate(train["episode_id"]) if episode_partition(e) == "fit"]
    expected = train["values"][ids].mean(0)
    train["values"][
        [i for i, e in enumerate(train["episode_id"]) if episode_partition(e) == "tune"]
    ] = 100000
    mean, std = fit_statistics(train, ids)
    assert torch.equal(mean, expected)
    assert (std > 0).all()


def test_episode_leakage_and_feature_identity_rejected():
    train, val = feature_table(), feature_table(True)
    initial = feature_fingerprint(train)
    train["values"][0, 0] += 1
    assert feature_fingerprint(train) != initial
    val["episode_id"][0] = train["episode_id"][0]
    with pytest.raises(ValueError, match="Episode leakage"):
        validate_features(train, val)


def test_probe_fit_cache_and_viewer_prediction_roundtrip(tmp_path):
    torch.set_num_threads(1)
    train, val = feature_table(), feature_table(True)
    result = fit_probe_sweep(train, val, tmp_path / "probe", config(), torch.device("cpu"))
    assert result["head"] == [4, 8, 2]
    assert result["validation"]["bootstrap_replicates"] == 32
    assert len(result["validation"]["per_channel_mae"]) == 2
    assert "delta_variance_skill" in result["validation"]["dynamics"]
    network, state = load_probe(tmp_path / "probe/seed_45/head.pt")
    predicted = predict_probe(network, state, val["values"][:, None])
    assert predicted.shape == (len(val["values"]), 1, 2)
    direct = network((val["values"] - state["mean"]) / state["std"])
    torch.testing.assert_close(predicted[:, 0], direct)
    repeat = fit_probe_sweep(train, val, tmp_path / "probe", config(), torch.device("cpu"))
    assert repeat["selected_learning_rate"] == result["selected_learning_rate"]
    assert repeat["validation"] == result["validation"]
    val["target"][0, 0] += 0.1
    with pytest.raises(ValueError, match="protocol changed"):
        fit_probe_sweep(train, val, tmp_path / "probe", config(), torch.device("cpu"))


def test_validation_labels_never_select_learning_rate_or_duration(tmp_path):
    torch.set_num_threads(1)
    train, val = feature_table(), feature_table(True)
    first = fit_probe_sweep(train, val, tmp_path / "first", config(), torch.device("cpu"))
    changed = {**val, "target": 1 - val["target"]}
    second = fit_probe_sweep(train, changed, tmp_path / "second", config(), torch.device("cpu"))
    assert first["selected_learning_rate"] == second["selected_learning_rate"]
    assert first["selected_head_epochs"] == second["selected_head_epochs"]
    assert first["validation"]["mae"] != second["validation"]["mae"]


def test_missing_targets_do_not_become_zero_measurements(tmp_path):
    train, val = feature_table(), feature_table(True)
    val["valid"][:, 1] = False
    val["target"][:, 1] = float("nan")
    network = nn.Sequential(nn.Linear(4, 2), nn.Sigmoid())
    result = score(network, train, val, torch.device("cpu"), bootstrap=8)
    assert result["per_channel_valid_counts"][1] == 0
    assert result["mae"] >= 0
    assert torch.isfinite(torch.tensor(result["mae"]))


def test_dataset_classifier_is_generic_balanced_and_disjoint(tmp_path):
    torch.set_num_threads(1)
    fit = {}
    tune = {}
    validation = {}
    for i, source in enumerate(("sensor_z", "sensor_a")):

        def group(name, count):
            return dict(values=torch.ones(count, 4) * i, episode_id=[f"{source}-{name}"] * count)

        fit[source] = group("fit", 5 + i)
        tune[source] = group("tune", 3)
        validation[source] = group("val", 4)
    result = fit_dataset_probe(
        fit,
        tune,
        validation,
        dict(seed=45, hidden_dims=[4], epochs=2, patience=1),
        tmp_path,
        torch.device("cpu"),
    )
    assert result["sources"] == ["sensor_a", "sensor_z"]
    assert result["head"] == [4, 4, 2]
    assert result["validation"]["chance_accuracy"] == 0.5
    tune["sensor_a"]["episode_id"] = fit["sensor_a"]["episode_id"][:3]
    with pytest.raises(ValueError, match="episode leakage"):
        fit_dataset_probe(fit, tune, validation, {}, tmp_path / "invalid", torch.device("cpu"))


def test_epoch_selection_and_producer_hash_validation(tmp_path):
    run = tmp_path / "run"
    (run / "viewer_epochs").mkdir(parents=True)
    (run / "viewer_epochs/epoch_005.pt").write_bytes(b"fixture")
    assert checkpoint_paths(run, [5]) == [(5, run / "viewer_epochs/epoch_005.pt")]
    with pytest.raises(FileNotFoundError):
        checkpoint_paths(run, [10])
    registry = SimpleNamespace(
        heldout_source="heldout", excluded_sources={"removed"}, sources={"source"}
    )
    state = dict(
        epoch=4,
        model_config={
            "prediction_mode": "current_frame",
            "latent_mode": "deterministic",
            "sensor_code_dims": {"heldout": 2},
        },
        provenance={},
    )
    with pytest.raises(ValueError, match="held-out"):
        verify_checkpoint(state, registry, 5)
    state["model_config"]["sensor_code_dims"] = {"source": 2}
    with pytest.raises(ValueError, match="implementation provenance"):
        verify_checkpoint(state, registry, 5)


class SyntheticDataset:
    def __init__(self, split):
        self.split = split
        self.names = (
            [f"validation-{i}" for i in range(3)]
            if split == "val"
            else [episode("fit", 0), episode("fit", 1), episode("tune", 0)]
        )
        self.episodes = [{"episode_id": name} for name in self.names]
        self.windows = [(i, 15 * j) for i in range(len(self.names)) for j in range(2)]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        ep, start = self.windows[index]
        x = torch.ones(8, 3, 3, 2, 2) * (index + 1) / 10
        target = torch.full((1, 2, 1, 1), 0.2 + index / 20)
        return ClipSample(
            VisualClip(
                x, torch.ones(8, 3, dtype=torch.bool), torch.arange(8), camera_roles=torch.arange(3)
            ),
            SensorTarget(target, torch.ones_like(target, dtype=torch.bool), "force_tanh"),
            Supervision(target, torch.ones_like(target, dtype=torch.bool), "direct"),
            "heldout",
            self.names[ep],
            self.split,
            torch.tensor([start + 14]),
            torch.tensor([start + 14]),
        )


class SyntheticEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.sensor_code_dims = {"source": 2}
        self.numeric_shapes = {}

    def encode(self, features, mask, camera_roles=None):
        mu = features.reshape(-1, 1, 1).expand(-1, 1, 4)
        return mu, torch.full_like(mu, -torch.inf)


class SyntheticBackbone(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.provenance = {}


def test_actual_evaluation_api_runs_only_synthetic_cpu_contract(tmp_path, monkeypatch):
    import tactilelatent.configuration as configuration_module
    import tactilelatent.dataset_management as management
    import tactilelatent.models as models
    import tactilelatent.models.backbone as backbone_module

    root = tmp_path
    run = root / "processing/runs/synthetic"
    run.mkdir(parents=True)
    signature = hashlib.sha256(Path(configuration_module.__file__).read_bytes()).hexdigest()
    torch.save(
        dict(
            schema_version=2,
            epoch=0,
            model_type="fixture",
            model={},
            model_config={
                "prediction_mode": "current_frame",
                "latent_mode": "deterministic",
                "sensor_code_dims": {"source": 2},
            },
            provenance={"implementation": {"configuration.py": signature}},
        ),
        run / "last.pt",
    )
    registry = SimpleNamespace(
        heldout_source="heldout",
        excluded_sources=set(),
        sources={"source", "heldout"},
        provenance={"native": "fixture"},
    )
    monkeypatch.setattr(management, "build_sources", lambda *args: registry)

    def dataset(registry, split, sources=None, allow_heldout=False):
        assert sources == ["heldout"] and allow_heldout
        return SyntheticDataset(split)

    monkeypatch.setattr(management, "build_dataset", dataset)
    monkeypatch.setattr(models, "model_from_state", lambda state: SyntheticEncoder())
    monkeypatch.setattr(models, "VJEPABackbone", SyntheticBackbone)
    monkeypatch.setattr(
        backbone_module,
        "spatial_video_features",
        lambda backbone, batch, device: batch.visual.rgb[:, 0, 0, 0, 0, 0],
    )
    experiment = {
        "model": {"backbone": {"repo": "unused", "checkpoint": "unused"}},
        "datasets": {},
        "training": {"device": "cpu", "workers": 0},
        "transfer": {**config(), "source": "heldout", "output": "processing/evaluations/fixture"},
    }
    result = evaluate_experiment(experiment, project_root=root, run=run)
    assert result["results"]["1"]["head"] == [4, 8, 2]
    registration = Path(result["output"]) / "epoch_001/viewer_registration.json"
    assert json.loads(registration.read_text())["epoch"] == 1
    assert (run / "last.pt").is_file()
    assert not any(p.name == "video_features" for p in root.rglob("*"))


def test_classifier_feature_cache_never_stores_pressure_images(tmp_path, monkeypatch):
    import tactilelatent.models.backbone as backbone_module
    from tactilelatent.evaluation.transfer import extract_features

    class PressureDataset(SyntheticDataset):
        def __getitem__(self, index):
            sample = super().__getitem__(index)
            values = torch.ones(1, 1, 224, 224)
            sample.target = SensorTarget(
                values, torch.ones_like(values, dtype=torch.bool), "pressure_grid"
            )
            return sample

    data = PressureDataset("val")
    registry = SimpleNamespace(provenance={"source": "pressure-fixture"})
    monkeypatch.setattr(
        backbone_module,
        "spatial_video_features",
        lambda backbone, batch, device: batch.visual.rgb[:, 0, 0, 0, 0, 0],
    )
    cached = extract_features(
        SyntheticEncoder(),
        SyntheticBackbone(),
        registry,
        data,
        list(range(len(data))),
        tmp_path / "features.pt",
        "checkpoint",
        "pressure-fixture",
        "val",
        torch.device("cpu"),
        with_targets=False,
    )
    assert cached["target"].shape == (len(data), 1)
    assert cached["values"].shape == (len(data), 4)
    assert cached["signature"]["with_targets"] is False
    assert (tmp_path / "features.pt").stat().st_size < 20000


def test_scientific_signature_excludes_ui_and_pins_model_code(tmp_path):
    from tactilelatent.dataset_management.artifacts import (
        scientific_implementation_path,
        scientific_implementation_signature,
    )

    def put(relative, text):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    put("configuration.py", "VALUE=1\n")
    put("contracts.py", "VALUE=2\n")
    put("models/attention.py", "VALUE=3\n")
    put("datasets/sources/example.py", "VALUE=4\n")
    put("training/engine.py", "VALUE=5\n")
    put("visualization/server.py", "VALUE=6\n")
    put("evaluation/transfer.py", "VALUE=7\n")
    before = scientific_implementation_signature(tmp_path)
    put("visualization/server.py", "VALUE=100\n")
    put("evaluation/transfer.py", "VALUE=200\n")
    assert scientific_implementation_signature(tmp_path) == before
    put("models/attention.py", "VALUE=300\n")
    after = scientific_implementation_signature(tmp_path)
    assert after["models/attention.py"] != before["models/attention.py"]
    assert not scientific_implementation_path("../models/attention.py")
    assert not scientific_implementation_path("/models/attention.py")
    assert not scientific_implementation_path("visualization/server.py")
    assert not scientific_implementation_path("models/assets.json")
