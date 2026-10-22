import json
from pathlib import Path

import pytest
import torch
from torch import nn

import tactilelatent.frozen_probe as frozen_probe
from tactilelatent.evaluate_direct_vla_touch import load_direct_model
from tactilelatent.evaluate_holdout_decoder import evaluate
from tactilelatent.frozen_probe import episode_partition, head
from tactilelatent.joint_all_config import ALL_TARGETS
from tactilelatent.joint_data import JointClipDataset, SourceBatchSampler
from tactilelatent.model import MODEL_ID, REVISION
from tactilelatent.train_joint import resolve_train_sources


ROOT = Path(__file__).resolve().parents[1]


def test_new_configs_are_fresh_and_sensor_disjoint():
    sources = set(ALL_TARGETS)
    for suffix in ("fresh_8k_50ep", "smoke"):
        cfg = json.loads((ROOT / f"configs/joint_all_holdout_motiontrans_{suffix}.json").read_text())
        selected = resolve_train_sources(cfg, ALL_TARGETS, sources)
        assert set(selected) == sources - {"ftp_motiontrans"}
        assert "ftp_vla_touch" in selected
        assert cfg["init_checkpoint"] is cfg["init_joint_checkpoint"] is None
    cfg = json.loads((ROOT / "configs/motiontrans_direct_vae32_50ep.json").read_text())
    assert resolve_train_sources(cfg, ALL_TARGETS, sources) == ["ftp_motiontrans"]
    assert cfg["epochs"] == 50 and cfg["latent_dim"] == 32
    assert cfg["init_checkpoint"] is cfg["init_joint_checkpoint"] is None


def test_motiontrans_decoder_shapes_sampling_and_coverage(tmp_path):
    network = head(32, 6, dropout=0)
    assert network[0].weight.shape == (512, 32)
    assert network[3].weight.shape == (6, 512)
    # Constant output: a zero-width interval must include exactly matching
    # targets, and exclude readings outside the interval, for every channel.
    for parameter in network.parameters():
        nn.init.zeros_(parameter)
    cache = {"mu": torch.zeros(2, 8, 32), "logvar": torch.zeros(2, 8, 32),
             "target": torch.full((2, 8, 6, 1, 1), .5),
             "valid": torch.ones(2, 8, 6, 1, 1, dtype=torch.bool)}
    cache["target"][1] = .7
    a = evaluate(network, cache, cache, torch.device("cpu"), 64, 45, tmp_path, (6, 1, 1))
    b = evaluate(network, cache, cache, torch.device("cpu"), 64, 45, tmp_path, (6, 1, 1))
    assert a == b
    assert a["sample_count"] == 64
    assert a["deterministic_mae"] == pytest.approx(.1)
    assert len(a["per_channel"]) == 6
    for values in a["per_channel"].values():
        assert values["intervals"]["0.9"]["empirical_coverage"] == .5
        assert values["intervals"]["0.9"]["mean_width"] == 0
    assert (tmp_path / "examples/worst_error.png").is_file()


def test_motiontrans_loader_excludes_sensor_and_freezes_weights(tmp_path, monkeypatch):
    captured = {}

    class TinyBackbone(nn.Module):
        dim = 12

        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))

    class TinyJoint(nn.Module):
        def __init__(self, input_dim, targets, **kwargs):
            super().__init__()
            captured["sources"] = set(targets)
            self.visual = nn.Linear(input_dim, kwargs["latent_dim"])

    cfg = {"latent_objective": "vae", "heldout_source": "ftp_motiontrans",
           "width": 16, "heads": 2, "temporal_layers": 1, "cross_view_layers": 1,
           "latent_dim": 32, "dropout": .1, "stochastic": True}
    checkpoint = tmp_path / "last.pt"
    torch.save({"backbone": MODEL_ID, "backbone_revision": REVISION,
                "config": cfg, "model": TinyJoint(12, {}, **cfg).state_dict()}, checkpoint)
    monkeypatch.setattr(frozen_probe, "DinoBackbone", TinyBackbone)
    monkeypatch.setattr(frozen_probe, "JointTactileVAE", TinyJoint)
    backbone, model, _ = frozen_probe.load_encoder(checkpoint, torch.device("cpu"), "ftp_motiontrans")
    assert captured["sources"] == set(ALL_TARGETS) - {"ftp_motiontrans"}
    assert not backbone.training and not model.training
    assert all(not p.requires_grad for module in (backbone, model) for p in module.parameters())
    with pytest.raises(ValueError, match="held out"):
        frozen_probe.load_encoder(checkpoint, torch.device("cpu"))


def test_direct_motiontrans_refuses_incomplete_checkpoint(tmp_path):
    checkpoint = tmp_path / "partial.pt"
    torch.save({"backbone": MODEL_ID, "backbone_revision": REVISION,
                "config": {"train_sources": ["ftp_motiontrans"], "heldout_source": None,
                           "latent_objective": "vae", "epochs": 50}, "epoch": 1}, checkpoint)
    with pytest.raises(ValueError, match="not completed"):
        load_direct_model(checkpoint, torch.device("cpu"), "ftp_motiontrans")


@pytest.mark.skipif(not (ROOT / "processing/joint_all/manifest.json").exists(), reason="Local dataset manifest absent")
def test_manifest_motiontrans_splits_and_joint_schedule():
    manifest = ROOT / "processing/joint_all/manifest.json"
    data = {split: JointClipDataset(manifest, split=split, sources=["ftp_motiontrans"])
            for split in ("train", "val")}
    assert len(data["train"]) == 18613 and len(data["val"]) == 1972
    train_ids = {row["episode_id"] for row in data["train"].episodes}
    val_ids = {row["episode_id"] for row in data["val"].episodes}
    assert train_ids.isdisjoint(val_ids)
    fit = {name for name in train_ids if episode_partition(name) == "fit"}
    tune = {name for name in train_ids if episode_partition(name) == "tune"}
    assert fit and tune and fit.isdisjoint(tune)
    joint = JointClipDataset(manifest, split="train", sources=sorted(set(ALL_TARGETS) - {"ftp_motiontrans"}))
    assert "ftp_motiontrans" not in joint.indices_by_source
    sampler = SourceBatchSampler(joint, 2, max_windows_per_source=8000, sampling_mode="cycle")
    counts = {source: 0 for source in joint.indices_by_source}
    for batch in sampler:
        source = joint.episodes[joint.windows[batch[0]][0]]["source"]
        assert all(joint.episodes[joint.windows[i][0]]["source"] == source for i in batch)
        counts[source] += len(batch)
    assert counts == {source: min(8000, len(indices)) for source, indices in joint.indices_by_source.items()}
