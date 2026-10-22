import json

import torch

from tactilelatent.joint_all_config import ALL_TARGETS, EXTRA_TARGETS
from tactilelatent.joint_data import JointClipDataset
from tactilelatent.joint_model import JointTactileVAE


def test_twelve_source_manifest_and_complete_holdout():
    path = "processing/joint_all/manifest.json"
    manifest = json.load(open(path))
    assert {row["source"] for row in manifest["episodes"]} == set(ALL_TARGETS)
    holdout = "ftp_vla_touch"
    training = set(ALL_TARGETS) - {holdout}
    for split in ("train", "val"):
        dataset = JointClipDataset(path, split=split, sources=training)
        assert set(dataset.indices_by_source) == training
        assert holdout not in dataset.indices_by_source
    heldout = JointClipDataset(path, split=None, sources=[holdout])
    assert {row["source"] for row in heldout.episodes} == {holdout}
    assert len(heldout.episodes) == 380


def test_extra_source_native_targets_decode_and_model_excludes_holdout():
    path = "processing/joint_all/manifest.json"
    for source, (_, expected, _) in EXTRA_TARGETS.items():
        item = JointClipDataset(path, split="train", sources=[source])[0]
        assert list(item["target"].shape) == [8, *expected]
        assert bool(torch.isfinite(item["target"]).all())
        assert bool(item["view_mask"].any(dim=-1).all())
    training_targets = {name: target for name, target in ALL_TARGETS.items()
                        if name != "ftp_vla_touch"}
    model = JointTactileVAE(input_dim=12, width=16, heads=2,
                            temporal_layers=1, cross_view_layers=1,
                            latent_dim=7, dropout=0, targets=training_targets)
    assert "ftp_vla_touch" not in model.decoders
    assert set(model.decoders) == set(training_targets) - {"egotouch"}
