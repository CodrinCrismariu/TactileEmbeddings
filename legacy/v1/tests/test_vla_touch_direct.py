import pytest
import torch

from tactilelatent.joint_all_config import ALL_TARGETS
from tactilelatent.model import MODEL_ID, REVISION
from tactilelatent.train_joint import resolve_train_sources
from tactilelatent.evaluate_direct_vla_touch import load_direct_model


def test_explicit_vla_training_source_keeps_prior_holdout_behavior():
    sources = set(ALL_TARGETS)
    cfg = {"dataset_family": "joint_all", "train_sources": ["ftp_vla_touch"],
           "heldout_source": None}
    assert resolve_train_sources(cfg, ALL_TARGETS, sources) == ["ftp_vla_touch"]
    prior = {"dataset_family": "joint_all", "heldout_source": "ftp_vla_touch"}
    assert set(resolve_train_sources(prior, ALL_TARGETS, sources)) == sources - {"ftp_vla_touch"}
    with pytest.raises(ValueError):
        resolve_train_sources({**cfg, "heldout_source": "egotouch"}, ALL_TARGETS, sources)
    with pytest.raises(ValueError):
        resolve_train_sources({**cfg, "train_sources": ["ftp_vla_touch", "ftp_vla_touch"]},
                              ALL_TARGETS, sources)


def test_direct_evaluator_refuses_incomplete_checkpoint_before_loading_dino(tmp_path):
    path = tmp_path / "partial.pt"
    torch.save({"backbone": MODEL_ID, "backbone_revision": REVISION,
                "config": {"train_sources": ["ftp_vla_touch"],
                           "heldout_source": None, "latent_objective": "vae",
                           "epochs": 50}, "epoch": 20}, path)
    with pytest.raises(ValueError, match="not completed"):
        load_direct_model(path, torch.device("cpu"))
