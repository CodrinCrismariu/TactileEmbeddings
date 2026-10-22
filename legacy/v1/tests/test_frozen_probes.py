import torch
from torch import nn

import tactilelatent.frozen_probe as frozen_probe
from tactilelatent.evaluate_dataset_classifier import (SOURCES, class_table,
                                                       probability_set, update_metrics,
                                                       metric_state)
from tactilelatent.evaluate_holdout_decoder import (finish_group, new_group,
                                                     quantile_sorted, update_group)
from tactilelatent.frozen_probe import (episode_partition, fit_head, head,
                                         latent_draw, select_indices)
from tactilelatent.joint_model import JointTactileVAE
from tactilelatent.model import MODEL_ID, REVISION


def test_matching_head_shapes_and_fixed_latent_sampling():
    decoder = head(32, 3072)
    classifier = head(32, len(SOURCES), classification=True)
    assert decoder[0].weight.shape == classifier[0].weight.shape == (512, 32)
    assert decoder(torch.zeros(4, 32)).shape == (4, 3072)
    assert classifier(torch.zeros(4, 32)).shape == (4, 11)
    mu, logvar = torch.zeros(2, 3, 32), torch.zeros(2, 3, 32)
    a = latent_draw(mu, logvar, torch.Generator().manual_seed(9))
    b = latent_draw(mu, logvar, torch.Generator().manual_seed(9))
    torch.testing.assert_close(a, b)
    assert not torch.equal(a, mu)


def test_episode_partition_and_cap_do_not_split_episode():
    class Dataset:
        episodes = [{"episode_id": f"episode-{i}"} for i in range(100)]
        windows = [(episode, frame) for episode in range(100) for frame in range(3)]
    dataset = Dataset()
    fit = select_indices(dataset, "fit", cap=20)
    tune = select_indices(dataset, "tune", cap=20)
    assert set(fit).isdisjoint(tune)
    assert all(episode_partition(dataset.episodes[dataset.windows[i][0]]["episode_id"]) == "fit"
               for i in fit)
    assert all(episode_partition(dataset.episodes[dataset.windows[i][0]]["episode_id"]) == "tune"
               for i in tune)


def test_frozen_encoder_stays_unchanged_during_probe_fit():
    encoder = JointTactileVAE(input_dim=12, width=16, heads=2,
                              temporal_layers=1, cross_view_layers=1,
                              latent_dim=7, dropout=0, targets={"egotouch": ("pressure_grid", [2, 21, 21], 1.)})
    encoder.requires_grad_(False).eval()
    frozen = {key: value.clone() for key, value in encoder.state_dict().items()}
    probe = head(7, 3, classification=True)
    table = {"mu": torch.randn(16, 7), "logvar": torch.zeros(16, 7),
             "y": torch.randint(0, 3, (16,))}
    fit_head(probe, table, None, torch.nn.functional.cross_entropy,
             fixed_epochs=1, batch_size=8)
    for key, value in encoder.state_dict().items():
        torch.testing.assert_close(value, frozen[key])
    assert all(parameter.grad is None for parameter in encoder.parameters())


def test_checkpoint_loader_freezes_both_encoders(tmp_path, monkeypatch):
    class TinyBackbone(nn.Module):
        dim = 12

        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))

    class TinyJoint(nn.Module):
        def __init__(self, input_dim, targets, **kwargs):
            super().__init__()
            self.visual = nn.Linear(input_dim, kwargs["latent_dim"])

    monkeypatch.setattr(frozen_probe, "DinoBackbone", TinyBackbone)
    monkeypatch.setattr(frozen_probe, "JointTactileVAE", TinyJoint)
    cfg = {"latent_objective": "vae", "heldout_source": "ftp_vla_touch",
           "width": 16, "heads": 2, "temporal_layers": 1,
           "cross_view_layers": 1, "latent_dim": 7, "dropout": .1,
           "stochastic": True}
    path = tmp_path / "last.pt"
    torch.save({"backbone": MODEL_ID, "backbone_revision": REVISION,
                "config": cfg, "model": TinyJoint(12, {}, **cfg).state_dict()}, path)
    backbone, encoder, _ = frozen_probe.load_encoder(path, torch.device("cpu"))
    assert not backbone.training and not encoder.training
    assert all(not parameter.requires_grad for parameter in backbone.parameters())
    assert all(not parameter.requires_grad for parameter in encoder.parameters())


def test_interval_coverage_and_classifier_probability_sets():
    ordered = torch.tensor([.1, .2, .3, .4]).reshape(4, 1)
    assert quantile_sorted(ordered, .5).item() == torch.tensor(.25).item()
    stats = new_group()
    y = torch.tensor([.25, .9])
    mean = torch.tensor([.25, .25])
    std = torch.tensor([.1, .1])
    intervals = {level: (torch.tensor([.1, .1]), torch.tensor([.4, .4]))
                 for level in (.5, .8, .9, .95)}
    update_group(stats, torch.tensor([True, True]), y, mean, std,
                 torch.tensor([.1, .5]), intervals)
    result = finish_group(stats)
    assert result["intervals"]["0.9"]["empirical_coverage"] == .5
    assert result["intervals"]["0.9"]["above_fraction"] == .5
    probabilities = torch.tensor([[.8, .15, .05], [.4, .35, .25]])
    truth = torch.tensor([0, 2])
    covered, size, mass = probability_set(probabilities, truth, .8)
    assert covered.tolist() == [True, True]
    assert size.tolist() == [1, 3]
    torch.testing.assert_close(mass, torch.tensor([.8, 1.]))
    state = metric_state()
    update_metrics(state, probabilities, truth)
    assert state["correct"] == 1


def test_balanced_classifier_table():
    parts = [{"mu": torch.zeros(2, 8, 32), "logvar": torch.zeros(2, 8, 32)}
             for _ in SOURCES]
    table = class_table(parts, torch.device("cpu"))
    counts = torch.bincount(table["y"], minlength=len(SOURCES))
    assert counts.tolist() == [16] * len(SOURCES)
