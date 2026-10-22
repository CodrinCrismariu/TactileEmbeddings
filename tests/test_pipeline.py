import json

import h5py
import numpy as np
import pytest
import torch

from tactilelatent.data import EgoTouch, resolve_splits
from tactilelatent.evaluate_uncertainty import _finish, _new_stats, _update
from tactilelatent.losses import objective
from tactilelatent.model import EgoTouchLatentModel, load_vae_weights
from tactilelatent.train import features, kl_weight, make_scheduler, train_one_epoch


def small_model():
    torch.manual_seed(7)
    return EgoTouchLatentModel(input_dim=12, width=16, heads=2, temporal_layers=1,
                               cross_view_layers=1, latent_dim=7, dropout=0)


def test_image_only_latent_shapes_and_masked_views():
    model = small_model().eval()
    x = torch.randn(2, 3, 3, 4, 12)
    mask = torch.ones(2, 3, 3, dtype=torch.bool)
    mask[:, :, 2] = False
    first = model(x, mask)
    assert set(first) == {"pressure", "mu", "logvar"}
    assert first["mu"].shape == first["logvar"].shape == (2, 3, 7)
    assert first["pressure"].shape == (2, 3, 2, 21, 21)
    changed = x.clone()
    changed[:, :, 2] = 1e4
    torch.testing.assert_close(first["pressure"], model(changed, mask)["pressure"])


def test_stochastic_training_and_deterministic_mean_inference():
    model = small_model().eval()
    x, mask = torch.randn(1, 2, 3, 4, 12), torch.ones(1, 2, 3, dtype=torch.bool)
    a, b = model(x, mask), model(x, mask)
    torch.testing.assert_close(a["pressure"], b["pressure"])
    assert not torch.allclose(model(x, mask, sample_latent=True)["pressure"],
                              model(x, mask, sample_latent=True)["pressure"])


def test_loss_has_gradients_and_standard_normal_kl():
    model = small_model()
    x = torch.randn(2, 2, 3, 4, 12)
    mask = torch.ones(2, 2, 3, dtype=torch.bool)
    y = torch.rand(2, 2, 2, 21, 21)
    valid = torch.ones_like(y, dtype=torch.bool)
    y[:, :, :, 0] = float("nan")
    valid[:, :, :, 0] = False
    out = model(x, mask)
    loss = objective(out, y, valid, beta=.01)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.prior_mu.weight.grad is not None
    assert model.prior_logvar.weight.grad is not None
    assert torch.isfinite(model.prior_mu.weight.grad).all()


def test_old_vae_checkpoint_ignores_only_inactive_weights():
    model = small_model().eval()
    weights = model.state_dict()
    legacy = {**weights, "posterior_mu.weight": torch.zeros(1, 1)}
    restored = small_model().eval()
    assert load_vae_weights(restored, legacy) == 1
    x, mask = torch.randn(1, 2, 3, 4, 12), torch.ones(1, 2, 3, dtype=torch.bool)
    torch.testing.assert_close(model(x, mask)["pressure"], restored(x, mask)["pressure"])
    with pytest.raises(ValueError, match="Unsupported"):
        load_vae_weights(restored, {**weights, "unknown.weight": torch.zeros(1)})


def test_hdf5_and_missing_files(tmp_path):
    path = tmp_path / "Home" / "task" / "episode.hdf5"
    path.parent.mkdir(parents=True)
    (tmp_path / "split.json").write_text(json.dumps({"train": ["/upstream/Home/task/episode.hdf5"]}))
    with pytest.raises(FileNotFoundError):
        EgoTouch(tmp_path)
    with h5py.File(path, "w") as f:
        f["timestamps"] = np.arange(20) / 30
        f["images/chest_color"] = np.zeros((20, 24, 32, 3), dtype=np.uint8)
        for side in ("left", "right"):
            pressure = np.zeros((20, 21, 21), np.float32)
            pressure[:, 0] = np.nan
            f[f"pressure/{side}_pressure_grid"] = pressure
    item = EgoTouch(tmp_path, size=32)[0]
    assert item["rgb"].shape == (8, 3, 3, 32, 32)
    assert not item["view_mask"][:, 1:].any()
    assert not item["target_mask"][:, 0].any()
    (tmp_path / "split.json").write_text(json.dumps({"train": [str(path)], "val": [str(path)]}))
    with pytest.raises(ValueError, match="Duplicate"):
        resolve_splits(tmp_path)


def test_uncertainty_metrics_reward_a_covering_distribution():
    target = torch.tensor([[[[[.25]]]]])
    mask = torch.ones_like(target, dtype=torch.bool)
    samples = torch.tensor([.1, .2, .3, .4]).reshape(4, 1, 1, 1, 1, 1)
    stats = _new_stats(torch.device("cpu"))
    _update(stats, samples, target, mask, torch.zeros(1, 1, 2), .05)
    result = _finish(stats, 4)
    assert result["crps"] >= 0
    assert result["mean_pressure_std"] > 0
    assert result["central_interval_calibration"]["0.5"]["empirical_coverage"] == 1


def test_training_helpers_and_one_optimizer_update():
    class TinyBackbone(torch.nn.Module):
        def forward(self, rgb):
            color_means = rgb.mean(dim=(-1, -2))
            return color_means.repeat(1, 4).unsqueeze(1).expand(-1, 4, -1)

    torch.manual_seed(11)
    model = small_model()
    backbone = TinyBackbone()
    item = {
        "rgb": torch.rand(2, 3, 3, 16, 16),
        "view_mask": torch.ones(2, 3, dtype=torch.bool),
        "target": torch.rand(2, 2, 21, 21),
        "target_mask": torch.ones(2, 2, 21, 21, dtype=torch.bool),
    }
    batch = {key: value.unsqueeze(0) for key, value in item.items()}
    assert features(backbone, batch, torch.device("cpu")).shape == (1, 2, 3, 4, 12)

    cfg = {
        "accumulation": 2, "epochs": 1, "warmup_epochs": 0,
        "learning_rate": .01, "min_learning_rate": .001,
        "beta_max": .001, "beta_warmup_epochs": 2,
        "wrist_dropout": 0., "pressure_threshold": .1,
        "contact_weight": 3., "tv_weight": .01, "log_every": 1,
    }
    assert kl_weight(cfg, 0) == 0
    assert kl_weight(cfg, 1) == pytest.approx(.0005)
    assert kl_weight(cfg, 3) == pytest.approx(.001)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])
    scheduler = make_scheduler(optimizer, cfg, batches_per_epoch=2)
    before = model.prior_mu.weight.detach().clone()
    average_loss = train_one_epoch(
        model, backbone, [batch, batch], optimizer, scheduler, cfg,
        torch.device("cpu"), epoch=0,
    )
    assert np.isfinite(average_loss)
    assert not torch.equal(before, model.prior_mu.weight)
