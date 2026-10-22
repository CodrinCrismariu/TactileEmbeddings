import numpy as np
import pytest
import torch

from tactilelatent.fgw_mixer import (MixerConfig, _solve, check_plan, distances,
    fit_mixer, pull_to_source, push_to_support, state_loss)
from tactilelatent.fgw_data import BankSampler, CodeOnlyDataset, sample_banks
from tactilelatent.fgw_objective import visual_extra_loss
from tactilelatent.evaluate_holdout_decoder import training_fraction, episode_diagnostics


def test_mass_normalized_push_pull_constants_and_unequal_banks():
    plan = np.outer(np.ones(3) / 3, np.ones(7) / 7)
    x = np.full((7, 4), 8.)
    support, valid = push_to_support(plan, x)
    np.testing.assert_allclose(support, 8.)
    recovered, matched = pull_to_source(plan, support, x)
    np.testing.assert_allclose(recovered, x)
    assert valid.all() and matched.all()
    result = fit_mixer({"a": x, "b": np.full((5, 4), 8.)},
        MixerConfig(bank_size=7, barycenter_size=3, rounds=2))
    for target in result["targets"].values():
        torch.testing.assert_close(target, torch.full_like(target, 8.))


def test_partial_fgw_supports_16d_codes_and_finite_training_gradients():
    rng = np.random.default_rng(45)
    banks = {source: rng.normal(size=(16, 16)) for source in ("a", "b", "c")}
    cfg = MixerConfig(variant="partial_stochastic_fgw", bank_size=16, barycenter_size=16, rounds=2)
    fit = fit_mixer(banks, cfg, seed=45)
    assert fit["barycenter"].shape == (16, 16)
    assert fit["diagnostics"]["solver_failures"] == 0
    for source in banks:
        assert fit["targets"][source].shape == (16, 16)
        value = torch.from_numpy(banks[source]).float().requires_grad_(True)
        loss, _ = state_loss(value, fit["targets"][source], fit["valid"][source],
                             fit["banks"][source], torch.arange(16), fit["initial_variance"][source], cfg, 0)
        loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(value.grad).all() and value.grad.abs().sum() > 0


def test_permutation_and_rotation_recover_relational_geometry():
    rng = np.random.default_rng(22)
    x = rng.normal(size=(8, 3))
    permutation = rng.permutation(len(x))
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    y = x[permutation] @ rotation
    expected = np.eye(len(x))[:, permutation] / len(x)
    plan = _solve(x, y, MixerConfig(variant="gw", max_iter=500), 1., initial=expected)
    # A known initialization verifies the correct solution, not a guarantee
    # that nonconvex GW recovers every unknown rotation from arbitrary starts.
    mapping = plan.argmax(1)
    np.testing.assert_allclose(distances(x), distances(y)[mapping][:, mapping], atol=1e-7)
    unrotated = _solve(x, x[permutation], MixerConfig(variant="fgw"), 1.)
    np.testing.assert_allclose(x, x[permutation][unrotated.argmax(1)], atol=1e-7)


def test_symmetric_geometry_has_multiple_equally_valid_correspondences():
    x = np.array([[1., 0], [0, 1.], [-1, 0], [0, -1.]])
    d = distances(x)
    for permutation in (np.arange(4), np.array([1, 2, 3, 0])):
        np.testing.assert_allclose(d, d[permutation][:, permutation])
    # Pure GW cannot identify which of these rotations is semantically right.
    assert not np.array_equal(x, x[[1, 2, 3, 0]])


def test_partial_transport_excludes_outlier_and_unmatched_loss():
    x = np.array([[0., 0], [1., 0], [0., 2.]])
    y = np.vstack([x, [100., 100.]])
    cfg = MixerConfig(variant="partial_stochastic_fgw", transported_mass=.75)
    plan = _solve(x, y, cfg, 1.)
    check_plan(plan, np.ones(3) / 3, np.ones(4) / 4, .75, True)
    assert plan[:, -1].sum() < 1e-10
    target, valid = pull_to_source(plan, x, y)
    assert not valid[-1]
    np.testing.assert_array_equal(target[-1], y[-1])


def test_heldout_and_solver_failures_fail_closed(monkeypatch):
    with pytest.raises(ValueError, match="excluding MotionTrans"):
        fit_mixer({"ftp_motiontrans": np.ones((3, 2)), "a": np.ones((3, 2))}, MixerConfig())
    import ot
    monkeypatch.setattr(ot.gromov, "fused_gromov_wasserstein",
                        lambda *a, **k: (np.zeros((3, 3)), {}))
    with pytest.raises(FloatingPointError, match="mass"):
        fit_mixer({"a": np.ones((3, 2)), "b": np.ones((3, 2))},
                  MixerConfig(barycenter_size=3))


def test_detached_targets_spread_gradient_and_repeatable_masks():
    rng = np.random.default_rng(2)
    banks = {"a": rng.normal(size=(11, 4)), "b": rng.normal(size=(9, 4))}
    cfg = MixerConfig(barycenter_size=7, rounds=2)
    first, second = fit_mixer(banks, cfg, seed=3), fit_mixer(banks, cfg, seed=3)
    for source in banks:
        torch.testing.assert_close(first["targets"][source], second["targets"][source])
        assert not first["targets"][source].requires_grad
    value = torch.ones(2, 4, requires_grad=True)
    target = torch.zeros(2, 4, requires_grad=True)
    loss, _ = state_loss(value, target, torch.tensor([True, False]), torch.zeros(5, 4),
                         torch.tensor([0, 1]), 1., cfg, 0)
    loss.backward()
    assert target.grad is None and value.grad[0].abs().sum() > 0
    assert value.grad[1].abs().sum() > 0  # Unmatched state participates in spread only.


class FixtureDataset:
    def __init__(self):
        self.episodes = [{"source": s, "episode_id": f"{s}{i}"} for s in ("a", "b") for i in range(4)]
        self.windows = [(e, start) for e in range(8) for start in range(5)]
        self.indices_by_source = {s: [i for i, (e, _) in enumerate(self.windows)
                                     if self.episodes[e]["source"] == s] for s in ("a", "b")}


def test_episode_balancing_and_identical_source_sampling(monkeypatch):
    import tactilelatent.fgw_data as module
    data = FixtureDataset()
    monkeypatch.setattr(module, "code_point", lambda d, index: None if index == 0 else (7, torch.zeros(4)))
    banks = sample_banks(data, 8, 45, 0)
    assert 0 not in banks["a"]
    for indices in banks.values():
        counts = np.bincount([data.windows[i][0] % 4 for i in indices], minlength=4)
        np.testing.assert_array_equal(counts, 2)
    sampler = BankSampler(data, 2, 45, 20, .25); sampler.set_epoch(0, banks)
    first, second = list(sampler), list(sampler)
    assert first == second
    assert sum(anchor for batch in first for _, anchor in batch) == 10
    for batch in first:
        assert len({data.episodes[data.windows[i][0]]["source"] for i, _ in batch}) == 1


def test_code_only_loader_never_calls_scene_rgb(monkeypatch):
    import tactilelatent.alignment_data as module
    monkeypatch.setattr(module, "scene_rgb", lambda *a: pytest.fail("Decoded scene image"))
    class Cache:
        def read(self, episode, frames):
            return torch.ones(len(frames), 3), torch.tensor([False] + [True] * (len(frames) - 1))
    dataset = object.__new__(CodeOnlyDataset)
    dataset.windows = [(0, 0)]; dataset.episodes = [{"source": "egotouch", "episode_id": "train"}]
    dataset.span, dataset.stride, dataset.caches = 15, 2, {"egotouch": Cache()}
    item = dataset[0]
    assert item["frame_indices"].tolist() == list(range(0, 15, 2))
    assert not item["valid"][0] and "rgb" not in item


def test_fraction_nested_episode_disjoint_and_bootstrap():
    episodes = [f"episode{i}" for i in range(100)]
    cache = {"episode_id": [e for e in episodes for _ in range(2)], "mu": torch.randn(200, 3, 4),
             "signature": {"a": 1}, "indices": list(range(200))}
    small = training_fraction(cache, .1, 45); medium = training_fraction(cache, .25, 45)
    assert set(small["episode_id"]) < set(medium["episode_id"]) < set(episodes)
    assert len(small["mu"]) == len(small["episode_id"]) == len(small["indices"])
    y = torch.randn(10, 8, 6)
    stats = episode_diagnostics(y + .1, y, [f"e{i // 2}" for i in range(10)], 45, bootstrap=100)
    np.testing.assert_allclose(stats["episode_bootstrap_mae_ci95"], .1, atol=1e-7)
    np.testing.assert_allclose(stats["within_episode_channel_correlations"], 1., atol=1e-6)


def test_sam_replays_same_transport_payload_without_target_gradients():
    from tactilelatent.alignment_model import JointSensorAlignment
    from tactilelatent.train_alignment import alignment_loss, sam_replay
    from tactilelatent.sam import SAMStep, rng_snapshot
    model = JointSensorAlignment({"egotouch": 3}, {},
        {"egotouch": {"mean": [0.] * 3, "std": [1.] * 3}}, input_dim=8, width=16,
        heads=4, temporal_layers=1, cross_view_layers=1, latent_dim=4,
        adapter_hidden_dims=[8], dropout=.1)
    features = torch.randn(2, 3, 3, 4, 8); mask = torch.ones(2, 3, 3, dtype=torch.bool)
    batch = {"code_target": torch.randn(2, 3, 3), "code_valid": torch.ones(2, 3, dtype=torch.bool)}
    payload = {"selected": torch.arange(2), "frames": torch.tensor([2, 2]),
               "positions": torch.tensor([0, 1]), "bank": torch.randn(4, 4),
               "target": torch.randn(2, 4), "valid": torch.ones(2, dtype=torch.bool), "initial_variance": 1.}
    cfg = {"fgw": {"placement": "visual"}}
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4); sam = SAMStep(optimizer)
    before = rng_snapshot(); output = model(features, mask, "egotouch")
    loss, _, _ = alignment_loss(output, batch, .001)
    extra, _ = visual_extra_loss(output, payload, cfg, 0); (loss + extra).backward()
    after = rng_snapshot(); target_before = payload["target"].clone()
    sam_replay(model, optimizer, sam, [{"features": features, "mask": mask, "source": "egotouch",
        "batch": batch, "rng": before, "beta": .001, "alignment_payload": payload,
        "config": cfg, "alignment_epoch": 0}], torch.device("cpu"))
    assert sam.steps == 1 and torch.equal(after["cpu"], torch.get_rng_state())
    torch.testing.assert_close(target_before, payload["target"])


def test_calibration_partitions_quantiles_and_joint_episode_coverage():
    from tactilelatent.evaluate_fgw_calibration import calibration_partitions, conformal_quantile, calibrated_metrics
    episodes = [f"e{i}" for i in range(100) for _ in range(2)]
    partitions = calibration_partitions(episodes, 45)
    sets = {k: {episodes[i] for i in ids} for k, ids in partitions.items()}
    assert len(sets["fit"]) == 70 and len(sets["tune"]) == len(sets["calibration"]) == 15
    assert not sets["fit"] & sets["tune"] and not sets["fit"] & sets["calibration"]
    assert conformal_quantile(np.arange(20), .9) == 18
    assert np.isinf(conformal_quantile([1., 2.], .95))
    point = torch.full((4, 8, 6), .5); truth = point + .1
    result = calibrated_metrics(point, torch.ones_like(point), truth, ["a", "a", "b", "b"], [1.] * 20)
    assert result["levels"]["0.9"]["joint_episode_coverage"] == 1.
