import json

import numpy as np
import pytest
import torch

from tactilelatent.tactile_autoencoder import TactileAutoencoder, HurdlePressureOutput, ClippedPressureOutput
from tactilelatent.pretrain_egotouch_autoencoder import (
    PressureFrames, frame_batches, main, reconstruction_loss, selection_score, autoencoder_loss,
)


def pressure_manifest(tmp_path):
    rows = []
    for split in ("train", "val", "test_unseen"):
        path = tmp_path / f"{split}.npz"
        pressure = np.zeros((10, 21, 21), dtype=np.float32)
        pressure[:, 6:12, 8:15] = .7
        pressure[:, 0, :] = np.nan
        np.savez(path, left_pressure_grid=pressure, right_pressure_grid=pressure * .5)
        rows.append({"source": "egotouch", "episode_id": split, "split": split,
                     "frames": 10, "target_path": str(path)})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"episodes": rows}))
    return manifest


@pytest.mark.parametrize("architecture,channels", [("mlp", [4]), ("conv", [4, 8]), ("conv", [4, 8, 8])])
def test_reconstruct_and_freeze_decoder(architecture, channels):
    model = TactileAutoencoder(architecture, 5, channels, [16])
    pressure = torch.rand(3, 2, 21, 21)
    valid = torch.ones_like(pressure, dtype=torch.bool)
    valid[..., 0, :] = False
    first = model.encode(pressure, valid)
    pressure[..., 0, :] = float("nan")
    torch.testing.assert_close(model.encode(pressure, valid), first)
    output = model(pressure, valid)
    assert output.shape == pressure.shape
    assert torch.isfinite(output).all()
    assert ((output >= 0) & (output <= 1)).all()
    # Frozen decoder must still pass gradients into a future RGB -> code adapter.
    model.decoder.requires_grad_(False)
    code = first.detach().requires_grad_()
    reconstruction_loss(model.decode(code), pressure, valid).backward()
    assert torch.isfinite(code.grad).all() and code.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.decoder.parameters())


def test_pressure_only_loader_sampling_and_split_guard(tmp_path):
    manifest = pressure_manifest(tmp_path)
    dataset = PressureFrames(manifest, "train", stride=2)
    assert len(dataset) == 5
    assert dataset[0]["target"].shape == (2, 21, 21)
    assert not dataset[0]["valid"][..., 0, :].any()
    assert torch.isfinite(dataset[0]["target"]).all()
    batches = list(frame_batches(dataset, 2, 3, 45, True))
    assert batches == list(frame_batches(dataset, 2, 3, 45, True))
    assert len({i for batch in batches for i in batch}) == 3
    assert sorted(i for batch in frame_batches(dataset, 2, 0, 45, True) for i in batch) == list(range(5))
    data = json.loads(manifest.read_text())
    data["episodes"][1]["target_path"] = data["episodes"][0]["target_path"]
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Duplicate"):
        PressureFrames(manifest, "train")


def test_masked_loss_and_contact_weight():
    pred = torch.tensor([.2, .4, .9], requires_grad=True)
    target = torch.tensor([0., .8, float("nan")])
    valid = torch.tensor([True, True, False])
    loss = reconstruction_loss(pred, target, valid)
    torch.testing.assert_close(loss, torch.tensor((.04 + .1 + 3 * (.16 + .2)) / 2))
    loss.backward()
    assert pred.grad[2] == 0


def test_training_exports_resume_and_refuses_overwrite(tmp_path):
    manifest = pressure_manifest(tmp_path)
    output = tmp_path / "run"
    args = ["--manifest", str(manifest), "--output", str(output), "--architecture", "mlp",
            "--hidden-dims", "16", "--latent-dim", "4", "--epochs", "2", "--workers", "0",
            "--device", "cpu", "--batch-size", "3", "--max-train-frames", "0", "--max-val-frames", "0"]
    main(args)
    best = torch.load(output / "best.pt", weights_only=False)
    model = TactileAutoencoder(**best["model_config"])
    model.load_state_dict(best["model"])
    restored = TactileAutoencoder(**best["model_config"])
    restored.decoder.load_state_dict(best["decoder"])
    code = torch.randn(2, 4)
    torch.testing.assert_close(restored.decode(code), model.decode(code))
    assert best["validation"]["frames"] == 5
    with pytest.raises(FileExistsError):
        main(args)
    main(args + ["--resume"])
    assert len((output / "metrics.jsonl").read_text().splitlines()) == 2
    with pytest.raises(ValueError, match="configuration"):
        main(args + ["--resume", "--latent-dim", "8"])


def test_tiny_model_can_learn_pressure_reconstruction():
    torch.manual_seed(45)
    model = TactileAutoencoder("mlp", 4, hidden_dims=[16])
    pressure = torch.zeros(4, 2, 21, 21)
    pressure[:, :, 5:10, 8:13] = .8
    valid = torch.ones_like(pressure, dtype=torch.bool)
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    initial = reconstruction_loss(model(pressure), pressure, valid).item()
    for _ in range(30):
        optimizer.zero_grad()
        loss = reconstruction_loss(model(pressure), pressure, valid)
        loss.backward()
        optimizer.step()
    assert reconstruction_loss(model(pressure), pressure, valid).item() < initial * .25


def test_interrupted_run_resumes_to_same_weights(tmp_path, monkeypatch):
    import tactilelatent.pretrain_egotouch_autoencoder as trainer

    manifest = pressure_manifest(tmp_path)
    common = ["--manifest", str(manifest), "--architecture", "mlp", "--hidden-dims", "8",
              "--latent-dim", "3", "--epochs", "2", "--workers", "0", "--device", "cpu",
              "--batch-size", "2", "--max-train-frames", "4", "--max-val-frames", "4"]
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    main(common + ["--output", str(full)])
    original = trainer.run_epoch
    calls = 0

    def interrupt_second_epoch(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:  # First epoch train + validation have completed and been saved.
            raise RuntimeError("Simulated interruption")
        return original(*args, **kwargs)

    monkeypatch.setattr(trainer, "run_epoch", interrupt_second_epoch)
    with pytest.raises(RuntimeError, match="Simulated"):
        main(common + ["--output", str(resumed)])
    monkeypatch.setattr(trainer, "run_epoch", original)
    main(common + ["--output", str(resumed), "--resume"])
    expected = torch.load(full / "last.pt", weights_only=False)
    actual = torch.load(resumed / "last.pt", weights_only=False)
    assert actual["epoch"] == expected["epoch"] == 1
    for name in expected["model"]:
        torch.testing.assert_close(actual["model"][name], expected["model"][name], rtol=0, atol=0)


def test_balanced_selection_and_custom_loss():
    metrics = {"mae": .01, "contact_mae": .04, "high_pressure_mae": .10}
    assert selection_score(metrics, "balanced_mae") == pytest.approx(.05)
    assert selection_score(metrics, "mae") == .01
    assert selection_score({**metrics, "high_pressure_mae": None}, "balanced_mae") == pytest.approx(.02)
    pred, target = torch.tensor([.2]), torch.tensor([.8])
    assert reconstruction_loss(pred, target, torch.tensor([True]), contact_weight=1, mae_weight=0).item() == pytest.approx(.36)


def test_extended_budget_preserves_optimizer_and_plateau_scheduler(tmp_path):
    manifest = pressure_manifest(tmp_path)
    common = ["--manifest", str(manifest), "--architecture", "mlp", "--hidden-dims", "8",
              "--sensor-code-dim", "3", "--workers", "0", "--device", "cpu", "--batch-size", "2",
              "--max-train-frames", "4", "--max-val-frames", "4", "--scheduler", "plateau",
              "--lr-patience", "0", "--selection-metric", "balanced_mae"]
    full, extended = tmp_path / "full", tmp_path / "extended"
    main(common + ["--epochs", "3", "--output", str(full)])
    main(common + ["--epochs", "1", "--output", str(extended)])
    main(common + ["--epochs", "3", "--output", str(extended), "--resume"])
    a = torch.load(full / "last.pt", weights_only=False)
    b = torch.load(extended / "last.pt", weights_only=False)
    assert a["scheduler"] == b["scheduler"]
    assert a["best_score"] == b["best_score"]
    for name in a["model"]:
        torch.testing.assert_close(a["model"][name], b["model"][name], rtol=0, atol=0)
    with pytest.raises(ValueError, match="cannot reduce"):
        main(common + ["--epochs", "2", "--output", str(extended), "--resume"])


def test_hurdle_exact_zeros_without_destroying_positive_values():
    head = HurdlePressureOutput(.5).eval()
    logits = torch.zeros(1, 4, 21, 21)
    logits[:, :2, :, :10] = -20  # Confident zero state; magnitude logits stay zero (.5).
    logits[:, :2, :, 10:] = 20
    output = head(logits)
    assert torch.equal(output[..., :10], torch.zeros_like(output[..., :10]))
    torch.testing.assert_close(output[..., 10:], torch.full_like(output[..., 10:], .5))
    head.train()
    assert (head(logits)[..., :10] > 0).all()  # Differentiable soft path remains available.


@pytest.mark.parametrize("architecture", ["conv", "mlp"])
def test_hurdle_decoder_gradients_and_export(architecture):
    model = TactileAutoencoder(architecture, 4, channels=[4, 8], hidden_dims=[8], output_mode="hurdle")
    pressure = torch.zeros(2, 2, 21, 21)
    pressure[..., 5:10, 5:10] = .8
    valid = torch.ones_like(pressure, dtype=torch.bool)
    valid[..., 0, :] = False
    parts = model.decode_components(model.encode(pressure, valid))
    cfg = {"contact_weight": 3., "loss_threshold": .1, "mae_weight": .5, "nonzero_loss_weight": 1.}
    loss = autoencoder_loss(parts, pressure, valid, cfg)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    # Once frozen/eval, soft decoding still sends gradients into future visual codes.
    model.zero_grad(set_to_none=True)
    model.eval().requires_grad_(False)
    code = torch.randn(2, 4, requires_grad=True)
    model.decode(code, hard=False).sum().backward()
    assert code.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.parameters())
    restored = TactileAutoencoder(**model.config).eval()
    restored.decoder.load_state_dict(model.decoder.state_dict())
    torch.testing.assert_close(restored.decoder(code), model.decode(code, hard=True))


def test_hurdle_learns_zero_and_positive_pressure_and_masks_invalid_cells():
    torch.manual_seed(3)
    model = TactileAutoencoder("mlp", 4, hidden_dims=[16], output_mode="hurdle")
    pressure = torch.zeros(2, 2, 21, 21)
    pressure[..., 5:10, 5:10] = .8
    valid = torch.ones_like(pressure, dtype=torch.bool)
    valid[..., 0, :] = False
    pressure[..., 0, :] = float("nan")
    cfg = {"contact_weight": 3., "loss_threshold": .1}
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    for _ in range(80):
        optimizer.zero_grad()
        parts = model.decode_components(model.encode(pressure, valid))
        loss = autoencoder_loss(parts, pressure, valid, cfg)
        loss.backward()
        optimizer.step()
    model.eval()
    pred = model(pressure, valid)
    assert (pred[(pressure == 0) & valid] == 0).float().mean() > .95
    assert (pred[(pressure > 0) & valid] - .8).abs().mean() < .1


def test_single_pressure_clipping_preserves_training_gradients():
    head = ClippedPressureOutput()
    raw = torch.tensor([-.3, 0., .001, .4, 1.2], requires_grad=True)
    # Train on raw values: negative/over-range outputs receive corrective gradients.
    head(raw).square().sum().backward()
    assert raw.grad[0] < 0 and raw.grad[-1] > 0
    head.eval()
    torch.testing.assert_close(head(raw), torch.tensor([0., 0., .001, .4, 1.]))
    assert torch.equal(head(raw)[:2], torch.zeros(2))


@pytest.mark.parametrize("architecture", ["conv", "mlp"])
def test_single_clipped_decoder_export_and_frozen_adapter_gradient(architecture):
    model = TactileAutoencoder(architecture, 4, channels=[4, 8], hidden_dims=[8], output_mode="clamp")
    model.eval().requires_grad_(False)
    code = torch.randn(2, 4, requires_grad=True)
    raw = model.decode(code, hard=False)
    assert raw.shape == (2, 2, 21, 21)  # One pressure output per hand/taxel, not two heads.
    parts = model.decode_components(code)
    assert set(parts) == {"soft", "hard"}
    torch.testing.assert_close(model.decode(code), raw.clamp(0, 1))
    raw.square().sum().backward()
    assert code.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.parameters())
    restored = TactileAutoencoder(**model.config).eval()
    restored.decoder.load_state_dict(model.decoder.state_dict())
    torch.testing.assert_close(restored.decoder(code), model.decode(code))
