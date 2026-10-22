import json
from pathlib import Path

import av
import numpy as np
import pytest
import torch

from tactilelatent.aligned_frames import CONTRACT, sensor_indices, targets, write_json, file_hash
from tactilelatent.alignment_data import AlignedClipDataset
from tactilelatent.alignment_model import JointSensorAlignment, FrozenSensorDecoders, model_from_state, MODEL_TYPE
from tactilelatent.sensor_autoencoder import SensorAutoencoder
from tactilelatent.sensor_code_cache import CodeCache, build_source, cache_signature, train_statistics
from tactilelatent.train_alignment import (alignment_loss, sam_replay, extension_config,
    check_resume_contract, alignment_scheduler, EXTENSION_TRAINER_HASHES, bank_sampling_config,
    model_weight_hash)
from tactilelatent.sam import SAMStep, rng_snapshot


@pytest.fixture
def tiny_data(tmp_path):
    rows = []
    generator = np.random.default_rng(12)
    for split in ("train", "val"):
        values = generator.uniform(.0 if split == "train" else .8,
                                   .2 if split == "train" else 1., (20, 2, 21, 21)).astype(np.float32)
        if split == "train":
            values[0] = np.nan
        path = tmp_path / f"{split}.npz"
        np.savez(path, left_pressure_grid=values[:, 0], right_pressure_grid=values[:, 1])
        video = tmp_path / f"{split}.mkv"
        with av.open(str(video), "w") as container:
            stream = container.add_stream("ffv1", rate=30)
            stream.width, stream.height, stream.pix_fmt = 24, 16, "bgr0"
            for frame in range(20):
                image = av.VideoFrame.from_ndarray(np.full((16, 24, 3), frame, np.uint8), format="rgb24")
                for packet in stream.encode(image):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        rows.append({"source": "egotouch", "episode_id": split, "frames": 20, "split": split,
                     "views": {"0": str(video)}, "target_path": str(path), "target_kind": "pressure_grid",
                     "target_shape": [2, 21, 21], "target_scale": 1.})
    manifest = tmp_path / "manifest.json"
    write_json(manifest, {"schema_version": 2, "frame_contract": CONTRACT, "episodes": rows})
    torch.manual_seed(12)
    teacher = SensorAutoencoder([2, 21, 21], latent_dim=3, channels=[4])
    checkpoint = tmp_path / "teacher.pt"
    torch.save({"model_config": teacher.config, "model": teacher.state_dict()}, checkpoint)
    registry = {"schema_version": 1, "sources": {"egotouch": {
        "checkpoint": str(checkpoint), "sha256": file_hash(checkpoint), "code_dim": 3,
        "target_shape": [2, 21, 21], "model_config": teacher.config}}}
    return manifest, rows, teacher, registry


def test_cache_exact_codes_masks_all_frames_and_train_only_statistics(tiny_data, tmp_path):
    manifest, rows, teacher, registry = tiny_data
    root = tmp_path / "cache"
    index = build_source(manifest, registry, "egotouch", root, "cpu", workers=0, chunk_frames=7)
    assert index["frames"] == 40
    assert index["statistics"]["train_frames"] == 19
    cache = CodeCache(root, "egotouch", cache_signature(manifest, registry, "egotouch"))
    for row in rows:
        # Out-of-order requests and duplicates retain exact code/frame pairing.
        frames = [19, 1, 1, 0, 9]
        code, valid = cache.read(row["episode_id"], frames, normalized=False)
        target = torch.from_numpy(targets(row, frames))
        with torch.no_grad():
            expected = teacher.encode(target)
        torch.testing.assert_close(code, expected, rtol=1e-5, atol=1e-6)
        assert torch.equal(valid, torch.isfinite(target).flatten(1).any(1))
    raw = np.load(root / "egotouch" / "codes.npy")
    np.testing.assert_allclose(index["statistics"]["mean"], raw[1:20].mean(0), rtol=1e-5)
    dataset = AlignedClipDataset(manifest, "train", code_cache=root, registry=registry, include_targets=False)
    item = dataset[0]
    assert item["frame_indices"].tolist() == list(range(0, 15, 2))
    assert "target" not in item
    expected, valid = cache.read("train", item["frame_indices"].numpy())
    torch.testing.assert_close(item["code_target"], expected)
    assert item["rgb"].shape == (8, 3, 3, 224, 224)
    assert not item["code_valid"][0]


def test_resume_after_completed_episode_and_stale_loader(tiny_data, tmp_path, monkeypatch):
    import tactilelatent.sensor_code_cache as module
    manifest, rows, teacher, registry = tiny_data
    root = tmp_path / "cache"
    original = module.write_json
    def interrupted(path, value):
        original(path, value)
        if Path(path).name == "progress.json" and len(value["completed_episodes"]) == 1:
            raise RuntimeError("simulated interruption")
    monkeypatch.setattr(module, "write_json", interrupted)
    with pytest.raises(RuntimeError, match="interruption"):
        build_source(manifest, registry, "egotouch", root, "cpu", workers=0)
    first = np.load(root / "egotouch" / "codes.partial.npy")[:20].copy()
    monkeypatch.setattr(module, "write_json", original)
    build_source(manifest, registry, "egotouch", root, "cpu", workers=0)
    np.testing.assert_array_equal(first, np.load(root / "egotouch" / "codes.npy")[:20])
    expected = cache_signature(manifest, registry, "egotouch")
    expected["teacher_sha256"] = "wrong"
    with pytest.raises(ValueError, match="Stale"):
        CodeCache(root, "egotouch", expected)


def test_vla_permutation_never_reorders_scene_and_preserves_native_pixels(monkeypatch):
    import tactilelatent.native_optical as native
    import tactilelatent.aligned_frames as aligned
    stored = np.stack([np.full((12, 18, 3), x, np.uint8) for x in [200, 20, 80]])[:, None]
    monkeypatch.setattr(native, "ftp_group", lambda row: {"data/sensor": stored})
    row = {"source": "ftp_vla_touch", "episode_id": "fixture", "frames": 3,
           "target_start": 0, "target_keys": ["sensor"], "target_kind": "optical_rgb",
           "target_shape": [3, 12, 18], "sensor_frame_order": [1, 2, 0],
           "views": {"0": "main"}, "raw_scene": {"rotation_quarters": {"0": 0}}}
    calls = []
    def scenes(row, indices, camera):
        calls.append(indices.tolist())
        return [np.full((30, 40, 3), int(frame), np.uint8) for frame in indices]
    monkeypatch.setattr(aligned, "raw_scene_images", scenes)
    np.testing.assert_array_equal(targets(row, [0, 1, 2])[:, 0, 0, 0], np.array([20, 80, 200], np.float32) / 255)
    scene, _ = aligned.scene_rgb(row, [0, 1, 2])
    assert calls == [[0, 1, 2]]
    np.testing.assert_allclose(scene[:, 0, 0, 0, 0].numpy(), np.array([0, 1, 2]) / 255)
    with pytest.raises(ValueError, match="verified"):
        sensor_indices({**row, "sensor_frame_order": None}, [0])


def small_model():
    return JointSensorAlignment({"egotouch": 3}, {"ftp_rdp": [1, 1, 1]},
        {"egotouch": {"mean": [1., 2., 3.], "std": [2., 3., 4.]}},
        input_dim=8, width=16, heads=4, temporal_layers=1,
        cross_view_layers=1, latent_dim=4, adapter_hidden_dims=[8], dropout=.1)


def inputs():
    value = torch.randn(2, 3, 3, 4, 8)
    mask = torch.zeros(2, 3, 3, dtype=torch.bool); mask[:, :, 0] = True
    batch = {"code_target": torch.randn(2, 3, 3, requires_grad=True),
             "code_valid": torch.tensor([[True, True, False], [True, True, True]])}
    return value, mask, batch


def test_teacher_detachment_gaussian_gradients_numeric_head_and_checkpoint():
    model = small_model(); features, mask, batch = inputs()
    output = model(features, mask, "egotouch")
    assert set(output) == {"mu", "logvar", "sensor_code"}
    loss, _, _ = alignment_loss(output, batch, .001); loss.backward()
    assert batch["code_target"].grad is None
    assert model.visual.prior_mu.weight.grad.abs().sum() > 0
    assert model.visual.prior_logvar.weight.grad.abs().sum() > 0
    assert model.adapters["egotouch"][0].weight.grad.abs().sum() > 0
    assert all(not name.startswith("visual.decoder.") for name in model.state_dict())
    numeric = model(features, mask, "ftp_rdp")["pred"]
    assert numeric.shape == (2, 3, 1, 1, 1)
    assert ((numeric > 0) & (numeric < 1)).all()
    torch.testing.assert_close(model.raw_code("egotouch", torch.zeros(2, 3)), torch.tensor([[1., 2., 3.]]).expand(2, -1))


def test_sam_reuses_samples_and_restores_rng():
    model = small_model(); features, mask, batch = inputs()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    sam = SAMStep(optimizer)
    before = rng_snapshot()
    output = model(features, mask, "egotouch")
    loss, _, _ = alignment_loss(output, batch, .001); loss.backward()
    after = rng_snapshot()
    norm = sam_replay(model, optimizer, sam, [{"features": features, "mask": mask,
        "source": "egotouch", "batch": batch, "beta": .001, "rng": before}], torch.device("cpu"))
    assert norm > 0 and sam.steps == 1 and not sam.originals
    assert torch.equal(torch.get_rng_state(), after["cpu"])


def test_frozen_decoder_matches_teacher_after_unstandardization(tiny_data, tmp_path):
    manifest, rows, teacher, registry = tiny_data
    teacher.eval()
    value = torch.from_numpy(targets(rows[0], [1, 3]))
    with torch.inference_mode():
        raw = teacher.encode(value)
        expected = teacher.decode(raw)
        decoder = FrozenSensorDecoders(registry, "cpu")
        actual = decoder.decode("egotouch", raw)
    torch.testing.assert_close(actual, expected)
    assert not any(parameter.requires_grad for parameter in decoder.decoder.parameters())


def test_alignment_checkpoint_dispatch_and_teacher_normalization():
    cfg = dict(sensor_code_dims={"egotouch": 3}, numeric_shapes={"ftp_rdp": [1, 1, 1]},
               statistics={"egotouch": {"mean": [1., 2., 3.], "std": [2., 3., 4.]}},
               input_dim=8, width=16, heads=4, temporal_layers=1, cross_view_layers=1,
               latent_dim=4, adapter_hidden_dims=[8], dropout=.1)
    model = JointSensorAlignment(**cfg).eval()
    state = {"model_type": MODEL_TYPE, "model_config": cfg, "model": model.state_dict()}
    loaded = model_from_state(state).eval()
    features, mask, _ = inputs()
    torch.testing.assert_close(model(features, mask, "egotouch")["sensor_code"],
                               loaded(features, mask, "egotouch")["sensor_code"])
    with pytest.raises(ValueError, match="alignment checkpoint"):
        model_from_state({**state, "model_type": "legacy"})


def test_isolated_deterministic_fitting_reduces_code_error():
    torch.manual_seed(7)
    model = small_model().eval()
    features, mask, batch = inputs()
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    def objective():
        return alignment_loss(model(features, mask, "egotouch", sample_latent=False), batch)[0]
    first = objective().item()
    for _ in range(20):
        optimizer.zero_grad(); loss = objective(); loss.backward(); optimizer.step()
    assert objective().item() < first * .5


@pytest.mark.parametrize('fgw', [False, True])
def test_epoch_extension_preserves_optimizer_and_lr_and_rejects_other_changes(fgw):
    cfg = {'epochs': 50, 'learning_rate': 5e-5, 'min_learning_rate': 1e-6,
           'warmup_epochs': 2, 'accumulation': 2, 'width': 16}
    if fgw:
        cfg.update(fgw={'placement': 'visual'}, fork_parent_epochs=10)
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([parameter], lr=cfg['learning_rate'])
    scheduler = alignment_scheduler(optimizer, cfg, 4)
    saved_updates = 60 if fgw else 80
    for _ in range(saved_updates):
        optimizer.zero_grad(); parameter.square().sum().backward(); optimizer.step(); scheduler.step()
    lr = optimizer.param_groups[0]['lr']
    provenance = {'trainer_sha256': next(iter(EXTENSION_TRAINER_HASHES)), 'data': 'unchanged'}
    state = {'config': cfg, 'epoch': 39, 'scheduler': scheduler.state_dict(),
             'optimizer': optimizer.state_dict(), 'provenance': provenance}
    extended = extension_config(cfg, state, 200)
    check_resume_contract(state, extended, {**provenance, 'trainer_sha256': 'new-trainer'}, extending=True)
    resumed_optimizer = torch.optim.AdamW([parameter], lr=cfg['learning_rate'])
    resumed_scheduler = alignment_scheduler(resumed_optimizer, extended, 4)
    resumed_optimizer.load_state_dict(state['optimizer']); resumed_scheduler.load_state_dict(state['scheduler'])
    assert resumed_optimizer.param_groups[0]['lr'] == lr
    assert resumed_optimizer.state_dict()['state'][0]['step'] == saved_updates
    resumed_optimizer.step(); resumed_scheduler.step()
    assert lr * .999 < resumed_optimizer.param_groups[0]['lr'] <= lr
    end_updates = 2 * (extended['epochs'] - cfg.get('fork_parent_epochs', 0))
    for _ in range(end_updates - saved_updates - 1):
        resumed_optimizer.step(); resumed_scheduler.step()
    assert resumed_optimizer.param_groups[0]['lr'] == pytest.approx(cfg['min_learning_rate'])
    with pytest.raises(ValueError, match='cannot change'):
        check_resume_contract(state, {**extended, 'width': 32}, provenance, extending=True)
    with pytest.raises(ValueError, match='cannot change'):
        check_resume_contract(state, extended, {**provenance, 'data': 'changed'}, extending=True)
    with pytest.raises(ValueError, match='cannot change'):
        check_resume_contract({**state, 'provenance': {**provenance, 'trainer_sha256': 'unknown'}},
                              extended, provenance, extending=True)
    with pytest.raises(ValueError, match='Extension'):
        extension_config(cfg, state, 30)


def test_16d_bottleneck_keeps_sensor_code_and_numeric_shapes():
    cfg = dict(sensor_code_dims={"egotouch": 256},
               numeric_shapes={"ftp_rdp": [1, 1, 1], "ftp_rdp_bimanual": [2, 1, 1]},
               statistics={"egotouch": {"mean": [0.] * 256, "std": [1.] * 256}},
               input_dim=8, width=16, heads=4, temporal_layers=1, cross_view_layers=1,
               latent_dim=16, adapter_hidden_dims=[128, 128], dropout=.1)
    torch.manual_seed(45); model = JointSensorAlignment(**cfg)
    torch.manual_seed(45); control = JointSensorAlignment(**cfg)
    assert model_weight_hash(model) == model_weight_hash(control)
    features, mask, _ = inputs()
    output = model(features, mask, "egotouch")
    assert output["mu"].shape[-1] == output["logvar"].shape[-1] == 16
    assert output["sensor_code"].shape[-1] == 256
    loss, _, _ = alignment_loss(output, {"code_target": torch.zeros_like(output["sensor_code"]),
        "code_valid": torch.ones(output["mu"].shape[:2], dtype=torch.bool)}, .001)
    loss.backward()
    assert torch.isfinite(loss) and any(p.grad is not None for p in model.visual.parameters())
    for source, shape in cfg["numeric_shapes"].items():
        assert model(features, mask, source)["pred"].shape[2:] == torch.Size(shape)


def test_plain_and_fgw_use_identical_bank_sampling_and_lr():
    plain = {"matched_bank_sampling": {"bank_size": 256, "fraction": .25},
             "epochs": 30, "learning_rate": 5e-5, "min_learning_rate": 1e-6,
             "warmup_epochs": 2, "accumulation": 16}
    fgw = {**plain, "fgw": {"placement": "visual", "bank_size": 256, "bank_fraction": .25}}
    assert bank_sampling_config(plain) == bank_sampling_config(fgw)
    assert bank_sampling_config({}) is None
    a = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=5e-5)
    b = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=5e-5)
    sa, sb = alignment_scheduler(a, plain, 32), alignment_scheduler(b, fgw, 32)
    for _ in range(60):
        assert sa.get_last_lr() == sb.get_last_lr()
        a.step(); b.step(); sa.step(); sb.step()
    with pytest.raises(ValueError, match="agree"):
        bank_sampling_config({**fgw, "matched_bank_sampling": {"bank_size": 128, "fraction": .25}})
