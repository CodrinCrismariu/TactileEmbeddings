"""Scientific parity of objectives, actual SAM implementation and v2 resume."""

import random
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_core_models import assert_same_tensors, baseline_module, config, inputs

from tactilelatent.contracts import ClipBatch, SensorTarget, Supervision, VisualClip
from tactilelatent.models import JointModel
from tactilelatent.training.checkpointing import restore_checkpoint, save_checkpoint
from tactilelatent.training.engine import train_epoch, wrist_dropout
from tactilelatent.training.objectives import forward_batch, supervision_loss, valid_frames
from tactilelatent.training.optim import SAMStep, make_scheduler, replay_update, rng_snapshot


def batch(kind="code", source="ftp_unit"):
    _, mask = inputs()
    target = torch.full((2, 1, 3), 0.2)
    valid = (
        torch.ones((2, 1), dtype=torch.bool)
        if kind == "code"
        else torch.ones_like(target, dtype=torch.bool)
    )
    return ClipBatch(
        VisualClip(torch.zeros(2, 8, 3, 3, 2, 2), mask, torch.arange(8)),
        SensorTarget(target, torch.ones_like(target, dtype=torch.bool), "native_numeric"),
        Supervision(target, valid, kind),
        source,
        ["a", "b"],
        "train",
        torch.tensor([[14], [14]]),
        torch.tensor([[14], [14]]),
    )


@pytest.mark.parametrize("kind", ["code", "direct"])
def test_objective_missing_values_and_gradients_match_legacy(kind):
    item = batch(kind)
    item.supervision.valid[0] = False
    if kind == "direct":
        item.supervision.values[0] = float("nan")
    predicted = torch.full((2, 1, 3), 0.4, requires_grad=True)
    output = {
        "mu": predicted,
        "logvar": torch.zeros_like(predicted),
        "latent_mode": "deterministic",
        "sensor_code" if kind == "code" else "pred": predicted,
    }
    old = (
        {"code_target": item.supervision.values, "code_valid": item.supervision.valid}
        if kind == "code"
        else {"target": item.supervision.values, "target_mask": item.supervision.valid}
    )
    expected = baseline_module("train_alignment").alignment_loss(output, old, 0.0)[0]
    actual = supervision_loss(output, item)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(
        torch.autograd.grad(actual, predicted, retain_graph=True)[0],
        torch.autograd.grad(expected, predicted)[0],
        atol=0,
        rtol=0,
    )
    assert valid_frames(item).tolist() == [[False], [True]]


def update(model, item, features, *, legacy=False, optimizer=None, scheduler=None, sam=None):
    optimizer = optimizer or torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.05)
    sam = sam or (baseline_module("sam").SAMStep if legacy else SAMStep)(optimizer, 0.05)
    before = rng_snapshot()
    output = (
        model(features, item.visual.view_mask, item.source)
        if legacy
        else forward_batch(model, features, item)
    )
    loss = supervision_loss(output, item)
    loss.backward()
    if legacy:
        baseline_module("train_alignment").sam_replay(
            model,
            optimizer,
            sam,
            [
                dict(
                    features=features,
                    mask=item.visual.view_mask,
                    source=item.source,
                    batch={
                        "code_target": item.supervision.values,
                        "code_valid": item.supervision.valid,
                    },
                    rng=before,
                    beta=0.0,
                )
            ],
            torch.device("cpu"),
        )
    else:
        replay_update(
            model,
            optimizer,
            sam,
            [dict(features=features, batch=item, rng=before)],
            torch.device("cpu"),
        )
    if scheduler:
        scheduler.step()
    return optimizer, sam


def test_actual_core_sam_replay_has_exact_weights_optimizer_and_rng_parity():
    torch.set_num_threads(1)
    features, _ = inputs()
    torch.manual_seed(45)
    old = baseline_module("acquisition_joint").AllSourceVJEPA(**config()).train()
    torch.manual_seed(45)
    new = JointModel(**config()).train()
    torch.manual_seed(91)
    oa, sa = update(old, batch(), features, legacy=True)
    rng = torch.get_rng_state().clone()
    torch.manual_seed(91)
    ob, sb = update(new, batch(), features)
    assert_same_tensors(old.state_dict(), new.state_dict())
    assert torch.equal(rng, torch.get_rng_state())
    assert sa.state_dict() == sb.state_dict()
    assert oa.state_dict()["param_groups"] == ob.state_dict()["param_groups"]
    for key in oa.state_dict()["state"]:
        assert_same_tensors(oa.state_dict()["state"][key], ob.state_dict()["state"][key])


def test_checkpoint_resume_includes_all_random_and_optimizer_state(tmp_path):
    torch.manual_seed(45)
    random.seed(45)
    np.random.seed(45)
    model = JointModel(**config()).train()
    tc = dict(
        epochs=3, accumulation=1, warmup_epochs=1, min_learning_rate=1e-6, learning_rate=0.0003
    )
    experiment, provenance = {"training": tc}, {"teacher": "123", "data": "abc"}
    opt = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.05)
    sched = make_scheduler(opt, tc, 2)
    sam, generator = SAMStep(opt), torch.Generator().manual_seed(45)
    features, _ = inputs()
    update(model, batch(), features, optimizer=opt, scheduler=sched, sam=sam)
    save_checkpoint(
        tmp_path,
        1,
        model,
        config(),
        experiment,
        provenance,
        opt,
        sched,
        sam,
        generator,
        1.0,
        {},
        2.0,
    )
    random_expected, numpy_expected = random.random(), np.random.rand()
    loader_expected = torch.rand(3, generator=generator)
    update(model, batch(), features, optimizer=opt, scheduler=sched, sam=sam)
    expected = deepcopy(model.state_dict())
    expected_rng = torch.get_rng_state().clone()
    fresh = JointModel(**config()).train()
    other = torch.optim.AdamW(fresh.parameters(), lr=0.0003, weight_decay=0.05)
    other_sched, other_sam = make_scheduler(other, tc, 2), SAMStep(other)
    state = restore_checkpoint(
        tmp_path / "last.pt",
        fresh,
        config(),
        experiment,
        provenance,
        other,
        other_sched,
        other_sam,
        generator,
    )
    assert state["epoch"] == 0
    assert random.random() == random_expected and np.random.rand() == numpy_expected
    assert torch.equal(torch.rand(3, generator=generator), loader_expected)
    update(fresh, batch(), features, optimizer=other, scheduler=other_sched, sam=other_sam)
    assert_same_tensors(expected, fresh.state_dict())
    assert torch.equal(expected_rng, torch.get_rng_state())
    assert sched.state_dict() == other_sched.state_dict()
    with pytest.raises(ValueError, match="provenance changed"):
        restore_checkpoint(
            tmp_path / "last.pt",
            fresh,
            config(),
            experiment,
            {"teacher": "changed"},
            other,
            other_sched,
            other_sam,
            generator,
        )
    with pytest.raises(ValueError, match="Only resumable"):
        restore_checkpoint(
            tmp_path / "viewer_epochs/epoch_001.pt",
            fresh,
            config(),
            experiment,
            provenance,
            other,
            other_sched,
            other_sam,
            generator,
        )


def test_train_epoch_uses_frozen_features_once_and_replays_exact_augmented_batch(monkeypatch):
    import tactilelatent.models.backbone as module

    features, _ = inputs()
    calls, augmented = [], []

    def extract(backbone, item, device, chunk_size=2):
        calls.append(item)
        return features

    def augment(item, **kwargs):
        augmented.append(kwargs)
        return item

    monkeypatch.setattr(module, "spatial_video_features", extract)
    model = JointModel(**config())
    tc = dict(
        seed=45,
        epochs=1,
        wrist_dropout=0.3,
        accumulation=2,
        warmup_epochs=0,
        min_learning_rate=1e-6,
        learning_rate=0.0003,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=0.0003)
    sched, sam = make_scheduler(opt, tc, 3), SAMStep(opt)
    result, stats = train_epoch(
        model,
        None,
        [batch(), batch(), batch()],
        SimpleNamespace(augment=augment),
        opt,
        sched,
        sam,
        tc,
        {},
        torch.device("cpu"),
        0,
    )
    assert len(calls) == len(augmented) == 3
    assert sam.steps == 2 and sched.last_epoch == 2
    assert result["ftp_unit"]["supervision"] >= 0 and stats


def test_wrist_dropout_is_schedule_deterministic_and_does_not_consume_global_rng():
    item = batch()
    item.visual.view_mask[:] = True
    before = torch.get_rng_state().clone()
    a = wrist_dropout(item, 0.3, 45, 2, 3, torch.device("cpu"))
    b = wrist_dropout(item, 0.3, 45, 2, 3, torch.device("cpu"))
    assert torch.equal(a.visual.view_mask, b.visual.view_mask)
    assert torch.equal(before, torch.get_rng_state())
    assert a.visual.view_mask[:, :, 0].all()


def test_native_validation_reduction_and_paired_equivariance(monkeypatch):
    import tactilelatent.models.backbone as module
    from tactilelatent.training.validation import paired_validate, validate

    class Model(torch.nn.Module):
        direct_numeric = {}

        def forward(self, features, *args, **kwargs):
            return {"pred": features, "mu": features, "logvar": torch.zeros_like(features)}

    def extract(backbone, item, device, **kwargs):
        return item.target.values + 1

    monkeypatch.setattr(module, "spatial_video_features", extract)
    item = batch("direct", "numeric")
    item.supervision.values = item.target.values
    spec = SimpleNamespace(
        metadata={"native_metric_reduction": "cells"},
        target_kind="native_numeric",
        metric_groups={},
        mirror={"verified": True},
    )

    def augment(item, **kwargs):
        return replace(
            item,
            target=replace(
                item.target, values=item.target.values.flip(-1), valid=item.target.valid.flip(-1)
            ),
        )

    registry = SimpleNamespace(specs={"numeric": spec}, augment=augment)
    result = validate(Model(), None, [item], registry, ".", torch.device("cpu"))
    assert result["numeric"]["native_mae"] == 1.0
    pair = paired_validate(Model(), None, [item], registry, ".", torch.device("cpu"))
    assert pair == {"numeric": {"original": 1.0, "mirrored": 1.0, "equivariance": 0.0}}


def test_lazy_validation_teacher_loading_preserves_training_rng_and_frozen_weights(tmp_path):
    from tactilelatent.models.autoencoders import make_autoencoder
    from tactilelatent.models.autoencoders.factory import checkpoint_hash
    from tactilelatent.training.validation import FrozenDecoders

    teacher_config = dict(
        input_dim=3,
        latent_dim=2,
        hidden_dims=[4],
        scale=[1.0, 2.0, 3.0],
        nonnegative=[False, False, True],
    )
    teacher = make_autoencoder(teacher_config, "native-numeric-ae-v1")
    path = tmp_path / "teacher.pt"
    torch.save(
        {
            "model_type": "native-numeric-ae-v1",
            "model_config": teacher_config,
            "model": teacher.state_dict(),
        },
        path,
    )
    registry = SimpleNamespace(
        specs={
            "numeric": SimpleNamespace(
                teacher={"checkpoint": path.name, "sha256": checkpoint_hash(path)}
            )
        }
    )
    decoders = FrozenDecoders(registry, tmp_path, torch.device("cpu"))
    torch.manual_seed(73)
    before = rng_snapshot()
    loaded = decoders.teacher("numeric")
    after = rng_snapshot()
    assert torch.equal(before["cpu"], after["cpu"])
    assert all(torch.equal(a, b) for a, b in zip(before["cuda"], after["cuda"]))
    assert_same_tensors(teacher.state_dict(), loaded.state_dict())
    assert not loaded.training and not any(p.requires_grad for p in loaded.parameters())
    assert decoders.teacher("numeric") is loaded
    assert torch.equal(before["cpu"], torch.get_rng_state())
