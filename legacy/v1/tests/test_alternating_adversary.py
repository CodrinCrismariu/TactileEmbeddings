import random

import numpy as np
import pytest
import torch

from tactilelatent.alternating_adversary import (
    BalancedLatentBank, choose_challenger, fit_classifiers, reset_challenger,
    reversed_loss,
)
from tactilelatent.frozen_probe import head
from tactilelatent.train_joint_alternating import (
    checkpoint_state, restore_checkpoint, save_checkpoint,
)


def test_balanced_bank_is_bounded_recent_and_restorable():
    bank = BalancedLatentBank(("a", "b"), capacity=2, minimum=1)
    for value in range(3):
        bank.add("a", torch.full((1, 2, 3), float(value)),
                 torch.full((1, 2, 3), float(value) + .5))
    assert not bank.ready()
    bank.add("b", torch.full((1, 2, 3), 9.), torch.full((1, 2, 3), 9.5))
    assert bank.ready() and bank.counts() == {"a": 2, "b": 1}
    for _ in range(10):
        means, samples, labels = bank.sample(3, torch.device("cpu"))
        assert means.shape == samples.shape == (12, 3)
        assert torch.bincount(labels).tolist() == [6, 6]
        assert set(means[labels == 0, 0].tolist()) <= {1., 2.}
    restored = BalancedLatentBank(("a", "b"), capacity=2, minimum=1)
    restored.load_state_dict(bank.state_dict())
    assert restored.counts() == bank.counts()
    with pytest.raises(ValueError):
        BalancedLatentBank(("b", "a"), capacity=2, minimum=1).load_state_dict(
            bank.state_dict())


def test_classifier_only_then_reversed_encoder_only():
    torch.manual_seed(4)
    encoder = torch.nn.Linear(3, 2)
    persistent = head(2, 2, dropout=0, classification=True)
    challenger = head(2, 2, dropout=0, classification=True)
    classifier_optimizer = torch.optim.AdamW(
        list(persistent.parameters()) + list(challenger.parameters()), lr=.01)
    bank = BalancedLatentBank(("a", "b"), capacity=2, minimum=1)
    for label, source in enumerate(("a", "b")):
        visual = torch.full((1, 2, 3), float(label))
        encoded = encoder(visual)
        bank.add(source, encoded, encoded + .1)
    encoder_before = encoder.weight.detach().clone()
    assert fit_classifiers((persistent, challenger), classifier_optimizer, bank,
                           steps=3, clips_per_source=1, device=torch.device("cpu")) == 3
    assert encoder.weight.grad is None
    torch.testing.assert_close(encoder.weight, encoder_before)
    assert not any(p.requires_grad for p in persistent.parameters())
    x = torch.randn(1, 2, 3)
    mu = encoder(x)
    direct = reversed_loss(persistent, mu, mu + .1, 0, -1.)
    direct.backward(retain_graph=True)
    direct_gradient = encoder.weight.grad.detach().clone()
    encoder.zero_grad(set_to_none=True)
    reversed_loss(persistent, mu, mu + .1, 0, .4).backward()
    torch.testing.assert_close(encoder.weight.grad, -.4 * direct_gradient)
    assert all(p.grad is None for p in persistent.parameters())


def test_challenger_reset_clears_its_optimizer_history():
    bank = BalancedLatentBank(("a", "b"), capacity=1, minimum=1)
    for i, source in enumerate(("a", "b")):
        bank.add(source, torch.ones(1, 2, 2) * i, torch.ones(1, 2, 2) * i)
    persistent = head(2, 2, classification=True)
    challenger = head(2, 2, classification=True)
    optimizer = torch.optim.AdamW(list(persistent.parameters()) + list(challenger.parameters()))
    fit_classifiers((persistent, challenger), optimizer, bank, steps=1,
                    clips_per_source=1, device=torch.device("cpu"))
    assert all(p in optimizer.state for p in challenger.parameters())
    reset_challenger(challenger, optimizer)
    assert all(p not in optimizer.state for p in challenger.parameters())
    assert all(p in optimizer.state for p in persistent.parameters())


def test_strongest_classifier_is_selected():
    first = torch.nn.Linear(2, 2)
    second = torch.nn.Linear(2, 2)
    with torch.no_grad():
        first.weight.zero_(); first.bias[:] = torch.tensor([5., -5.])
        second.weight.zero_(); second.bias[:] = torch.tensor([-5., 5.])
    mu = torch.zeros(1, 2, 2)
    selected, losses, accuracies = choose_challenger((first, second), mu, mu, 0)
    assert selected == 0 and losses[0] < losses[1]
    assert accuracies == [1., 0.]


def test_epoch_checkpoint_restores_bank_optimizers_and_rng(tmp_path):
    torch.manual_seed(6); random.seed(6); np.random.seed(6)
    model = torch.nn.Linear(2, 2)
    persistent = head(2, 2, classification=True)
    challenger = head(2, 2, classification=True)
    model_opt = torch.optim.AdamW(model.parameters(), lr=.01)
    class_opt = torch.optim.AdamW(
        list(persistent.parameters()) + list(challenger.parameters()), lr=.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(model_opt, lambda _: 1.)
    bank = BalancedLatentBank(("a", "b"), capacity=2, minimum=1)
    bank.add("a", torch.ones(1, 2, 2), torch.ones(1, 2, 2) * 2)
    bank.add("b", torch.zeros(1, 2, 2), torch.ones(1, 2, 2))
    loss = model(torch.ones(1, 2)).sum(); loss.backward(); model_opt.step(); scheduler.step()
    fit_classifiers((persistent, challenger), class_opt, bank, steps=1,
                    clips_per_source=1, device=torch.device("cpu"))
    cfg = {"test": 1}
    state = checkpoint_state(model, persistent, challenger, model_opt, class_opt,
        scheduler, bank, 4, cfg, {"target": .2}, {}, {"epoch": 4}, torch.device("cpu"))
    path = tmp_path / "last.pt"
    save_checkpoint(path, state)
    expected_torch, expected_python, expected_numpy = (
        torch.rand(1).item(), random.random(), np.random.random())
    model.weight.data.zero_()
    bank.add("a", torch.zeros(1, 2, 2), torch.zeros(1, 2, 2))
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    epoch, reversal, _ = restore_checkpoint(loaded, model, persistent, challenger,
        model_opt, class_opt, scheduler, bank, cfg, torch.device("cpu"))
    assert epoch == 5 and reversal == {"target": .2}
    assert bank.counts() == {"a": 1, "b": 1}
    assert model.weight.abs().sum() > 0
    assert torch.rand(1).item() == expected_torch
    assert random.random() == expected_python
    assert np.random.random() == expected_numpy
    assert class_opt.state and model_opt.state
