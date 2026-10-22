import torch
import pytest

from tactilelatent.frozen_probe import head
from tactilelatent.train_joint_adversarial import (
    classification_loss, escalate, reverse, reversal_strength, source_weights,
)


def test_gradient_reversal_flips_only_encoder_gradient():
    x = torch.tensor([[1., 2.]], requires_grad=True)
    classifier = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        classifier.weight.copy_(torch.tensor([[1., 0.], [0., 1.]]))
    direct = classifier(x).sum()
    direct.backward()
    input_gradient = x.grad.clone()
    weight_gradient = classifier.weight.grad.clone()
    x.grad = None
    classifier.weight.grad = None
    classifier(reverse(x, .3)).sum().backward()
    torch.testing.assert_close(x.grad, -.3 * input_gradient)
    torch.testing.assert_close(classifier.weight.grad, weight_gradient)


def test_classifier_uses_mean_and_sample_and_blocks_encoder_during_warmup():
    classifier = head(7, 3, classification=True)
    mu = torch.randn(2, 3, 7, requires_grad=True)
    logvar = torch.zeros(2, 3, 7, requires_grad=True)
    loss, raw, accuracy = classification_loss(classifier, mu, logvar, 1, 0., 1.)
    loss.backward()
    assert mu.grad is not None and torch.count_nonzero(mu.grad) == 0
    assert logvar.grad is not None and torch.count_nonzero(logvar.grad) == 0
    assert classifier[0].weight.grad is not None
    assert raw.ndim == 0 and 0 <= accuracy <= 1


def test_balancing_and_escalation():
    class Dataset:
        indices_by_source = {"a": range(100), "b": range(20)}
    weights = source_weights(Dataset(), cap=50, sources=("a", "b"))
    assert 50 * weights["a"] == 20 * weights["b"]
    state = {"start": 0., "target": .2, "ramp_start": 1, "ramp_epochs": 5}
    assert reversal_strength(0, state) == 0
    assert reversal_strength(5, state) == .2
    stronger = escalate(state, epoch=9, diagnostic_macro=.9)
    assert stronger["target"] == .4
    assert reversal_strength(10, stronger) == pytest.approx(.3)
    assert escalate(stronger, epoch=19, diagnostic_macro=.1) == stronger
