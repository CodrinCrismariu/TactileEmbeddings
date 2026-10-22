import copy

import pytest
import torch
from torch import nn

from tactilelatent.continuation_objectives import SourceStops, objective, retention_loss


class Visual(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean = nn.Linear(4, 3)
        self.variance = nn.Linear(4, 3)
        self.dropout = nn.Dropout(.2)

    def encode(self, x, mask):
        x = self.dropout(x)
        return self.mean(x), self.variance(x).clamp(-8, 4)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = Visual()
        self.stochastic = True
        self.adapters = nn.ModuleDict({'source': nn.Linear(3, 2)})
        self.numeric_heads = nn.ModuleDict()


def setup_item(stop=False, anchor=False):
    torch.manual_seed(45)
    model = Model().train()
    item = dict(features=torch.randn(2, 3, 4), mask=None, source='source', stop_trunk=stop,
        beta=.001, continuation_epoch=0,
        batch=dict(code_target=torch.randn(2, 3, 2), code_valid=torch.ones(2, 3, dtype=torch.bool)),
        anchor=torch.randn(2, 3, 3) if anchor else None)
    cfg = dict(retention=dict(weight=.1, parent_variance=2., ramp_epochs=2))
    return model, item, cfg


def test_task_and_kl_gate_preserves_adapter_gradients():
    model, item, cfg = setup_item(stop=True)
    objective(model, item, cfg)[1].backward()
    assert all(p.grad is None for p in model.visual.parameters())
    assert all(p.grad is not None and p.grad.abs().sum()>0 for p in model.adapters.parameters())


def test_combined_gate_trains_trunk_only_through_anchor():
    model, item, cfg = setup_item(stop=True, anchor=True)
    objective(model, item, cfg)[1].backward()
    assert model.visual.mean.weight.grad is not None
    assert model.visual.variance.weight.grad is None
    assert model.adapters['source'].weight.grad is not None


def test_ungated_control_trains_mean_and_variance():
    model, item, cfg = setup_item()
    objective(model, item, cfg)[1].backward()
    assert all(p.grad is not None for p in model.visual.parameters())


def test_retention_masks_missing_targets_and_detaches_teacher():
    mu = torch.zeros(1, 2, 3, requires_grad=True)
    target = torch.ones_like(mu, requires_grad=True)
    target.data[:, 1] = 1000
    loss = retention_loss(mu, target, dict(code_valid=torch.tensor([[True, False]])), 2.)
    assert loss.item() == .5
    loss.backward()
    assert target.grad is None
    assert mu.grad[:, 1].abs().sum() == 0


def test_no_valid_numeric_target_is_zero_and_differentiable():
    mu = torch.randn(1, 2, 3, requires_grad=True)
    loss = retention_loss(mu, torch.zeros_like(mu), dict(target_mask=torch.zeros(1, 2, 1, 1, 1)), 2.)
    assert loss.item() == 0
    loss.backward()
    assert mu.grad.abs().sum() == 0


def test_stops_deterioration_not_plateaus_and_never_restarts():
    cfg = dict(enabled=True, patience=3, relative_tolerance=.01)
    gate = SourceStops(['source'], cfg, [])
    for epoch, value in enumerate([1., 1., 1., 1., 1.02, 1.03, 1.04], 1):
        gate.update(dict(source=dict(supervision_loss=value)), epoch)
    assert gate.rows['source']['stopped_epoch'] == 7
    gate.update(dict(source=dict(supervision_loss=.5)), 8)
    assert gate.stopped('source')
    assert not SourceStops(['source'], dict(cfg, enabled=False), [dict(epoch=7, validation=dict(source=dict(supervision_loss=1.))) ]).stopped('source')


def test_motiontrans_cannot_control_source_gates_or_training():
    with pytest.raises(ValueError):
        SourceStops(['ftp_motiontrans'], dict(enabled=True, patience=3, relative_tolerance=.01), [])
    model, item, cfg = setup_item()
    item['source'] = 'ftp_motiontrans'
    with pytest.raises(ValueError):
        objective(model, item, cfg)


def test_replay_uses_same_noise_and_immutable_anchor():
    model, item, cfg = setup_item(anchor=True)
    before = torch.get_rng_state()
    target = item['anchor'].clone()
    first = objective(model, item, cfg)[1]
    torch.set_rng_state(before)
    second = objective(model, item, cfg)[1]
    assert torch.equal(first, second)
    assert torch.equal(target, item['anchor'])
    state = torch.get_rng_state()
    copy.deepcopy(model.visual).eval().requires_grad_(False)
    assert torch.equal(state, torch.get_rng_state())


def test_sam_replay_keeps_gated_trunk_weights_exactly_unchanged():
    from tactilelatent.sam import SAMStep, rng_snapshot
    from tactilelatent.train_continuation import replay_step
    model, item, cfg = setup_item(stop=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.05)
    sam = SAMStep(optimizer, .05)
    item.update(rng=rng_snapshot(), group_size=1)
    initial = {n:p.detach().clone() for n,p in model.visual.named_parameters()}
    adapter = model.adapters['source'].weight.detach().clone()
    objective(model, item, cfg)[1].backward()
    replay_step(model, optimizer, sam, [item], cfg, torch.device('cpu'))
    assert all(torch.equal(initial[n], p) for n,p in model.visual.named_parameters())
    assert not torch.equal(adapter, model.adapters['source'].weight)
    assert sam.steps == 1
