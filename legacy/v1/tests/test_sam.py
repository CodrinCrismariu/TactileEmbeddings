import pytest
import torch
from tactilelatent.sam import SAMStep, rng_snapshot, restore_rng
from tactilelatent.train_joint import joint_loss, sam_replay_step


def test_sam_direction_radius_exact_restore_and_update():
    p = torch.nn.Parameter(torch.tensor([3., 4.]))
    frozen = torch.nn.Parameter(torch.tensor([2.]), requires_grad=False)
    opt = torch.optim.SGD([p, frozen], lr=.1)
    sam = SAMStep(opt, rho=.5)
    (p.square().sum() / 2).backward()
    original = p.detach().clone()
    assert sam.perturb() == 5.
    assert torch.allclose(p, torch.tensor([3.3, 4.4]))
    assert torch.allclose((p - original).norm(), torch.tensor(.5))
    assert frozen.item() == 2.
    (p.square().sum() / 2).backward()
    sam.restore()
    assert torch.equal(p, original)
    opt.step()
    assert torch.allclose(p, torch.tensor([2.67, 3.56]))


def test_rng_replay_and_sam_state():
    state = rng_snapshot()
    first = torch.randn(8)
    restore_rng(state)
    assert torch.equal(first, torch.randn(8))
    p = torch.nn.Parameter(torch.ones(1))
    sam = SAMStep(torch.optim.AdamW([p]))
    sam.steps = 7
    restored = SAMStep(torch.optim.AdamW([p]))
    restored.load_state_dict(sam.state_dict())
    assert restored.steps == 7
    p.sum().backward()
    sam.perturb()
    with pytest.raises(RuntimeError):
        sam.state_dict()
    sam.restore()


def test_sam_finite_gradient_guard():
    p = torch.nn.Parameter(torch.ones(1))
    sam = SAMStep(torch.optim.AdamW([p]))
    p.grad = torch.tensor([float('nan')])
    with pytest.raises(FloatingPointError):
        sam.perturb()
    assert p.item() == 1. and not sam.originals


def test_accumulated_replay_stochasticity_and_exact_resume():
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(4, 2)
            self.dropout = torch.nn.Dropout(.2)

        def forward(self, features, mask, source):
            mu = self.layer(self.dropout(features))
            pred = (mu + .1 * torch.randn_like(mu)).sigmoid()
            return {"pred": pred, "mu": mu, "logvar": torch.zeros_like(mu)}

    def step(model, opt, sam):
        opt.zero_grad(set_to_none=True)
        replay = []
        for value in (1., 2.):
            x = torch.full((2, 4), value)
            y = torch.zeros(2, 2)
            valid = torch.ones_like(y, dtype=torch.bool)
            state = rng_snapshot()
            result = model(x, None, 'test')
            (joint_loss(result, y, valid, 'test', .001) / 2).backward()
            replay.append({'features': x, 'target': y, 'valid': valid,
                           'rng': state, 'mask': None, 'source': 'test', 'beta': .001})
        after = rng_snapshot()
        diagnostics = sam_replay_step(model, opt, sam, replay, 2, torch.device('cpu'))
        assert torch.equal(after['cpu'], torch.get_rng_state())
        assert diagnostics['sam_first_gradient_norm'] > 0

    import copy
    torch.manual_seed(987)
    model = Tiny()
    opt = torch.optim.AdamW(model.parameters(), lr=.01)
    sam = SAMStep(opt)
    step(model, opt, sam)
    saved = copy.deepcopy({'model': model.state_dict(), 'optimizer': opt.state_dict(),
                           'sam': sam.state_dict(), 'rng': rng_snapshot()})
    step(model, opt, sam)
    expected = copy.deepcopy(model.state_dict())
    model2 = Tiny()
    opt2 = torch.optim.AdamW(model2.parameters(), lr=.01)
    sam2 = SAMStep(opt2)
    model2.load_state_dict(saved['model'])
    opt2.load_state_dict(saved['optimizer'])
    sam2.load_state_dict(saved['sam'])
    restore_rng(saved['rng'])
    step(model2, opt2, sam2)
    assert all(torch.equal(value, model2.state_dict()[key]) for key, value in expected.items())
    assert sam2.steps == sam.steps == 2
