import importlib.util
import json
from pathlib import Path

import pytest
import torch

from tactilelatent.alignment_model import JointSensorAlignment, model_from_state, MODEL_TYPE
from tactilelatent.frozen_probe import latent_draw
from tactilelatent.model import EgoTouchLatentModel
from tactilelatent.train_alignment import alignment_loss, sam_replay
from tactilelatent.train_invariance import check_config
from tactilelatent.sam import SAMStep, rng_snapshot


def config():
    return dict(sensor_code_dims={'egotouch': 3}, numeric_shapes={'ftp_rdp': [1, 1, 1]},
        statistics={'egotouch': {'mean': [0.] * 3, 'std': [1.] * 3}},
        input_dim=8, width=16, heads=4, temporal_layers=1, cross_view_layers=1,
        latent_dim=16, adapter_hidden_dims=[8], dropout=0., stochastic=False, latent_mode='deterministic')


def inputs():
    return (torch.randn(2, 3, 3, 4, 8), torch.ones(2, 3, 3, dtype=torch.bool),
            {'code_target': torch.randn(2, 3, 3), 'code_valid': torch.ones(2, 3, dtype=torch.bool)})


def test_no_variance_head_noise_or_kl_and_mean_projection_gets_gradients():
    model = JointSensorAlignment(**config()).train()
    features, mask, batch = inputs()
    assert not hasattr(model.visual, 'prior_logvar')
    assert not any('prior_logvar' in name for name in model.state_dict())
    before = torch.get_rng_state()
    a = model(features, mask, 'egotouch', sample_latent=True)
    assert torch.equal(before, torch.get_rng_state())
    b = model(features, mask, 'egotouch')
    torch.testing.assert_close(a['sensor_code'], b['sensor_code'], atol=0, rtol=0)
    assert torch.isneginf(a['logvar']).all()
    torch.testing.assert_close(latent_draw(a['mu'], a['logvar']), a['mu'], atol=0, rtol=0)
    loss, task, kl = alignment_loss(a, batch, 0.)
    assert torch.isfinite(loss) and kl == 0 and loss == task
    loss.backward()
    assert model.visual.prior_mu.weight.grad.abs().sum() > 0
    with pytest.raises(ValueError, match='KL penalty'):
        alignment_loss(b, batch, .001)
    numeric = model(features, mask, 'ftp_rdp')['pred']
    assert torch.isfinite(numeric).all() and ((numeric >= 0) & (numeric <= 1)).all()


def test_deterministic_checkpoint_roundtrip_and_sam_replay():
    model = JointSensorAlignment(**config()); features, mask, batch = inputs()
    state = {'model_type': MODEL_TYPE, 'model_config': config(), 'model': model.state_dict()}
    restored = model_from_state(state)
    torch.testing.assert_close(model(features, mask, 'egotouch')['sensor_code'],
                               restored(features, mask, 'egotouch')['sensor_code'], atol=0, rtol=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    sam = SAMStep(optimizer)
    rng = rng_snapshot()
    output = model(features, mask, 'egotouch'); loss, _, _ = alignment_loss(output, batch, 0.)
    loss.backward()
    after = torch.get_rng_state()
    sam_replay(model, optimizer, sam, [{'features': features, 'mask': mask, 'source': 'egotouch',
        'batch': batch, 'rng': rng, 'beta': 0.}], torch.device('cpu'))
    assert torch.equal(after, torch.get_rng_state()) and sam.steps == 1


def test_fresh_config_rejects_parent_sampling_and_kl():
    cfg = json.loads(Path('configs/joint_16d_deterministic_control.json').read_text())
    check_config(cfg)
    assert cfg['parent_epochs'] == 0 and cfg['epochs'] == 20 and cfg['init_checkpoint'] is None
    for changed in ({'stochastic': True}, {'beta_max': .001}, {'init_checkpoint': 'old.pt'},
                    {'latent_objective': 'vae'}):
        with pytest.raises(ValueError):
            check_config({**cfg, **changed})


def test_gaussian_checkpoint_encoder_is_exactly_unchanged_by_context_refactor():
    path = Path('processing/runs/invariance_16d_v1/stopped_code_snapshot/model.py')
    if not path.exists():
        pytest.skip('Local archived Gaussian implementation unavailable')
    spec = importlib.util.spec_from_file_location('legacy_gaussian_model', path)
    archived = importlib.util.module_from_spec(spec); spec.loader.exec_module(archived)
    kwargs = dict(input_dim=8, width=16, heads=4, temporal_layers=1, cross_view_layers=1,
                  latent_dim=16, dropout=0., stochastic=True)
    old = archived.EgoTouchLatentModel(**kwargs).eval()
    new = EgoTouchLatentModel(**kwargs).eval();new.load_state_dict(old.state_dict(), strict=True)
    features, mask, _ = inputs()
    for a,b in zip(old.encode(features, mask),new.encode(features, mask)):
        torch.testing.assert_close(a,b,atol=0,rtol=0)
