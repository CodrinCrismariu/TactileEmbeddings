import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.func import functional_call

from tactilelatent.relational import relational_loss, read_floors, extra_loss
from tactilelatent.meta_learning import adapt_head, make_head, target_loss, TaskSampler, meta_loss
from tactilelatent.trajectory_matching import match_trajectories, transport, trajectory_loss
from tactilelatent.train_invariance import check_config, replay_step
from tactilelatent.sam import SAMStep, rng_snapshot


def test_relational_rotation_scale_translation_and_missing_frames():
    torch.manual_seed(3)
    code = torch.randn(2, 8, 3, requires_grad=True)
    rotation, _ = torch.linalg.qr(torch.randn(3, 3))
    mu = (4 * code.detach() @ rotation + 7).requires_grad_()
    valid = torch.ones(2, 8, dtype=torch.bool)
    valid[0, 2] = False
    with torch.no_grad():
        code[0, 2] = float('nan')
    loss, stats = relational_loss(mu, code, valid, .001)
    assert loss < 1e-10 and stats['relational_clips'] == 2
    loss.backward()
    assert code.grad is None and torch.isfinite(mu.grad).all()
    mu2 = torch.randn(2, 8, 5, requires_grad=True)
    value, _ = relational_loss(mu2, code, valid, .001)
    value.backward()
    assert value > .01 and mu2.grad.abs().sum() > 0


def test_constant_invalid_and_subfloor_clips_do_not_amplify_noise():
    mu = torch.randn(3, 8, 4, requires_grad=True)
    code = torch.zeros(3, 8, 2)
    code[1] = torch.randn(8, 2) * 1e-8
    code[2] = float('nan')
    loss, stats = relational_loss(mu, code, torch.ones(3, 8, dtype=torch.bool), .01)
    loss.backward()
    assert loss == 0 and stats['relational_skipped'] == 3
    assert torch.equal(mu.grad, torch.zeros_like(mu))
    with pytest.raises(ValueError):
        relational_loss(mu, code, torch.ones(3, 8, dtype=torch.bool), 0.)


def test_floors_reject_stale_heldout_or_incomplete_artifacts(tmp_path):
    data, caches = {'hash': 'x'}, {'a': {'hash': 'c'}}
    artifact = dict(schema_version=1, data=data, cache_signatures=caches, split='train',
                    sources={'a': {'distance_floor': .1}})
    path = tmp_path / 'floors.json'
    path.write_text(json.dumps(artifact))
    assert read_floors(path, data, caches) == {'a': .1}
    for modified in ({**artifact, 'split': 'val'}, {**artifact, 'data': {}},
                     {**artifact, 'sources': {}},
                     {**artifact, 'sources': {'ftp_motiontrans': {'distance_floor': .1}}}):
        path.write_text(json.dumps(modified))
        with pytest.raises(ValueError):
            read_floors(path, data, caches)


def test_meta_gradient_includes_support_adaptation_finite_difference():
    torch.manual_seed(1)
    head = make_head(2, 1, (3,), dtype=torch.float64)
    sx = torch.randn(4, 2, dtype=torch.float64)
    qx = torch.randn(3, 2, dtype=torch.float64)
    sy, qy = torch.randn(4, 1, dtype=torch.float64), torch.randn(3, 1, dtype=torch.float64)
    sm, qm = torch.ones_like(sy, dtype=torch.bool), torch.ones_like(qy, dtype=torch.bool)
    def objective(theta, detach_support=False):
        support = sx @ theta
        weights = adapt_head(head, support.detach() if detach_support else support, sy, sm, 2, .1)
        return target_loss(functional_call(head, weights, (qx @ theta,)), qy, qm)
    theta = torch.eye(2, dtype=torch.float64, requires_grad=True)
    derivative = torch.autograd.grad(objective(theta), theta)[0]
    numerical = torch.zeros_like(theta)
    for i in range(2):
        for j in range(2):
            delta = torch.zeros_like(theta); delta[i, j] = 1e-5
            numerical[i, j] = (objective(theta + delta) - objective(theta - delta)).detach() / 2e-5
    torch.testing.assert_close(derivative, numerical, rtol=1e-5, atol=1e-7)
    first_order = torch.autograd.grad(objective(theta, True), theta)[0]
    assert not torch.allclose(first_order, derivative, atol=1e-4)


def test_numeric_activation_and_code_head_unbounded():
    numeric = make_head(2, 1, (4,), numeric=True)
    code = make_head(2, 1, (4,), numeric=False)
    assert isinstance(numeric[-1], nn.Sigmoid)
    assert isinstance(code[-1], nn.Linear)
    with torch.no_grad():
        code[-1].weight.zero_(); code[-1].bias.fill_(-3)
    assert (code(torch.ones(2, 2)) < 0).all()
    with pytest.raises(ValueError, match='valid targets'):
        target_loss(torch.zeros(2, 1), torch.zeros(2, 1), torch.zeros(2, 1, dtype=torch.bool))


def task_dataset():
    episodes, windows, indices = [], [], {}
    for source in ('a', 'b'):
        indices[source] = []
        for e in range(8):
            ep = len(episodes)
            episodes.append(dict(source=source, episode_id=f'{source}{e}', split='train', group=f'{source}{e//2}'))
            for start in (0, 15):
                indices[source].append(len(windows)); windows.append((ep, start))
    return SimpleNamespace(episodes=episodes, windows=windows, indices_by_source=indices)


def test_meta_source_rotation_episode_group_disjointness_and_holdout():
    dataset = task_dataset()
    sampler = TaskSampler(dataset, 45, 2, 2, 'group')
    for number in range(6):
        task = sampler.sample(number)
        assert task == sampler.sample(number)
        assert task['source'] == ('a', 'b')[number % 2]
        assert not set(task['support_episodes']) & set(task['query_episodes'])
        assert not set(task['support_groups']) & set(task['query_groups'])
    dataset.episodes[0]['split'] = 'val'
    with pytest.raises(ValueError, match='training episodes'):
        TaskSampler(dataset)
    dataset = task_dataset(); dataset.indices_by_source['ftp_motiontrans'] = []
    with pytest.raises(ValueError, match='training episodes'):
        TaskSampler(dataset)


def test_transport_constant_unequal_sizes_and_unsupported_rows():
    plan = torch.rand(4, 7); plan[-1].zero_()
    result, valid = transport(plan, torch.full((7, 3), 5.))
    torch.testing.assert_close(result[:3], torch.full((3, 3), 5.))
    assert valid.tolist() == [True, True, True, False]


def distances(x):
    d = np.abs(x[:, None] - x[None, :])
    return d / np.sqrt(np.mean(d[~np.eye(len(x), dtype=bool)] ** 2))


def test_teacher_gw_ordered_and_reversed_and_ambiguous():
    a = distances(np.array([0., .1, .35, .8, 1.7, 3.4]))
    plan, stats = match_trajectories(a, a, epsilon=.08, max_cost=.5)
    assert plan is not None, stats
    assert stats['ordered'] and stats['reliability'] > 0
    reversed_plan, stats = match_trajectories(a, a[::-1, ::-1].copy(), epsilon=.08, max_cost=.5)
    assert reversed_plan is None and not stats.get('ordered', True)
    ambiguous = np.ones((6, 6)) - np.eye(6)
    plan, stats = match_trajectories(ambiguous, ambiguous)
    assert plan is None and stats['reason'] == 'ambiguous_or_poor_fit'


def test_teacher_solver_failure_and_invalid_input(monkeypatch):
    import ot
    monkeypatch.setattr(ot.gromov, 'entropic_gromov_wasserstein', lambda *a, **k: (np.zeros((3, 3)), {'err': [0.]}))
    plan, stats = match_trajectories(np.eye(3), np.eye(3))
    assert plan is None and stats['reason'] == 'solver_failure'
    with pytest.raises(ValueError):
        match_trajectories(np.full((3, 3), np.nan), np.eye(3))


def test_unrelated_outlier_trajectory_is_not_forced_to_match():
    a = distances(np.array([0., .1, .35, .8, 1.7, 3.4]))
    b = distances(np.array([0., .1, .2, .3, .4, 100.]))
    plan, stats = match_trajectories(a, b, epsilon=.2, max_cost=.001)
    assert plan is None


def test_round_trip_floor_native_shape_and_train_only():
    from tactilelatent.prepare_relational import estimate_source
    from tactilelatent.sensor_autoencoder import SensorAutoencoder
    ae = SensorAutoencoder([2, 21, 21], latent_dim=3, channels=[4]).eval()
    codes = torch.randn(20, 3)
    class Cache:
        index = {'statistics': {'std': [1., 1., 1.]}}
        def valid_frames(self, episode, frames):
            return np.ones(len(frames), dtype=bool)
        def read(self, episode, frames, normalized=True):
            return codes[frames], torch.ones(len(frames), dtype=torch.bool)
    row = dict(split='train', source='egotouch', episode_id='one', frames=20, target_shape=[2, 21, 21])
    result = estimate_source([row], Cache(), ae, 'cpu', episodes=1, frames=4)
    assert result['distance_floor'] > 0 and result['frames'] == 4
    assert result['samples'][0]['episode_id'] == 'one'
    with pytest.raises(ValueError, match='training episodes'):
        estimate_source([{**row, 'split': 'val'}], Cache(), ae, 'cpu')
    with pytest.raises(ValueError, match='native tactile shape'):
        estimate_source([{**row, 'target_shape': [2, 10, 10]}], Cache(), ae, 'cpu')


class TinyVisual(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean, self.logvar = nn.Linear(3, 2), nn.Linear(3, 2)
    def encode(self, x, mask):
        return self.mean(x), self.logvar(x).clamp(-8, 4)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = TinyVisual()
        self.adapter = nn.Linear(2, 3)
    def forward(self, x, mask, source):
        mu, logvar = self.visual.encode(x, mask)
        z = mu + torch.randn_like(mu) * (.5 * logvar).exp()
        return dict(mu=mu, logvar=logvar, sensor_code=self.adapter(z))


def test_meta_never_updates_persistent_adapter_and_sam_replay_preserves_rng():
    torch.manual_seed(4)
    model = TinyModel()
    prepared = {'task': {'source': 'a', 'number': 0}, 'numeric': False}
    for name in ('support', 'query'):
        prepared[name] = dict(features=torch.randn(2, 4, 3), mask=torch.ones(2, 4, dtype=torch.bool),
                              target=torch.randn(2, 4, 3), valid=torch.ones(2, 4, 3, dtype=torch.bool))
    cfg = {'meta': {'hidden_dims': [4], 'inner_steps': 2, 'inner_lr': .1, 'sample_latent': True}}
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    sam = SAMStep(optimizer)
    rng = rng_snapshot()
    loss, _ = meta_loss(model, prepared, cfg['meta']); loss.backward()
    assert model.adapter.weight.grad is None
    assert model.visual.mean.weight.grad.abs().sum() > 0
    after = rng_snapshot()
    replay_step(model, optimizer, sam, [{'kind': 'meta', 'prepared': prepared, 'rng': rng, 'scale': 1.}],
                cfg, {}, torch.device('cpu'))
    assert torch.equal(torch.get_rng_state(), after['cpu']) and not sam.originals


def test_trajectory_targets_detached_and_motiontrans_rejected():
    mu, target = torch.randn(1, 5, 3, requires_grad=True), torch.randn(3, 3, requires_grad=True)
    loss, _ = trajectory_loss(mu, [{'batch': 0, 'positions': torch.tensor([0, 2, 4]),
                                   'target': target, 'reliability': .5}])
    loss.backward()
    assert target.grad is None and mu.grad.abs().sum() > 0
    with pytest.raises(ValueError, match='MotionTrans'):
        extra_loss({'mu': mu}, {}, 'ftp_motiontrans', {}, {}, 0)


def test_trajectory_refresh_uses_teacher_plans_and_builds_runnable_targets(tmp_path, monkeypatch):
    import tactilelatent.trajectory_matching as module
    class Dataset(torch.utils.data.Dataset):
        episodes = [{'source': 'a', 'split': 'train'}, {'source': 'b', 'split': 'train'}]
        indices_by_source = {'a': [0], 'b': [1]}
        def __len__(self):
            return 2
        def __getitem__(self, index):
            x = torch.tensor([0., .1, .35, .8, 1.7, 3.4])[:, None].expand(-1, 3).clone()
            return {'source': ('a', 'b')[index], 'episode_id': str(index), 'frame_indices': torch.arange(6),
                    'value': x * (index + 1), 'valid': torch.ones(6, dtype=torch.bool),
                    'input': x, 'view_mask': torch.ones(6, 3, dtype=torch.bool)}
    dataset = Dataset()
    monkeypatch.setattr(module.CodeOnlyDataset, '__getitem__', lambda self, index: dataset[index])
    monkeypatch.setattr(module, 'features', lambda backbone, original, device: original['input'].to(device))
    cfg = {'seed': 45, 'trajectory': {'weight': .1, 'clips_per_source': 1, 'candidates': 1, 'epsilon': .08, 'max_cost': .5}}
    model = TinyModel()
    objective = module.TrajectoryObjective(cfg, dataset, {'a': .01, 'b': .01}, torch.device('cpu'), tmp_path)
    result = objective.refresh(model, None, dataset.indices_by_source, 0)
    assert result['accepted_pairs'] == 1 and model.training
    snapshot = torch.load(tmp_path / 'trajectory/epoch_001.pt', weights_only=False)
    payload = objective.payload({'window_index': torch.tensor([0]), 'alignment_anchor': torch.tensor([True])})
    mu = torch.randn(1, 6, 2, requires_grad=True)
    loss, metrics = trajectory_loss(mu, payload); loss.backward()
    assert metrics['trajectory_matched_clips'] == 1 and mu.grad.abs().sum() > 0
    with torch.no_grad():
        model.visual.mean.weight.add_(3)
    objective.refresh(model, None, dataset.indices_by_source, 0)
    changed = torch.load(tmp_path / 'trajectory/epoch_001.pt', weights_only=False)
    torch.testing.assert_close(snapshot['pairs'][0][2], changed['pairs'][0][2])
    assert not torch.equal(snapshot['targets'][0][0]['target'], changed['targets'][0][0]['target'])


def test_checked_in_configs_keep_architecture_and_distinct_runs():
    configs = [json.loads(p.read_text()) for p in Path('configs').glob('invariance_16d_*.json')]
    assert len(configs) == 5
    for cfg in configs:
        check_config(cfg)
        assert cfg['epochs'] - cfg['parent_epochs'] == 10
        assert not cfg.get('private_dim')
    assert len({c['output'] for c in configs}) == len(configs)


@pytest.mark.parametrize('latent_mode', ['gaussian', 'deterministic'])
def test_trainer_resume_matches_uninterrupted_with_relational_meta_and_sam(tmp_path, monkeypatch, latent_mode):
    """Exercise real model/optimizer/checkpoints without downloading a backbone."""
    import sys
    import tactilelatent.train_invariance as trainer
    from tactilelatent.alignment_model import JointSensorAlignment, model_from_state
    sources = ['egotouch', 'opengraph']
    class Dataset(torch.utils.data.Dataset):
        def __init__(self, manifest, split, selected, **kwargs):
            self.episodes, self.windows, self.indices_by_source = [], [], {}
            for source in sources:
                self.indices_by_source[source] = []
                for ep in range(4):
                    self.windows.append((len(self.episodes), 0))
                    self.episodes.append(dict(source=source, episode_id=f'{source}{ep}', split=split))
                    self.indices_by_source[source].append(len(self.windows) - 1)
        def __len__(self):
            return len(self.windows)
        def __getitem__(self, key):
            index, anchor = key if isinstance(key, tuple) else (key, False)
            row = self.episodes[index]
            rng = torch.Generator().manual_seed(index + 21)
            return dict(rgb=torch.randn(3, 3, 3, 2, 2, generator=rng),
                view_mask=torch.ones(3, 3, dtype=torch.bool), source=row['source'], episode_id=row['episode_id'],
                frame_indices=torch.arange(3), code_target=torch.randn(3, 3, generator=rng),
                code_valid=torch.ones(3, dtype=torch.bool), window_index=index, alignment_anchor=anchor)
    class Backbone(nn.Module):
        def forward(self, rgb):
            return rgb.mean((-1, -2)).unsqueeze(1)
    config = dict(input_dim=3, width=8, heads=2, temporal_layers=1, cross_view_layers=1,
                  latent_dim=16, adapter_hidden_dims=[8], dropout=.1, stochastic=True,
                  sensor_code_dims={s: 3 for s in sources}, numeric_shapes={},
                  statistics={s: {'mean': [0.] * 3, 'std': [1.] * 3} for s in sources})
    parent = {'model': copy.deepcopy(JointSensorAlignment(**config).state_dict())}
    cfg = json.loads(Path('configs/invariance_16d_meta.json').read_text())
    cfg.update(parent_epochs=1, epochs=3, warmup_epochs=1, batch_size=1, accumulation=2,
               workers=0, train_windows_per_source_per_epoch=4, validation_windows_per_source=1)
    cfg['meta'].update(support_episodes=1, query_episodes=1, every_updates=2, hidden_dims=[4], inner_steps=2)
    if latent_mode == 'deterministic':
        config.update(latent_mode='deterministic', stochastic=False)
        cfg.update(latent_mode='deterministic', stochastic=False, beta_max=0.,
                   latent_objective='deterministic', parent_epochs=0, epochs=2, init_checkpoint=None, parent_sha256=None)
        parent = None
    cfg['matched_bank_sampling']['bank_size'] = 4
    provenance = {'data': {}, 'parent_sha256': 'fixture'}
    monkeypatch.setattr(trainer, 'prepare', lambda cfg: (sources, {'sources': {}}, config, provenance, {s: .01 for s in sources}, parent))
    monkeypatch.setattr(trainer, 'BankDataset', Dataset)
    monkeypatch.setattr(trainer, 'AlignedClipDataset', Dataset)
    monkeypatch.setattr(trainer, 'DinoBackbone', Backbone)
    monkeypatch.setattr(trainer, 'sample_banks', lambda ds, *args: dict(ds.indices_by_source))
    validation_calls = [0]
    def validation(*args):
        validation_calls[0] += 1
        return {s: {'supervision_loss': .5} for s in sources}
    monkeypatch.setattr(trainer, 'validate', validation)
    def run(folder, resume=False):
        cfg['output'] = str(tmp_path / folder)
        path = tmp_path / f'{folder}.json'; path.write_text(json.dumps(cfg))
        monkeypatch.setattr(sys, 'argv', ['train', '--config', str(path), '--device', 'cpu'] + (['--resume'] if resume else []))
        trainer.main()
    run('reference')
    def interrupted(*args):
        validation_calls[0] += 1
        if validation_calls[0] == 4:
            raise RuntimeError('simulated failure before second checkpoint')
        return {s: {'supervision_loss': .5} for s in sources}
    monkeypatch.setattr(trainer, 'validate', interrupted)
    with pytest.raises(RuntimeError, match='simulated failure'):
        run('resumed')
    monkeypatch.setattr(trainer, 'validate', validation)
    run('resumed', True)
    a = torch.load(tmp_path / 'reference/last.pt', weights_only=False)
    b = torch.load(tmp_path / 'resumed/last.pt', weights_only=False)
    for key in a['model']:
        torch.testing.assert_close(a['model'][key], b['model'][key], atol=0, rtol=0)
    assert a['meta_task_number'] == b['meta_task_number'] == 4
    assert a['updates'] == b['updates'] == 8
    assert len((tmp_path / 'resumed/meta_tasks.jsonl').read_text().splitlines()) == 4
    assert a['model_type'] == 'joint_sensor_alignment_v1'
    assert (tmp_path / f"resumed/viewer_epochs/epoch_{cfg['epochs']:03d}.pt").exists()
    loaded = model_from_state(b)
    assert loaded.visual.prior_mu.out_features == 16
