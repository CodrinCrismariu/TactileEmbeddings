import torch
from torch import nn

import tactilelatent.paired_probe_cache as paired
import tactilelatent.frozen_probe as probe


def test_shared_dino_cache_matches_existing_probe_format_and_values(tmp_path, monkeypatch):
    class Dataset(torch.utils.data.Dataset):
        def __len__(self): return 4
        def __getitem__(self, index):
            rng = torch.Generator().manual_seed(index)
            return {'rgb': torch.rand(2, 3, 3, 2, 2, generator=rng),
                    'view_mask': torch.ones(2, 3, dtype=torch.bool), 'source': 'ftp_motiontrans',
                    'episode_id': f'ep{index}', 'target': torch.rand(2, 6, 1, 1, generator=rng),
                    'target_mask': torch.ones(2, 6, 1, 1, dtype=torch.bool)}
    class Backbone(nn.Module):
        calls = 0
        def forward(self, rgb):
            self.calls += 1
            return rgb.mean((-2, -1)).unsqueeze(1)
    class Visual(nn.Module):
        def __init__(self):
            super().__init__(); self.project = nn.Linear(3, 4)
        def encode(self, x, mask):
            mu = self.project(x.mean((2, 3)))
            return mu, mu * .1
    class Model(nn.Module):
        def __init__(self):
            super().__init__(); self.visual = Visual()
    monkeypatch.setattr(paired, 'evaluation_dataset', lambda *a, **kw: Dataset())
    monkeypatch.setattr(probe, 'evaluation_dataset', lambda *a, **kw: Dataset())
    models = {'a': Model().eval(), 'b': Model().eval()}
    indices = [2, 0, 3]
    destinations = {name: tmp_path / f'paired_{name}.pt' for name in models}
    hashes = {'a': 'hash-a', 'b': 'hash-b'}
    backbone = Backbone()
    paired.extract_split(backbone, models, 'fixture', 'ftp_motiontrans', 'train', indices,
                         destinations, hashes, 'manifest-hash', torch.device('cpu'), batch_size=2, workers=0)
    shared_calls = backbone.calls
    for name, model in models.items():
        expected = probe.cached_latents(backbone, model, 'fixture', 'ftp_motiontrans', 'train', indices,
            tmp_path / f'original_{name}.pt', hashes[name], 'manifest-hash', torch.device('cpu'),
            with_targets=True, batch_size=2, workers=0)
        actual = torch.load(destinations[name], weights_only=False)
        assert actual['signature'] == expected['signature']
        assert actual['episode_id'] == ['ep2', 'ep0', 'ep3'] == expected['episode_id']
        for key in ('mu', 'logvar', 'target', 'valid'):
            torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
    assert backbone.calls == shared_calls * 3
    before = backbone.calls
    paired.extract_split(backbone, models, 'fixture', 'ftp_motiontrans', 'train', indices,
                         destinations, hashes, 'manifest-hash', torch.device('cpu'), batch_size=2, workers=0)
    assert backbone.calls == before


def test_paired_episode_bootstrap_preserves_pairing_and_checks_counts():
    import runpy
    from pathlib import Path
    import pytest
    difference = runpy.run_path(str(Path('scripts/evaluate_invariance_holdout.py')))['paired_difference']
    a = {'validation': {'episode_errors': [
        {'episode_id': 'a', 'count': 6, 'absolute_sum': .6},
        {'episode_id': 'b', 'count': 12, 'absolute_sum': 1.2}]}}
    b = {'validation': {'episode_errors': [
        {'episode_id': 'b', 'count': 12, 'absolute_sum': 1.44},
        {'episode_id': 'a', 'count': 6, 'absolute_sum': .72}]}}
    result = difference(a, b)
    assert result['relational_minus_control_mae'] == pytest.approx(.02)
    assert result['paired_episode_bootstrap_ci95'] == pytest.approx([.02, .02])
    b['validation']['episode_errors'][0]['count'] = 13
    with pytest.raises(ValueError, match='pairing'):
        difference(a, b)


def test_explicit_viewer_snapshot_selection_when_full_checkpoint_exists(tmp_path):
    import runpy
    from pathlib import Path
    choose = runpy.run_path(str(Path('scripts/evaluate_alignment_checkpoint_sweep.py')))['choose_checkpoint']
    viewer = tmp_path / 'viewer_epochs/epoch_020.pt'
    viewer.parent.mkdir(); viewer.touch()
    assert choose(tmp_path, 20) == viewer
    full = tmp_path / 'epoch_020.pt'; full.touch()
    assert choose(tmp_path, 20) == full
    assert choose(tmp_path, 20, 'viewer') == viewer
