"""Saved transfer heads must match their actual encoder latent dimension."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tactilelatent.frozen_probe import head


spec = importlib.util.spec_from_file_location('checkpoint_sweep',
    Path(__file__).parents[1] / 'scripts/evaluate_alignment_checkpoint_sweep.py')
sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)


@pytest.mark.parametrize('latent_dim', [16, 32])
@pytest.mark.parametrize('name', ['shallow', 'deep', 'deeper', 'deepest'])
def test_reuse_requires_matching_latent_dimension_and_weights(tmp_path, latent_dim, name):
    snapshot = tmp_path / 'checkpoint.pt'
    output = tmp_path / 'head'; output.mkdir()
    args = SimpleNamespace(seed=45, head_epochs=100, patience=10, samples=64)
    result = {'checkpoint_sha256': 'checkpoint', 'manifest_sha256': 'manifest',
        'checkpoint_epoch': 9, 'source': 'ftp_motiontrans', 'frozen_encoder': True,
        'head': '-'.join(map(str, [latent_dim, *sweep.HEADS[name], 6])), 'seed': 45, 'target_shape': [6, 1, 1],
        'selected_epochs': 5, 'validation': {'sample_count': 64}}
    saved = {'checkpoint_sha256': 'checkpoint', 'selected_epochs': 5,
        'head': head(latent_dim, 6, hidden_dims=sweep.HEADS[name]).state_dict(),
        'config': {'checkpoint': str(snapshot), 'hidden_dims': sweep.HEADS[name],
                   'epochs': 100, 'patience': 10, 'samples': 64, 'seed': 45}}
    (output / 'results.json').write_text(json.dumps(result))
    torch.save(saved, output / 'decoder.pt')
    assert sweep.completed_result(output, snapshot, 'checkpoint', 'manifest', 10,
                                  name, args, latent_dim) == result
    other_dim = 32 if latent_dim == 16 else 16
    with pytest.raises(ValueError, match='Incompatible'):
        sweep.completed_result(output, snapshot, 'checkpoint', 'manifest', 10,
                               name, args, other_dim)
    saved['head'] = head(other_dim, 6, hidden_dims=sweep.HEADS[name]).state_dict()
    torch.save(saved, output / 'decoder.pt')
    with pytest.raises(ValueError, match='structure differs'):
        sweep.completed_result(output, snapshot, 'checkpoint', 'manifest', 10,
                               name, args, latent_dim)
