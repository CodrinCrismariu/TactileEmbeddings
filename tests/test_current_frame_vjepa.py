from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tactilelatent.current_frame import CurrentFrameCodeClips, causal_clip_indices
from tactilelatent.marker_code_model import MarkerCodeVJEPA, model_from_state, MODEL_TYPE
from tactilelatent.sam import SAMStep, rng_snapshot
from tactilelatent.train_alignment import alignment_loss
from tactilelatent.train_vjepa_markers import replay_update


def test_causal_history_has_exact_current_frame_and_never_future_input():
    for frame in (0, 1, 7, 14, 63):
        indices, position = causal_clip_indices(frame, 64)
        assert len(indices) == 8 and indices[-1] == frame and indices.max() <= frame
        assert (np.diff(indices) >= 0).all() and position == 0
    np.testing.assert_array_equal(causal_clip_indices(63, 64)[0], np.arange(49, 64, 2))
    np.testing.assert_array_equal(causal_clip_indices(0, 64)[0], np.zeros(8, dtype=int))
    with pytest.raises(ValueError):
        causal_clip_indices(64, 64)


def test_single_128d_query_preserves_spatial_temporal_sensitivity_and_sam_resume():
    torch.set_num_threads(2); torch.manual_seed(45)
    config = dict(sensor_code_dims={'ftp_unit': 4}, numeric_shapes={'ftp_rdp': [1, 1, 1]},
        statistics={'ftp_unit': dict(mean=[0.]*4, std=[1.]*4)}, latent_dim=128,
        input_dim=12, width=16, heads=4, layers=2, feedforward=32, grid_size=2,
        dropout=0., prediction_mode='current_frame')
    model = MarkerCodeVJEPA(**config).eval()
    features = torch.randn(2, 4, 3, 4, 12)
    mask = torch.ones(2, 8, 3, dtype=torch.bool); mask[:, :, 1:] = False
    original = model(features, mask, 'ftp_unit')
    assert model.visual.frame_queries.shape == (1, 16)
    assert original['mu'].shape == (2, 1, 128) and original['sensor_code'].shape == (2, 1, 4)
    assert torch.equal(original['mu'], model(features, mask, 'ftp_unit')['mu'])
    assert torch.isneginf(original['logvar']).all()
    changed = features.clone(); changed[:, :, 1:] = float('nan')
    torch.testing.assert_close(original['mu'], model(changed, mask, 'ftp_unit')['mu'], rtol=0, atol=0)
    # Exact mean is unchanged: position-sensitive attention must distinguish it.
    assert not torch.allclose(original['mu'], model(features.flip(3), mask, 'ftp_unit')['mu'])
    assert not torch.allclose(original['mu'], model(features.flip(1), mask, 'ftp_unit')['mu'])
    batch = dict(source=['ftp_unit']*2, view_mask=mask,
        code_target=torch.randn(2, 1, 4), code_valid=torch.ones(2, 1, dtype=torch.bool))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001); sam = SAMStep(optimizer, .05)
    before = rng_snapshot(); loss, _, kl = alignment_loss(original, batch, 0.)
    assert kl == 0
    loss.backward(); assert model.visual.frame_queries.grad.abs().sum() > 0
    replay_update(model, optimizer, sam, [dict(features=features, batch=batch, rng=before)], torch.device('cpu'))
    assert sam.steps == 1
    restored = model_from_state(dict(model_type=MODEL_TYPE, model_config=config, model=model.state_dict())).eval()
    torch.testing.assert_close(model(features, mask, 'ftp_unit')['mu'], restored(features, mask, 'ftp_unit')['mu'], rtol=0, atol=0)
    assert restored(features, mask, 'ftp_rdp')['pred'].shape == (2, 1, 1, 1, 1)


def test_only_current_sensor_targets_are_loaded_with_full_rgb_history(monkeypatch):
    import tactilelatent.current_frame as module
    reads = []
    class Codes:
        def valid_frames(self, episode, indices):
            return np.asarray(indices) != 14  # History is valid, current is not.
        def read(self, episode, indices):
            reads.append(('code', indices.copy()))
            return torch.zeros(len(indices), 4), torch.ones(len(indices), dtype=torch.bool)
    class Markers:
        def read(self, row, indices):
            reads.append(('marker', indices.copy()))
            return dict(marker_target=torch.zeros(len(indices), 9, 2),
                marker_valid=torch.ones(len(indices), 9, dtype=torch.bool),
                marker_reference=torch.ones(9, 2), marker_surface=torch.zeros(9, dtype=torch.long),
                marker_spacing=torch.ones(9))
    rows = [dict(source=s, episode_id=s, frames=40) for s in ('ftp_unit', 'ftp_rdp')]
    dataset = SimpleNamespace(dataset=SimpleNamespace(span=15, stride=2, rgb_size=384),
        episodes=rows, windows=[(0, 0), (0, 15), (1, 0)],
        indices_by_source={'ftp_unit': [0, 1], 'ftp_rdp': [2]},
        code_caches={'ftp_unit': Codes()}, cache=Markers())
    def rgb(row, indices, size):
        reads.append(('rgb', indices.copy()))
        return torch.zeros(len(indices), 3, 3, 4, 4), torch.ones(len(indices), 3, dtype=torch.bool)
    def targets(row, indices):
        reads.append(('target', indices.copy()))
        return np.zeros((len(indices), 1, 1, 1), np.float32)
    monkeypatch.setattr(module, 'scene_rgb', rgb); monkeypatch.setattr(module, 'targets', targets)
    current = CurrentFrameCodeClips(dataset)
    assert len(current) == 2  # No supervision for the unsupported current label.
    marker = current[0]
    assert marker['rgb'].shape[0] == 8 and marker['marker_target'].shape[0] == 1
    assert marker['code_target'].shape == (1, 4)
    assert marker['frame_indices'].tolist() == [29]
    assert marker['history_frame_indices'].tolist() == list(range(15, 30, 2))
    for kind, indices in reads:
        np.testing.assert_array_equal(indices, np.arange(15, 30, 2) if kind == 'rgb' else [29])
    reads.clear(); numeric = current[1]
    assert numeric['target'].shape == (1, 1, 1, 1)
    np.testing.assert_array_equal(reads[-1][1], [14])
