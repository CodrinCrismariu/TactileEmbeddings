"""Attention must inspect the exact scene clip used for native prediction."""
from collections import OrderedDict
from io import BytesIO
import threading
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from tactilelatent import acquisition_scene_viewer as scenes
from tactilelatent.autoencoder_viewer import Viewer
from tactilelatent.current_frame import causal_clip_indices
from tactilelatent.vjepa_spatial import SpatialFrameEncoder


@pytest.fixture
def decoder(monkeypatch):
    calls = []

    def decode(path, indices, size):
        calls.append((path, list(indices)))
        return torch.tensor(np.asarray(indices) % 200 / 200, dtype=torch.float32)[:, None, None, None].expand(-1, 3, size, size)

    monkeypatch.setattr(scenes, 'video_frames', decode)
    return calls


def native_row():
    return dict(source='trex', frames=20, targets=dict(path='native.npz', sha256='pinned', shape=[20, 60]),
                views={'0': dict(path='shard.mp4', role='ego', frame_offset=1000, frames=20)})


def test_expansion_rgb_keeps_history_camera_masks_and_boundary_padding(decoder):
    row = dict(source='baihu', frames=20, views={'1': 'left.mp4', '2': 'right.mp4'})
    indices, _ = causal_clip_indices(6, 20)
    rgb, mask = scenes.mapped_scene_rgb(row, indices, 'all', size=8)
    assert rgb.shape == (8, 3, 3, 8, 8)
    assert not mask[:, 0].any() and mask[:, 1:].all()
    torch.testing.assert_close(rgb[:, 1, 0, 0, 0], torch.tensor(indices / 200, dtype=torch.float32))
    assert decoder == [('left.mp4', indices.tolist()), ('right.mp4', indices.tolist())]
    with pytest.raises(ValueError, match='no ego'):
        scenes.mapped_scene_rgb(row, indices, 'ego', size=8)


def test_shard_offsets_and_explicit_maps_are_shared_with_overlay_images(monkeypatch, decoder):
    data = dict(timestamps=np.arange(20) / 30)
    monkeypatch.setattr(scenes, '_targets', lambda *args: data)
    row = native_row()
    indices, _ = causal_clip_indices(18, 20)
    scenes.mapped_scene_rgb(row, indices, 'ego', size=8)
    assert decoder[-1][1] == (1000 + indices).tolist()
    scenes.mapped_scene_rgb(row, [18], 'ego', size=8)
    assert decoder[-1][1] == [1018]
    data.update(rgb_frame_indices=2 * np.arange(20) + 7, rgb_frame_valid=np.ones(20, bool))
    row['views']['0'] = dict(path='mapped.mp4', role='fixed_scene', frames=50,
                            frame_indices_key='rgb_frame_indices', frame_indices_artifact='native.npz')
    scenes.mapped_scene_rgb(row, indices, 'all', size=8)
    assert decoder[-1][1] == data['rgb_frame_indices'][indices].tolist()
    scenes.mapped_scene_rgb(row, [18], 'all', size=8)
    assert decoder[-1][1] == [43]
    data['rgb_frame_valid'][10] = False
    with pytest.raises(ValueError, match='missing|ambiguous'):
        scenes.mapped_scene_rgb(row, indices, 'all', size=8)
    data['rgb_frame_valid'][10] = True
    data['rgb_frame_indices'][10] = data['rgb_frame_indices'][8]
    with pytest.raises(ValueError, match='Nonchronological'):
        scenes.mapped_scene_rgb(row, indices, 'all', size=8)
    row['views']['0']['frame_indices_artifact'] = 'untrusted.npz'
    with pytest.raises(ValueError, match='external'):
        scenes.mapped_scene_rgb(row, [18], 'all', size=8)


def test_feel_layout_aliases_and_attention_catalog_do_not_advertise_untrained_heads():
    row = dict(source='feel', targets=dict(shape=[20, 5]))
    assert scenes.joint_row(row)['source'] == 'feel_lab' and row['source'] == 'feel'
    row['targets']['shape'] = [20, 12]
    assert scenes.joint_row(row)['source'] == 'feel_kitchen'
    sources = ['egopressure', 'baihu', 'feel_lab', 'feel_kitchen', 'hrdexdb_allegro_v5', 'hrdexdb_inspire_f1', 'trex']
    models = [dict(attention_supported=True, source_shapes={s: [] for s in sources}),
              dict(attention_supported=True, source_shapes={'egotouch': []})]
    scenes.add_attention_sources(models, [dict(source=s) for s in sources + ['egotouch', 'ftp_motiontrans']])
    assert models[0]['attention_sources'] == sorted(sources)
    assert models[1]['attention_sources'] == ['egotouch']


@pytest.fixture
def joint_class(monkeypatch):
    from tactilelatent import joint_viewer as module
    from tactilelatent.acquisition_joint_viewer import install
    # install() supplies process-local factories; restore all globals afterwards.
    for name in ('JointViewer', 'MarkerCodeVJEPA', 'marker_code_from_state', 'MixedMarkerTeachers', 'source_shapes'):
        monkeypatch.setattr(module, name, getattr(module, name))
    install()
    return module.JointViewer


@pytest.mark.parametrize('source,channels', [('egopressure', None), ('baihu', None), ('feel_lab', 5),
    ('feel_kitchen', 12), ('hrdexdb_allegro_v5', 4), ('hrdexdb_inspire_f1', 24), ('trex', 60)])
def test_new_sensor_attention_uses_prediction_features_and_matching_image(source, channels, joint_class, monkeypatch, decoder):
    from tactilelatent import joint_viewer as module
    torch.set_num_threads(2)
    row = dict(source='feel' if source.startswith('feel_') else source, frames=20, views={'0': 'ego.mp4'})
    if channels is not None:
        row.update(targets=dict(path='native.npz', sha256='pinned', shape=[20, channels]))
        row['views']['0'] = dict(path='shard.mp4', role='ego', frame_offset=1000, frames=20)
    monkeypatch.setattr(scenes, '_targets', lambda *args: dict(timestamps=np.arange(20) / 30))
    visual = SpatialFrameEncoder(input_dim=12, width=16, heads=4, layers=2,
        feedforward=32, dropout=0., grid_size=2, prediction_mode='current_frame').eval()
    network = SimpleNamespace(visual=visual)
    features = torch.randn(1, 4, 3, 4, 12)
    encodings = []

    def encode(backbone, batch, device, chunk_size):
        encodings.append(batch)
        return features

    monkeypatch.setattr(module, 'spatial_video_features', encode)
    run = dict(id='model', name='integrated', source_shapes={source: []},
        config=dict(model_type=module.MARKER_CODE_TYPE, prediction_mode='current_frame'),
        snapshots=[dict(id='fixed.pt')])
    viewer = SimpleNamespace(episodes=[], _episode=lambda *_: row, _keep_recent=Viewer._keep_recent)
    joint = object.__new__(joint_class)
    joint.viewer = viewer; joint.device = torch.device('cpu'); joint.lock = threading.RLock()
    joint.runs = {'model': run}; joint.features = OrderedDict(); joint.attentions = OrderedDict()
    joint.video_backbone = object()
    joint._refresh_storage = lambda: None
    joint._load = lambda *_: ((network, None, 7, run['source_shapes']), (1, 2, 3))
    indices, _ = causal_clip_indices(18, 20)
    # Simulate prediction's extraction first: attention must reuse those features.
    prediction_features, mask = joint._video_features(12345, indices, 'ego', run['config'])
    result = joint.attention(12345, 18, 'model', 'fixed.pt', 'ego', expected_epoch=7)
    assert result['epoch'] == 7 and result['clip_frames'] == indices.tolist()
    assert result['camera_slots'] == ['0'] and result['default_image_slot'] == 7
    assert result['shape'] == [2, 4, 4, 3, 2, 2]
    assert len(encodings) == 1 and prediction_features is features
    image = joint.attention_image(12345, 18, 'model', 'fixed.pt', 'ego', '0', 7)
    decoded = np.asarray(Image.open(BytesIO(image)))
    expected = round(((1018 if channels is not None else 18) % 200) / 200 * 255)
    assert abs(int(decoded[0, 0, 0]) - expected) <= 1
    assert decoder[-1][1] == [1018 if channels is not None else 18]
    assert not mask[:, :, 1:].any() and all(p.grad is None for p in visual.parameters())
    with pytest.raises(ValueError, match='updated'):
        joint.attention(12345, 18, 'model', 'fixed.pt', 'ego', expected_epoch=6)
