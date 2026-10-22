import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tactilelatent.aligned_frames import file_hash, json_hash, write_json
from tactilelatent.marker_autoencoder import MarkerAutoencoder
from tactilelatent.marker_data import episode_key
from tactilelatent.marker_targets import VERSION
from tactilelatent.sensor_code_cache import CodeCache
import tactilelatent.marker_code_cache as module


@pytest.mark.parametrize('source', ['ftp_unit', 'ftp_vla_touch'])
def test_marker_codes_preserve_chronology_masks_and_train_only_statistics(tmp_path, monkeypatch, source):
    import tactilelatent.marker_data as data
    import tactilelatent.marker_ae_data as ae_data
    torch.set_num_threads(2); torch.manual_seed(45)
    signature = {'test': True}
    monkeypatch.setattr(ae_data, 'manifest_signature', lambda _: signature)
    rows = [dict(source=source, split=split, episode_id=split, frames=17) for split in ('train', 'val')]
    manifest = tmp_path / 'manifest.json'; write_json(manifest, {'episodes': rows})
    targets = tmp_path / 'tracks'
    for e, row in enumerate(rows):
        if source == 'ftp_vla_touch': row['sensor_frame_order'] = list(range(16, -1, -1))
        n = 3 + e; folder = targets / source / row['split'] / episode_key(row); folder.mkdir(parents=True)
        vectors = np.random.default_rng(e).normal(size=(17, n, 2)).astype('float32') + e * 10
        valid = np.ones((17, n), bool); valid[1] = False; vectors[1] = np.nan
        spacing = np.full(n, 30., np.float32)
        np.savez(folder / 'tracks.npz', displacement_px=vectors, valid=valid,
                 reference_uv=np.full((n, 2), .5, np.float32), surface_id=np.zeros(n, np.int64),
                 spacing_px=spacing, frame_indices=np.arange(17),
                 sensor_frame_indices=np.array(row.get('sensor_frame_order', list(range(17)))))
        write_json(folder / 'complete.json', dict(version=VERSION, row_sha256=json_hash(row),
                   extractor_sha256=file_hash(Path(data.__file__).with_name('marker_targets.py')),
                   loader_sha256=file_hash(data.__file__), tracks_sha256=file_hash(folder / 'tracks.npz'),
                   loss_statistics=dict(square_sum=float(np.nansum((vectors / 30.)**2)), coordinate_count=int(valid.sum()) * 2)))
    write_json(manifest, {'episodes': rows})
    model = MarkerAutoencoder(latent_dim=4, width=16, queries=4)
    teacher = tmp_path / 'teacher.pt'
    torch.save(dict(model_type='marker_autoencoder_v1', model_config=model.config, model=model.state_dict()), teacher)
    registry = dict(sources={source: dict(checkpoint=str(teacher), sha256=file_hash(teacher),
                                        code_dim=4, target_shape=[3, 240, 320], representation='marker_displacement')})
    root = tmp_path / 'codes'
    index = module.build_source(str(manifest), registry, source, targets, root, 'cpu', 5, signature)
    cache = CodeCache(root, source); assert index['frames'] == 34 and index['statistics']['train_frames'] == 16
    assert not cache.read('train', [1])[1].any()
    np.testing.assert_allclose(index['statistics']['mean'], cache.codes[:17][cache.valid[:17]].mean(0), atol=1e-7)
    module.verify_samples(str(manifest), registry, source, targets, root, 'cpu')
    # Re-entry validates completed files; changed teachers cannot silently reuse codes.
    assert module.build_source(str(manifest), registry, source, targets, root, 'cpu', 5, signature) == index
    altered = {**registry, 'sources': {source: {**registry['sources'][source], 'representation': 'optical_rgb'}}}
    with pytest.raises(ValueError, match='marker-displacement teacher'):
        module.build_source(str(manifest), altered, source, targets, root, 'cpu', 5, signature)
    # Lost progress can safely recompute an interrupted episode without changing codes.
    (root / source / 'index.json').unlink()
    progress = json.loads((root / source / 'progress.json').read_text()); progress['completed_episodes'] = [0]
    write_json(root / source / 'progress.json', progress)
    resumed = module.build_source(str(manifest), registry, source, targets, root, 'cpu', 5, signature)
    assert resumed['array_sha256'] == index['array_sha256']


def test_cached_supervision_skips_every_live_tactile_encoder():
    from tactilelatent.marker_code_model import MixedMarkerTeachers
    from types import SimpleNamespace
    teacher = MixedMarkerTeachers.__new__(MixedMarkerTeachers)
    teacher.device = torch.device('cpu'); teacher.marker_sources = {'ftp_unit'}; teacher.models = {}
    model = SimpleNamespace(normalization={'ftp_unit': SimpleNamespace(mean=torch.zeros(8)),
                                          'egotouch': SimpleNamespace(mean=torch.zeros(256))})
    for source, dimension in [('ftp_unit', 8), ('egotouch', 256)]:
        batch = dict(source=[source], code_target=torch.randn(1, 8, dimension), code_valid=torch.ones(1, 8, dtype=torch.bool))
        result = teacher.supervision(batch, model)
        assert torch.equal(result['code_target'], batch['code_target'])
        assert torch.equal(result['code_valid'], batch['code_valid'])


def test_cached_marker_clips_never_load_old_optical_images(monkeypatch):
    import tactilelatent.aligned_frames as frames
    class Dataset:
        episodes = [dict(source='ftp_vla_touch', episode_id='test', frames=15, sensor_frame_order=list(range(14, -1, -1)))]
        windows = [(0, 0)]; indices_by_source = {'ftp_vla_touch': [0]}; span = 15; stride = 2; rgb_size = 384
        def __getitem__(self, index): raise AssertionError('Old image loader called')
    class Tracks:
        def read(self, row, indices): return dict(marker_target=torch.zeros(8, 3, 2))
    class Codes:
        def read(self, episode_id, indices):
            assert episode_id == 'test'; np.testing.assert_array_equal(indices, np.arange(0, 15, 2))
            return torch.ones(8, 8), torch.ones(8, dtype=torch.bool)
    monkeypatch.setattr(frames, 'scene_rgb', lambda row, indices, size: (torch.zeros(8, 3, 3, 2, 2), torch.ones(8, 3, dtype=torch.bool)))
    item = module.CachedMarkerCodeClips(Dataset(), Tracks(), {'ftp_vla_touch': Codes()})[0]
    assert 'target' not in item and item['code_target'].shape == (8, 8)
    assert item['sensor_frame_indices'].tolist() == list(range(14, -1, -2))
