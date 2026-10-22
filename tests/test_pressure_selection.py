import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tactilelatent.aligned_frames import file_hash, json_hash, write_json
from tactilelatent.pressure_selection import (
    VERSION, MANIFEST, task_episodes, is_calibration_episode, read_task_manifest,
    build_task_code_index,
)


def runner():
    spec = importlib.util.spec_from_file_location('pressure_runner', 'scripts/train_egopressure_teacher.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_selection_excludes_only_pressure_calibration_recordings_and_preserves_pairing():
    names = ['p_001/p_001_calibration_routine_left', 'p_021/p_021_calibration_routine_right',
             'p_001/p_001_index_press_high_x5_left', 'p_021/p_021_draw_word_5x_right']
    rows = [dict(source='egopressure', episode_id=n, source_frame_ids=[2, 4, 7],
                 split='train' if i % 2 == 0 else 'val') for i, n in enumerate(names)]
    other = dict(source='another_sensor', episode_id=names[0])
    assert task_episodes([*rows, other]) == [rows[2], rows[3], other]
    assert task_episodes(rows)[0] is rows[2]
    assert not is_calibration_episode(dict(source='egopressure', episode_id='camera_calibration_task'))


def test_versioned_manifest_blocks_stale_acquisition_and_calibration_reentry(tmp_path):
    source = tmp_path / 'egopressure_manifest.json'
    rows = [dict(source='egopressure', episode_id='p_001_calibration_routine_left'),
            dict(source='egopressure', episode_id='task')]
    write_json(source, dict(episodes=rows, complete=True))
    assert len(read_task_manifest(tmp_path)['episodes']) == 1
    active = dict(episodes=[rows[1]], selection_version=VERSION,
                  acquisition_manifest_sha256=file_hash(source))
    write_json(tmp_path / MANIFEST, active)
    assert read_task_manifest(tmp_path)['episodes'] == [rows[1]]
    active['episodes'] = rows
    write_json(tmp_path / MANIFEST, active)
    with pytest.raises(ValueError, match='excluded'):
        read_task_manifest(tmp_path)
    active['episodes'] = [rows[1]]
    active['acquisition_manifest_sha256'] = 'changed'
    write_json(tmp_path / MANIFEST, active)
    with pytest.raises(ValueError, match='changed'):
        read_task_manifest(tmp_path)


def test_both_teacher_train_and_validation_samplers_never_load_calibration(monkeypatch):
    module = runner()
    from tactilelatent import expansion_training as training
    rows = [dict(source='egopressure', episode_id=f'p_001_calibration_routine_{split}',
                 split=split, frames=4) for split in ('train', 'val')]
    rows += [dict(source='egopressure', episode_id=f'task_{split}', split=split, frames=4)
             for split in ('train', 'val')]
    reads = []

    def targets(row, ix):
        assert not is_calibration_episode(row)
        reads.append(row['episode_id'])
        return np.zeros((len(ix), 1, 8, 8), np.float32)

    monkeypatch.setattr(module, 'pressure_targets', targets)
    monkeypatch.setattr(training, 'pressure_targets', targets)
    index = {r['episode_id']: dict(valid=list(range(4)), positive=[]) for r in rows}
    for split in ('train', 'val'):
        assert list(module.batches(rows, index, split, 0, 2, 4))
        assert list(training.batches(rows, 'egopressure', split, batch_size=2, limit=4))
    assert set(reads) == {'task_train', 'task_val'}


def test_joint_sampler_rejects_calibration_in_cached_statistics():
    from tactilelatent.expansion_joint import ExpandedCurrentClips
    class Base:
        episodes = []
        windows = []
        indices_by_source = {}
    index = dict(egopressure=dict(episodes=[dict(episode_id='p_001_calibration_routine_left')]))
    with pytest.raises(ValueError, match='excluded'):
        ExpandedCurrentClips(Base(), [], index, 'train')


def test_subset_cache_refits_stats_without_calibration_or_validation_and_reuses_bytes(tmp_path, monkeypatch):
    from tactilelatent import expansion_models, data_expansion
    checkpoint = tmp_path / 'teacher.pt'
    checkpoint.write_bytes(b'fixed historical teacher')
    digest = file_hash(checkpoint)
    rows, records = [], []
    magnitudes = [10000., 1., 3., 999.]
    for i, magnitude in enumerate(magnitudes):
        name = 'p_001_calibration_routine_left' if i == 0 else f'task_{i}'
        source = tmp_path / f'target_{i}.bin'
        source.write_text(str(magnitude))
        row = dict(source='egopressure', episode_id=name, frames=2,
                   split='val' if i == 3 else 'train', target_path=str(source),
                   source_frame_ids=[2, 4])
        rows.append(row)
        path = tmp_path / f'codes_{i}.npz'
        codes = np.full((2, 2), magnitude, np.float32)
        np.savez(path, codes=codes, valid=np.ones(2, bool), frame_indices=np.arange(2))
        h = file_hash(path)
        write_json(path.with_suffix('.json'), dict(sha256=h, signature=dict(
            row_sha256=json_hash(row), teacher_sha256=digest, dimension=2)))
        records.append(dict(episode_id=name, split=row['split'], frames=2, path=str(path), sha256=h))
    parent = tmp_path / 'parent.json'
    write_json(parent, dict(complete=True, source='egopressure', code_dim=2, teacher_sha256=digest,
        rows_sha256=json_hash(rows), target_snapshot_sha256=json_hash({r['episode_id']: file_hash(r['target_path']) for r in rows}),
        episodes=records))

    class Model:
        def encode(self, y, valid):
            return y[:, 0, 0, 0, None].expand(-1, 2)

    monkeypatch.setattr(expansion_models, 'load_teacher', lambda *a: (Model(), {}))
    monkeypatch.setattr(data_expansion, 'pressure_targets', lambda row, ix:
        np.full((len(ix), 1, 8, 8), float(Path(row['target_path']).read_text()), np.float32))
    result = build_task_code_index(rows, parent, checkpoint, tmp_path / 'subset', device='cpu')
    assert len(result['episodes']) == 3 and result['frames'] == 6
    np.testing.assert_allclose(result['statistics']['mean'], [2, 2])
    np.testing.assert_allclose(result['statistics']['std'], [1, 1])
    assert result['statistics']['train_frames'] == 4
    assert result['teacher_training_scope'].startswith('historical')
    assert json.loads((tmp_path / 'subset/verification.json').read_text())['passed']
    for record in records:
        assert file_hash(record['path']) == record['sha256']
    Path(rows[1]['target_path']).write_text('modified')
    with pytest.raises(ValueError, match='Native pressure targets changed'):
        build_task_code_index(rows, parent, checkpoint, tmp_path / 'bad', device='cpu')
