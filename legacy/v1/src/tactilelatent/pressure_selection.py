"""Task-only EgoPressure sampling; keep acquisition and historical runs intact."""
import json
from pathlib import Path
import re

from .aligned_frames import file_hash, json_hash, write_json


VERSION = 'egopressure-task-episodes-v1'
MANIFEST = 'egopressure_task_manifest_v1.json'
CALIBRATION = re.compile(r'(?:^|_)calibration_routine(?:_|$)', re.IGNORECASE)


def is_calibration_episode(row):
    return row.get('source') == 'egopressure' and bool(
        CALIBRATION.search(row['episode_id'].rsplit('/', 1)[-1]))


def task_episodes(rows):
    """Exclude whole calibration recordings, preserving all other row fields."""
    return [row for row in rows if not is_calibration_episode(row)]


def require_task_episodes(rows):
    if any(is_calibration_episode(row) for row in rows):
        raise ValueError('EgoPressure calibration routines are excluded from training and validation')


def read_task_manifest(root):
    root = Path(root)
    path = root / MANIFEST
    if path.exists():
        document = json.loads(path.read_text())
        if (document.get('selection_version') != VERSION or
                document['acquisition_manifest_sha256'] != file_hash(root / 'egopressure_manifest.json')):
            raise ValueError('EgoPressure task selection changed; rebuild its versioned manifest')
        require_task_episodes(document['episodes'])
        return document
    document = json.loads((root / 'egopressure_manifest.json').read_text())
    return {**document, 'episodes': task_episodes(document['episodes']), 'selection_version': VERSION}


def build_task_code_index(rows, parent_path, checkpoint, destination, device='cuda', progress=None):
    """Reuse unchanged episode codes; refit statistics on task-training rows.

    The producer's manifest can include unsupported recordings. Its actual
    supervised contribution must be audited separately from manifest membership.
    """
    import numpy as np
    import torch
    from .data_expansion import pressure_targets
    from .expansion_models import load_teacher

    parent_path, checkpoint, destination = Path(parent_path), Path(checkpoint), Path(destination)
    parent = json.loads(parent_path.read_text())
    digest = file_hash(checkpoint)
    if not parent['complete'] or parent['teacher_sha256'] != digest:
        raise ValueError('Historical pressure code cache/teacher mismatch')
    if parent['source'] != 'egopressure' or parent['rows_sha256'] != json_hash(rows):
        raise ValueError('Parent code index does not describe the original EgoPressure rows')
    hashes = {r['episode_id']: file_hash(r['target_path']) for r in rows}
    if json_hash(hashes) != parent['target_snapshot_sha256']:
        raise ValueError('Native pressure targets changed after historical code caching')
    selected = task_episodes(rows)
    require_task_episodes(selected)
    records = {r['episode_id']: r for r in parent['episodes']}
    dimension = parent['code_dim']
    total, squares = np.zeros(dimension, np.float64), np.zeros(dimension, np.float64)
    count, result = 0, []
    for i, row in enumerate(selected):
        record = records[row['episode_id']]
        path = Path(record['path'])
        receipt = json.loads(path.with_suffix('.json').read_text())
        if (file_hash(path) != record['sha256'] or receipt['sha256'] != record['sha256'] or
                receipt['signature'] != dict(row_sha256=json_hash(row), teacher_sha256=digest,
                                             dimension=dimension) or
                record['split'] != row['split'] or record['frames'] != row['frames']):
            raise ValueError('Selected sensor code changed or no longer matches its native episode')
        with np.load(path, allow_pickle=False) as values:
            codes, valid = values['codes'], values['valid']
            if (codes.shape != (row['frames'], dimension) or codes.dtype != np.float32 or
                    valid.shape != (row['frames'],) or valid.dtype != np.bool_ or
                    not np.isfinite(codes).all() or
                    not np.array_equal(values['frame_indices'], np.arange(row['frames']))):
                raise ValueError('Selected sensor-code geometry or chronology changed')
        if row['split'] == 'train':
            x = codes[valid].astype(np.float64)
            total += x.sum(0)
            squares += (x * x).sum(0)
            count += len(x)
        result.append(record)
        if progress is not None and i % 100 == 0:
            progress(stage='checking_task_codes', episodes_done=i + 1, episodes=len(selected))
    if count < 2:
        raise ValueError('No supported task-training codes for normalization')
    mean = total / count
    std = np.maximum(np.sqrt(np.maximum(squares / count - mean * mean, 1e-12)), 1e-6)
    model, _ = load_teacher(checkpoint, device)
    checks = []
    # Replay the actual 16-frame producer batches, including final remainders.
    with torch.inference_mode():
        for split in ('train', 'val'):
            row = next(r for r in selected if r['split'] == split)
            with np.load(records[row['episode_id']]['path'], allow_pickle=False) as values:
                cached, supported = values['codes'], values['valid']
            for start in sorted({0, (row['frames'] - 1) // 16 * 16}):
                ix = np.arange(start, min(start + 16, row['frames']))
                y = torch.from_numpy(pressure_targets(row, ix)).to(device)
                valid = torch.isfinite(y)
                fresh = model.encode(y, valid).float().cpu()
                torch.testing.assert_close(torch.from_numpy(cached[ix]), fresh, rtol=2e-5, atol=2e-5)
                np.testing.assert_array_equal(supported[ix], valid.flatten(1).any(-1).cpu().numpy())
                checks.append(dict(split=split, episode_id=row['episode_id'], frames=ix.tolist(),
                    max_absolute_error=float((fresh - torch.from_numpy(cached[ix])).abs().max())))
    destination.mkdir(parents=True, exist_ok=True)
    verification = dict(passed=True, checks=checks, teacher_sha256=digest, selection_version=VERSION)
    write_json(destination / 'verification.json', verification)
    index = dict(complete=True, source='egopressure', code_dim=dimension, teacher_sha256=digest,
        episodes=result, rows_sha256=json_hash(selected),
        target_snapshot_sha256=json_hash({r['episode_id']: hashes[r['episode_id']] for r in selected}),
        frames=sum(r['frames'] for r in selected),
        statistics=dict(mean=mean.tolist(), std=std.tolist(), normalization_split='train',
                        train_frames=count, episode_selection=VERSION),
        selection_version=VERSION, parent_code_index=str(parent_path),
        parent_code_index_sha256=file_hash(parent_path),
        teacher_training_scope='historical producer manifest; audit actual supported calibration samples separately',
        pressure_normalization='unchanged frozen-teacher target convention; no per-episode code values modified')
    write_json(destination / 'index.json', index)
    return index
