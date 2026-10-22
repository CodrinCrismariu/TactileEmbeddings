"""Cache frozen marker AE codes in chronological frame order, without RGB.

The four replacement caches share CodeCache's format. The other five native
sensor caches are verified and linked, never rebuilt or copied.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from .aligned_frames import file_hash, manifest_signature, write_json
from .marker_ae_data import snapshot
from .marker_autoencoder import load_marker_autoencoder
from .marker_data import MarkerCache, MarkerTargetClips
from .marker_targets import MARKER_SOURCES
from .sensor_code_cache import (CodeCache, cache_signature, train_statistics,
                                verify_teachers)


def marker_signature(manifest, registry, source, pinned, data_signature=None):
    info = registry['sources'][source]
    if source not in MARKER_SOURCES or info.get('representation') != 'marker_displacement':
        raise ValueError('Marker sources require a marker-displacement teacher')
    return {**(data_signature or manifest_signature(manifest)),
            'source': source, 'teacher_sha256': info['sha256'],
            'target_snapshot_sha256': pinned['source_snapshot_sha256'],
            'code_builder_sha256': file_hash(__file__),
            'encoder_sha256': file_hash(Path(__file__).with_name('marker_autoencoder.py')),
            'dtype': 'float32', 'encoder_mode': 'eval-fp32-no-grad',
            'representation': 'marker_displacement', 'frame_order': 'chronological'}


def build_source(manifest, registry, source, targets, root, device='cuda',
                 batch_size=256, data_signature=None, report=lambda **row: None):
    if batch_size < 1:
        raise ValueError('Positive encoder batch size required')
    info = registry['sources'][source]
    if file_hash(info['checkpoint']) != info['sha256']:
        raise ValueError('Pinned marker teacher changed')
    pinned = snapshot(manifest, targets, source)
    signature = marker_signature(manifest, registry, source, pinned, data_signature)
    folder = Path(root) / source
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'build.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        index_path = folder / 'index.json'
        if index_path.exists():
            index = json.loads(index_path.read_text())
            if index['signature'] != signature:
                raise ValueError(f'Stale marker codes: {source}; use a new cache root')
            for name, digest in index['array_sha256'].items():
                if file_hash(folder / name) != digest:
                    raise ValueError(f'Corrupt marker codes: {source}/{name}')
            return index
        episodes = [record['row'] for record in pinned['episodes']]
        if not episodes or any(r['source'] == 'ftp_motiontrans' for r in episodes):
            raise ValueError('Invalid marker source episodes')
        offsets = np.concatenate(([0], np.cumsum([r['frames'] for r in episodes]))).tolist()
        dimension = info['code_dim']
        if shutil.disk_usage(folder).free < offsets[-1] * (4 * dimension + 1) + 64 * 2**20:
            raise RuntimeError('Insufficient free storage for marker codes')
        progress_path = folder / 'progress.json'
        progress = json.loads(progress_path.read_text()) if progress_path.exists() else {
            'signature': signature, 'completed_episodes': []}
        if progress['signature'] != signature:
            raise ValueError('Partial marker-code provenance changed')
        completed = set(progress['completed_episodes'])
        arrays = []
        for name, shape, dtype in [('codes', (offsets[-1], dimension), np.float32),
                                   ('valid', (offsets[-1],), np.bool_)]:
            partial, final = folder / f'{name}.partial.npy', folder / f'{name}.npy'
            existing = partial if partial.exists() else final if final.exists() else None
            if existing is None:
                if completed:
                    raise ValueError('Partial marker-code storage is missing')
                value = np.lib.format.open_memmap(partial, mode='w+', dtype=dtype, shape=shape)
            else:
                value = np.load(existing, mmap_mode='r+')
            if value.shape != shape or value.dtype != dtype:
                raise ValueError('Marker-code storage shape/dtype mismatch')
            arrays.append(value)
        codes, valid = arrays
        write_json(folder / 'targets_snapshot.json', pinned)
        model, _ = load_marker_autoencoder(info['checkpoint'], device)
        cache = MarkerCache(targets, max_loaded_episodes=1)
        began = time.monotonic()
        with torch.inference_mode(), torch.autocast(torch.device(device).type, enabled=False):
            for number, row in enumerate(episodes):
                if number in completed:
                    continue
                for start in range(0, row['frames'], batch_size):
                    frames = np.arange(start, min(start + batch_size, row['frames']))
                    batch = cache.read(row, frames)
                    y = batch['marker_target'].to(device).float()
                    mask = batch['marker_valid'].to(device) & torch.isfinite(y).all(-1)
                    ref = batch['marker_reference'].to(device)[None].expand(len(y), -1, -1)
                    surface = batch['marker_surface'].to(device)[None].expand(len(y), -1)
                    values = model.encode(y, mask, ref, surface).float().cpu().numpy()
                    if values.shape != (len(y), dimension) or not np.isfinite(values).all():
                        raise ValueError('Invalid marker encoder output')
                    lo = offsets[number] + start
                    codes[lo:lo + len(y)] = values
                    valid[lo:lo + len(y)] = mask.any(-1).cpu().numpy()
                codes.flush(); valid.flush()
                completed.add(number)
                progress['completed_episodes'] = sorted(completed)
                write_json(progress_path, progress)
                if len(completed) % 25 == 0 or len(completed) == len(episodes):
                    seconds = time.monotonic() - began
                    report(stage='encoding', source=source, code_dim=dimension,
                           completed_episodes=len(completed), episodes=len(episodes),
                           frames_cached=sum(episodes[i]['frames'] for i in completed),
                           frames=offsets[-1], source_seconds=seconds)
        statistics = train_statistics(codes, valid, episodes, offsets)
        for name in ('codes', 'valid'):
            partial = folder / f'{name}.partial.npy'
            if partial.exists():
                partial.replace(folder / f'{name}.npy')
        index = dict(signature=signature, episodes=episodes, offsets=offsets, frames=offsets[-1],
                     code_dim=dimension, target_shape=info['target_shape'], statistics=statistics,
                     representation='marker_displacement',
                     array_sha256={f'{name}.npy': file_hash(folder / f'{name}.npy') for name in ('codes', 'valid')})
        write_json(index_path, index)
        return index


def verify_samples(manifest, registry, source, targets, root, device='cuda'):
    """Compare saved raw and normalized codes to fresh encodings, including repair."""
    cache = CodeCache(root, source)
    rows = cache.index['episodes']
    selected = [next(r for r in rows if r['split'] == split) for split in ('train', 'val')]
    selected += [r for r in rows if r['episode_id'] == 'Slot/Slot_164']
    model, _ = load_marker_autoencoder(registry['sources'][source]['checkpoint'], device)
    tracks = MarkerCache(targets, max_loaded_episodes=1)
    result = []
    for row in selected:
        frames = np.unique([0, min(44, row['frames'] - 1), min(45, row['frames'] - 1), row['frames'] // 2, row['frames'] - 1])
        b = tracks.read(row, frames); y = b['marker_target'].to(device)
        valid = b['marker_valid'].to(device) & torch.isfinite(y).all(-1)
        with torch.inference_mode(), torch.autocast(torch.device(device).type, enabled=False):
            fresh = model.encode(y, valid, b['marker_reference'].to(device)[None].expand(len(y), -1, -1),
                                 b['marker_surface'].to(device)[None].expand(len(y), -1)).cpu()
        saved, mask = cache.read(row['episode_id'], frames, normalized=False)
        torch.testing.assert_close(saved, fresh, rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(mask, valid.any(-1).cpu())
        normalized, _ = cache.read(row['episode_id'], frames)
        stats = cache.index['statistics']
        torch.testing.assert_close(normalized, (saved - torch.tensor(stats['mean'])) / torch.tensor(stats['std']))
        result.append(dict(episode_id=row['episode_id'], split=row['split'], frames=frames.tolist(),
                           max_absolute_error=float((saved - fresh).abs().max())))
    write_json(Path(root) / source / 'verification.json', dict(passed=True, episodes=result))


class CachedMarkerCodeClips(MarkerTargetClips):
    """Scene clips and native validation targets, supervised only by cached codes."""
    def __init__(self, dataset, marker_cache, caches):
        super().__init__(dataset, marker_cache)
        self.code_caches = caches

    def __getitem__(self, index):
        item = super().__getitem__(index)
        source = item['source']
        if source in self.code_caches:
            item['code_target'], item['code_valid'] = self.code_caches[source].read(
                item['episode_id'], item['frame_indices'].numpy())
        return item


def bind_cached_codes(cfg, registry, train, val, mc, provenance):
    root = Path(cfg['code_cache'])
    ready = json.loads((root / 'registry.json').read_text())
    if not ready.get('complete') or ready['teachers'] != registry or ready['data_signature'] != provenance['data']:
        raise ValueError('Incomplete or mismatched replacement code caches')
    if ready['marker_target_index_sha256'] != file_hash(Path(cfg['marker_targets']) / 'index.json'):
        raise ValueError('Marker targets changed after caching')
    caches = {}
    for source in registry['sources']:
        if source in MARKER_SOURCES:
            pinned = snapshot(cfg['manifest'], cfg['marker_targets'], source)
            expected = marker_signature(cfg['manifest'], registry, source, pinned, provenance['data'])
        else:
            expected = cache_signature(cfg['manifest'], registry, source, provenance['data'])
        cache = CodeCache(root, source, expected)
        if cache.index['code_dim'] != mc['sensor_code_dims'][source]:
            raise ValueError('Cached sensor-code dimension differs from adapter')
        if source in MARKER_SOURCES and cache.index.get('representation') != 'marker_displacement':
            raise ValueError('Old image codes cannot supervise marker datasets')
        caches[source] = cache
        mc['statistics'][source] = cache.index['statistics']
    provenance['code_caches'] = {s: c.index['signature'] for s, c in caches.items()}
    provenance['code_registry_sha256'] = file_hash(root / 'registry.json')
    provenance['sample_cache'] = 'frozen sensor codes; no visual feature cache'
    provenance['implementation']['marker_code_cache.py'] = file_hash(__file__)
    wrappers = []
    for dataset in (train, val):
        base = dataset.dataset
        # Exclude fully unsupported clips while retaining original frame indices.
        base.windows = [(e, start) for e, start in base.windows
                        if base.episodes[e]['source'] not in caches or caches[base.episodes[e]['source']].valid_frames(
                            base.episodes[e]['episode_id'], np.arange(start, start + base.span, base.stride)).any()]
        from collections import defaultdict
        base.indices_by_source = defaultdict(list)
        for i, (e, _) in enumerate(base.windows):
            base.indices_by_source[base.episodes[e]['source']].append(i)
        wrappers.append(CachedMarkerCodeClips(base, dataset.cache, caches))
    return wrappers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/joint_vjepa21_spatial_marker_codes_32d.json')
    parser.add_argument('--native-cache', default='processing/sensor_code_cache_v1')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text()); root = Path(cfg['code_cache'])
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'build.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.set_num_threads(4)
    signature = manifest_signature(cfg['manifest'])
    audit = json.loads(Path(cfg['audit']).read_text())
    if audit.get('signature') != signature or not audit.get('passed') or not audit.get('VLA_complete_pixels'):
        raise ValueError('Canonical chronological data audit failed')
    native = json.loads(Path(cfg['teachers']).read_text())
    registry = {**native, 'sources': {s: v for s, v in native['sources'].items() if s not in MARKER_SOURCES}}
    markers = json.loads(Path(cfg['marker_teachers']).read_text())
    if not markers.get('complete') or set(markers['sources']) != set(MARKER_SOURCES):
        raise ValueError('Four completed marker AE teachers required')
    registry['sources'].update(markers['sources']); verify_teachers(registry)
    if len(registry['sources']) != 9 or 'ftp_motiontrans' in registry['sources']:
        raise ValueError('Expected nine tactile AEs; numeric/held-out sources must not be cached')
    began = time.monotonic()
    def report(**details):
        row = dict(pid=os.getpid(), elapsed_seconds=time.monotonic() - began, updated_at=time.time(), **details)
        write_json(root / 'status.json', row); print(json.dumps(row), flush=True)
    try:
        plan = dict(teachers=registry, data_signature=signature, marker_targets=cfg['marker_targets'])
        path = root / 'plan.json'
        if path.exists() and json.loads(path.read_text()) != plan:
            raise ValueError('Cache plan changed; use a new output root')
        write_json(path, plan); report(stage='verifying_native_caches')
        for source in sorted(set(registry['sources']) - set(MARKER_SOURCES)):
            cache = CodeCache(args.native_cache, source, cache_signature(cfg['manifest'], registry, source, signature))
            if cache.index['code_dim'] != registry['sources'][source]['code_dim']:
                raise ValueError('Native cache dimension differs')
            origin = (Path(args.native_cache) / source).resolve(); destination = root / source
            if destination.is_symlink():
                if destination.resolve() != origin: raise ValueError('Wrong native cache link')
            elif destination.exists(): raise FileExistsError(destination)
            else: destination.symlink_to(origin, target_is_directory=True)
        # Cache completed small sources first, while FreeTacMan extraction finishes.
        for source in sorted(MARKER_SOURCES, key=lambda s: (s == 'freetacman', s)):
            while True:
                try:
                    index = build_source(cfg['manifest'], registry, source, cfg['marker_targets'], root,
                                         args.device, args.batch_size, signature, report)
                    break
                except FileNotFoundError as error:
                    if 'incomplete/untrackable episodes' not in str(error): raise
                    report(stage='waiting_for_marker_extraction', source=source, reason=str(error))
                    time.sleep(10)
            verify_samples(cfg['manifest'], registry, source, cfg['marker_targets'], root, args.device)
            report(stage='source_complete', source=source, code_dim=index['code_dim'], frames=index['frames'])
        while True:
            path = Path(cfg['marker_targets']) / 'index.json'
            if path.exists():
                target_index = json.loads(path.read_text())
                if target_index.get('all_episodes_complete') and target_index['provenance']['data'] == signature:
                    break
            report(stage='waiting_for_extraction_reconciliation'); time.sleep(10)
        write_json(root / 'registry.json', dict(complete=True, teachers=registry, data_signature=signature,
                   marker_target_index_sha256=file_hash(path), heldout_source='ftp_motiontrans'))
        report(stage='complete', marker_code_dims={s: registry['sources'][s]['code_dim'] for s in MARKER_SOURCES},
               marker_frames=sum(json.loads((root / s / 'index.json').read_text())['frames'] for s in MARKER_SOURCES),
               new_array_bytes=sum(p.stat().st_size for s in MARKER_SOURCES for p in (root / s).glob('*.npy')),
               reused_native_sources=sorted(set(registry['sources']) - set(MARKER_SOURCES)))
    except Exception as error:
        report(stage='failed', error=str(error)); raise


if __name__ == '__main__':
    main()
