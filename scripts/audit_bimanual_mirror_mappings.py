"""Read-only audit of native geometry; this never promotes a mirror mapping.

Run from the project root with .venv/bin/python. A valid file layout or a
round-trip permutation does not establish a physical reflection convention.
Only reports under --output are written; teachers, policies and caches stay put.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import time

import numpy as np
import pyarrow.parquet as pq
import torch

from tactilelatent.aligned_frames import file_hash, json_hash, write_json
from tactilelatent.bimanual_mirror import (
    load_policy, require_spec, transform_targets, verify_evidence,
)
from tactilelatent.marker_data import episode_key
from tactilelatent.mirror_preparation import NativeTargets


def check(condition, message):
    if not condition:
        raise ValueError(message)


def header(archive, key):
    """Read an NPY header without expanding the full trajectory."""
    with archive.zip.open(key + '.npy') as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError(f'Unsupported NPY header {version}')
    return tuple(shape), dtype


def inspect_markers(row):
    source = row['source']
    surfaces = 4 if source == 'baihu' else 2
    folder = (Path(row['marker_path']).parent if source == 'baihu' else
              Path('processing/marker_displacement_storage_v1/targets') /
              source / row['split'] / episode_key(row))
    path = folder / 'tracks.npz'
    info = json.loads((folder / 'complete.json').read_text())
    check(info['row_sha256'] == json_hash(row), 'Marker episode provenance changed')
    check(info['frames'] == row['frames'], 'Marker frame count changed')
    check(info['surfaces'] == surfaces, 'Wrong number of tactile surfaces')
    if source == 'baihu':
        order = ['tactile_left_l', 'tactile_left_r', 'tactile_right_l', 'tactile_right_r']
        check(info['surface_order'] == order, 'Baihu surface identity changed')
        check([Path(v['path']).stem for v in info['source_files']] == order,
              'Baihu source video order changed')
    with np.load(path, allow_pickle=False) as archive:
        sid = archive['surface_id']
        ref = archive['reference_uv']
        pixels = archive['reference_px']
        spacing = archive['spacing_px']
        frames = archive['frame_indices']
        sensor_frames = archive['sensor_frame_indices']
        n, k = row['frames'], len(sid)
        check(k == info['markers'], 'Marker point count changed')
        check(sid.shape == (k,) and set(sid.tolist()) == set(range(surfaces)),
              'Marker surface identity missing or invalid')
        check(ref.shape == (k, 2) and np.isfinite(ref).all() and
              ((ref >= 0) & (ref <= 1)).all(), 'Reference geometry out of native bounds')
        check(spacing.shape == (k,) and np.isfinite(spacing).all() and
              (spacing > 0).all(), 'Invalid marker spacing')
        check(np.array_equal(frames, np.arange(n)) and
              np.array_equal(sensor_frames, frames), 'Chronological marker pairing changed')
        check(header(archive, 'displacement_px') == ((n, k, 2), np.dtype('float32')),
              'Signed marker trajectory shape/dtype changed')
        check(header(archive, 'valid') == ((n, k), np.dtype('bool')),
              'Marker validity shape/dtype changed')
        for s in range(surfaces):
            h, w = info['sensors'][s]['shape'][:2]
            expected_shape = [240, 240] if source == 'baihu' else [240, 320]
            check([h, w] == expected_shape, 'Native tactile dimensions changed')
            np.testing.assert_allclose(ref[sid == s], pixels[sid == s] / [w - 1, h - 1],
                                       atol=1e-6, rtol=0)
    result = dict(source=source, episode_id=row['episode_id'], frames=n,
                  markers=k, per_surface_points=np.bincount(sid).tolist(),
                  platform=row.get('platform'), tracks_path=str(path),
                  completion_sha256=file_hash(folder / 'complete.json'))
    if source == 'baihu':
        meta = json.loads(Path(row['metadata_path']).read_text())
        check(meta['time_align_info']['frame_count'] == n, 'Baihu native metadata mismatch')
        hands = meta['data_info']['tactile_info']
        check(set(hands) == {'hand_left', 'hand_right'}, 'Baihu hand names changed')
        for hand in hands.values():
            for side in ('left_data', 'right_data'):
                check(hand[side]['shape'] == [n], 'Baihu tactile stream length changed')
        head = 'head' in meta['data_info']['camera_info']
        check(('0' in row['views']) == head, 'Baihu head-camera mask changed')
        check({'1', '2'} <= set(row['views']), 'Missing Baihu wrist-camera slot')
        params = []
        for p in sorted((Path(row['original_dir']) / 'params').glob('*.json')):
            d = json.loads(p.read_text())
            extrinsic = d.get('extrinsic', {})
            params.append(dict(camera=p.stem, keys=sorted(d),
                identity_extrinsic=(extrinsic.get('rotation_matrix') == np.eye(3).astype(int).tolist()
                                    and extrinsic.get('translation_vector') == [0, 0, 0])))
        result.update(head_camera_present=head, camera_parameters=params)
    return result


def inspect_opengraph():
    root = Path('datasets/OpenGraphLabs-Research__ego-tactile-manipulation')
    info = json.loads((root / 'meta/info.json').read_text())
    names = info['features']['observation.tactile.left']['names']
    check(names == info['features']['observation.tactile.right']['names'],
          'OpenGraph bilateral labeled channels differ')
    fingers = ['index', 'middle', 'pinky', 'ring', 'thumb']
    check(names == [f'{finger}_r{r}_c{c}' for finger in fingers
                    for r in range(4) for c in range(4)], 'Unexpected taxel channel contract')
    raw_names = [v.replace('_r', '_').replace('_c', '_') for v in names]
    raw = sorted((root / 'extra/raw_tactile').glob('*.parquet'))
    for p in raw:
        check(pq.ParquetFile(p).schema.names == ['capture_ns'] + raw_names,
              f'Raw channel identity differs: {p}')
    episodes = json.loads((root / 'meta/opengraph/episode_manifest.json').read_text())
    check(len(raw) == 2 * len(episodes), 'Missing raw left/right episode metadata')
    return dict(episodes=len(episodes), raw_streams_checked=len(raw),
                fingers_in_dataset_order=fingers, channels_per_hand=len(names),
                raw_and_aligned_names_match=True,
                physical_reflection_verified=False,
                missing='Across-finger row orientation and release-specific thumb correction')


def inspect_enabled(rows, policy):
    reader = NativeTargets()
    report = {}
    for source in ('egotouch', 'ftp_rdp_bimanual'):
        spec = require_spec(policy, source)
        selected = []
        for split in ('train', 'val'):
            candidates = [r for r in rows if r['source'] == source and r['split'] == split]
            ix = np.unique(np.linspace(0, len(candidates) - 1, min(8, len(candidates)), dtype=int))
            selected.extend(candidates[i] for i in ix)
        frame_count = 0
        for row in selected:
            indices = np.unique([0, row['frames'] // 2, row['frames'] - 1])
            batch = reader.read(row, indices)
            mirrored = transform_targets(batch, spec)
            twice = transform_targets(mirrored, spec)
            for key in batch:
                torch.testing.assert_close(batch[key], twice[key], atol=0, rtol=0, equal_nan=True)
            # Assert the forward map as well as involution: the identity map
            # would pass an involution-only check.
            expected = batch['target'][:, [1, 0]]
            if source == 'egotouch':
                expected = expected.flip(-1)
            torch.testing.assert_close(mirrored['target'], expected, atol=0, rtol=0,
                                       equal_nan=True)
            frame_count += len(indices)
        report[source] = dict(episodes=len(selected), frames=frame_count,
            native_forward_mapping_passed=True, exact_round_trip_passed=True,
            physical_evidence='Rechecked by bimanual_mirror.verify_evidence')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='processing/audits/bimanual_mirror_mapping_v2')
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    paths = ['configs/bimanual_mirror_policy_v1.json',
             'processing/bimanual_mirror_v1/ready.json',
             'processing/bimanual_mirror_v1/codes/egotouch/index.json',
             'processing/bimanual_mirror_v1/teachers/egotouch/best.pt']
    before = {p: file_hash(p) for p in paths}
    started = time.time()
    policy = load_policy(paths[0])
    evidence = verify_evidence(policy)
    rows = json.loads(Path('processing/storage_v1/manifest.json').read_text())['episodes']
    baihu = json.loads(Path('processing/data_expansion_v1/baihu_available.json').read_text())
    marker_rows = [r for r in rows if r['source'] == 'ftp_unit_bimanual'] + baihu['episodes']
    results, errors = [], []

    def inspect(row):
        try:
            return inspect_markers(row), None
        except Exception as exc:
            return None, dict(source=row['source'], episode_id=row['episode_id'],
                              error=f'{type(exc).__name__}: {exc}')

    with ThreadPoolExecutor(max_workers=6) as pool:
        for i, (result, error) in enumerate(pool.map(inspect, marker_rows), 1):
            (errors if error else results).append(error or result)
            if i % 250 == 0:
                state = dict(stage='native_marker_geometry', checked=i, total=len(marker_rows),
                             errors=len(errors), elapsed_seconds=time.time() - started)
                write_json(output / 'status.json', state)
                print(json.dumps(state), flush=True)
    summary = {}
    for source in ('ftp_unit_bimanual', 'baihu'):
        subset = [r for r in results if r['source'] == source]
        summary[source] = dict(episodes_checked=len(subset), frames=sum(r['frames'] for r in subset),
            platforms=dict(Counter(r['platform'] for r in subset if r['platform'])),
            marker_count_range=([min(r['markers'] for r in subset), max(r['markers'] for r in subset)]
                                if subset else None),
            physical_reflection_verified=False)
    b = [r for r in results if r['source'] == 'baihu']
    summary['baihu'].update(
        absent_head_camera_episodes=sum(not r['head_camera_present'] for r in b),
        camera_parameter_files=sum(len(r['camera_parameters']) for r in b),
        identity_camera_extrinsics=sum(p['identity_extrinsic'] for r in b
                                       for p in r['camera_parameters']),
        incomplete_preparation_episodes=baihu.get('failures', []))
    write_json(output / 'native_episode_checks.json', results)
    document = dict(version='bimanual-mirror-evidence-review-v2',
        created_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        physical_evidence=evidence, opengraph=inspect_opengraph(),
        native_markers=summary, enabled_native_checks=inspect_enabled(rows, policy),
        failures=errors, geometry_integrity_passed=not errors,
        payload_checksum_scope='Completion receipts hashed; full track payloads were not rehashed. '
                               'Small geometry arrays and trajectory headers were inspected.',
        promotions=[], enabled_sources_unchanged=['egotouch', 'ftp_rdp_bimanual'],
        elapsed_seconds=time.time() - started,
        pinned_before=before, pinned_after={p: file_hash(p) for p in paths},
        script_sha256=file_hash(__file__))
    check(document['pinned_before'] == document['pinned_after'], 'Existing mirror artifacts changed')
    write_json(output / 'audit.json', document)
    write_json(output / 'status.json', dict(stage='complete', errors=len(errors),
               geometry_integrity_passed=not errors, elapsed_seconds=time.time() - started))
    print(json.dumps(dict(native_markers=summary, failures=errors,
                          enabled_native_checks=document['enabled_native_checks'])), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
