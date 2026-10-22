#!/usr/bin/env python3
"""Verify applied gel mappings on native data and the running mirror viewer."""
import argparse
from io import BytesIO
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
from PIL import Image
import torch

from tactilelatent.aligned_frames import file_hash, write_json
from tactilelatent.bimanual_mirror import load_policy, require_spec, transform_targets
from tactilelatent.mirror_preparation import NativeTargets


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url', default='http://localhost:8080')
    p.add_argument('--output', type=Path, default=Path('processing/bimanual_mirror_v2/mapping_verification.json'))
    args = p.parse_args(); torch.set_num_threads(4)
    policy = load_policy('configs/bimanual_mirror_policy_v2.json')
    catalog = json.load(urlopen(args.url+'/api/catalog', timeout=60))
    assert catalog['mirror_policy'] == policy['version']
    base = json.loads(Path('processing/storage_v1/manifest.json').read_text())['episodes']
    extra = json.loads(Path('processing/data_expansion_v1/manifest.json').read_text())['episodes']
    selected = []
    for task in ('chips_grasping.zarr', 'chickenlegs_hanging.zarr'):
        selected.append(next(r for r in base if r['source']=='ftp_unit_bimanual'
                             and r['split']=='val' and task in r['episode_id']))
    selected.extend(next(r for r in extra if r['source']=='baihu' and r['split']=='val' and r['platform']==platform)
                    for platform in ('D-WHEEL', 'PIKA', 'QINGLOONG'))
    reader = NativeTargets(); records = []
    for row in selected:
        source = row['source']; spec = require_spec(policy, source)
        frames = np.unique([0, row['frames']//2, row['frames']-1])
        native = reader.read(row, frames)
        transformed = transform_targets(native, spec)
        restored = transform_targets(transformed, spec)
        for key in native:
            torch.testing.assert_close(restored[key], native[key], equal_nan=True)
        episode = next(e for e in catalog['episodes'] if e['source']==source and e['name']==row['episode_id'])
        models = [m for m in catalog['models'] if m['source']==source
                  and m.get('representation')=='marker_displacement']
        model = next((m for m in models if 'native v3' in m['name']), models[0])['id']
        for i, frame in enumerate(frames):
            query = dict(episode=episode['id'], frame=int(frame), model=model)
            result = json.load(urlopen(args.url+'/api/mirror/frame?'+urlencode(query), timeout=60))
            assert result['mirror']['verified'] and result['mirror']['verification_basis']=='user_visual_review'
            assert result['mirror']['surface_permutation']==spec['surface_permutation']
            points = result['marker_displacement']
            h, w = row.get('native_sensor_shape', row['target_shape'][-2:])
            reference = transformed['marker_reference'].numpy()*np.array([w-1, h-1])
            np.testing.assert_allclose(points['reference_px'], reference, rtol=1e-6, atol=1e-5)
            np.testing.assert_array_equal(points['surface'], transformed['marker_surface'].numpy())
            measured = transformed['marker_target'][i].numpy()*transformed['marker_spacing'].numpy()[:, None]
            valid = transformed['marker_valid'][i].numpy() & np.isfinite(measured).all(-1)
            for j, value in enumerate(points['measured_px']):
                if valid[j]:
                    np.testing.assert_allclose(value, measured[j], rtol=1e-6, atol=1e-5)
                else:
                    assert value is None
            assert result['mae'] is not None and np.isfinite(result['mae'])
        query = dict(episode=episode['id'], frame=int(frames[len(frames)//2]))
        for destination in episode['mirror_cameras']:
            recorded = {'0':'0', '1':'2', '2':'1'}[destination]
            original = np.asarray(Image.open(BytesIO(urlopen(args.url+'/api/video?'+urlencode(dict(query, camera=recorded)), timeout=60).read())))
            reflected = np.asarray(Image.open(BytesIO(urlopen(args.url+'/api/mirror/video?'+urlencode(dict(query, camera=destination)), timeout=60).read())))
            np.testing.assert_array_equal(reflected, original[:, ::-1])
        records.append(dict(source=source, episode_id=row['episode_id'], platform=row.get('platform'),
            frames=frames.tolist(), native_shape=result['target_shape'], model=model,
            reflected_reconstruction_mae_px=result['mae'], exact_reference_and_signed_vectors=True,
            exact_decoded_rgb_reflection=True, roundtrip=True,
            original_cameras=episode['cameras'], mirrored_cameras=episode['mirror_cameras']))
        print(json.dumps(records[-1]), flush=True)
    historical = json.loads(Path('processing/bimanual_mirror_v2/historical_before.json').read_text())
    for path, digest in historical['files'].items():
        if file_hash(path) != digest:
            raise ValueError(f'Historical artifact changed: {path}')
    write_json(args.output, dict(passed=True, episodes=records,
        historical_artifacts_unchanged=True, verification_basis='user_review_plus_software_consistency_checks',
        production_joint_run_started=False))


if __name__ == '__main__':
    main()
