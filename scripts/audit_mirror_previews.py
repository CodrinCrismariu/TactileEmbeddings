#!/usr/bin/env python3
"""Exercise real native target hypotheses without enabling augmentation."""
import argparse
from io import BytesIO
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
from PIL import Image
import torch

from tactilelatent.aligned_frames import file_hash, json_hash, write_json
from tactilelatent.bimanual_mirror import load_policy, transform_targets
from tactilelatent.mirror_mapping_review import candidates
from tactilelatent.mirror_preparation import NativeTargets


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',default='http://127.0.0.1:8080')
    parser.add_argument('--output',type=Path,default=Path('processing/audits/bimanual_mirror_mapping_v3'))
    args=parser.parse_args();reader=NativeTargets();results=[]
    catalog=json.load(urlopen(args.url+'/api/catalog',timeout=60))
    base=json.loads(Path('processing/storage_v1/manifest.json').read_text())['episodes']
    extra=json.loads(Path('processing/data_expansion_v1/baihu_available.json').read_text())['episodes']
    selected=[next(r for r in base if r['source']=='opengraph' and r['split']=='val')]
    for task in ('chips_grasping.zarr','chickenlegs_hanging.zarr'):
        selected.append(next(r for r in base if r['source']=='ftp_unit_bimanual' and r['split']=='val' and task in r['episode_id']))
    selected.extend(next(r for r in extra if r['split']=='val' and r['platform']==p)
                    for p in ('D-WHEEL','PIKA','QINGLOONG'))
    for row in selected:
        source=row['source'];indices=np.unique([0,row['frames']//2,row['frames']-1])
        batch=reader.read(row,indices);entries=candidates(source)
        for entry in entries:
            restored=transform_targets(transform_targets(batch,entry['spec']),entry['spec'])
            for key in batch:
                torch.testing.assert_close(restored[key],batch[key],equal_nan=True,rtol=2e-5,atol=2e-5)
        episode=next(e for e in catalog['episodes'] if e['source']==source and e['name']==row['episode_id'])
        models=[m for m in catalog['models'] if m['source']==source and
                (source=='opengraph' or m.get('representation')=='marker_displacement')]
        model=models[0]['id'] if models else -1
        params=dict(episode=episode['id'],frame=int(indices[1]),model=model,candidate=entries[-1]['id'])
        result=json.load(urlopen(args.url+'/api/mirror/frame?'+urlencode(params),timeout=60))
        if result['mirror']['verified'] or result['mirror']['training_enabled']:
            raise ValueError('Preview incorrectly promoted to training')
        try:
            unverified=dict(params);del unverified['candidate']
            urlopen(args.url+'/api/mirror/frame?'+urlencode(unverified),timeout=60)
        except HTTPError as exc:
            if exc.code!=400:raise
        else:raise ValueError('Unverified source accepted without explicit review request')
        # Compare the reflected PNG pixels against the very same decoded preview.
        for destination in episode['mirror_cameras']:
            recorded={'0':'0','1':'2','2':'1'}[destination]
            query=dict(params,camera=recorded)
            original=np.asarray(Image.open(BytesIO(urlopen(args.url+'/api/video?'+urlencode(query),timeout=60).read())))
            query['camera']=destination
            reflected=np.asarray(Image.open(BytesIO(urlopen(args.url+'/api/mirror/video?'+urlencode(query),timeout=60).read())))
            np.testing.assert_array_equal(reflected,original[:,::-1])
        results.append(dict(source=source,episode_id=row['episode_id'],platform=row.get('platform'),
            frames=indices.tolist(),roundtrip_hypotheses=len(entries),fresh_teacher_model=model,
            native_target_shape=result['target_shape'],reconstruction_mae=result['mae'],
            camera_slots=episode['cameras'],mirrored_camera_slots=episode['mirror_cameras'],
            exact_decoded_rgb_reflection=True,training_still_blocked=True))
    policy=load_policy('configs/bimanual_mirror_policy_v1.json')
    quality=json.loads(Path('processing/bimanual_mirror_v1/teachers/egotouch/quality.json').read_text())
    if quality['signature']['policy_sha256']!=json_hash(policy):raise ValueError('Historical mirror policy changed')
    for name,sha in quality['signature']['implementation'].items():
        if file_hash(Path('src/tactilelatent')/name)!=sha:raise ValueError('Historical mirror implementation changed')
    teacher=quality['teacher']
    if file_hash(teacher['checkpoint'])!=teacher['sha256']:raise ValueError('Historical mirror teacher changed')
    storage=json.loads(Path('processing/storage_v1/implementation.json').read_text())
    # Storage receipts may also include non-file metadata; check file entries.
    pinned=storage.get('implementation',storage.get('files',storage))
    checked=[]
    if isinstance(pinned,dict):
        for name,sha in pinned.items():
            path=Path(name)
            if not path.is_file():path=Path('src/tactilelatent')/name
            if path.is_file() and isinstance(sha,str) and len(sha)==64:
                if file_hash(path)!=sha:raise ValueError(f'Storage implementation changed: {name}')
                checked.append(str(path))
    write_json(args.output/'native_preview_checks.json',dict(passed=True,checks=results,
        physical_mapping_verified=False,historical_teacher_and_policy_preserved=True,
        storage_files_checked=checked))
    print(json.dumps(dict(passed=True,episode_checks=len(results),
                         roundtrip_checks=sum(x['roundtrip_hypotheses'] for x in results)),indent=2))


if __name__=='__main__':main()
