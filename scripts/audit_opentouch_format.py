#!/usr/bin/env python3
"""Read-only comparison of OpenTouch HDF5, loaders, caches and upstream layout."""
import json
from collections import Counter, defaultdict
from pathlib import Path
import time

import h5py
import numpy as np
import torch

from tactilelatent.aligned_frames import targets
from tactilelatent.sensor_code_cache import CodeCache
from tactilelatent.sensor_autoencoder import make_autoencoder


def main():
    torch.set_num_threads(2)
    began=time.monotonic()
    output=Path('processing/audits/opentouch_format')
    layout=json.loads((output/'handLayoutNewest_meshid.json').read_text())
    erased=set(layout['erasedNodes'])
    mask=np.zeros((16,16),bool)
    for node in layout['positions']:
        if node not in erased:
            r,c=map(int,node.split('-'));mask[r,c]=True
    rows=[r for r in json.loads(Path('processing/joint_alignment/manifest.json').read_text())['episodes'] if r['source']=='opentouch']
    by_file=defaultdict(list)
    for row in rows:by_file[row['views']['0']].append(row)
    caches=[]
    for root in ('processing/sensor_ae_cache_v1','processing/sensor_ae_cache_native_v2'):
        folder=Path(root)/'opentouch'
        index=json.loads((folder/'index.json').read_text())
        for split,info in index['splits'].items():
            data=np.load(folder/f'{split}.npy',mmap_mode='r')
            offsets={r['episode_id']:info['offsets'][i] for i,r in enumerate(info['episodes'])}
            caches.append((root,split,index['signature']['stride'],offsets,data))
    counts=Counter(); failures=[]; files=[]; examples=[]
    histogram=np.zeros(3073,np.int64)
    train_sum=np.zeros((16,16),np.float64);train_frames=0
    temporal_sum=0.;temporal_pairs=0
    for path,episodes in sorted(by_file.items()):
        record={'path':path,'episodes':len(episodes),'frames':0}
        with h5py.File(path,'r') as h5:
            known={r['hdf5_demo'] for r in episodes}
            record['excluded_source_demos']=[name for name in h5['data'] if name not in known]
            for row in episodes:
                demo=h5['data'][row['hdf5_demo']]
                raw=demo['right_pressure'][:]
                n=row['frames'];record['frames']+=n;counts['episodes']+=1;counts['frames']+=n
                if raw.shape!=(n,16,16) or len(demo['rgb_images_jpeg'])!=n or len(demo['timestamps'])!=n:
                    failures.append({'episode':row['episode_id'],'reason':'shape/length'})
                counts['nonfinite']+=int((~np.isfinite(raw)).sum())
                counts['below_zero']+=int((raw<0).sum());counts['above_3072']+=int((raw>3072).sum())
                counts['noninteger']+=int((raw!=raw.round()).sum())
                histogram+=np.bincount(np.clip(raw,0,3072).astype(np.int64).ravel(),minlength=3073)
                counts['nonincreasing_timestamp_pairs']+=int((np.diff(demo['timestamps'][:])<=0).sum())
                expected=raw[:,None].astype(np.float32)/np.float32(3072.)
                sample=np.unique([0,n//2,n-1])
                loaded=targets(row,sample)
                counts['loader_frames_checked']+=len(sample)
                if not np.array_equal(loaded,expected[sample],equal_nan=True):
                    failures.append({'episode':row['episode_id'],'reason':'loader values'})
                for root,split,stride,offsets,data in caches:
                    if row['split']!=split or row['episode_id'] not in offsets:continue
                    indices=np.arange(0,n,stride)
                    cached=data[offsets[row['episode_id']]:offsets[row['episode_id']]+len(indices)]
                    counts[root+'_frames_checked']+=len(indices)
                    if not np.array_equal(cached,expected[indices],equal_nan=True):
                        failures.append({'episode':row['episode_id'],'reason':'cache values','cache':root})
                if row['split']=='train':
                    train_sum+=raw.sum(axis=0,dtype=np.float64);train_frames+=n
                temporal_sum+=float(np.abs(np.diff(raw.astype(np.float64),axis=0)).sum())
                temporal_pairs+=max(n-1,0)
                for name,selection in [('mapped',mask),('erased',~mask)]:
                    a=raw[:,selection]
                    counts[name+'_values']+=a.size
                    counts[name+'_zeros']+=int((a==0).sum())
                    counts[name+'_saturated']+=int((a==3072).sum())
                    counts[name+'_contact_by_current_threshold']+=int((a/3072.>.05).sum())
                if len(examples)<len(files)+1:
                    examples.append({'episode':row['episode_id'],'frame':int(sample[1]),
                        'raw':raw[sample[1]].tolist(),'sample_frames':sample.tolist(),'row':row})
        files.append(record)
        print(json.dumps({'file':Path(path).name,'frames':record['frames'],'checked_files':len(files)}),flush=True)
    # Check latent caches against fresh frozen-encoder outputs, using the builder's GPU path.
    info=json.loads(Path('processing/joint_alignment/teachers.json').read_text())['sources']['opentouch']
    saved=torch.load(info['checkpoint'],map_location='cpu',weights_only=False)
    model=make_autoencoder(saved['model_config']).eval().requires_grad_(False)
    model.load_state_dict(saved['model'])
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model.encoder.to(device)
    codes=CodeCache('processing/sensor_code_cache_v1','opentouch')
    code_difference=0.
    with torch.inference_mode():
        for example in examples:
            row=example['row'];frames=example['sample_frames']
            value=torch.from_numpy(targets(row,frames)).to(device)
            fresh=model.encode(value,torch.isfinite(value)).cpu()
            cached,_=codes.read(row['episode_id'],frames,normalized=False)
            code_difference=max(code_difference,float((cached-fresh).abs().max()))
            if not torch.allclose(cached,fresh,atol=2e-5,rtol=2e-5):
                failures.append({'episode':row['episode_id'],'reason':'cached encoder codes'})
            counts['latent_code_frames_checked']+=len(frames)
    report={'raw_storage_passed':not failures,'counts':dict(counts),'failures':failures,'files':files,
        'raw_format':{'shape':['frames',16,16],'model_shape':['frames',1,16,16],
                      'normalization':'raw / 3072','raw_min':int(np.flatnonzero(histogram)[0]),
                      'raw_max':int(np.flatnonzero(histogram)[-1])},
        'upstream_commit':json.loads((output/'upstream_tree.json').read_text())['sha'],
        'layout':{'mapped_channels':int(mask.sum()),'erased_channels':int((~mask).sum()),
                  'erased_list_entries':len(layout['erasedNodes']),'unique_erased_entries':len(erased),
                  'current_training_validity':'finite values only; does not exclude upstream erasedNodes'},
        'global_zero_fraction':float(histogram[0]/histogram.sum()),
        'global_saturation_fraction':float(histogram[-1]/histogram.sum()),
        'current_contact_fraction':float(histogram[154:].sum()/histogram.sum()),
        'mean_adjacent_frame_change_raw':temporal_sum/(temporal_pairs*256),
        'train_frames':train_frames,'code_max_abs_difference':code_difference,
        'seconds':time.monotonic()-began}
    np.save(output/'histogram.npy',histogram)
    np.save(output/'mapped_mask.npy',mask)
    np.save(output/'train_mean_raw.npy',train_sum/train_frames)
    (output/'examples.json').write_text(json.dumps(examples,indent=2)+'\n')
    (output/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('files',)},indent=2))
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
