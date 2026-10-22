"""Read verified acquisition manifests without touching historical data loaders.

Eight-frame histories use explicit source mappings. Nonchronological camera
histories are rejected; missing targets remain masked. No V-JEPA cache is made.
"""
from functools import lru_cache
import json
from pathlib import Path

import numpy as np

from .aligned_frames import file_hash, write_json
from .media import video_frames


@lru_cache(maxsize=16)
def _targets(path, digest):
    if file_hash(path)!=digest:raise ValueError('Native target hash changed')
    with np.load(path,allow_pickle=False) as f:return {k:f[k] for k in f.files}


def clip_indices(row,end,frames=8,stride=2):
    indices=np.arange(end-(frames-1)*stride,end+1,stride,dtype=np.int64)
    if indices[0]<0 or indices[-1]>=row['frames']:raise IndexError('History crosses episode boundary')
    targets=_targets(row['targets']['path'],row['targets']['sha256'])
    if np.any(np.diff(targets['timestamps'][indices])<=0):raise ValueError('Nonchronological target history')
    if 'rgb_frame_valid' in targets and not targets['rgb_frame_valid'][indices].all():
        raise ValueError('Camera history contains ambiguous/missing acquisition timestamps')
    views={}
    for slot,view in row['views'].items():
        if 'frame_indices_key' in view:
            if view.get('frame_indices_artifact')!=row['targets']['path']:
                raise ValueError('Unexpected external frame map')
            mapped=targets[view['frame_indices_key']][indices]
        else:mapped=indices+int(view.get('frame_offset',0))
        if np.any(mapped<0) or ('frames' in view and np.any(mapped>=view['frames'])) or np.any(np.diff(mapped)<=0):
            raise ValueError('Nonchronological camera history; exclude this clip')
        views[slot]=mapped
    return indices,views,targets


def load_clip(row,end,frames=8,stride=2):
    indices,views,targets=clip_indices(row,end,frames,stride)
    images={slot:video_frames(Path(row['views'][slot]['path']),mapped.tolist(),384)
            for slot,mapped in views.items()}
    return dict(images=images,values=targets['values'][indices],valid=targets['valid'][indices],
        timestamps=targets['timestamps'][indices],frame_indices=indices,
        camera_roles={slot:r['role'] for slot,r in row['views'].items()},
        current_value=targets['values'][end],current_valid=targets['valid'][end])


def finalize_numeric(job):
    """Training-only z-score statistics and a decoded history acceptance check.

    Statistics remain an optional reversible transform. Persisted native values
    are never normalized in place. Different source channel schemas stay apart.
    """
    path=job.root/'manifest.json'
    if not path.exists():return
    manifest=json.loads(path.read_text())
    if manifest.get('schema')!='acquisition-native-v2':return
    accum={};verified=[];failures=[]
    for row in manifest['episodes']:
        target=row['targets'];key=json.dumps(target['channel_order'])
        if row['split']=='train':
            with np.load(target['path'],allow_pickle=False) as f:
                x=f['values'].reshape(row['frames'],-1).astype(np.float64)
                mask=f['valid'].reshape(x.shape)&np.isfinite(x)
            state=accum.setdefault(key,dict(count=np.zeros(x.shape[1],np.int64),
                total=np.zeros(x.shape[1]),squares=np.zeros(x.shape[1])))
            state['count']+=mask.sum(0);v=np.where(mask,x,0)
            state['total']+=v.sum(0);state['squares']+=(v*v).sum(0)
        # Decode one valid history for each distinct source/schema/split.
        group=(row['split'],key,row.get('embodiment'),row.get('environment'))
        if group in verified or row['frames']<15:continue
        try:
            valid_end=None
            native=_targets(target['path'],target['sha256'])
            candidates=np.flatnonzero(native['valid'].reshape(row['frames'],-1).any(1))
            for end in candidates[candidates>=14][:1000]:
                try:clip_indices(row,int(end));valid_end=int(end);break
                except ValueError:continue
            if valid_end is None:raise ValueError('No chronological history among first 1000 supervised candidates')
            batch=load_clip(row,valid_end)
            if len(batch['values'])!=8 or any(len(v)!=8 for v in batch['images'].values()):
                raise ValueError('History decode count differs')
            verified.append(group)
        except Exception as error:failures.append(dict(episode=row['episode_id'],error=str(error)))
    stats=[]
    for key,state in accum.items():
        count=state['count'];mean=state['total']/np.maximum(count,1)
        std=np.sqrt(np.maximum(state['squares']/np.maximum(count,1)-mean*mean,0))
        scale=np.where(std>1e-8,std,1.)
        stats.append(dict(channel_order=json.loads(key),count=count.tolist(),mean=mean.tolist(),
            scale=scale.tolist(),unsupported_channels=(count==0).tolist()))
    write_json(job.root/'normalization.json',dict(fit_split='train',method='per-channel zscore',
        missing_not_zero=True,native_artifacts_unchanged=True,schemas=stats))
    write_json(job.root/'acceptance.json',dict(decoded_history_groups=[list(g) for g in verified],
        failures=failures,normalization_fit_only_on_train=True,frames=8,stride=2,
        joint_training_started=False))
    if failures:raise ValueError('History acceptance failed; see acceptance.json')
