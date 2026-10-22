"""Fixed native Baihu frame pools with reproducible cross-episode shuffling."""
import json
from pathlib import Path

import numpy as np
import torch

from .aligned_frames import file_hash, json_hash
from .data_expansion import marker_arrays


class BaihuAEFrames:
    def __init__(self, rows, split, limit, seed=45):
        self.rows=sorted([r for r in rows if r['split']==split],key=lambda r:r['episode_id'])
        if not self.rows or any(r['source']!='baihu' for r in self.rows):
            raise ValueError('Expected episode-disjoint Baihu-only frame pool')
        if limit<1:raise ValueError('Frame budget must be positive')
        sampled={}
        if split=='train':
            counts=np.array([(r['frames']+1)//2 for r in self.rows],np.int64)
            offsets=np.r_[0,counts.cumsum()]
            chosen=np.sort(np.random.default_rng(seed).choice(int(offsets[-1]),
                           min(limit,int(offsets[-1])),replace=False))
            episodes=np.searchsorted(offsets,chosen,side='right')-1
            for e in np.unique(episodes):sampled[int(e)]=2*(chosen[episodes==e]-offsets[e])
        else:
            # Reproduce the previous 10k validation sampler exactly, including
            # its native frame IDs and equal per-episode quota.
            quota=max(1,limit//len(self.rows));remaining=limit
            for e,row in enumerate(self.rows):
                ix=np.arange(0,row['frames'],2)
                if len(ix)>quota:ix=np.unique(np.linspace(0,row['frames']-1,quota,dtype=int))
                ix=ix[:remaining]
                if len(ix):sampled[e]=ix;remaining-=len(ix)
        metadata=[json.loads(Path(r['marker_path']).with_name('complete.json').read_text()) for r in self.rows]
        maximum=max(m['markers'] for m in metadata);n=sum(len(v) for v in sampled.values());e=len(self.rows)
        self.data=dict(target=torch.zeros(n,maximum,2),valid=torch.zeros(n,maximum,dtype=torch.bool),
            reference=torch.zeros(e,maximum,2),surface=torch.zeros(e,maximum,dtype=torch.long),
            spacing=torch.ones(e,maximum),episode=torch.empty(n,dtype=torch.long),frame=torch.empty(n,dtype=torch.long))
        self.records=[];start=0
        for e,indices in sampled.items():
            row=self.rows[e];d,info=marker_arrays(row);k=len(d['surface_id']);end=start+len(indices)
            y=d['displacement_px'][indices]/d['spacing_px'][None,:,None]
            valid=d['valid'][indices]&np.isfinite(y).all(-1)
            self.data['target'][start:end,:k]=torch.from_numpy(y)
            self.data['valid'][start:end,:k]=torch.from_numpy(valid)
            self.data['reference'][e,:k]=torch.from_numpy(d['reference_uv'].astype(np.float32))
            self.data['surface'][e,:k]=torch.from_numpy(d['surface_id'])
            self.data['spacing'][e,:k]=torch.from_numpy(d['spacing_px'])
            self.data['episode'][start:end]=e;self.data['frame'][start:end]=torch.from_numpy(indices)
            self.records.append(dict(episode_id=row['episode_id'],split=split,indices=indices.tolist(),
                                     tracks_sha256=info['tracks_sha256']))
            start=end
        supported=self.data['valid'].any(-1)
        for key in ('target','valid','episode','frame'):self.data[key]=self.data[key][supported]
        self.signature=dict(split=split,seed=seed,limit=limit,frame_stride=2,
            sampling='global_uniform_without_replacement' if split=='train' else 'legacy_equal_episode_validation',
            records_sha256=json_hash(self.records),frames=len(self),episodes=len(self.records),
            native_marker_padding=maximum,unsupported_frames=int((~supported).sum()))

    def __len__(self):return len(self.data['target'])

    def to(self,device):
        self.data={k:v.to(device) for k,v in self.data.items()}
        return self

    def batch(self,indices):
        ix=torch.as_tensor(indices,device=self.data['target'].device,dtype=torch.long)
        episode=self.data['episode'][ix]
        return dict(target=self.data['target'][ix],valid=self.data['valid'][ix],
            reference=self.data['reference'][episode],surface=self.data['surface'][episode],
            spacing=self.data['spacing'][episode,...,None],episode=episode,frame=self.data['frame'][ix])


@torch.no_grad()
def evaluate_baihu(model,pool,batch_size=128):
    model.eval();absolute=baseline=cells=active=active_zero=active_count=dependence=loss_sum=frames=0.
    change=change_cells=0.;codes=[];episodes=[]
    for start in range(0,len(pool),batch_size):
        b=pool.batch(np.arange(start,min(start+batch_size,len(pool))))
        y,v=b['target'],b['valid'];p=model(y,v,b['reference'],b['surface'])
        code=model.encode(y,v,b['reference'],b['surface']);codes.append(code.cpu());episodes.append(b['episode'].cpu())
        pz=model(torch.zeros_like(y),v,b['reference'],b['surface'])
        mask=v[...,None].expand_as(y);safe=torch.where(mask,y,0.)
        error=(p-safe).abs()*b['spacing'];zero=safe.abs()*b['spacing']
        moving=v&(safe.norm(dim=-1)>.5*model.displacement_scale)
        mm=moving[...,None].expand_as(safe)
        absolute+=error[mask].sum().item();baseline+=zero[mask].sum().item();cells+=mask.sum().item()
        active+=error[mm].sum().item();active_zero+=zero[mm].sum().item();active_count+=mm.sum().item()
        dependence+=((p-pz).abs()*b['spacing'])[mask].sum().item()
        if len(y)>1:
            adjacent=mask[1:]&mask[:-1]&(b['episode'][1:]==b['episode'][:-1])[:,None,None]
            delta=((p[1:]-p[:-1])-(safe[1:]-safe[:-1])).abs()*b['spacing'][1:]
            change+=delta[adjacent].sum().item();change_cells+=adjacent.sum().item()
        loss_sum+=model.loss(p,y,v,b['surface']).item()*len(y);frames+=len(y)
    if not cells:raise ValueError('No valid Baihu validation coordinates')
    c=torch.cat(codes).double();e=torch.cat(episodes)
    centered=c-c.mean(0);eigen=torch.linalg.eigvalsh(centered.T@centered/max(len(c)-1,1)).clamp_min(0)
    probabilities=eigen/eigen.sum().clamp_min(1e-30)
    rank=float(torch.exp(-(probabilities*probabilities.clamp_min(1e-30).log()).sum())) if eigen.sum()>0 else 0.
    episode_spread=[float(c[e==sid].std(0).mean()) for sid in e.unique() if (e==sid).sum()>1]
    return dict(mae=absolute/cells,zero_baseline_mae=baseline/cells,loss=loss_sum/frames,
        active_mae=active/max(active_count,1),active_zero_baseline_mae=active_zero/max(active_count,1),
        active_coordinates=int(active_count),frames=frames,valid_coordinates=int(cells),units='native_pixels',
        zero_input_output_change_px=dependence/cells,code_std=float(c.std(0).mean()),
        within_episode_code_std=float(np.mean(episode_spread)),effective_rank=rank,
        sampled_temporal_change_mae=change/max(change_cells,1))
