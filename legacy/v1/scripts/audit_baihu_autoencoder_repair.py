#!/usr/bin/env python3
"""Compare frozen Baihu teachers and ablate frame codes on the same episodes."""
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from tactilelatent.aligned_frames import file_hash, write_json
from tactilelatent.baihu_ae_data import BaihuAEFrames,evaluate_baihu
from tactilelatent.expansion_training import source_rows
from tactilelatent.expansion_models import load_teacher


def main():
    torch.set_num_threads(4);rows=source_rows('baihu')
    pool=BaihuAEFrames(rows,'val',10000).to('cuda')
    output=Path('processing/audits/baihu_autoencoder_investigation_v1');results={}
    rng=np.random.default_rng(45);episode=pool.data['episode'].cpu().numpy()
    permutation=np.arange(len(pool))
    for e in np.unique(episode):
        ix=np.flatnonzero(episode==e);permutation[ix]=rng.permutation(ix)
    for name,checkpoint in [('failed_v2','processing/data_expansion_v1/teachers/baihu_native_v2/best.pt'),
                            ('repaired_v3','processing/data_expansion_v1/teachers/baihu_native_v3/best.pt')]:
        model,state=load_teacher(checkpoint,'cuda');metrics=evaluate_baihu(model,pool)
        codes=[]
        with torch.inference_mode():
            for start in range(0,len(pool),128):
                b=pool.batch(np.arange(start,min(start+128,len(pool))))
                codes.append(model.encode(b['target'],b['valid'],b['reference'],b['surface']))
            code=torch.cat(codes);mean_code=code.clone()
            for e in pool.data['episode'].unique():
                mask=pool.data['episode']==e;mean_code[mask]=code[mask].mean(0)
            sums=defaultdict(lambda:defaultdict(float))
            for start in range(0,len(pool),128):
                ix=np.arange(start,min(start+128,len(pool)));b=pool.batch(ix)
                target=torch.where(b['valid'][...,None],b['target'],0.)
                predictions={
                    'frame_code':model.decode(code[ix],b['reference'],b['surface']),
                    'within_episode_shuffled_code':model.decode(code[permutation[ix]],b['reference'],b['surface']),
                    'episode_mean_code':model.decode(mean_code[ix],b['reference'],b['surface'])}
                groups={'all':b['valid']}
                for sid in range(4):groups['surface_'+str(sid)]=b['valid']&(b['surface']==sid)
                for platform in ('D-WHEEL','PIKA','QINGLOONG'):
                    m=torch.tensor([pool.rows[int(e)]['platform']==platform for e in b['episode'].cpu()],device='cuda')
                    groups[platform]=b['valid']&m[:,None]
                for group,valid in groups.items():
                    mask=valid[...,None].expand_as(target)
                    sums[group]['coordinates']+=mask.sum().item()
                    sums[group]['zero_mae']+=(target.abs()*b['spacing'])[mask].sum().item()
                    for key,p in predictions.items():
                        sums[group][key]+=((p-target).abs()*b['spacing'])[mask].sum().item()
            comparisons={key:{field:value/max(s['coordinates'],1) for field,value in s.items() if field!='coordinates'}
                         for key,s in sums.items()}
        results[name]=dict(checkpoint=checkpoint,sha256=file_hash(checkpoint),epoch=state['epoch'],
                           metrics=metrics,comparisons=comparisons)
        print(json.dumps(dict(model=name,**results[name])),flush=True)
    results['same_validation_frames']=pool.signature
    results['frame_ablation']='Permutations and means are within each episode; reference geometry and surface identities remain fixed.'
    write_json(output/'full_validation_comparison.json',results)


if __name__=='__main__':main()
