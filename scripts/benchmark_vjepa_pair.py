#!/usr/bin/env python3
"""Time complete frozen-feature + SAM training updates on matched real clips."""
import gc
import json
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from tactilelatent.aligned_frames import write_json
from tactilelatent.alignment_data import AlignedClipDataset
from tactilelatent.alignment_model import JointSensorAlignment
from tactilelatent.model import DinoBackbone
from tactilelatent.train import features
from tactilelatent.train_alignment import sam_replay
from tactilelatent.train_vjepa import prepare_data, make_training_state, train_microbatch, to_device
from tactilelatent.vjepa_model import VJEPABackbone, matched_models, video_features


def main():
    torch.set_num_threads(4)
    cfg=json.loads(Path('configs/joint_vjepa21_pair_32d_v1.json').read_text())
    data,_,registry,model_cfg,provenance=prepare_data(cfg)
    cases={}
    for index,(ep,_) in enumerate(data.windows):
        views=len(data.episodes[ep]['views'])
        if views==1 and 'one_view' not in cases:cases['one_view']=data.episodes[ep]['source']
        if views==3 and 'three_views' not in cases:cases['three_views']=data.episodes[ep]['source']
        if len(cases)==2:break
    print(json.dumps(dict(event='cases',cases=cases)),flush=True)
    report={'config':cfg,'cases':cases,'measurements':[],'warmup_batches':16,'measured_batches':32,
            'batch_size':2,'accumulation':16,'includes':'data loading + frozen forward + task forward/backward + SAM replay/update',
            'excludes':'startup, checkpoint writes, native decoder validation'}
    device=torch.device('cuda')
    for architecture in ('dino_single','vjepa_single','vjepa_pair'):
        is_dino=architecture.startswith('dino')
        if is_dino:
            backbone=DinoBackbone().to(device).eval()
            models={'stochastic':JointSensorAlignment(**model_cfg,width=768,heads=8,temporal_layers=3,
                      cross_view_layers=2,dropout=.1,stochastic=True).to(device)}
        else:
            backbone=VJEPABackbone().to(device).eval()
            models={n:m.to(device) for n,m in matched_models(model_cfg).items() if architecture=='vjepa_pair' or n=='stochastic'}
        for label,source in cases.items():
            dataset=AlignedClipDataset(cfg['manifest'],'train',[source],include_targets=False,
                code_cache=cfg['code_cache'],registry=registry,data_signature=provenance['data'],rgb_size=224 if is_dino else 384)
            # Match exact episode/start indices across architectures, retaining all views.
            eligible=[i for i,(ep,_) in enumerate(dataset.windows) if len(dataset.episodes[ep]['views'])==(1 if label=='one_view' else 3)]
            batches=[[eligible[(2*i+j)%len(eligible)] for j in range(2)] for i in range(48)]
            loader=DataLoader(dataset,batch_sampler=batches,num_workers=4,pin_memory=True)
            opts,schedulers,sams=make_training_state(models,cfg,len(loader))
            replay={n:[] for n in models}
            for m in models.values():m.train()
            began=None;peak=0
            for step,original in enumerate(loader):
                if step==16:
                    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();began=time.monotonic()
                batch=to_device(original,device)
                with torch.no_grad(),torch.autocast('cuda',torch.bfloat16):
                    encoded=features(backbone,original,device) if is_dino else video_features(backbone,original,device)
                values=train_microbatch(models,opts,sams,replay,encoded,batch['view_mask'],batch,source,.001,16,device)
                if (step+1)%16==0:
                    for n,m in models.items():
                        sam_replay(m,opts[n],sams[n],replay[n],device);replay[n].clear();schedulers[n].step()
            torch.cuda.synchronize();elapsed=time.monotonic()-began
            item=dict(architecture=architecture,case=label,source=source,seconds=elapsed,
                seconds_per_batch=elapsed/32,clips_per_second=64/elapsed,peak_gpu_bytes=torch.cuda.max_memory_allocated())
            report['measurements'].append(item);print(json.dumps(item),flush=True)
            write_json('processing/vjepa2_1/benchmark.json',report)
            del opts,schedulers,sams,replay,values,encoded,batch,original
        del backbone,models,m
        gc.collect();torch.cuda.empty_cache()
    report['completed']=True
    write_json('processing/vjepa2_1/benchmark.json',report)


if __name__=='__main__':main()
