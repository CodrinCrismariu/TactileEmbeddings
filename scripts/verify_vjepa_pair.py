#!/usr/bin/env python3
"""Production-data preflight: all sources, frozen gradients, native decoding."""
import json
from pathlib import Path
import time
import torch
from torch.utils.data import DataLoader
from tactilelatent.aligned_frames import write_json
from tactilelatent.joint_data import SourceBatchSampler
from tactilelatent.train_alignment import sam_replay
from tactilelatent.train_vjepa import (prepare_data,make_training_state,train_microbatch,to_device,validate_pair)
from tactilelatent.vjepa_model import VJEPABackbone,matched_models,video_features

cfg=json.loads(Path('configs/joint_vjepa21_pair_32d_v1.json').read_text())
torch.set_num_threads(4);device=torch.device('cuda')
train,val,registry,mc,provenance=prepare_data(cfg)
sampler=SourceBatchSampler(train,2,seed=45,max_windows_per_source=2,sampling_mode='cycle')
loader=DataLoader(train,batch_sampler=sampler,num_workers=4,pin_memory=True)
backbone=VJEPABackbone().to(device)
models={n:m.to(device) for n,m in matched_models(mc).items()}
opts,sched,sams=make_training_state(models,cfg,len(loader));replay={n:[] for n in models}
losses={};began=time.monotonic()
for original in loader:
    source=original['source'][0];assert source!='ftp_motiontrans'
    batch=to_device(original,device)
    with torch.no_grad(),torch.autocast('cuda',torch.bfloat16):
        encoded=video_features(backbone,original,device)
    values=train_microbatch(models,opts,sams,replay,encoded,batch['view_mask'],batch,source,.001,len(loader),device)
    losses[source]={n:v[1] for n,v in values.items()}
    print('forward/backward',source,flush=True)
for n,m in models.items():
    sam_replay(m,opts[n],sams[n],replay[n],device);replay[n].clear()
assert all(p.grad is None and not p.requires_grad for p in backbone.parameters())
validation_loader=DataLoader(val,batch_sampler=SourceBatchSampler(val,2,seed=45,max_windows_per_source=2),num_workers=4,pin_memory=True)
validation=validate_pair(models,backbone,validation_loader,registry,device)
assert all(len(rows)==11 and 'ftp_motiontrans' not in rows for rows in validation.values())
report=dict(passed=True,train_losses=losses,validation=validation,seconds=time.monotonic()-began,
    production_batches_per_epoch=len(SourceBatchSampler(train,2,seed=45,max_windows_per_source=8000,sampling_mode='cycle')),
    clips_per_source={s:min(len(ids),8000) for s,ids in train.indices_by_source.items()},
    trainable_parameters={n:sum(p.numel() for p in m.parameters()) for n,m in models.items()},
    frozen_parameters=sum(p.numel() for p in backbone.parameters()),peak_gpu_bytes=torch.cuda.max_memory_allocated())
write_json('processing/vjepa2_1/preflight.json',report)
print(json.dumps(report),flush=True)
