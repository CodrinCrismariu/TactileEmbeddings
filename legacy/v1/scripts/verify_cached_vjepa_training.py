"""Check complete cache integrity, source exclusion, real SAM steps and decoding."""
import json
from pathlib import Path
import time
import torch
from torch.utils.data import DataLoader

from tactilelatent.aligned_frames import write_json, file_hash
from tactilelatent.joint_data import SourceBatchSampler
from tactilelatent.train_vjepa import (prepare_data,make_training_state,train_microbatch,to_device,validate_pair)
from tactilelatent.train_alignment import sam_replay
from tactilelatent.vjepa_model import matched_models
from tactilelatent.vjepa_cached_training import bind_feature_cache,encoded_inputs,check_resume_contract

torch.set_num_threads(4)
cfg=json.loads(Path('configs/joint_vjepa21_pair_32d_cached_v1.json').read_text())
began=time.monotonic();device=torch.device('cuda')
train,val,registry,mc,provenance=prepare_data(cfg)
train,val,backbone_info,cache_info=bind_feature_cache(cfg,train,val,provenance)
provenance['backbone']=backbone_info;provenance['feature_cache']=cache_info
provenance['implementation']={name:file_hash(Path('src/tactilelatent')/name) for name in
 ('train_vjepa.py','vjepa_model.py','alignment_data.py','train_alignment.py','sam.py','train.py','joint_data.py','vjepa_cached_training.py')}
saved=torch.load(Path(cfg['output'])/'pair_state.pt',map_location='cpu',weights_only=False)
check_resume_contract(saved,cfg,provenance,True)
models={n:m.cuda().train() for n,m in matched_models(mc).items()}
for n,m in models.items():
 state=torch.load(Path(cfg['output'])/n/f"epoch_{saved['completed_epochs']:03d}.pt",map_location='cpu',weights_only=False)
 m.load_state_dict(state['model'])
sampler=SourceBatchSampler(train,2,seed=45,max_windows_per_source=64,sampling_mode='cycle')
loader=DataLoader(train,batch_sampler=sampler,num_workers=4,pin_memory=True)
opts,schedulers,sams=make_training_state(models,cfg,len(loader));replay={n:[] for n in models};sources=set()
start=time.monotonic();warm=None;total_batches=0
for step,original in enumerate(loader):
 assert 'rgb' not in original and original['source'][0]!='ftp_motiontrans'
 if step==32:torch.cuda.synchronize();warm=time.monotonic()
 source=original['source'][0];sources.add(source)
 batch=to_device(original,device);encoded=encoded_inputs(None,original,device)
 group_size=min(16,len(loader)-(step//16)*16)
 values=train_microbatch(models,opts,sams,replay,encoded,batch['view_mask'],batch,source,.001,group_size,device)
 assert values['deterministic'][1][2]==0
 if (step+1)%16==0 or step+1==len(loader):
  for n,m in models.items():
   sam_replay(m,opts[n],sams[n],replay[n],device);replay[n].clear();schedulers[n].step()
 total_batches+=1
 assert not any('rgb' in k for k in original)
torch.cuda.synchronize();elapsed=time.monotonic()-start;steady=time.monotonic()-warm
assert len(sources)==11
val_loader=DataLoader(val,batch_sampler=SourceBatchSampler(val,2,seed=45,max_windows_per_source=2),num_workers=4,pin_memory=True)
validation=validate_pair(models,None,val_loader,registry,device)
assert all(len(v)==11 and 'ftp_motiontrans' not in v for v in validation.values())
report=dict(passed=True,source_datasets=sorted(sources),cache_complete=True,backbone_instantiated=False,
 completed_training_batches=total_batches,training_seconds=elapsed,steady_seconds_per_batch=steady/(total_batches-32),
 projected_training_seconds_per_epoch=(steady/(total_batches-32))*31497,
 validation=validation,total_seconds=time.monotonic()-began,cache_provenance=cache_info)
write_json('processing/vjepa2_1/cached_training_preflight.json',report)
print(json.dumps({k:v for k,v in report.items() if k not in ('validation','cache_provenance')}),flush=True)
