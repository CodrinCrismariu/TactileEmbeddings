"""Bounded source-only diagnostics on copies; no production training resumes."""
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from tactilelatent.aligned_frames import write_json
from tactilelatent.joint_data import SourceBatchSampler
from tactilelatent.train_vjepa import prepare_data,valid_frames
from tactilelatent.vjepa_cache import VJEPAFeatureCache,CachedAlignedClips
from tactilelatent.vjepa_model import model_from_state
from tactilelatent.alignment_model import JointSensorAlignment
from tactilelatent.train_alignment import alignment_loss,model_weight_hash,sam_replay
from tactilelatent.sam import SAMStep,rng_snapshot

ROOT=Path('processing/diagnostics/vjepa_learning_v1')
ROOT.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(4);device=torch.device('cuda');began=time.monotonic()
cfg=json.load(open('configs/joint_vjepa21_pair_32d_cached_v1.json'))
train,val,registry,mc,provenance=prepare_data(cfg)
cache=VJEPAFeatureCache(cfg['feature_cache'])
train,val=CachedAlignedClips(train,cache),CachedAlignedClips(val,cache)
parent=Path(cfg['output'])/'deterministic'
initial_state=torch.load(parent/'epoch_000.pt',map_location='cpu',weights_only=False,mmap=True)
final_state=torch.load(parent/'epoch_030.pt',map_location='cpu',weights_only=False,mmap=True)
model=model_from_state(final_state).to(device).eval()
weight_hash_before=model_weight_hash(model)
report=dict(scope='Source-only bounded diagnostics. No MotionTrans evaluation, no production model changes.',
            weights_before=weight_hash_before,parameter_updates={},gradients={},sources={},overfit_copies={})
for name,p in model.named_parameters():
    old=initial_state['model'][name].float();now=p.detach().cpu().float()
    report['parameter_updates'][name]=dict(relative_l2=float((now-old).norm()/old.norm().clamp_min(1e-12)),
        absolute_l2=float((now-old).norm()),requires_grad=p.requires_grad)
old_dino=torch.load('processing/runs/joint_alignment_holdout_motiontrans/viewer_epochs/epoch_030.pt',map_location='cpu',weights_only=False,mmap=True)
dino=JointSensorAlignment(**old_dino['model_config'])
report['capacity']=dict(vjepa_trainable=sum(p.numel() for p in model.parameters()),dino_trainable=sum(p.numel() for p in dino.parameters()),
    vjepa_lr_horizon=final_state['config']['epochs'],dino_epoch30_lr_horizon=old_dino['config']['epochs'],
    optimizer_updates=final_state['sam']['steps'],final_lr=final_state['optimizer']['param_groups'][0]['lr'])
del dino,old_dino
samples={'train':defaultdict(lambda:defaultdict(list)),'val':defaultdict(lambda:defaultdict(list))}
mini={};gradient_vectors={}
for split,dataset,limit in [('train',train,256),('val',val,64)]:
    sampler=SourceBatchSampler(dataset,16,seed=45,max_windows_per_source=limit)
    loader=DataLoader(dataset,batch_sampler=sampler,num_workers=4,pin_memory=True)
    for original in loader:
        source=original['source'][0]
        assert source!='ftp_motiontrans' and 'rgb' not in original
        x=original['video_features'].to(device);mask=original['view_mask'].to(device)
        batch={k:v.to(device) if torch.is_tensor(v) else v for k,v in original.items() if k!='video_features'}
        with torch.no_grad():
            context=model.visual.encode_context(x,mask)
            out=model(x,mask,source,sample_latent=False)
        y=batch['code_target'] if 'code_target' in batch else batch['target'].flatten(2)
        valid=valid_frames(batch)
        row=samples[split][source]
        row['raw'].append(context.repeat_interleave(2,dim=1).cpu())
        row['mu'].append(out['mu'].cpu());row['target'].append(y.cpu());row['valid'].append(valid.cpu())
        row['pred'].append(out['sensor_code'].cpu() if 'sensor_code' in out else out['pred'].flatten(2).cpu())
        if split=='train' and source not in mini:
            mini[source]=(x[:4].detach(),mask[:4].detach(),{k:v[:4].detach() for k,v in batch.items() if torch.is_tensor(v) and k in ('code_target','code_valid','target','target_mask')})
            model.zero_grad(set_to_none=True)
            result=model(x,mask,source,sample_latent=False)
            loss,_,_=alignment_loss(result,batch,0.)
            loss.backward()
            vector=model.visual.prior_mu.weight.grad.detach().flatten().float().cpu()
            gradient_vectors[source]=vector
            report['gradients'][source]=dict(loss=float(loss.detach()),projection_gradient_norm=float(vector.norm()),
                head_gradient_norm=float(torch.stack([p.grad.norm() for n,p in model.named_parameters() if p.grad is not None and n.startswith(('adapters.','numeric_heads.'))]).norm()),
                finite=all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None))
            model.zero_grad(set_to_none=True)
    for source,row in samples[split].items():
        for key,parts in list(row.items()):row[key]=torch.cat(parts)
    print('loaded',split,flush=True)


def metrics(pred,y,valid,numeric):
    error=pred[valid]-y[valid]
    return dict(loss=float((error.square()+(.5*error.abs() if numeric else 0)).mean()),mae=float(error.abs().mean()),mse=float(error.square().mean()))


def ridge_predict(a,b,key):
    x=a[key].flatten(0,1).to(device);v=b[key].flatten(0,1).to(device)
    y=a['target'].flatten(0,1).to(device)
    valid=a['valid'].flatten().to(device)
    slots=torch.arange(8).repeat(len(a[key])).to(device)%2
    val_slots=torch.arange(8).repeat(len(b[key])).to(device)%2
    prediction=torch.empty(len(v),y.shape[-1],device=device)
    for slot in (0,1):
        selected=valid&(slots==slot);xx=x[selected];yy=y[selected]
        mean=xx.mean(0);std=xx.std(0,unbiased=False).clamp_min(1e-6)
        xx=(xx-mean)/std;vv=(v[val_slots==slot]-mean)/std
        ym=yy.mean(0);matrix=xx.T@xx+len(xx)*.01*torch.eye(xx.shape[1],device=device)
        beta=torch.linalg.solve(matrix,xx.T@(yy-ym))
        prediction[val_slots==slot]=vv@beta+ym
    return prediction.reshape(len(b[key]),8,-1).cpu()

for source,a in samples['train'].items():
    b=samples['val'][source];numeric=source not in mc['sensor_code_dims']
    y=a['target'][a['valid']]
    mean=y.mean(0)
    base=torch.zeros_like(b['target']) if not numeric else mean.expand_as(b['target'])
    raw=ridge_predict(a,b,'raw');z=ridge_predict(a,b,'mu')
    if numeric:raw=raw.clamp(0,1);z=z.clamp(0,1)
    adjacent=b['valid'][:,1:]&b['valid'][:,:-1]
    pred_change=(b['pred'][:,1:]-b['pred'][:,:-1])[adjacent]
    true_change=(b['target'][:,1:]-b['target'][:,:-1])[adjacent]
    pooled=a['raw'][:,::2]
    within=(pooled-pooled.mean(1,keepdim=True)).square().mean()
    total=(pooled-pooled.mean((0,1),keepdim=True)).square().mean()
    report['sources'][source]=dict(numeric=numeric,train_frames=int(a['valid'].sum()),validation_frames=int(b['valid'].sum()),
        current_train=metrics(a['pred'],a['target'],a['valid'],numeric),current_validation=metrics(b['pred'],b['target'],b['valid'],numeric),
        zero_code_or_train_mean_validation=metrics(base,b['target'],b['valid'],numeric),
        ridge_raw768_validation=metrics(raw,b['target'],b['valid'],numeric),ridge_latent32_validation=metrics(z,b['target'],b['valid'],numeric),
        raw_feature_mean_norm=float(pooled.mean((0,1)).norm()),raw_feature_coordinate_std_median=float(pooled.flatten(0,1).std(0).median()),
        raw_within_clip_variance_fraction=float(within/total.clamp_min(1e-12)),
        predicted_temporal_rms=float(pred_change.square().mean().sqrt()),target_temporal_rms=float(true_change.square().mean().sqrt()),
        temporal_prediction_error_rms=float((pred_change-true_change).square().mean().sqrt()))
    print('ridge',source,report['sources'][source]['current_validation']['loss'],report['sources'][source]['ridge_raw768_validation']['loss'],flush=True)
    write_json(ROOT/'partial.json',report)
labels=sorted(gradient_vectors)
g=torch.stack([gradient_vectors[s] for s in labels]);g=g/g.norm(dim=1,keepdim=True).clamp_min(1e-12)
cosine=g@g.T;off=~torch.eye(len(labels),dtype=torch.bool)
report['projection_gradient_cosines']=dict(sources=labels,matrix=cosine.tolist(),negative_pair_fraction=float((cosine[off]<0).float().mean()),mean_offdiagonal=float(cosine[off].mean()))

# Local optimization test on copies, retaining SAM and the same loss/weight decay.
# A repeated four-clip batch tests fit capacity and optimizer speed, not generalization.
for source in ('egotouch','ftp_rdp','opengraph'):
    x,mask,batch=mini[source]
    results={}
    for lr in (5e-5,1e-3):
        copy_model=model_from_state(final_state).to(device).train()
        opt=torch.optim.AdamW(copy_model.parameters(),lr=lr,weight_decay=.05);sam=SAMStep(opt,.05)
        trajectory=[]
        for step in range(201):
            opt.zero_grad(set_to_none=True);rng=rng_snapshot()
            out=copy_model(x,mask,source,sample_latent=False);loss,_,_=alignment_loss(out,batch,0.)
            if step in (0,20,50,100,200):trajectory.append(dict(update=step,loss=float(loss.detach())))
            if step==200:break
            loss.backward()
            sam_replay(copy_model,opt,sam,[dict(features=x,mask=mask,source=source,batch=batch,rng=rng,beta=0.)],device)
        results[str(lr)]=trajectory;del copy_model,opt,sam;torch.cuda.empty_cache()
    report['overfit_copies'][source]=results
    print('bounded_overfit',source,results,flush=True)
report['weights_after']=model_weight_hash(model)
assert report['weights_before']==report['weights_after']
report['seconds']=time.monotonic()-began
report['note']='Ridge uses 256 sampled source-training clips and fixed 64 validation clips, separate frame-pair slots, fit-only standardization, fixed alpha=.01. It is a diagnostic, not an optimized source baseline. Tiny-batch fits use copies and cannot establish validation gains.'
write_json(ROOT/'results.json',report)
print('complete',report['seconds'],flush=True)
