#!/usr/bin/env python3
"""Repair EgoPressure AE training, preserving the original teacher and codes.

One seed, 64D code, native 224x224 targets and single sigmoid pressure output.
The new run starts at epoch zero and never launches joint visual training.
"""
import argparse
from collections import defaultdict
import fcntl
import json
import math
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq
import torch

from tactilelatent.aligned_frames import file_hash,json_hash,write_json
from tactilelatent.data_expansion import ROOT,pressure_targets,pressure_counts,denormalize_pressure
from tactilelatent.expansion_models import load_teacher
from tactilelatent.pressure_teacher import (PressureAutoencoder,pressure_logits,reconstruction_loss,
    initialize_pressure_bias,choose_training_indices,metric_sums,summarize_metrics)
from tactilelatent.train_alignment import atomic_checkpoint
from tactilelatent.pressure_selection import read_task_manifest,task_episodes,VERSION as PRESSURE_SELECTION


def report(folder,**fields):
    status=folder/'status.json';row=json.loads(status.read_text()) if status.exists() else {}
    row.update(fields,updated_at=time.time(),source='egopressure')
    write_json(status,row)
    print(json.dumps(row),flush=True)


def sampling_index(rows,folder):
    rows=task_episodes(rows)
    path=folder/'sampling_index.json';signature=json_hash(rows)
    if path.exists():
        saved=json.loads(path.read_text())
        if saved['rows_sha256']!=signature:raise ValueError('Sampling index native rows changed')
        return saved['episodes']
    result={}
    for number,row in enumerate(rows):
        table=pq.read_table(row['target_path'],columns=['frame','pressure_map_range'])
        ids=table['frame'].to_pylist();ranges=table['pressure_map_range'].to_pylist()
        lookup={int(frame):value for frame,value in zip(ids,ranges)}
        if len(lookup)!=len(ids):raise ValueError('Duplicate native pressure frame')
        valid=[];positive=[]
        for i,frame in enumerate(row['source_frame_ids']):
            value=lookup.get(int(frame))
            if value is None or len(value)!=2 or not np.isfinite(value).all() or value[1]<0:continue
            valid.append(i)
            if value[1]>0:positive.append(i)
        result[row['episode_id']]=dict(valid=valid,positive=positive)
        if number%100==0:report(folder,stage='indexing_native_frames',episodes_done=number+1,episodes=len(rows))
    write_json(path,dict(rows_sha256=signature,episodes=result,
        state_definition='released pressure_map_range peak > 0; actual target masks checked again during loading'))
    return result


def fit_input_statistics(rows,folder):
    rows=task_episodes(rows)
    path=folder/'input_statistics.json'
    if path.exists():
        saved=json.loads(path.read_text())
        if saved.get('rows_sha256')!=json_hash(rows):
            raise ValueError('Encoder input statistics use a different episode selection; use a fresh run folder')
        return saved
    total=squares=0.;count=0
    train=[row for row in rows if row['split']=='train']
    for i,row in enumerate(train):
        ix=np.unique(np.linspace(0,row['frames']-1,8,dtype=int));y=pressure_targets(row,ix)
        x=y[np.isfinite(y)].astype(np.float64);total+=float(x.sum());squares+=float((x*x).sum());count+=len(x)
        if i%100==0:report(folder,stage='fitting_train_input_statistics',episodes_done=i+1,episodes=len(train))
    if not count:raise ValueError('No valid training pressure')
    mean=total/count;scale=max(math.sqrt(max(squares/count-mean*mean,0)),1e-6)
    result=dict(mean=mean,scale=scale,count=count,fit_split='train',frames_per_episode=8,rows_sha256=json_hash(rows),
        decoder_targets_unchanged=True,method='fixed encoder-input zscore after existing bounded normalization')
    write_json(path,result);return result


def batches(rows,index,split,epoch,batch_size,limit):
    selected=[r for r in task_episodes(rows) if r['split']==split and index[r['episode_id']]['valid']]
    rng=np.random.default_rng(45+epoch if split=='train' else 100045)
    if split=='train':rng.shuffle(selected)
    budget=max(1,math.ceil(limit/max(len(selected),1)));seen=0
    for row in selected:
        if seen>=limit:break
        candidates=index[row['episode_id']]
        if split=='train':
            ix=choose_training_indices(candidates['valid'],candidates['positive'],min(budget,limit-seen),rng)
        else:
            valid=np.asarray(candidates['valid'],np.int64)
            ix=valid[np.unique(np.linspace(0,len(valid)-1,min(budget,limit-seen,len(valid)),dtype=int))]
        for start in range(0,len(ix),batch_size):
            selected_indices=ix[start:start+batch_size]
            # Read sorted native frames for bounded Parquet access, then restore
            # the sampled order. This never pairs across episode boundaries.
            order=np.argsort(selected_indices);reverse=np.argsort(order)
            target=pressure_targets(row,selected_indices[order])[reverse]
            valid=np.isfinite(target);supported=valid.reshape(len(target),-1).any(1)
            if not supported.any():continue
            yield torch.from_numpy(target[supported]),torch.from_numpy(valid[supported])
            seen+=int(supported.sum())


@torch.no_grad()
def evaluate(model,rows,index,device,limit=10000):
    model.eval();totals=defaultdict(float)
    for y,v in batches(rows,index,'val',0,32,limit):
        y=y.to(device);v=v.to(device);pred=model(y,v)
        for key,value in metric_sums(pred,y,v).items():totals[key]+=value
    result=summarize_metrics(totals)
    if not result['contact_count']:raise ValueError('Validation contains no supported pressure contacts')
    result.update(mae=result['all_mae'],zero_baseline_mae=result['all_zero_baseline_mae'],
        units='unchanged reversibly bounded pressure')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs',type=int,default=30)
    parser.add_argument('--train-frames',type=int,default=100000)
    parser.add_argument('--val-frames',type=int,default=10000)
    parser.add_argument('--output',type=Path,default=ROOT/'teachers/egopressure_balanced_tasks_v1')
    parser.add_argument('--code-output',type=Path,default=ROOT/'codes/egopressure_balanced_tasks_v1')
    args=parser.parse_args();folder=args.output;folder.mkdir(parents=True,exist_ok=True)
    lock=(folder/'train.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if not torch.cuda.is_available():raise RuntimeError('Production repair requires CUDA')
    torch.set_num_threads(4);torch.manual_seed(45);np.random.seed(45)
    rows=sorted(read_task_manifest(ROOT)['episodes'],key=lambda r:(r['split'],r['episode_id']))
    if any(r['source']!='egopressure' for r in rows):raise ValueError('Unexpected source in pressure teacher')
    if any(int(r['participant'].split('_')[1])>18 for r in rows if r['split']=='train'):
        raise ValueError('Validation participant leaked into training')
    index=sampling_index(rows,folder);stats=fit_input_statistics(rows,folder)
    model=PressureAutoencoder(target_shape=[1,224,224],channels=[16,32,64,128],latent_dim=64,
        input_mean=stats['mean'],input_scale=stats['scale']).cuda();initialize_pressure_bias(model)
    config=dict(source='egopressure',representation='pressure_uv',seed=45,epochs=args.epochs,
        model=model.config,training_version='balanced_v2',learning_rate=1e-4,weight_decay=1e-4,
        batch_size=32,max_train_frames=args.train_frames,max_val_frames=args.val_frames,
        loss='per-frame balanced soft-target logit BCE + balanced L1/0.5MSE + 0.5 global logit BCE',
        positive_sampling_fraction=.75,initial_output_probability=.02,selection='balanced_contact_background_mae',
        rows_sha256=json_hash(rows),pressure_normalization_sha256=file_hash(ROOT/'pressure_normalization.json'))
    config['episode_selection']=PRESSURE_SELECTION
    report(folder,stage='checking_native_target_provenance')
    config['target_snapshot_sha256']=json_hash({r['episode_id']:file_hash(r['target_path']) for r in rows})
    config['implementation']={str(p):file_hash(p) for p in [Path(__file__),
        Path('src/tactilelatent/pressure_teacher.py'),Path('src/tactilelatent/sensor_autoencoder.py'),
        Path('src/tactilelatent/data_expansion.py'),Path('src/tactilelatent/pressure_selection.py')]}
    if (folder/'config.json').exists() and json.loads((folder/'config.json').read_text())!=config:
        raise ValueError('Repair configuration/provenance differs; use a new output folder')
    write_json(folder/'config.json',config)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs,eta_min=1e-6)
    start=0;best=float('inf');last=folder/'last.pt'
    if last.exists():
        state=torch.load(last,map_location='cpu',weights_only=False)
        if state['config']!=config:raise ValueError('Repair checkpoint configuration differs')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        start=state['epoch'];best=state['best_score'];torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
    for epoch in range(start,args.epochs):
        model.train();began=time.monotonic();total=0.;frames=0
        for step,(y,v) in enumerate(batches(rows,index,'train',epoch,32,args.train_frames)):
            y=y.cuda();v=v.cuda();optimizer.zero_grad(set_to_none=True)
            loss=reconstruction_loss(pressure_logits(model,y,v),y,v)
            if not torch.isfinite(loss):raise ValueError('Nonfinite repaired pressure loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            total+=float(loss.item())*len(y);frames+=len(y)
            if step%100==0:report(folder,stage='training',epoch=epoch+1,total_epochs=args.epochs,
                frames=frames,train_loss=total/max(frames,1),elapsed_seconds=time.monotonic()-began)
        report(folder,stage='validation',epoch=epoch+1)
        val=evaluate(model,rows,index,'cuda',args.val_frames);scheduler.step()
        score=val['balanced_mae'];improved=score<best;best=min(best,score)
        record=dict(epoch=epoch+1,source='egopressure',train_loss=total/max(frames,1),val=val,
                    selection_score=score,seconds=time.monotonic()-began)
        state=dict(model_type='expansion_pressure_autoencoder_v2',model_config=model.config,config=config,
            model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
            epoch=epoch+1,best_score=best,validation=val,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
        if improved:
            atomic_checkpoint(folder/'best.pt',state);write_json(folder/'best_metrics.json',record)
        atomic_checkpoint(last,state)
        with (folder/'metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        report(folder,stage='training',**record,total_epochs=args.epochs,
            eta_seconds=record['seconds']*(args.epochs-epoch-1))
    best_record=json.loads((folder/'best_metrics.json').read_text());v=best_record['val']
    accepted=v['contact_to_zero_ratio']<.9 and v['all_mae']<v['all_zero_baseline_mae']
    write_json(folder/'quality.json',dict(accepted=accepted,teacher_sha256=file_hash(folder/'best.pt'),
        reason='must improve both contact MAE and global MAE over zero',metrics=v))
    if not accepted:
        report(folder,stage='blocked_quality',reason='Not yet better than zero on both contact and global MAE; code caching blocked')
        return
    from tactilelatent.expansion_training import build_codes
    report(folder,stage='caching_repaired_codes')
    code_root=args.code_output
    build_codes('egopressure',rows,folder/'best.pt',destination=code_root)
    write_json(folder/'ready.json',dict(ready=True,checkpoint=str(folder/'best.pt'),code_index=str(code_root/'index.json'),
        teacher_sha256=file_hash(folder/'best.pt'),source='egopressure',joint_training_started=False))
    selection=ROOT/'source_teacher_overrides.json'
    overrides=json.loads(selection.read_text()) if selection.exists() else {}
    overrides['egopressure']=dict(checkpoint=str(folder/'best.pt'),code_index=str(code_root/'index.json'),
        teacher_sha256=file_hash(folder/'best.pt'),code_index_sha256=file_hash(code_root/'index.json'),quality=str(folder/'quality.json'))
    write_json(selection,overrides)
    report(folder,stage='complete',quality_accepted=True,joint_training_started=False)


if __name__=='__main__':main()
