#!/usr/bin/env python3
"""Bounded matched diagnostics for Baihu's collapsed native marker AE.

No production checkpoint/cache is replaced. Native points, vector precision,
surface IDs, and train/validation episode boundaries are preserved.
"""
import argparse
from collections import defaultdict
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

from tactilelatent.aligned_frames import file_hash, json_hash, write_json
from tactilelatent.data_expansion import marker_arrays
from tactilelatent.expansion_models import ExpansionMarkerAutoencoder
from tactilelatent.baihu_autoencoder import BaihuMarkerAutoencoder

SCALE=.06337994120925243


def select(rows, split, count):
    groups=defaultdict(list)
    for row in sorted(rows,key=lambda r:r['episode_id']):
        if row['split']==split:groups[(row['task'],row['platform'])].append(row)
    rng=np.random.default_rng(45 if split=='train' else 100045)
    return [group[int(i)] for key,group in sorted(groups.items())
            for i in rng.choice(len(group),min(count,len(group)),replace=False)]


def pack(rows, quota=128):
    pieces=[];geometry=[];surfaces=[];spacing=[];episode=[];frames=[];groups=[]
    maximum=0
    for row in rows:
        d,_=marker_arrays(row);ix=np.arange(0,row['frames'],2)
        if len(ix)>quota:ix=ix[np.linspace(0,len(ix)-1,quota,dtype=int)]
        y=d['displacement_px'][ix]/d['spacing_px'][None,:,None]
        v=d['valid'][ix]&np.isfinite(y).all(-1)
        pieces.append((torch.from_numpy(y),torch.from_numpy(v)))
        geometry.append(torch.from_numpy(d['reference_uv'].astype(np.float32)))
        surfaces.append(torch.from_numpy(d['surface_id']))
        spacing.append(torch.from_numpy(d['spacing_px']))
        start=len(episode);episode.extend([len(geometry)-1]*len(ix));frames.extend(ix.tolist())
        groups.append(np.arange(start,len(episode)));maximum=max(maximum,y.shape[1])
    n=len(episode);target=torch.zeros(n,maximum,2);valid=torch.zeros(n,maximum,dtype=torch.bool)
    ref=torch.zeros(len(rows),maximum,2);sid=torch.zeros(len(rows),maximum,dtype=torch.long)
    pitch=torch.ones(len(rows),maximum)
    for i,((y,v),indices) in enumerate(zip(pieces,groups)):
        k=y.shape[1];target[indices,:k]=y;valid[indices,:k]=v
        ref[i,:k]=geometry[i];sid[i,:k]=surfaces[i];pitch[i,:k]=spacing[i]
    signature=dict(episodes=[dict(episode_id=r['episode_id'],split=r['split'],
        platform=r['platform'],task=r['task'],tracks_sha256=file_hash(r['marker_path'])) for r in rows],
        frames=frames,episode=episode,native_marker_padding=maximum,quota=quota)
    tensors=dict(target=target,valid=valid,reference=ref,surface=sid,spacing=pitch,
                 episode=torch.tensor(episode,dtype=torch.long))
    return tensors,groups,signature


def batch(data, ids):
    ix=torch.as_tensor(ids,device=data['target'].device,dtype=torch.long)
    e=data['episode'][ix]
    return dict(y=data['target'][ix],v=data['valid'][ix],ref=data['reference'][e],
                surface=data['surface'][e],spacing=data['spacing'][e,...,None])


@torch.no_grad()
def evaluate(model,data,batch_size=128):
    model.eval();total=zero=active=active_zero=delta=cells=active_cells=0.
    codes=[]
    for start in range(0,len(data['target']),batch_size):
        b=batch(data,np.arange(start,min(start+batch_size,len(data['target']))))
        p=model(b['y'],b['v'],b['ref'],b['surface'])
        pz=model(torch.zeros_like(b['y']),b['v'],b['ref'],b['surface'])
        mask=b['v'][...,None].expand_as(b['y']);safe=torch.where(mask,b['y'],0.)
        error=(p-safe).abs()*b['spacing'];baseline=safe.abs()*b['spacing']
        moving=b['v']&(safe.norm(dim=-1)>.5*SCALE);m=moving[...,None].expand_as(safe)
        total+=error[mask].sum().item();zero+=baseline[mask].sum().item();cells+=mask.sum().item()
        active+=error[m].sum().item();active_zero+=baseline[m].sum().item();active_cells+=m.sum().item()
        delta+=((p-pz).abs()*b['spacing'])[mask].sum().item()
        codes.append(model.encode(b['y'],b['v'],b['ref'],b['surface']).cpu())
    c=torch.cat(codes)
    return dict(mae_px=total/cells,zero_mae_px=zero/cells,
        active_mae_px=active/max(active_cells,1),active_zero_mae_px=active_zero/max(active_cells,1),
        zero_input_output_change_px=delta/cells,code_std=float(c.std(0).mean()),
        valid_coordinates=int(cells),frames=len(c),active_coordinates=int(active_cells))


def activations(model,b):
    # Same observation point in both architectures: the output of the GELU
    # directly before the sensor-code linear layer.
    gelu=next(layer for layer in model.to_code if isinstance(layer,torch.nn.GELU))
    captured={}
    def hook(module,args,out):
        a=args[0].detach();h=out.detach()
        captured.update(pre_min=float(a.min()),pre_max=float(a.max()),
                        hidden_zero_fraction=float((h==0).float().mean()))
    handle=gelu.register_forward_hook(hook)
    model(b['y'],b['v'],b['ref'],b['surface']);handle.remove()
    return captured


def affine_oracle(data):
    """24 per-frame coefficients use target data: capacity diagnostic only."""
    total=baseline=cells=0.
    for start in range(0,len(data['target']),128):
        b=batch(data,np.arange(start,min(start+128,len(data['target']))))
        pred=torch.zeros_like(b['y'])
        for s in range(4):
            mask=b['v']&(b['surface']==s)
            x=torch.cat((torch.ones_like(b['ref'][...,:1]),b['ref']),dim=-1)
            xm=x*mask[...,None];ym=torch.nan_to_num(b['y'])*mask[...,None]
            coefficients=torch.linalg.pinv(xm)@ym
            field=x@coefficients;pred=torch.where((b['surface']==s)[...,None],field,pred)
        mask=b['v'][...,None].expand_as(b['y']);safe=torch.where(mask,b['y'],0.)
        total+=((pred-safe).abs()*b['spacing'])[mask].sum().item()
        baseline+=(safe.abs()*b['spacing'])[mask].sum().item();cells+=mask.sum().item()
    return dict(mae_px=total/cells,zero_mae_px=baseline/cells,coefficients=24,
                note='Target-conditioned least-squares oracle; not a trained teacher or transfer result')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('processing/audits/baihu_autoencoder_investigation_v1'))
    p.add_argument('--epochs',type=int,default=8)
    p.add_argument('--overfit-steps',type=int,default=400)
    args=p.parse_args();root=args.output;root.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);began=time.monotonic()
    rows=json.loads(Path('processing/data_expansion_v1/baihu_available.json').read_text())['episodes']
    train,groups,train_sig=pack(select(rows,'train',6));val,_,val_sig=pack(select(rows,'val',2))
    assert not ({r['episode_id'] for r in train_sig['episodes']}&{r['episode_id'] for r in val_sig['episodes']})
    write_json(root/'pilot_sampling.json',dict(train=train_sig,val=val_sig))
    train={k:v.cuda() for k,v in train.items()};val={k:v.cuda() for k,v in val.items()}
    print(json.dumps(dict(stage='data_ready',train_frames=len(train['target']),val_frames=len(val['target']),
                          train_episodes=len(groups),seconds=time.monotonic()-began)),flush=True)
    results={};sizes=[len(g) for g in groups]
    for normalization,mixed in [(False,False),(False,True),(True,False),(True,True)]:
        name=('layernorm' if normalization else 'original')+('_mixed' if mixed else '_episode_batches')
        torch.manual_seed(45)
        cls=BaihuMarkerAutoencoder if normalization else ExpansionMarkerAutoencoder
        model=cls(displacement_scale=SCALE).cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        history=[]
        for epoch in range(args.epochs):
            model.train();rng=np.random.default_rng(45+epoch)
            if mixed:
                order=rng.permutation(len(train['target']));offsets=np.cumsum([0]+sizes)
                batches=[order[offsets[i]:offsets[i+1]] for i in range(len(sizes))]
            else:batches=[groups[i] for i in rng.permutation(len(groups))]
            total=count=0
            for ix in batches:
                b=batch(train,ix);optimizer.zero_grad(set_to_none=True)
                pred=model(b['y'],b['v'],b['ref'],b['surface']);loss=model.loss(pred,b['y'],b['v'],b['surface'])
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
                total+=loss.item()*len(ix);count+=len(ix)
            record=dict(epoch=epoch+1,train_loss=total/count,val=evaluate(model,val),
                        activation=activations(model,b),seconds=time.monotonic()-began)
            history.append(record);print(json.dumps(dict(arm=name,**record)),flush=True)
            write_json(root/(name+'.json'),dict(history=history,seed=45,initialization='same base component weights'))
        results[name]=history[-1]
        torch.save(dict(model=model.state_dict(),model_config=model.config,diagnostic_only=True),root/(name+'.pt'))
        del model,optimizer;torch.cuda.empty_cache()
    # Can either architecture fit one genuinely dynamic native episode at all?
    energy=[]
    for g in groups:
        b=batch(train,g);energy.append(float(torch.nan_to_num(b['y']).square().sum()/b['v'].sum()))
    group=groups[int(np.argmax(energy))];ix=group[np.linspace(0,len(group)-1,min(32,len(group)),dtype=int)]
    b=batch(train,ix)
    for normalization in (False,True):
        name='overfit_'+('layernorm' if normalization else 'original');torch.manual_seed(45)
        cls=BaihuMarkerAutoencoder if normalization else ExpansionMarkerAutoencoder
        model=cls(displacement_scale=SCALE).cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        history=[]
        for step in range(args.overfit_steps):
            model.train();optimizer.zero_grad(set_to_none=True)
            pred=model(b['y'],b['v'],b['ref'],b['surface']);loss=model.loss(pred,b['y'],b['v'],b['surface']);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            if step%50==0 or step+1==args.overfit_steps:
                mask=b['v'][...,None].expand_as(b['y']);safe=torch.where(mask,b['y'],0.)
                with torch.no_grad():pred=model(b['y'],b['v'],b['ref'],b['surface'])
                record=dict(step=step+1,loss=float(loss.detach()),
                    mae_px=float(((pred-safe).abs()*b['spacing'])[mask].mean()),
                    zero_mae_px=float((safe.abs()*b['spacing'])[mask].mean()),activation=activations(model,b))
                history.append(record);print(json.dumps(dict(arm=name,**record)),flush=True)
        results[name]=history[-1];write_json(root/(name+'.json'),history)
    results['affine_oracle']=affine_oracle(val)
    results['sampling']=dict(train_frames=len(train['target']),val_frames=len(val['target']),
        train_episodes=len(groups),val_episodes=len(val_sig['episodes']),sampling_sha256=json_hash([train_sig,val_sig]))
    results['runtime_seconds']=time.monotonic()-began
    write_json(root/'pilot_results.json',results)
    print(json.dumps(dict(stage='complete',results=results)),flush=True)


if __name__=='__main__':main()
