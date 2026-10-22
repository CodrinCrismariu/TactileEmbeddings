"""Two small code sizes per marker source; screen 30, extend one to 100."""
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .aligned_frames import file_hash,write_json
from .marker_ae_data import snapshot,MarkerAEFrames
from .marker_autoencoder import MarkerAutoencoder,load_marker_autoencoder
from .sam import rng_snapshot,restore_rng
from .train_alignment import atomic_checkpoint


def device_batch(original,device):
    return {k:v.to(device,non_blocking=True) for k,v in original.items()}


@torch.no_grad()
def evaluate(model,loader,device):
    model.eval();absolute=pixels=endpoints=baseline=motion=motion_count=loss_sum=frames=coordinates=0.
    for original in loader:
        b=device_batch(original,device);y=b['marker_target'];valid=b['marker_valid'];pred=model(y,valid,b['marker_reference'],b['marker_surface'])
        loss_sum+=model.loss(pred,y,valid,b['marker_surface']).item()*len(y);frames+=len(y)
        error=(pred-y).abs();spacing=b['marker_spacing'][...,None]
        absolute+=error[valid].sum().item();pixels+=(error*spacing)[valid].sum().item();coordinates+=2*valid.sum().item()
        endpoints+=torch.linalg.vector_norm((pred-y)*spacing,dim=-1)[valid].sum().item()
        baseline+=(y.abs()*spacing)[valid].sum().item()
        active=valid & (torch.linalg.vector_norm(y,dim=-1)>.5*model.displacement_scale)
        motion+=(error*spacing)[active].sum().item();motion_count+=2*active.sum().item()
    return dict(loss=loss_sum/max(frames,1),mae= pixels/max(coordinates,1),mae_px=pixels/max(coordinates,1),
        mae_spacing=absolute/max(coordinates,1),endpoint_error_px=endpoints/max(coordinates/2,1),
        zero_baseline_mae_px=baseline/max(coordinates,1),active_mae_px=motion/max(motion_count,1),
        active_coordinates=int(motion_count),valid_coordinates=int(coordinates),frames=int(frames))


def train_trial(cfg,source,code,pinned,train,val,output,target_epochs,report):
    folder=output/'trials'/source/f'code{code}';folder.mkdir(parents=True,exist_ok=True)
    config=dict(source=source,seed=cfg['seed'],epochs=cfg['final_epochs'],representation='marker_displacement',
        marker_targets=cfg['marker_targets'],manifest=cfg['manifest'],learning_rate=cfg['learning_rate'],
        weight_decay=cfg['weight_decay'],batch_size=cfg['batch_size'],frame_stride=cfg['frame_stride'],
        max_train_frames=cfg['max_train_frames'],latent_dim=code)
    mc=dict(architecture='marker_set',latent_dim=code,width=cfg['width'],queries=cfg['queries'],heads=4,
            displacement_scale=pinned['displacement_scale'])
    provenance=dict(source_snapshot_sha256=pinned['source_snapshot_sha256'],data=pinned['data'],
        train_sampling=train.signature,val_sampling=val.signature,
        implementation={p:file_hash(Path(__file__).with_name(p)) for p in ('marker_autoencoder.py','marker_ae_data.py','train_marker_autoencoders.py','marker_model.py')})
    device=torch.device('cuda');random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
    model=MarkerAutoencoder(**mc).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=cfg['final_epochs'],eta_min=cfg['min_learning_rate'])
    generator=torch.Generator().manual_seed(cfg['seed'])
    loader=DataLoader(train,batch_size=cfg['batch_size'],shuffle=True,generator=generator,num_workers=0,pin_memory=True)
    vloader=DataLoader(val,batch_size=cfg['batch_size'],shuffle=False,num_workers=0,pin_memory=True)
    start=0;best=float('inf')
    if (folder/'last.pt').exists():
        state=torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
        if state['config']!=config or state['model_config']!=mc or state['provenance']!=provenance:raise ValueError('AE resume provenance changed')
        model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        start=state['epoch'];best=state['best_score'];generator.set_state(state['loader_rng']);restore_rng(state['rng'])
        random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng'])
        path=folder/'metrics.jsonl'
        if path.exists():path.write_text(''.join(line+'\n' for line in path.read_text().splitlines() if json.loads(line)['epoch']<=start))
    write_json(folder/'config.json',config);write_json(folder/'source_snapshot.json',pinned)
    for epoch in range(start,target_epochs):
        began=time.monotonic();model.train();total=0.;seen=0
        for step,original in enumerate(loader):
            b=device_batch(original,device);pred=model(b['marker_target'],b['marker_valid'],b['marker_reference'],b['marker_surface'])
            loss=model.loss(pred,b['marker_target'],b['marker_valid'],b['marker_surface'])
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite marker AE loss')
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
            total+=loss.item()*len(pred);seen+=len(pred)
            if step%100==0:report('training',source=source,code_dim=code,epoch=epoch+1,target_epochs=target_epochs,batch=step+1,batches=len(loader),loss=loss.item())
        report('validation',source=source,code_dim=code,epoch=epoch+1,target_epochs=target_epochs)
        metrics=evaluate(model,vloader,device);scheduler.step();best=min(best,metrics['mae_px'])
        record=dict(epoch=epoch+1,train_loss=total/max(seen,1),val=metrics,selection_score=metrics['mae_px'],seconds=time.monotonic()-began)
        state=dict(model_type='marker_autoencoder_v1',model_config=mc,model=model.state_dict(),config=config,
            provenance=provenance,epoch=epoch+1,best_score=best,optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),
            rng=rng_snapshot(),python_rng=random.getstate(),numpy_rng=np.random.get_state(),loader_rng=generator.get_state(),validation=metrics)
        atomic_checkpoint(folder/'last.pt',state)
        if metrics['mae_px']<=best:
            atomic_checkpoint(folder/'best.pt',state);write_json(folder/'best_metrics.json',record)
        with (folder/'metrics.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
        report('epoch_complete',source=source,code_dim=code,**record)
    del model,opt,scheduler,loader,vloader;gc.collect();torch.cuda.empty_cache()
    return json.loads((folder/'best_metrics.json').read_text())['selection_score']


@torch.no_grad()
def pin_teacher(output,source,code,pinned,train,cfg):
    folder=output/'trials'/source/f'code{code}';model,state=load_marker_autoencoder(folder/'best.pt','cuda')
    total=torch.zeros(code,dtype=torch.float64);square=total.clone();count=0
    for original in DataLoader(train,batch_size=cfg['batch_size'],num_workers=0,pin_memory=True):
        b=device_batch(original,'cuda');value=model.encode(b['marker_target'],b['marker_valid'],b['marker_reference'],b['marker_surface']).double().cpu()
        total+=value.sum(0);square+=value.square().sum(0);count+=len(value)
    mean=total/count;std=(square/count-mean.square()).clamp_min(1e-12).sqrt()
    digest=file_hash(folder/'best.pt');destination=output/'teachers'/f'{source}_{digest[:16]}.pt';destination.parent.mkdir(parents=True,exist_ok=True)
    if not destination.exists():shutil.copyfile(folder/'best.pt',destination)
    info=dict(checkpoint=str(destination.resolve()),sha256=digest,model_config=state['model_config'],
        code_dim=code,representation='marker_displacement',source_snapshot_sha256=pinned['source_snapshot_sha256'],
        completed_epochs=cfg['final_epochs'],best_epoch=state['epoch'],validation=state['validation'],
        statistics=dict(mean=mean.float().tolist(),std=std.float().tolist(),fit_frames=count,split='train'),
        target_shape=next(r['row']['target_shape'] for r in pinned['episodes']))
    registry_path=output/'teachers.json';registry=json.loads(registry_path.read_text()) if registry_path.exists() else dict(schema_version=1,sources={},complete=False)
    registry['sources'][source]=info;registry['complete']=set(registry['sources'])==set(cfg['sources']);write_json(registry_path,registry)
    del model;torch.cuda.empty_cache();return info


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',default='configs/marker_autoencoders_v1.json');args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text());output=Path(cfg['output']);output.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(output/'training.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(4);torch.set_float32_matmul_precision('high');started=time.monotonic()
    plan=output/'plan.json'
    if plan.exists() and json.loads(plan.read_text())!=cfg:raise ValueError('AE plan changed')
    write_json(plan,cfg)
    def report(stage,**details):
        row=dict(stage=stage,pid=os.getpid(),elapsed_seconds=time.monotonic()-started,updated_at=time.time(),**details)
        write_json(output/'status.json',row);print(json.dumps(row),flush=True)
    try:
        for source,codes in cfg['sources'].items():
            registry_path=output/'teachers.json'
            if registry_path.exists() and source in json.loads(registry_path.read_text())['sources']:
                info=json.loads(registry_path.read_text())['sources'][source]
                if file_hash(info['checkpoint'])!=info['sha256']:raise ValueError('Pinned AE teacher changed')
                continue
            while True:
                try:pinned=snapshot(cfg['manifest'],cfg['marker_targets'],source);break
                except FileNotFoundError as error:report('waiting_for_marker_targets',source=source,reason=str(error));time.sleep(30)
            report('loading_marker_vectors',source=source)
            train=MarkerAEFrames(pinned,'train',cfg['frame_stride'],cfg['max_train_frames'],cfg['seed'])
            val=MarkerAEFrames(pinned,'val',cfg['frame_stride'],0,cfg['seed'])
            report('data_ready',source=source,train_frames=len(train),validation_frames=len(val),max_markers=train.shape[1],code_candidates=codes)
            scores={code:train_trial(cfg,source,code,pinned,train,val,output,cfg['screen_epochs'],report) for code in codes}
            chosen=min(scores,key=scores.get);report('finalist_selected',source=source,code_dim=chosen,scores=scores)
            train_trial(cfg,source,chosen,pinned,train,val,output,cfg['final_epochs'],report)
            info=pin_teacher(output,source,chosen,pinned,train,cfg);report('teacher_complete',source=source,code_dim=chosen,validation=info['validation'])
            del train,val,pinned;gc.collect()
        report('complete',teachers=str(output/'teachers.json'))
    except Exception as error:report('failed',error=str(error));raise


if __name__=='__main__':main()
