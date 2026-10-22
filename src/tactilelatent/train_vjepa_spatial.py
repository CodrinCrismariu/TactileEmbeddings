"""Live native-data joint training: full V-JEPA tokens, deterministic attention."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, default_collate

from .aligned_frames import file_hash, manifest_signature, write_json
from .alignment_data import AlignedClipDataset
from .joint_all_config import ALL_TARGETS
from .joint_data import SourceBatchSampler
from .sam import SAMStep, rng_snapshot, restore_rng
from .sensor_autoencoder import make_autoencoder
from .sensor_code_cache import verify_teachers
from .train import make_scheduler
from .train_alignment import alignment_loss, atomic_checkpoint, sam_replay, model_weight_hash
from .train_vjepa import LatentStats, valid_frames
from .viewer_checkpoints import save_viewer_epoch
from .vjepa_model import VJEPABackbone, BACKBONE_ID, REVISION
from .vjepa_spatial import MODEL_TYPE, JointSpatialVJEPA, spatial_video_features


def prepare(cfg):
    if (cfg['latent_dim']!=32 or cfg['heldout_source']!='ftp_motiontrans'
            or cfg.get('feature_cache') or cfg.get('code_cache') or cfg.get('beta_max',0)!=0):
        raise ValueError('Expected 32D deterministic live-data experiment without dataset caches')
    signature=manifest_signature(cfg['manifest'])
    audit=json.loads(Path(cfg['audit']).read_text())
    if audit.get('signature')!=signature or not audit.get('passed') or not audit.get('VLA_complete_pixels'):
        raise ValueError('Native chronological data audit failed')
    registry=json.loads(Path(cfg['teachers']).read_text());verify_teachers(registry)
    # Reuse fixed TRAIN-ONLY normalization constants, not cached sample codes.
    # This file is the old epoch-0 configuration, not pretrained task weights.
    normalization=json.loads(Path(cfg['normalization']).read_text())
    if normalization['data']!=signature:
        raise ValueError('Train-only normalization belongs to a different data manifest')
    if normalization['teacher_hashes']!={s:i['sha256'] for s,i in registry['sources'].items()}:
        raise ValueError('Normalization constants refer to different frozen teachers')
    statistics=normalization['statistics']
    sources=sorted(set(registry['sources'])|{'ftp_rdp','ftp_rdp_bimanual'})
    if len(sources)!=11 or cfg['heldout_source'] in sources:raise ValueError('Invalid source split')
    kwargs=dict(sources=sources,frames=8,stride=2,rgb_size=384,include_targets=True)
    train=AlignedClipDataset(cfg['manifest'],'train',**kwargs)
    val=AlignedClipDataset(cfg['manifest'],'val',**kwargs)
    for dataset in (train,val):
        if dataset.caches or set(dataset.indices_by_source)!=set(sources):
            raise ValueError('Live sources incomplete or sample cache enabled')
    mc=dict(sensor_code_dims={s:i['code_dim'] for s,i in registry['sources'].items()},
            numeric_shapes={s:ALL_TARGETS[s][1] for s in sources if s not in registry['sources']},
            statistics=statistics,latent_dim=32,adapter_hidden_dims=cfg['adapter_hidden_dims'],
            latent_mode='deterministic',**cfg['attention'])
    provenance=dict(data=signature,teachers=registry,
        normalization_sha256=file_hash(cfg['normalization']),sample_cache='none',
        tactile_encoding='frozen_encoder_live_fp32_native_targets',
        camera_roles={'default':['scene_or_main','left','right'],'freetacman_slot_1':'scene_or_main'},
        implementation={name:file_hash(Path(__file__).with_name(name)) for name in
            ('vjepa_spatial.py','train_vjepa_spatial.py','vjepa_model.py','alignment_data.py',
             'aligned_frames.py','native_optical.py','sensor_data.py','sensor_autoencoder.py',
             'tactile_autoencoder.py','train_alignment.py','sam.py','train.py','joint_data.py')})
    return train,val,registry,mc,provenance


class LiveTeachers:
    def __init__(self,registry,device,chunk_frames=1):
        self.models={};self.chunk_frames=chunk_frames;self.device=device
        for source,info in registry['sources'].items():
            state=torch.load(info['checkpoint'],map_location='cpu',weights_only=False)
            model=make_autoencoder(state['model_config']);model.load_state_dict(state['model'],strict=True)
            self.models[source]=model.to(device).eval().requires_grad_(False)

    @torch.no_grad()
    def supervision(self,original,model):
        source=original['source'][0]
        y=original['target'];valid=original['target_mask']
        if source not in self.models:
            return dict(target=y.to(self.device),target_mask=valid.to(self.device))
        codes=[]
        with torch.autocast(self.device.type,enabled=False):
            for target,mask in zip(y.flatten(0,1).split(self.chunk_frames),valid.flatten(0,1).split(self.chunk_frames)):
                codes.append(self.models[source].encode(target.to(self.device).float(),mask.to(self.device)))
        raw=torch.cat(codes).reshape(y.shape[0],y.shape[1],-1)
        norm=model.normalization[source]
        return dict(code_target=(raw-norm.mean)/norm.std,
                    code_valid=valid.flatten(2).any(-1).to(self.device))


def training_step(model,backbone,teachers,original,device,group_size,replay,mask=None):
    source=original['source'][0]
    if len(set(original['source']))!=1 or source=='ftp_motiontrans':
        raise ValueError('Mixed sources or held-out data in joint training')
    mask=original['view_mask'].to(device) if mask is None else mask
    with torch.no_grad(),torch.autocast(device.type,torch.bfloat16,enabled=device.type=='cuda'):
        encoded=spatial_video_features(backbone,original,device)
    batch=teachers.supervision(original,model)
    before=rng_snapshot()
    with torch.autocast(device.type,torch.bfloat16,enabled=device.type=='cuda'):
        output=model(encoded,mask,source)
        loss,task,kl=alignment_loss(output,batch,0.)
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite joint loss')
    (loss/group_size).backward()
    # In-memory replay of this optimizer group only; never written as a cache.
    replay.append(dict(features=encoded.detach(),mask=mask,source=source,batch=batch,rng=before,beta=0.))
    return output,batch,dict(loss=loss.item(),supervision=task.item(),kl=kl.item())


@torch.no_grad()
def validate(model,backbone,teachers,loader,device):
    model.eval();totals=defaultdict(lambda:defaultdict(float))
    for original in loader:
        source=original['source'][0]
        with torch.autocast(device.type,torch.bfloat16):
            encoded=spatial_video_features(backbone,original,device)
            output=model(encoded,original['view_mask'].to(device),source)
        supervision=teachers.supervision(original,model)
        _,task,_=alignment_loss(output,supervision,0.)
        row=totals[source];frames=valid_frames(supervision).sum().item()
        row['task']+=task.item()*frames;row['frames']+=frames
        target=original['target'].flatten(0,1);mask=original['target_mask'].flatten(0,1)
        if source in teachers.models:
            predicted=model.raw_code(source,output['sensor_code']).flatten(0,1)
            truth=model.raw_code(source,supervision['code_target']).flatten(0,1)
        else:predicted=output['pred'].float().flatten(0,1)
        for start in range(0,len(target),teachers.chunk_frames):
            end=start+teachers.chunk_frames;y=target[start:end].to(device).float();m=mask[start:end].to(device)
            if source in teachers.models:
                with torch.autocast(device.type,enabled=False):
                    pred=teachers.models[source].decode(predicted[start:end].float())
                    oracle=teachers.models[source].decode(truth[start:end].float())
                row['oracle_absolute']+=((oracle-y).abs()*m).sum().item()
            else:pred=predicted[start:end]
            row['absolute']+=((pred-y).abs()*m).sum().item()
            row['square']+=((pred-y).square()*m).sum().item();row['cells']+=m.sum().item()
    result={}
    for source,row in totals.items():
        result[source]=dict(supervision_loss=row['task']/max(row['frames'],1),
            normalized_mae=row['absolute']/max(row['cells'],1),valid_cells=int(row['cells']))
        if source in teachers.models:
            result[source]['frozen_ae_mae']=row['oracle_absolute']/max(row['cells'],1)
        if ALL_TARGETS[source][0]=='optical_rgb':
            result[source]['psnr']=-10*np.log10(max(row['square']/max(row['cells'],1),1e-15))
    return result


def checkpoint(root,epoch,model,optimizer,scheduler,sam,cfg,mc,provenance,registry,best,validation,seconds,generator):
    state=dict(model_type=MODEL_TYPE,model_config=mc,model=model.state_dict(),config=cfg,
        provenance=provenance,teachers=registry,backbone=BACKBONE_ID,backbone_revision=REVISION,
        epoch=epoch-1,optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),sam=sam.state_dict(),
        best_score=best,validation=validation,elapsed_seconds=seconds,rng=rng_snapshot(),
        python_rng=random.getstate(),numpy_rng=np.random.get_state(),loader_rng=generator.get_state())
    atomic_checkpoint(root/f'epoch_{epoch:03d}.pt',state)
    atomic_checkpoint(root/'last.pt',state);save_viewer_epoch(root/'last.pt',state)
    if validation and np.mean([x['supervision_loss'] for x in validation.values()])<=best:
        atomic_checkpoint(root/'best.pt',state)


def preflight(model,backbone,teachers,train,device,cfg,root):
    """Real native data from every source; test copies, no production update."""
    import copy
    model=copy.deepcopy(model).train();rows=[]
    torch.cuda.reset_peak_memory_stats();start=time.monotonic()
    for source,indices in sorted(train.indices_by_source.items()):
        original=default_collate([train[i] for i in indices[:2]])
        optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
        sam=SAMStep(optimizer,cfg['sam_rho']);replay=[];began=time.monotonic()
        output,batch,loss=training_step(model,backbone,teachers,original,device,1,replay)
        if output['mu'].shape!=(len(original['source']),8,32) or loss['kl']!=0:
            raise AssertionError('Wrong per-frame latent/KL')
        sam_replay(model,optimizer,sam,replay,device);replay.clear()
        torch.cuda.synchronize()
        row=dict(source=source,target_shape=list(original['target'].shape[2:]),
                 cameras=int(original['view_mask'][0,0].sum()),seconds=time.monotonic()-began,**loss)
        rows.append(row);print(json.dumps(dict(event='preflight',**row)),flush=True)
    if any(p.grad is not None for p in backbone.parameters()) or any(
            p.grad is not None for teacher in teachers.models.values() for p in teacher.parameters()):
        raise AssertionError('Frozen encoder received gradients')
    # Measure warmed full input pipeline with a real accumulation-16 SAM group.
    timing=[];selected=['egotouch','ftp_unit','freetacman']
    for source in selected:
        ids=train.indices_by_source[source];optimizer.zero_grad(set_to_none=True);replay=[]
        began=time.monotonic()
        for i in range(cfg['accumulation']):
            original=default_collate([train[ids[(i*2+j)%len(ids)]] for j in range(2)])
            training_step(model,backbone,teachers,original,device,cfg['accumulation'],replay)
        sam_replay(model,optimizer,sam,replay,device);replay.clear();torch.cuda.synchronize()
        timing.append(dict(source=source,batches=cfg['accumulation'],seconds=time.monotonic()-began))
        print(json.dumps(dict(event='benchmark',**timing[-1])),flush=True)
    result=dict(passed=True,sources=rows,timing=timing,seconds=time.monotonic()-start,
                peak_gpu_bytes=torch.cuda.max_memory_allocated(),sample_cache='none',
                config=cfg,implementation={name:file_hash(Path(__file__).with_name(name)) for name in
                    ('train_vjepa_spatial.py','vjepa_spatial.py')})
    validation_items=[default_collate([train[ids[0]]]) for _,ids in sorted(train.indices_by_source.items())]
    result['native_reconstruction_check']=validate(model,backbone,teachers,validation_items,device)
    write_json(root/'preflight.json',result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/joint_vjepa21_spatial_live_32d.json')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--preflight-only',action='store_true')
    args=parser.parse_args();cfg=json.loads(Path(args.config).read_text());root=Path(cfg['output']);root.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(root/'training.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (root/'last.pt').exists() and not args.resume and not args.preflight_only:raise FileExistsError('Use --resume')
    torch.set_num_threads(4);random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
    device=torch.device('cuda')
    write_json(root/'status.json',dict(stage='initializing',pid=os.getpid(),sample_cache='none'))
    train,val,registry,mc,provenance=prepare(cfg)
    backbone=VJEPABackbone(cfg['backbone_repo'],cfg['backbone_checkpoint']).to(device).eval()
    provenance['backbone']={**backbone.provenance,'spatial_pool':'none','camera_pool':'learned_cross_attention',
        'temporal_mapping':'eight_distinct_frame_queries','feature_shape':[4,3,576,768]}
    teachers=LiveTeachers(registry,device,cfg['teacher_chunk_frames'])
    torch.manual_seed(cfg['seed']);model=JointSpatialVJEPA(**mc).to(device)
    if args.preflight_only:
        preflight(model,backbone,teachers,train,device,cfg,root);return
    if not (root/'preflight.json').exists() or not json.loads((root/'preflight.json').read_text()).get('passed'):
        raise RuntimeError('Run and inspect real-data preflight before launch')
    checked=json.loads((root/'preflight.json').read_text())
    if checked.get('config')!=cfg or any(checked.get('implementation',{}).get(name)!=provenance['implementation'][name]
            for name in ('train_vjepa_spatial.py','vjepa_spatial.py')):
        raise RuntimeError('Configuration/implementation changed after preflight')
    train_sampler=SourceBatchSampler(train,cfg['batch_size'],seed=cfg['seed'],max_windows_per_source=cfg['train_windows_per_source_per_epoch'],sampling_mode='cycle')
    val_sampler=SourceBatchSampler(val,cfg['batch_size'],seed=cfg['seed'],max_windows_per_source=cfg['validation_windows_per_source'])
    generator=torch.Generator().manual_seed(cfg['seed'])
    kw=dict(num_workers=cfg['workers'],pin_memory=True,generator=generator)
    train_loader=DataLoader(train,batch_sampler=train_sampler,**kw);val_loader=DataLoader(val,batch_sampler=val_sampler,**kw)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    scheduler=make_scheduler(optimizer,cfg,len(train_loader));sam=SAMStep(optimizer,cfg['sam_rho'])
    start=0;best=float('inf');elapsed=0.
    if args.resume:
        state=torch.load(root/'last.pt',map_location='cpu',weights_only=False)
        if state['config']!=cfg or state['provenance']!=provenance or state['model_config']!=mc:
            raise ValueError('Resume configuration/provenance changed')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler']);sam.load_state_dict(state['sam'])
        start=state['epoch']+1;best=state['best_score'];elapsed=state['elapsed_seconds']
        restore_rng(state['rng']);random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng']);generator.set_state(state['loader_rng'])
        path=root/'metrics.jsonl'
        if path.exists():path.write_text(''.join(line+'\n' for line in path.read_text().splitlines() if json.loads(line)['epoch']<=start))
    else:
        checkpoint(root,0,model,optimizer,scheduler,sam,cfg,mc,provenance,registry,best,{},0.,generator)
        write_json(root/'initialization.json',dict(seed=cfg['seed'],model_sha256=model_weight_hash(model),parameters=sum(p.numel() for p in model.parameters())))
    write_json(root/'config.json',cfg);write_json(root/'provenance.json',provenance)
    print(json.dumps(dict(event='indexed',batches_per_epoch=len(train_loader),trainable_parameters=sum(p.numel() for p in model.parameters()),sample_cache='none',start_epoch=start+1)),flush=True)
    began=time.monotonic()-elapsed
    for epoch in range(start,cfg['epochs']):
        train_sampler.set_epoch(epoch);model.train();optimizer.zero_grad(set_to_none=True)
        totals=defaultdict(lambda:np.zeros(3));counts=defaultdict(int);stats=LatentStats(32);replay=[];epoch_start=time.monotonic();torch.cuda.reset_peak_memory_stats()
        for step,original in enumerate(train_loader):
            source=original['source'][0];mask=original['view_mask'].to(device).clone()
            g=torch.Generator(device=device).manual_seed(cfg['seed']+epoch*1000000+step)
            if mask[:,:,0].all():mask[:,:,1:] &= torch.rand(len(mask),1,2,device=device,generator=g)>=cfg['wrist_dropout']
            group=min(cfg['accumulation'],len(train_loader)-(step//cfg['accumulation'])*cfg['accumulation'])
            output,batch,loss=training_step(model,backbone,teachers,original,device,group,replay,mask)
            stats.add(output,valid_frames(batch));totals[source]+=list(loss.values());counts[source]+=1
            if (step+1)%cfg['accumulation']==0 or step+1==len(train_loader):
                sam_replay(model,optimizer,sam,replay,device);replay.clear();scheduler.step()
            if step%cfg['log_every']==0:
                seconds=time.monotonic()-epoch_start
                status=dict(stage='training',pid=os.getpid(),epoch=epoch+1,completed_epochs=epoch,batch=step+1,batches_per_epoch=len(train_loader),
                    total_epochs=cfg['epochs'],epoch_seconds=seconds,seconds_per_batch=seconds/(step+1),loss=loss,source=source,
                    learning_rate=scheduler.get_last_lr()[0],peak_gpu_bytes=torch.cuda.max_memory_allocated(),sample_cache='none',
                    projected_training_seconds_remaining=(len(train_loader)*(cfg['epochs']-epoch)-step-1)*seconds/(step+1))
                write_json(root/'status.json',status);print(json.dumps(status),flush=True)
        write_json(root/'status.json',dict(stage='validation',pid=os.getpid(),epoch=epoch+1,total_epochs=cfg['epochs']))
        validation=validate(model,backbone,teachers,val_loader,device)
        score=float(np.mean([x['supervision_loss'] for x in validation.values()]));best=min(best,score)
        record=dict(epoch=epoch+1,validation=validation,selection_score=score,
            train={s:dict(zip(('loss','supervision','kl'),v/counts[s])) for s,v in totals.items()},
            latent=stats.result(),seconds=time.monotonic()-began,epoch_seconds=time.monotonic()-epoch_start,uncertainty='none')
        with (root/'metrics.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
        checkpoint(root,epoch+1,model,optimizer,scheduler,sam,cfg,mc,provenance,registry,best,validation,record['seconds'],generator)
        print(json.dumps(record),flush=True)
    write_json(root/'status.json',dict(stage='complete',completed_epochs=cfg['epochs'],total_epochs=cfg['epochs'],seconds=time.monotonic()-began))


if __name__=='__main__':main()
