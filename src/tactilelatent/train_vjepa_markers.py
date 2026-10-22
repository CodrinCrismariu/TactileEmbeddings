"""Live full-spatial V-JEPA training with optical marker-displacement targets."""
import argparse
from collections import defaultdict
import copy
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .aligned_frames import file_hash,write_json,manifest_signature
from .alignment_data import AlignedClipDataset
from .sensor_code_cache import verify_teachers
from .joint_all_config import ALL_TARGETS
from .joint_data import SourceBatchSampler
from .marker_targets import MARKER_SOURCES
from .marker_data import MarkerCache,MarkerTargetClips,marker_collate
from .marker_model import MODEL_TYPE,MarkerSpatialVJEPA,spatial_marker_loss
from .sam import SAMStep,rng_snapshot,restore_rng
from .train import make_scheduler
from .train_alignment import atomic_checkpoint,model_weight_hash,alignment_scheduler
from .train_vjepa import LatentStats
from .train_vjepa_spatial import prepare as prepare_native,LiveTeachers,validate as validate_native
from .viewer_checkpoints import save_viewer_epoch
from .vjepa_model import VJEPABackbone,BACKBONE_ID,REVISION
from .vjepa_spatial import spatial_video_features
from .marker_code_model import MODEL_TYPE as CODE_MODEL_TYPE,MarkerCodeVJEPA,MixedMarkerTeachers
from .train_alignment import alignment_loss
from .mirror_joint import augment as augment_mirror_batch, paired_validate as validate_mirror_pairs


def prepare_code_inputs(cfg):
    """Keep only the five retained native teachers; never require old RGB AEs."""
    if (type(cfg['latent_dim']) is not int or cfg['latent_dim']<1
            or cfg['heldout_source']!='ftp_motiontrans' or cfg.get('stochastic')
            or cfg['beta_max']!=0 or cfg.get('feature_cache')):
        raise ValueError('Expected a positive-dimensional deterministic live visual model with MotionTrans held out')
    signature=manifest_signature(cfg['manifest']);audit=json.loads(Path(cfg['audit']).read_text())
    if signature != audit.get('signature'):
        # Explicit archive-retirement receipts preserve the logical marker and
        # code identities while physical archive sizes and loader code change.
        from .storage import compatible_data
        if compatible_data(cfg['manifest'],audit.get('signature')):
            signature=audit['signature']
    if audit.get('signature')!=signature or not audit.get('passed') or not audit.get('VLA_complete_pixels'):
        raise ValueError('Canonical chronological data audit failed')
    registry=json.loads(Path(cfg['teachers']).read_text())
    registry={**registry,'sources':{s:v for s,v in registry['sources'].items() if s not in MARKER_SOURCES}}
    verify_teachers(registry)
    normalization=json.loads(Path(cfg['normalization']).read_text())
    if normalization['data']!=signature or any(normalization['teacher_hashes'].get(s)!=i['sha256'] for s,i in registry['sources'].items()):
        raise ValueError('Native teacher normalization changed')
    excluded=set(cfg.get('excluded_sources',[]))
    registry['sources']={s:v for s,v in registry['sources'].items() if s not in excluded}
    sources=sorted((set(registry['sources'])|set(MARKER_SOURCES)|{'ftp_rdp','ftp_rdp_bimanual'})-excluded)
    if len(sources)!=11-len(excluded) or 'ftp_motiontrans' in sources:raise ValueError('Invalid source split')
    options=dict(sources=sources,frames=8,stride=2,rgb_size=384,include_targets=True)
    train=AlignedClipDataset(cfg['manifest'],'train',**options);val=AlignedClipDataset(cfg['manifest'],'val',**options)
    mc=dict(sensor_code_dims={s:i['code_dim'] for s,i in registry['sources'].items()},
        numeric_shapes={s:ALL_TARGETS[s][1] for s in ('ftp_rdp','ftp_rdp_bimanual')},
        statistics={s:normalization['statistics'][s] for s in registry['sources']},
        latent_dim=cfg['latent_dim'],adapter_hidden_dims=cfg['adapter_hidden_dims'],latent_mode='deterministic',**cfg['attention'])
    if cfg.get('prediction_mode','all_frames')!='all_frames':
        if cfg['prediction_mode']!='current_frame':raise ValueError('Unknown prediction mode')
        mc['prediction_mode']=cfg['prediction_mode']
    files=('vjepa_spatial.py','train_vjepa_spatial.py','vjepa_model.py','alignment_data.py','aligned_frames.py',
           'native_optical.py','sensor_data.py','sensor_autoencoder.py','tactile_autoencoder.py','train_alignment.py','sam.py','train.py','joint_data.py')
    provenance=dict(data=signature,teachers=registry,normalization_sha256=file_hash(cfg['normalization']),
        camera_roles={'default':['scene_or_main','left','right'],'freetacman_slot_1':'scene_or_main'},
        implementation={name:file_hash(Path(__file__).with_name(name)) for name in files})
    return train,val,registry,mc,provenance


def prepare(cfg,allow_pilot=False):
    train,val,registry,mc,provenance=(prepare_code_inputs(cfg) if cfg.get('marker_teachers') else prepare_native(cfg))
    active_markers=set(MARKER_SOURCES)-set(cfg.get('excluded_sources',[]))
    root=Path(cfg['marker_targets']);index=json.loads((root/'index.json').read_text())
    if index['provenance']['data']!=provenance['data'] or not index['all_episodes_complete']:
        raise ValueError('Marker extraction is incomplete, rejected episodes exist, or data provenance differs')
    if index['provenance']['pilot'] and not allow_pilot:
        raise ValueError('Pilot targets are for preflight only')
    available={(r['source'],r['split'],r['episode_id']) for r in index['episodes']}
    for dataset in (train,val):
        expected={(r['source'],r['split'],r['episode_id']) for r in dataset.episodes if r['source'] in MARKER_SOURCES}
        if not expected<=available and not allow_pilot:
            raise ValueError('Missing marker episodes; do not silently drop or substitute RGB supervision')
        if allow_pilot:
            dataset.windows=[(e,s) for e,s in dataset.windows if dataset.episodes[e]['source'] not in MARKER_SOURCES
                or (dataset.episodes[e]['source'],dataset.episodes[e]['split'],dataset.episodes[e]['episode_id']) in available]
            dataset.indices_by_source=defaultdict(list)
            for i,(e,_) in enumerate(dataset.windows):dataset.indices_by_source[dataset.episodes[e]['source']].append(i)
    registry=copy.deepcopy(registry)
    registry['sources']={s:v for s,v in registry['sources'].items() if s not in MARKER_SOURCES}
    mc['sensor_code_dims']={s:v for s,v in mc['sensor_code_dims'].items() if s not in MARKER_SOURCES}
    mc['statistics']={s:v for s,v in mc['statistics'].items() if s not in MARKER_SOURCES}
    mc['marker_hidden']=cfg.get('marker_hidden',128)
    mc['marker_scales']={s:max(index['statistics'][s]['rms_spacing_units'],1e-6) for s in active_markers}
    if cfg.get('marker_teachers'):
        markers=json.loads(Path(cfg['marker_teachers']).read_text())
        markers['sources']={s:v for s,v in markers['sources'].items() if s not in set(cfg.get('excluded_sources',[]))}
        if not markers.get('complete') or set(markers['sources'])!=active_markers:
            raise ValueError('Every active marker source must have a completed AE teacher before joint training')
        for source,info in markers['sources'].items():
            if info['representation']!='marker_displacement' or file_hash(info['checkpoint'])!=info['sha256']:
                raise ValueError('Expected a pinned marker-displacement AE teacher')
            registry['sources'][source]=info;mc['sensor_code_dims'][source]=info['code_dim']
            mc['statistics'][source]={key:info['statistics'][key] for key in ('mean','std')}
        mc.pop('marker_hidden');mc.pop('marker_scales')
    provenance['teachers']=registry
    provenance['sample_cache']='marker_targets_only; no visual feature or AE-code cache'
    provenance['tactile_encoding']=f'{len(active_markers)} native marker-displacement sources; {len(registry["sources"])} live frozen AE encoders; two numeric sources'
    provenance['marker_targets']=dict(root=str(root),index_sha256=file_hash(root/'index.json'),
        provenance=index['provenance'],normalization='train-only displacement RMS in marker-spacing units',
        scales={s:max(index['statistics'][s]['rms_spacing_units'],1e-6) for s in active_markers})
    if cfg.get('marker_teachers'):
        provenance['marker_teachers_sha256']=file_hash(cfg['marker_teachers'])
        provenance['tactile_encoding']=f'{len(active_markers)} frozen marker AE codes; {len(registry["sources"])-len(active_markers)} frozen native AE codes; two numeric sources'
        provenance['implementation'].update({n:file_hash(Path(__file__).with_name(n)) for n in ('marker_autoencoder.py','marker_code_model.py')})
    provenance['implementation'].update({n:file_hash(Path(__file__).with_name(n)) for n in
        ('train_vjepa_markers.py','marker_targets.py','marker_data.py','marker_model.py')})
    train=MarkerTargetClips(train,MarkerCache(root));val=MarkerTargetClips(val,MarkerCache(root))
    if cfg.get('code_cache'):
        if not cfg.get('marker_teachers'):raise ValueError('Code caching requires marker AE supervision')
        from .marker_code_cache import bind_cached_codes
        train,val=bind_cached_codes(cfg,registry,train,val,mc,provenance)
    if cfg.get('prediction_mode')=='current_frame':
        from .current_frame import CurrentFrameCodeClips
        train,val=CurrentFrameCodeClips(train),CurrentFrameCodeClips(val)
        provenance['prediction']=dict(mode='current_frame',rgb_frames=8,history_frames=7,
            input_stride=2,target='last sampled frame only',future_frames=False,
            unsupported_current_clips_excluded=True)
        provenance['implementation']['current_frame.py']=file_hash(Path(__file__).with_name('current_frame.py'))
    return train,val,registry,mc,provenance


def on_device(batch,device):
    return {k:v.to(device,non_blocking=True) if torch.is_tensor(v) else v for k,v in batch.items() if k!='rgb'}


def forward(model,features,batch):
    return model(features,batch['view_mask'],batch['source'][0],sample_latent=False,
                 marker_reference=batch.get('marker_reference'),marker_surface=batch.get('marker_surface'))


def supervision(original,teachers,model,device):
    batch=on_device(original,device)
    if original['source'][0] not in MARKER_SOURCES or isinstance(teachers,MixedMarkerTeachers):batch.update(teachers.supervision(original,model))
    return batch


def task_loss(model,output,batch):
    return alignment_loss(output,batch,0.) if model.model_type==CODE_MODEL_TYPE else spatial_marker_loss(model,output,batch)


def replay_update(model,optimizer,sam,replay,device):
    after=rng_snapshot();sam.perturb()
    try:
        for item in replay:
            restore_rng(item['rng'])
            with torch.autocast(device.type,torch.bfloat16,enabled=device.type=='cuda'):
                out=forward(model,item['features'],item['batch'])
                loss,_,_=task_loss(model,out,item['batch'])
            (loss/len(replay)).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    finally:
        sam.restore();restore_rng(after)
    optimizer.step();optimizer.zero_grad(set_to_none=True);sam.steps+=1


@torch.no_grad()
def validate(model,backbone,teachers,loader,device):
    model.eval();totals=defaultdict(lambda:defaultdict(float));native={}
    for original in loader:
        source=original['source'][0]
        if source not in MARKER_SOURCES:
            # Reuse native AE/numeric validation without changing old metrics.
            one=validate_native(model,backbone,teachers,[original],device)[source]
            row=native.setdefault(source,defaultdict(float));row['batches']+=1
            for k,v in one.items():row[k]+=v
            continue
        batch=supervision(original,teachers,model,device)
        with torch.autocast(device.type,torch.bfloat16):
            features=spatial_video_features(backbone,original,device)
            output=forward(model,features,batch)
        _,loss,_=task_loss(model,output,batch)
        if model.model_type==CODE_MODEL_TYPE:
            output['marker_displacement']=teachers.decode_marker(source,model.raw_code(source,output['sensor_code']),batch)
        target=batch['marker_target'];valid=batch['marker_valid'] & torch.isfinite(target).all(-1)
        error=(output['marker_displacement'].float()-torch.nan_to_num(target)).abs()
        row=totals[source];row['loss']+=loss.item();row['batches']+=1
        row['absolute']+=error[valid].sum().item();row['coordinates']+=2*valid.sum().item()
        pixels=error*batch['marker_spacing'][:,None,:,None]
        row['pixels']+=pixels[valid].sum().item()
        row['endpoint_pixels']+=torch.linalg.vector_norm(pixels,dim=-1)[valid].sum().item()
        row['baseline_pixels']+=(torch.nan_to_num(target).abs()*batch['marker_spacing'][:,None,:,None])[valid].sum().item()
        # Padded marker positions have reference=(0,0); actual detections are
        # inside the detector border, so they can be excluded from coverage.
        real=batch['marker_reference'].abs().sum(-1)>0
        row['possible']+=real.sum().item()*target.shape[1]
        adjacent=valid[:,1:] & valid[:,:-1]
        delta=(output['marker_displacement'][:,1:].float()-output['marker_displacement'][:,:-1].float()
               -(torch.nan_to_num(target[:,1:])-torch.nan_to_num(target[:,:-1]))).abs()
        row['change_pixels']+=(delta*batch['marker_spacing'][:,None,:,None])[adjacent].sum().item()
        row['change_coordinates']+=2*adjacent.sum().item()
    result={s:{k:v/row['batches'] if k!='valid_cells' else int(v) for k,v in row.items() if k!='batches'} for s,row in native.items()}
    for s,row in totals.items():
        n=max(row['coordinates'],1)
        result[s]=dict(target='marker_displacement',supervision_loss=row['loss']/row['batches'],
            marker_mae_spacing=row['absolute']/n,marker_mae_px=row['pixels']/n,
            endpoint_error_px=row['endpoint_pixels']/max(n/2,1),
            zero_displacement_baseline_mae_px=row['baseline_pixels']/n,
            temporal_change_mae_px=(row['change_pixels']/row['change_coordinates'] if row['change_coordinates'] else None),
            tracked_fraction=(n/2)/max(row['possible'],1),valid_marker_coordinates=int(row['coordinates']))
    return result


def save(root,epoch,model,optimizer,scheduler,sam,cfg,mc,provenance,registry,best,validation,seconds,generator):
    state=dict(model_type=model.model_type,model_config=mc,model=model.state_dict(),config=cfg,
        provenance=provenance,teachers=registry,backbone=BACKBONE_ID,backbone_revision=REVISION,
        epoch=epoch-1,optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),sam=sam.state_dict(),
        best_score=best,validation=validation,elapsed_seconds=seconds,rng=rng_snapshot(),
        python_rng=random.getstate(),numpy_rng=np.random.get_state(),loader_rng=generator.get_state())
    if cfg.get('save_full_epoch_checkpoints',True):atomic_checkpoint(root/f'epoch_{epoch:03d}.pt',state)
    atomic_checkpoint(root/'last.pt',state)
    save_viewer_epoch(root/'last.pt',state)
    if validation and np.mean([x['supervision_loss'] for x in validation.values()])<=best:
        atomic_checkpoint(root/'best.pt',state)


def check_epoch_extension(state,cfg,mc,provenance,request):
    """Permit only an audited epoch-budget/LR-tail extension of this trainer."""
    old=state['config'];before=copy.deepcopy(state['provenance']);after=copy.deepcopy(provenance)
    name='train_vjepa_markers.py'
    previous=before['implementation'].pop(name);current=after['implementation'].pop(name)
    expected={**request['original_config'],'epochs':request['target_epochs'],'lr_extension':request['lr_extension']}
    if (old!=request['original_config'] or cfg!=expected or cfg['epochs']<=old['epochs']
            or state['epoch']+1!=request['after_epoch'] or state['model_config']!=mc
            or state['model_type']!=CODE_MODEL_TYPE or before!=after
            or previous!=request['previous_trainer_sha256'] or current!=request['updated_trainer_sha256']
            or cfg['lr_extension']['anchor_update']!=state['scheduler']['last_epoch']
            or cfg['lr_extension']['anchor_lr']!=state['optimizer']['param_groups'][0]['lr']):
        raise ValueError('Epoch extension cannot change model, data, teachers, caches, or prior optimizer schedule')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/joint_vjepa21_spatial_storage_32d.json')
    parser.add_argument('--resume',action='store_true');parser.add_argument('--preflight-only',action='store_true')
    parser.add_argument('--extension-plan',help='Audited first handover only; requires --resume')
    parser.add_argument('--pilot-targets',help='Preflight ONLY: test available pilot episodes')
    args=parser.parse_args();cfg=json.loads(Path(args.config).read_text())
    if (cfg.get('storage_preparation')
            and not Path(cfg['storage_preparation']).exists()):
        raise RuntimeError('Finish RGB migration and native-target verification before joint training; see processing/storage_v1/status.json')
    if (args.config=='configs/joint_vjepa21_spatial_storage_32d.json'
            and not Path('processing/storage_v1/preparation.json').exists()):
        raise RuntimeError('Finish RGB migration and native-target verification before joint training; see processing/storage_v1/status.json')
    if args.extension_plan and (not args.resume or args.preflight_only):raise ValueError('Extension requires a training resume')
    if args.pilot_targets:
        if not args.preflight_only:raise ValueError('Pilot targets cannot be used for production training')
        cfg['marker_targets']=args.pilot_targets
    root=Path(cfg['output']);root.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(root/'training.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (root/'last.pt').exists() and not args.resume and not args.preflight_only:raise FileExistsError('Use --resume')
    torch.set_num_threads(4);random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
    device=torch.device('cuda');train,val,registry,mc,provenance=prepare(cfg,allow_pilot=bool(args.pilot_targets))
    backbone=VJEPABackbone(cfg['backbone_repo'],cfg['backbone_checkpoint']).to(device).eval()
    provenance['backbone']={**backbone.provenance,'spatial_pool':'none','camera_pool':'learned_cross_attention',
                           'temporal_mapping':('one_current_frame_query_with_seven_history_frames'
                               if cfg.get('prediction_mode')=='current_frame' else 'eight_distinct_frame_queries'),
                           'feature_shape':[4,3,576,768]}
    teachers=(MixedMarkerTeachers(registry,device) if cfg.get('marker_teachers') else LiveTeachers(registry,device,cfg['teacher_chunk_frames']))
    torch.manual_seed(cfg['seed']);model=(MarkerCodeVJEPA(**mc) if cfg.get('marker_teachers') else MarkerSpatialVJEPA(**mc)).to(device)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    sam=SAMStep(optimizer,cfg['sam_rho'])
    if args.preflight_only:
        rows=[];preflight_start=time.monotonic();torch.cuda.reset_peak_memory_stats()
        for source,ids in sorted(train.indices_by_source.items()):
            source_start=time.monotonic()
            original=marker_collate([train[i] for i in ids[:2]])
            if source in cfg.get('mirror_augmentation',{}).get('sources',[]) and cfg.get('mirror_augmentation',{}).get('enabled'):
                original=augment_mirror_batch(original,cfg,0,0,flags=[True]*len(original['source']))
            batch=supervision(original,teachers,model,device);before=rng_snapshot()
            with torch.no_grad(),torch.autocast('cuda',torch.bfloat16):features=spatial_video_features(backbone,original,device)
            with torch.autocast('cuda',torch.bfloat16):out=forward(model,features,batch);loss,_,_=task_loss(model,out,batch)
            output_frames=1 if cfg.get('prediction_mode')=='current_frame' else 8
            if (out['mu'].shape!=(len(original['source']),output_frames,cfg['latent_dim'])
                    or not torch.isfinite(out['mu']).all() or not torch.isneginf(out['logvar']).all()):
                raise ValueError('Expected one finite deterministic latent per original frame')
            if not torch.isfinite(loss):raise FloatingPointError(source)
            loss.backward();replay_update(model,optimizer,sam,[dict(features=features,batch=batch,rng=before)],device)
            rows.append(dict(source=source,loss=loss.item(),latent_shape=list(out['mu'].shape),
                cached_sensor_codes='code_target' in original,target='markers' if source in MARKER_SOURCES else 'native',
                seconds=time.monotonic()-source_start))
            print(json.dumps(dict(event='preflight_source',**rows[-1])),flush=True)
        if any(p.grad is not None for p in backbone.parameters()) or any(p.grad is not None for t in teachers.models.values() for p in t.parameters()):
            raise ValueError('Frozen weights acquired gradients')
        write_json(root/('pilot_preflight.json' if args.pilot_targets else 'preflight.json'),
            dict(passed=True,config=cfg,provenance=provenance,rows=rows,
                 marker_sources=sorted(set(MARKER_SOURCES)&set(train.indices_by_source)),
                 teacher_sources=list(teachers.models),production_weights_modified=False,
                 seconds=time.monotonic()-preflight_start,peak_gpu_bytes=torch.cuda.max_memory_allocated()))
        print(json.dumps(rows),flush=True);return
    state=torch.load(root/'last.pt',map_location='cpu',weights_only=False) if args.resume else None
    if args.extension_plan:
        request=json.loads(Path(args.extension_plan).read_text())
        check_epoch_extension(state,cfg,mc,provenance,request)
        checked=json.loads((root/'preflight.json').read_text())
        if (not checked['passed'] or checked['config']!=state['config'] or checked['provenance']!=state['provenance']
                or file_hash(root/'preflight.json')!=request['original_preflight_sha256']):
            raise ValueError('Original production preflight differs from the extended checkpoint')
        write_json(root/'preflight_extension.json',dict(passed=True,config=cfg,provenance=provenance,
            inherited_gpu_preflight=str(root/'preflight.json'),extension_only=True,request=request))
    else:
        checked_path=root/('preflight_extension.json' if (root/'preflight_extension.json').exists() else 'preflight.json')
        checked=json.loads(checked_path.read_text())
        if not checked['passed'] or checked['config']!=cfg or checked['provenance']!=provenance:
            raise ValueError('Run production preflight on complete marker targets before training')
    sampler=SourceBatchSampler(train,cfg['batch_size'],seed=cfg['seed'],max_windows_per_source=cfg['train_windows_per_source_per_epoch'],sampling_mode='cycle')
    vsampler=SourceBatchSampler(val,cfg['batch_size'],seed=cfg['seed'],max_windows_per_source=cfg['validation_windows_per_source'])
    generator=torch.Generator().manual_seed(cfg['seed'])
    options=dict(num_workers=cfg['workers'],pin_memory=True,generator=generator,collate_fn=marker_collate)
    loader=DataLoader(train,batch_sampler=sampler,**options);vloader=DataLoader(val,batch_sampler=vsampler,**options)
    scheduler=alignment_scheduler(optimizer,cfg,len(loader));start=0;best=float('inf');elapsed=0.
    if args.resume:
        if not args.extension_plan and (state['config']!=cfg or state['provenance']!=provenance or state['model_config']!=mc):raise ValueError('Resume provenance changed')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler']);sam.load_state_dict(state['sam'])
        start=state['epoch']+1;best=state['best_score'];elapsed=state['elapsed_seconds']
        restore_rng(state['rng']);random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng']);generator.set_state(state['loader_rng'])
    else:save(root,0,model,optimizer,scheduler,sam,cfg,mc,provenance,registry,best,{},0.,generator)
    write_json(root/'config.json',cfg);write_json(root/'provenance.json',provenance)
    began=time.monotonic()-elapsed
    for epoch in range(start,cfg['epochs']):
        sampler.set_epoch(epoch);model.train();optimizer.zero_grad(set_to_none=True);replay=[]
        totals=defaultdict(list);stats=LatentStats(cfg['latent_dim']);epoch_start=time.monotonic()
        for step,original in enumerate(loader):
            source=original['source'][0]
            if source=='ftp_motiontrans':raise ValueError('Held-out source in joint training')
            original=augment_mirror_batch(original,cfg,epoch,step)
            batch=supervision(original,teachers,model,device);mask=batch['view_mask'].clone()
            g=torch.Generator(device=device).manual_seed(cfg['seed']+epoch*1000000+step)
            if mask[:,:,0].all():mask[:,:,1:] &= torch.rand(len(mask),1,2,device=device,generator=g)>=cfg['wrist_dropout']
            batch['view_mask']=mask
            with torch.no_grad(),torch.autocast('cuda',torch.bfloat16):features=spatial_video_features(backbone,original,device)
            before=rng_snapshot()
            with torch.autocast('cuda',torch.bfloat16):out=forward(model,features,batch);loss,task,_=task_loss(model,out,batch)
            if not torch.isfinite(loss):raise FloatingPointError(source)
            group=min(cfg['accumulation'],len(loader)-(step//cfg['accumulation'])*cfg['accumulation'])
            (loss/group).backward();totals[source].append(task.item())
            valid=batch['marker_valid'].any(-1) if source in MARKER_SOURCES else (batch['code_valid'] if 'code_valid' in batch else batch['target_mask'].flatten(2).any(-1))
            stats.add(out,valid)
            replay.append(dict(features=features.detach(),batch=batch,rng=before))
            if (step+1)%cfg['accumulation']==0 or step+1==len(loader):
                replay_update(model,optimizer,sam,replay,device);replay.clear();scheduler.step()
            if step%cfg['log_every']==0:
                write_json(root/'status.json',dict(stage='training',pid=os.getpid(),epoch=epoch+1,completed_epochs=epoch,
                    total_epochs=cfg['epochs'],batch=step+1,batches_per_epoch=len(loader),source=source,
                    loss=loss.item(),epoch_seconds=time.monotonic()-epoch_start,visual_cache='none',
                    marker_target_cache=cfg['marker_targets'],sensor_code_cache=cfg.get('code_cache'),
                    latent_dim=cfg['latent_dim'],prediction_mode=cfg.get('prediction_mode','all_frames')))
        write_json(root/'status.json',dict(stage='validation',epoch=epoch+1,pid=os.getpid()))
        validation=validate(model,backbone,teachers,vloader,device)
        mirror_validation=validate_mirror_pairs(model,backbone,teachers,vloader,device,cfg)
        score=float(np.mean([r['supervision_loss'] for r in validation.values()]));best=min(best,score)
        record=dict(epoch=epoch+1,validation=validation,selection_score=score,
            train={s:dict(supervision=float(np.mean(v))) for s,v in totals.items()},latent=stats.result(),
            seconds=time.monotonic()-began,epoch_seconds=time.monotonic()-epoch_start,uncertainty='none')
        if cfg.get('mirror_augmentation',{}).get('enabled'):
            record['mirror_validation']=mirror_validation
        save(root,epoch+1,model,optimizer,scheduler,sam,cfg,mc,provenance,registry,best,validation,record['seconds'],generator)
        with (root/'metrics.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
    write_json(root/'status.json',dict(stage='complete',completed_epochs=cfg['epochs'],total_epochs=cfg['epochs']))


if __name__=='__main__':main()
