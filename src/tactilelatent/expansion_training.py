"""Single-seed 30-epoch tactile teachers and chronological code caches.

Waits for complete acquisition and CUDA; never substitutes CPU production
training or starts a joint visual-model experiment.
"""
from collections import defaultdict
import copy
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch

from .aligned_frames import CONTRACT,file_hash,json_hash,write_json
from .data_expansion import (ROOT,marker_arrays,pressure_targets,pressure_counts,
    denormalize_pressure,atomic_npz,reserve,report)
from .expansion_models import ExpansionMarkerAutoencoder,load_teacher
from .sensor_autoencoder import SensorAutoencoder
from .train_alignment import atomic_checkpoint
from .sensor_code_cache import train_statistics
from .pressure_selection import read_task_manifest,task_episodes,require_task_episodes,VERSION as PRESSURE_SELECTION


def source_rows(source):
    path=ROOT/('baihu_available.json' if source=='baihu' else 'egopressure_manifest.json')
    if not path.exists():raise FileNotFoundError(f'Waiting for {source} acquisition')
    content=read_task_manifest(ROOT) if source=='egopressure' else json.loads(path.read_text())
    if not content.get('complete') or content.get('failures'):
        raise FileNotFoundError(f'Waiting for complete, verified {source} acquisition')
    if source=='egopressure' and not (ROOT/'pressure_normalization.json').exists():
        raise FileNotFoundError('Waiting for train-only pressure normalization')
    rows=sorted(content['episodes'],key=lambda r:(r['split'],r['episode_id']))
    if not rows or {r['split'] for r in rows}!={'train','val'}:raise ValueError('Missing source split')
    if any(r['source']!=source for r in rows):raise ValueError('Mixed/held-out source in tactile teacher')
    return rows


def marker_scale(rows):
    stats=[marker_arrays(r)[1]['loss_statistics'] for r in rows if r['split']=='train']
    return max((sum(s['square_sum'] for s in stats)/max(sum(s['coordinate_count'] for s in stats),1))**.5,1e-6)


def batches(rows,source,split,epoch=0,batch_size=128,limit=100000):
    selected=[r for r in task_episodes(rows) if r['split']==split]
    rng=np.random.default_rng(45+epoch if split=='train' else 100045)
    if split=='train':rng.shuffle(selected)
    # Validation covers every episode, with a fixed per-episode budget.
    per_episode=None if split=='train' else max(1,limit//len(selected))
    seen=0
    for row in selected:
        if seen>=limit:break
        indices=np.arange(0,row['frames'],2)
        if per_episode is not None and len(indices)>per_episode:
            indices=np.unique(np.linspace(0,row['frames']-1,per_episode,dtype=int))
        indices=indices[:limit-seen]
        data=marker_arrays(row)[0] if source=='baihu' else None
        for start in range(0,len(indices),batch_size):
            ix=indices[start:start+batch_size]
            if source=='baihu':
                target=torch.from_numpy(data['displacement_px'][ix]/data['spacing_px'][None,:,None])
                valid=torch.from_numpy(data['valid'][ix]);n=len(ix)
                ref=torch.from_numpy(data['reference_uv'].astype(np.float32))[None].expand(n,-1,-1)
                surface=torch.from_numpy(data['surface_id'])[None].expand(n,-1)
                batch=dict(target=target,valid=valid,reference=ref,surface=surface,
                    spacing=torch.from_numpy(data['spacing_px']))
            else:
                target=torch.from_numpy(pressure_targets(row,ix));valid=torch.isfinite(target)
                batch=dict(target=target,valid=valid)
            supported=valid.flatten(1).any(-1)
            if not supported.any():continue
            # Keep full chronological arrays and masks; completely missing frames
            # contribute no loss, code statistics, or labels.
            batch.update(frame_indices=ix,row=row)
            yield batch
            seen+=len(ix)


def prediction(model,b,source,device):
    y=b['target'].to(device);v=b['valid'].to(device)
    if source=='baihu':
        ref=b['reference'].to(device);surface=b['surface'].to(device)
        pred=model(y,v,ref,surface);loss=model.loss(pred,y,v,surface)
    else:
        pred=model(y,v);clean=torch.where(v,y,0.)
        error=pred-clean
        weights=1.+2.*(clean>0).float()
        loss=((error.square()+.5*error.abs())*weights)[v].mean()
    return pred,loss,y,v


def evaluate(model,rows,source,device):
    model.eval();absolute=baseline=cells=changes=change_cells=losses=frames=0.
    physical_error=physical_cells=0.
    scale=json.loads((ROOT/'pressure_normalization.json').read_text())['scale_counts'] if source=='egopressure' else None
    with torch.no_grad():
        for b in batches(rows,source,'val',batch_size=64 if source=='baihu' else 16,limit=10000):
            pred,loss,y,v=prediction(model,b,source,device)
            safe=torch.where(v[...,None] if source=='baihu' else v,y,0.)
            error=(pred-safe).abs()
            if source=='baihu':
                spacing=b['spacing'].to(device)[None,:,None]
                error=error*spacing;zero=safe.abs()*spacing;mask=v[...,None].expand_as(y)
            else:zero=safe.abs();mask=v
            absolute+=error[mask].sum().item();baseline+=zero[mask].sum().item();cells+=mask.sum().item()
            if len(y)>1:
                adjacent=mask[1:]&mask[:-1]
                delta=((pred[1:]-pred[:-1])-(safe[1:]-safe[:-1])).abs()
                if source=='baihu':delta=delta*spacing
                changes+=delta[adjacent].sum().item();change_cells+=adjacent.sum().item()
            if scale is not None:
                p=denormalize_pressure(pred.cpu().numpy(),scale);t=denormalize_pressure(safe.cpu().numpy(),scale)
                m=mask.cpu().numpy()&np.isfinite(p)&np.isfinite(t)
                physical_error+=float(np.abs(p-t)[m].sum());physical_cells+=int(m.sum())
            losses+=loss.item()*len(y);frames+=len(y)
    if cells==0:raise ValueError('No supported validation targets')
    result=dict(mae=absolute/cells,zero_baseline_mae=baseline/cells,loss=losses/max(frames,1),
        frames=int(frames),valid_coordinates=int(cells),temporal_change_mae=changes/max(change_cells,1),
        units='native_pixels' if source=='baihu' else 'reversibly bounded pressure')
    if scale is not None:result['pressure_mae_counts']=physical_error/max(physical_cells,1)
    return result


def train_teacher(source,rows,device='cuda',epochs=30,folder=None):
    rows=task_episodes(rows)
    folder=Path(folder) if folder is not None else ROOT/'teachers'/source
    folder.mkdir(parents=True,exist_ok=True)
    seed=45;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(4)
    if source=='baihu':
        mc=dict(latent_dim=32,width=128,queries=8,heads=4,displacement_scale=marker_scale(rows),surfaces=4)
        model=ExpansionMarkerAutoencoder(**mc);model_type='expansion_marker_autoencoder_v1';batch_size=128;lr=.001
    else:
        mc=dict(target_shape=[1,224,224],architecture='conv',latent_dim=64,channels=[16,32,64,128],
            hidden_dims=[256,128],output_mode='sigmoid')
        model=SensorAutoencoder(**mc);model_type='expansion_pressure_autoencoder_v1';batch_size=32;lr=.0001
    config=dict(source=source,seed=45,epochs=epochs,model=mc,representation='marker_displacement' if source=='baihu' else 'pressure_uv',
        batch_size=batch_size,learning_rate=lr,weight_decay=.0001,frame_stride=2,max_train_frames=100000,
        max_val_frames=10000,rows_sha256=json_hash(rows),manifest=str(ROOT/'manifest.json'))
    config['implementation']={name:file_hash(Path(__file__).with_name(name)) for name in
        ('data_expansion.py','expansion_models.py','expansion_training.py','expansion_tracker.py',
         'marker_targets.py','marker_model.py','marker_autoencoder.py','sensor_autoencoder.py','pressure_selection.py')}
    config['target_snapshot_sha256']=json_hash({r['episode_id']:file_hash(r['marker_path'] if source=='baihu' else r['target_path']) for r in rows})
    if source=='egopressure':config['pressure_normalization_sha256']=file_hash(ROOT/'pressure_normalization.json')
    if (folder/'config.json').exists() and json.loads((folder/'config.json').read_text())!=config:
        raise ValueError('Teacher episode selection/provenance changed; preserve this run and use a fresh folder')
    write_json(folder/'config.json',config)
    device=torch.device(device);model=model.to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.0001)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs,eta_min=1e-6)
    start=0;best=float('inf')
    if (folder/'last.pt').exists():
        state=torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
        if state['config']!=config:raise ValueError('Teacher resume configuration/provenance differs')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        start=state['epoch'];best=state['best_score'];torch.set_rng_state(state['torch_rng'])
        if device.type=='cuda':torch.cuda.set_rng_state_all(state['cuda_rng'])
        if (folder/'metrics.jsonl').exists():
            records=[json.loads(l) for l in (folder/'metrics.jsonl').read_text().splitlines()]
            (folder/'metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records if r['epoch']<=start))
    for epoch in range(start,epochs):
        began=time.monotonic();model.train();total=frames=0
        for b in batches(rows,source,'train',epoch,batch_size):
            optimizer.zero_grad(set_to_none=True);_,loss,_,_=prediction(model,b,source,device)
            if not torch.isfinite(loss):raise ValueError('Nonfinite teacher loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            total+=loss.item()*len(b['target']);frames+=len(b['target'])
        val=evaluate(model,rows,source,device);scheduler.step();improved=val['mae']<best;best=min(best,val['mae'])
        record=dict(epoch=epoch+1,source=source,train_loss=total/max(frames,1),val=val,
            selection_score=val['mae'],seconds=time.monotonic()-began)
        state=dict(model_type=model_type,model_config=mc,config=config,model=model.state_dict(),epoch=epoch+1,
            optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),best_score=best,
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else [],validation=val)
        if improved:
            atomic_checkpoint(folder/'best.pt',state);write_json(folder/'best_metrics.json',record)
        atomic_checkpoint(folder/'last.pt',state)
        with (folder/'metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        report('teachers',stage='training',**record)
    del model,optimizer;torch.cuda.empty_cache()
    return folder/'best.pt'


def build_codes(source,rows,checkpoint,device='cuda',destination=None):
    require_task_episodes(rows)
    model,state=load_teacher(checkpoint,device);dimension=state['model_config']['latent_dim']
    current_snapshot=json_hash({r['episode_id']:file_hash(r['marker_path'] if source=='baihu' else r['target_path']) for r in rows})
    if current_snapshot!=state['config']['target_snapshot_sha256']:raise ValueError('Teacher targets changed before code caching')
    if source=='egopressure' and state['config']['pressure_normalization_sha256']!=file_hash(ROOT/'pressure_normalization.json'):
        raise ValueError('Teacher pressure normalization changed before caching')
    dest=Path(destination) if destination is not None else ROOT/'codes'/source
    dest.mkdir(parents=True,exist_ok=True);records=[]
    offsets=[0];means=np.zeros(dimension,np.float64);squares=np.zeros(dimension,np.float64);count=0
    for number,row in enumerate(rows):
        p=dest/(json_hash(row['episode_id'])[:24]+'.npz');receipt=p.with_suffix('.json')
        signature=dict(row_sha256=json_hash(row),teacher_sha256=file_hash(checkpoint),dimension=dimension)
        if receipt.exists():
            saved=json.loads(receipt.read_text())
            if saved['signature']!=signature or file_hash(p)!=saved['sha256']:raise ValueError('Changed cached teacher codes')
            with np.load(p) as f:codes=f['codes'];valid=f['valid']
        else:
            reserve(row['frames']*(dimension*4+1));codes=np.zeros((row['frames'],dimension),np.float32);valid=np.zeros(row['frames'],bool)
            d=marker_arrays(row)[0] if source=='baihu' else None
            for start in range(0,row['frames'],128 if source=='baihu' else 16):
                ix=np.arange(start,min(start+(128 if source=='baihu' else 16),row['frames']))
                with torch.inference_mode():
                    if source=='baihu':
                        y=torch.from_numpy(d['displacement_px'][ix]/d['spacing_px'][None,:,None]).to(device);v=torch.from_numpy(d['valid'][ix]).to(device)
                        ref=torch.from_numpy(d['reference_uv'].astype(np.float32)).to(device)[None].expand(len(ix),-1,-1)
                        surface=torch.from_numpy(d['surface_id']).to(device)[None].expand(len(ix),-1)
                        value=model.encode(y,v,ref,surface)
                    else:
                        y=torch.from_numpy(pressure_targets(row,ix)).to(device);v=torch.isfinite(y);value=model.encode(y,v)
                codes[ix]=value.float().cpu().numpy();valid[ix]=v.flatten(1).any(-1).cpu().numpy()
            if not np.isfinite(codes).all():raise ValueError('Nonfinite teacher codes')
            atomic_npz(p,codes=codes,valid=valid,frame_indices=np.arange(row['frames'],dtype=np.int64))
            write_json(receipt,dict(signature=signature,sha256=file_hash(p)))
        if row['split']=='train':
            x=codes[valid].astype(np.float64);means+=x.sum(0);squares+=(x*x).sum(0);count+=len(x)
        records.append(dict(episode_id=row['episode_id'],path=str(p),sha256=file_hash(p),frames=row['frames'],split=row['split']))
        if number%25==0:report('teachers',stage='caching',source=source,completed_episodes=number+1,episodes=len(rows))
    if count<2:raise ValueError('No supported training codes')
    mean=means/count;std=np.sqrt(np.maximum(squares/count-mean**2,1e-12));std=np.maximum(std,1e-6)
    info=dict(complete=True,source=source,code_dim=dimension,teacher_sha256=file_hash(checkpoint),
        episodes=records,statistics=dict(mean=mean.tolist(),std=std.tolist(),normalization_split='train'),
        rows_sha256=json_hash(rows),target_snapshot_sha256=current_snapshot,frames=sum(r['frames'] for r in rows))
    write_json(dest/'index.json',info)
    # Fresh encodings at both split boundaries verify the actual on-disk code
    # producer/consumer, independently of its checksum receipts.
    checks=[]
    for split in ('train','val'):
        row=next(r for r in rows if r['split']==split);ix=np.unique([0,row['frames']-1])
        record=next(r for r in records if r['episode_id']==row['episode_id'])
        with np.load(record['path']) as f:saved=f['codes'][ix];saved_valid=f['valid'][ix]
        with torch.inference_mode():
            if source=='baihu':
                # Replay the producer's complete chunks. Different attention
                # batch sizes can choose different CUDA kernels; an isolated
                # two-frame replay is not a faithful cache-integrity check.
                d=marker_arrays(row)[0];pieces=[];fresh_masks=[]
                for frame in ix:
                    start=int(frame)//128*128
                    chunk=np.arange(start,min(start+128,row['frames']))
                    y=torch.from_numpy(d['displacement_px'][chunk]/d['spacing_px'][None,:,None]).to(device)
                    v=torch.from_numpy(d['valid'][chunk]).to(device)
                    ref=torch.from_numpy(d['reference_uv'].astype(np.float32)).to(device)[None].expand(len(chunk),-1,-1)
                    surface=torch.from_numpy(d['surface_id']).to(device)[None].expand(len(chunk),-1)
                    pieces.append(model.encode(y,v,ref,surface)[int(frame)-start].cpu())
                    fresh_masks.append(v[int(frame)-start].any().cpu())
                fresh=torch.stack(pieces);fresh_valid=torch.stack(fresh_masks)
            else:
                y=torch.from_numpy(pressure_targets(row,ix)).to(device);v=torch.isfinite(y);fresh=model.encode(y,v).cpu()
                fresh_valid=v.flatten(1).any(-1).cpu()
        torch.testing.assert_close(torch.from_numpy(saved),fresh,rtol=2e-5,atol=2e-5)
        np.testing.assert_array_equal(saved_valid,fresh_valid.numpy())
        checks.append(dict(episode_id=row['episode_id'],split=split,frames=ix.tolist(),max_absolute_error=float((fresh-torch.from_numpy(saved)).abs().max())))
    write_json(dest/'verification.json',dict(passed=True,checks=checks,teacher_sha256=file_hash(checkpoint)))
    return info


def prepare_configuration():
    cfg=json.loads(Path('configs/joint_vjepa21_spatial_current_32d.json').read_text())
    cfg.update(output='processing/runs/joint_vjepa21_spatial_expanded32d_v1',data_expansion=str(ROOT),
        manifest=str(ROOT/'manifest.json'),training_entrypoint='scripts/train_expanded_vjepa.py',
        storage_preparation=str(ROOT/'ready.json'),
        base_config='configs/joint_vjepa21_spatial_current_32d.json',expanded_manifest=str(ROOT/'manifest.json'),
        new_teachers=str(ROOT/'teachers.json'),new_code_cache=str(ROOT/'codes'))
    cfg['egopressure_selection']=PRESSURE_SELECTION
    write_json(Path('configs/joint_vjepa21_spatial_expanded_32d.json'),cfg)
    return cfg


def publish():
    base=json.loads(Path('processing/storage_v1/manifest.json').read_text())
    new=[];teachers={};statistics={};code_paths={}
    selection=ROOT/'source_teacher_overrides.json'
    overrides=json.loads(selection.read_text()) if selection.exists() else {}
    for source in ('baihu','egopressure'):
        rows=source_rows(source)
        choice=overrides.get(source)
        checkpoint=Path(choice['checkpoint']) if choice else ROOT/'teachers'/source/'best.pt'
        code_path=Path(choice['code_index']) if choice else ROOT/'codes'/source/'index.json'
        if choice:
            quality=json.loads(Path(choice['quality']).read_text())
            if (file_hash(checkpoint)!=choice['teacher_sha256'] or file_hash(code_path)!=choice['code_index_sha256']
                    or not quality['accepted'] or quality['teacher_sha256']!=choice['teacher_sha256']):
                raise ValueError('Replacement pressure teacher/cache quality or provenance differs')
        index=json.loads(code_path.read_text());code_paths[source]=str(code_path)
        rejected=ROOT/'quality'/f'{source}_rejected_teachers.json'
        if rejected.exists() and file_hash(checkpoint) in json.loads(rejected.read_text()).get('sha256',[]):
            raise ValueError(f'{source} teacher failed reconstruction quality; use a validated replacement before joint publication')
        last=torch.load(checkpoint.with_name('last.pt'),map_location='cpu',weights_only=False,mmap=True)
        if last['epoch']!=30 or last['config']['seed']!=45 or last['config']['epochs']!=30:
            raise ValueError('Both single-seed teachers must finish 30 epochs before publication')
        if not index['complete'] or index['rows_sha256']!=json_hash(rows) or index['teacher_sha256']!=file_hash(checkpoint):
            raise ValueError('Cannot publish incomplete/mismatched new teacher codes')
        verification=json.loads(code_path.with_name('verification.json').read_text())
        if not verification['passed'] or verification['teacher_sha256']!=index['teacher_sha256']:
            raise ValueError('New source code cache has not passed fresh-encoding verification')
        new.extend(rows);statistics[source]=index['statistics']
        teachers[source]=dict(checkpoint=str(checkpoint),sha256=file_hash(checkpoint),code_dim=index['code_dim'],
            representation='marker_displacement' if source=='baihu' else 'pressure_uv',
            target_shape=[12,240,240] if source=='baihu' else [1,224,224])
    episodes=[r for r in base['episodes'] if r['source']!='freetacman']+new
    write_json(ROOT/'manifest.json',dict(schema_version=2,frame_contract=CONTRACT,episodes=episodes,
        sources={**base['sources'],**{s:{} for s in ('baihu','egopressure')}},
        provenance=dict(base_manifest_sha256=file_hash('processing/storage_v1/manifest.json'),
            new_loader_sha256=file_hash(Path(__file__).with_name('data_expansion.py')))))
    write_json(ROOT/'teachers.json',dict(complete=True,sources=teachers,statistics=statistics))
    prepare_configuration()
    write_json(ROOT/'ready.json',dict(ready=True,training_sources=12,heldout_source='ftp_motiontrans',
        excluded_sources=['freetacman'],feel='pilot only',joint_training_started=False,
        manifest_sha256=file_hash(ROOT/'manifest.json'),teachers_sha256=file_hash(ROOT/'teachers.json'),
        code_index_paths=code_paths,code_index_sha256={s:file_hash(p) for s,p in code_paths.items()}))
    report('publish',stage='complete',sources=12,joint_training_started=False)


def cuda_ready():
    # Probe in a fresh interpreter: a failed CUDA initialization is sticky in
    # this process, and the driver may be repaired while acquisition runs.
    result=subprocess.run([sys.executable,'-c','import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)'],
        stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,text=True,timeout=30)
    return result.returncode==0,result.stderr[-500:]


def run_teachers():
    prepare_configuration()
    available,reason=cuda_ready()
    report('teachers',stage='queued',cuda_available=available,reason=reason,
        epochs=30,seed=45,source_code_dims={'baihu':32,'egopressure':64})
    done=set()
    for source in ('baihu','egopressure'):
        if (ROOT/'codes'/source/'index.json').exists():done.add(source)
    while len(done)<2:
        progressed=False
        for source in ('baihu','egopressure'):
            if source in done:continue
            try:rows=source_rows(source)
            except FileNotFoundError as e:
                report('teachers',stage='waiting_for_data',source=source,reason=str(e));continue
            available,reason=cuda_ready()
            if not available:
                report('teachers',stage='waiting_for_cuda',source=source,reason=reason);continue
            checkpoint=train_teacher(source,rows);build_codes(source,rows,checkpoint);done.add(source);progressed=True
        if not progressed:time.sleep(30)
    publish()
