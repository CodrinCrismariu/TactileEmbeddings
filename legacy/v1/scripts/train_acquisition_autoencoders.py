#!/usr/bin/env python3
"""Resumable queue of native tactile-vector teachers; no RGB/backbone loading.

Each job fits one format using every observed training/validation frame, existing
episode splits, scale-only train statistics, and a small masked MLP. Jobs run
sequentially on one GPU and save viewer-readable best checkpoints each epoch.
"""
import argparse
import fcntl
import json
from pathlib import Path
import random
import time
import traceback

import numpy as np
import torch

from tactilelatent.aligned_frames import file_hash,write_json
from tactilelatent.numeric_autoencoder import MODEL_TYPE,NumericAutoencoder,numeric_loss
from tactilelatent.pretrain_egotouch_autoencoder import atomic_save


def rows_for(spec):
    path=Path('processing/data_queue_v2')/spec['acquisition']/'manifest.json'
    manifest=json.loads(path.read_text())
    if manifest.get('schema')!='acquisition-native-v2':raise ValueError('Unexpected acquisition schema')
    rows=[r for r in manifest['episodes'] if all(r.get(k)==spec[k] for k in ('environment','embodiment') if k in spec)]
    if not rows or any(r['targets']['shape'][1:]!=[spec['input_dim']] for r in rows):
        raise ValueError('Wrong native layout for teacher')
    channels=rows[0]['targets']['channel_order']
    if any(r['targets']['channel_order']!=channels for r in rows):raise ValueError('Native channel order differs')
    if set(r['split'] for r in rows)!={'train','val'}:raise ValueError('Existing train/validation splits required')
    return path,manifest,rows,channels


def load_data(spec,device):
    path,manifest,rows,channels=rows_for(spec)
    arrays={'train':[],'val':[]};masks={'train':[],'val':[]};provenance=[]
    count=np.zeros(spec['input_dim'],np.int64);total=np.zeros(spec['input_dim']);squares=total.copy()
    minimum=np.full(spec['input_dim'],np.inf);maximum=-minimum
    for row in rows:
        t=row['targets']
        if file_hash(t['path'])!=t['sha256']:raise ValueError('Native target checksum mismatch')
        with np.load(t['path'],allow_pickle=False) as native:
            x=native['values'].reshape(-1,spec['input_dim']).astype(np.float32)
            mask=native['valid'].reshape(x.shape)&np.isfinite(x)
            if len(x)!=row['frames']:raise ValueError('Native frame count differs')
            times=native['timestamps']
            if len(times)!=len(x) or not np.isfinite(times).all() or np.any(np.diff(times)<0):
                raise ValueError('Invalid acquisition chronology')
        # No observation is created for missing coordinates/all-missing frames.
        observed=mask.any(1);x=x[observed];mask=mask[observed];safe=np.where(mask,x,0)
        arrays[row['split']].append(safe);masks[row['split']].append(mask)
        if row['split']=='train' and len(x):
            count+=mask.sum(0);total+=safe.sum(0,dtype=np.float64)
            squares+=(safe.astype(np.float64)**2).sum(0)
            minimum=np.minimum(minimum,np.where(mask,x,np.inf).min(0))
            maximum=np.maximum(maximum,np.where(mask,x,-np.inf).max(0))
        provenance.append(dict(episode=row['episode_id'],split=row['split'],frames=row['frames'],
            observed_frames=int(observed.sum()),target_path=t['path'],target_sha256=t['sha256']))
    scale=np.sqrt(squares/np.maximum(count,1));scale=np.where(scale>1e-8,scale,1.).astype(np.float32)
    mean=(total/np.maximum(count,1)).astype(np.float32)
    varying=(maximum>minimum)&(count>1)
    result={}
    for split in ('train','val'):
        result[split]=(torch.from_numpy(np.concatenate(arrays[split])).to(device),
            torch.from_numpy(np.concatenate(masks[split])).to(device))
    nonnegative=np.full(spec['input_dim'],spec['source']!='trex',bool)
    if spec['source']=='hrdexdb_inspire_f1':nonnegative[2::3]=False
    normalization=dict(method='asinh(native/train_channel_RMS)',offset=[0.]*spec['input_dim'],
        scale=scale.tolist(),count=count.tolist(),mean_baseline=mean.tolist(),varying=varying.tolist(),
        nonnegative=nonnegative.tolist(),fit_split='train',per_episode_normalization=False,
        unsupported_channels=np.flatnonzero(count==0).tolist(),native_channel_order=channels)
    snapshot=dict(schema=MODEL_TYPE,source=spec['source'],revision=manifest.get('revision'),
        manifest_path=str(path),manifest_sha256=file_hash(path),episodes=provenance,
        excluded_sources=['ftp_motiontrans','freetacman','tdex','tacquad'],
        fixed_zero_registry_sha256=file_hash('processing/zero_reference_v1/references.json'),
        normalization_fit_split='train',rgb_loaded=False,source_targets_unchanged=True)
    return result,normalization,snapshot


def metrics(model,data,mean,varying,batch_size,source):
    x,valid=data;device=x.device;d=x.shape[1]
    sums=torch.zeros((11,d),device=device,dtype=torch.float64)
    zsum=torch.zeros(model.latent_dim,device=device);zouter=torch.zeros((model.latent_dim,model.latent_dim),device=device)
    model.eval()
    with torch.inference_mode():
        for start in range(0,len(x),batch_size):
            target=x[start:start+batch_size];mask=valid[start:start+batch_size]
            code=model.encode(target,mask);prediction=model.decode(code)
            expected=model.normalize(target);actual=model.normalize(prediction)
            absolute=(prediction-target).abs();error=(actual-expected).abs()
            nonzero=mask & model.nonnegative & (target!=0);zero=mask & model.nonnegative & (target==0)
            constant=model.normalize(mean).expand_as(expected)
            parts=[mask.sum(0),(absolute*mask).sum(0),((prediction-target).square()*mask).sum(0),
                (error*mask).sum(0),(expected.abs()*mask).sum(0),((constant-expected).abs()*mask).sum(0),
                (absolute*nonzero).sum(0),nonzero.sum(0),(absolute*zero).sum(0),zero.sum(0),
                (target.abs()*nonzero).sum(0)]
            sums+=torch.stack(parts).double();zsum+=code.sum(0);zouter+=code.T@code
    s=sums.cpu().numpy();observed=s[0]>0;active=observed&varying
    if not active.any():active=observed
    per_mae=np.divide(s[1],s[0],out=np.zeros(d),where=observed)
    transformed=np.divide(s[3],s[0],out=np.zeros(d),where=observed)
    baseline_zero=np.divide(s[4],s[0],out=np.zeros(d),where=observed)
    baseline_mean=np.divide(s[5],s[0],out=np.zeros(d),where=observed)
    covariance=(zouter/len(x)-torch.outer(zsum/len(x),zsum/len(x))).cpu().numpy()
    eigen=np.maximum(np.linalg.eigvalsh(covariance),0);prob=eigen/max(eigen.sum(),1e-30);p=prob[prob>0]
    families={'all_native':list(range(d))}
    if source=='trex':families={name:[i for i in range(d) if (i%6<3)==force] for name,force in [('force',True),('moment',False)]}
    if source=='hrdexdb_inspire_f1':families={name:list(range(j,d,3)) for j,name in enumerate(['normal','tangential','direction'])}
    return dict(frames=len(x),mae=float(s[1].sum()/s[0].sum()),
        mse=float(s[2].sum()/s[0].sum()),score=float(transformed[active].mean()),
        zero_baseline_score=float(baseline_zero[active].mean()),constant_baseline_score=float(baseline_mean[active].mean()),
        per_channel_mae=[float(per_mae[i]) if observed[i] else None for i in range(d)],
        per_quantity_mae={k:float(s[1,v].sum()/max(s[0,v].sum(),1)) for k,v in families.items()},
        nonzero_mae=float(s[6].sum()/s[7].sum()) if s[7].sum() else None,
        nonzero_zero_baseline_mae=float(s[10].sum()/s[7].sum()) if s[7].sum() else None,
        zero_mae=float(s[8].sum()/s[9].sum()) if s[9].sum() else None,
        code_std=float(np.sqrt(np.maximum(np.diag(covariance),0)).mean()),
        effective_rank=float(np.exp(-(p*np.log(p)).sum())) if p.size else 0.,
        score_description='mean per-channel MAE in fixed asinh/RMS coordinates, varying observed channels',
        physical_force_calibration_verified=False)


def run_job(config,spec,folder,notify):
    began=time.monotonic();device=torch.device('cuda')
    random.seed(config['seed']);np.random.seed(config['seed']);torch.manual_seed(config['seed']);torch.cuda.manual_seed_all(config['seed'])
    data,normalization,provenance=load_data(spec,device)
    folder.mkdir(parents=True,exist_ok=True)
    model_config=dict(input_dim=spec['input_dim'],latent_dim=spec['latent_dim'],hidden_dims=spec['hidden_dims'],
        scale=normalization['scale'],nonnegative=normalization['nonnegative'],architecture='mlp',output_mode='native numeric')
    signature=dict(model=model_config,source=spec['source'],spec=spec,training={k:v for k,v in config.items() if k!='jobs'},
        provenance=provenance,normalization=normalization)
    config_path=folder/'config.json'
    if config_path.exists() and json.loads(config_path.read_text())!=signature:
        raise ValueError('Training/provenance changed; choose a new output version')
    write_json(config_path,signature);write_json(folder/'normalization.json',normalization)
    write_json(folder/'provenance.json',provenance)
    model=NumericAutoencoder(**model_config).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=config['epochs'],eta_min=config['minimum_learning_rate'])
    mean=torch.tensor(normalization['mean_baseline'],device=device);varying=np.array(normalization['varying'])
    start=0;best=float('inf');history=[]
    if (folder/'last.pt').exists():
        saved=torch.load(folder/'last.pt',map_location=device,weights_only=False)
        model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        torch.set_rng_state(saved['torch_rng'].cpu());torch.cuda.set_rng_state(saved['cuda_rng'].cpu())
        start=saved['epoch'];best=saved['best_score']
        if (folder/'metrics.jsonl').exists():history=[json.loads(l) for l in (folder/'metrics.jsonl').read_text().splitlines() if l.strip()]
    (folder/'epochs').mkdir(exist_ok=True)
    initial=dict(model_type=MODEL_TYPE,model_config=model_config,model=model.state_dict(),epoch=0,
        source=spec['source'],spec=spec,provenance=provenance)
    if not (folder/'epoch_000.pt').exists():atomic_save(initial,folder/'epoch_000.pt')
    x,valid=data['train'];epoch_seconds=[]
    print(json.dumps(dict(event='initialized',job=spec['id'],train_frames=len(x),val_frames=len(data['val'][0]),
        input_dim=spec['input_dim'],latent_dim=spec['latent_dim'],parameters=sum(p.numel() for p in model.parameters()),
        setup_seconds=time.monotonic()-began)),flush=True)
    for epoch in range(start,config['epochs']):
        tick=time.monotonic();model.train();order=torch.randperm(len(x),device=device)
        loss_sum=0.;steps=0
        for indices in order.split(spec['batch_size']):
            target=x[indices];mask=valid[indices]
            code=model.encode(target,mask);pred=model.decode_normalized(code)
            loss=numeric_loss(pred,model.normalize(target),mask,model.nonnegative,config['nonzero_weight'])
            if not torch.isfinite(loss):raise ValueError('Nonfinite numeric reconstruction loss')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            loss_sum+=float(loss.detach());steps+=1
        validation=metrics(model,data['val'],mean,varying,max(spec['batch_size'],8192),spec['source'])
        scheduler.step();seconds=time.monotonic()-tick;epoch_seconds.append(seconds)
        improved=validation['score']<best;best=min(best,validation['score'])
        record=dict(epoch=epoch+1,source=spec['source'],job=spec['id'],train_loss=loss_sum/max(steps,1),val=validation,
            selection_score=validation['score'],best_score=best,seconds=seconds,learning_rate=optimizer.param_groups[0]['lr'])
        # Only observation-preserving, baseline-beating teachers may be promoted.
        accepted=(validation['score']<.95*min(validation['zero_baseline_score'],validation['constant_baseline_score'])
                  and validation['code_std']>1e-4)
        if validation['nonzero_mae'] is not None:
            accepted=accepted and validation['nonzero_mae']<.9*validation['nonzero_zero_baseline_mae']
        compact=dict(model_type=MODEL_TYPE,model_config=model_config,model=model.state_dict(),epoch=epoch+1,
            source=spec['source'],spec=spec,provenance=provenance,validation=validation,accepted=bool(accepted))
        atomic_save(compact,folder/'epochs'/f'epoch_{epoch+1:03d}.pt')
        if improved:
            atomic_save(compact,folder/'best.pt');write_json(folder/'best_metrics.json',record)
            write_json(folder/'quality.json',dict(accepted=bool(accepted),epoch=epoch+1,validation=validation,
                rule='Beat both fixed zero/mean baselines by 5%, retain nonzero signal and noncollapsed code.'))
        atomic_save(dict(compact,optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),best_score=best,
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),folder/'last.pt')
        with (folder/'metrics.jsonl').open('a') as stream:stream.write(json.dumps(record,allow_nan=False)+'\n')
        state=dict(stage='training',job=spec['id'],source=spec['source'],epoch=epoch+1,epochs=config['epochs'],
            seconds_per_epoch=float(np.mean(epoch_seconds[-5:])),eta_seconds=float(np.mean(epoch_seconds[-5:])*(config['epochs']-epoch-1)),
            validation=validation)
        write_json(folder/'status.json',state);notify(state);print(json.dumps(record,allow_nan=False),flush=True)
    summary=json.loads((folder/'best_metrics.json').read_text());quality=json.loads((folder/'quality.json').read_text())
    state=dict(stage='complete',job=spec['id'],source=spec['source'],epoch=config['epochs'],epochs=config['epochs'],
        best_epoch=summary['epoch'],accepted=quality['accepted'],seconds=time.monotonic()-began)
    write_json(folder/'status.json',state);notify(state)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/acquisition_autoencoders_v1.json')
    parser.add_argument('--job',help='Run one configured job; default runs the full resumable queue')
    args=parser.parse_args();config=json.loads(Path(args.config).read_text())
    if not torch.cuda.is_available():raise RuntimeError('CUDA is required for the production queue')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    output=Path(config['output']);output.mkdir(parents=True,exist_ok=True)
    jobs=[s for s in config['jobs'] if args.job is None or s['id']==args.job]
    if not jobs:parser.error('Unknown configured job')
    removed=json.loads(Path('configs/data_queue_active.json').read_text()).get('removed_sources',{})
    if any(s['acquisition'] in removed or s['source'] in ['ftp_motiontrans','freetacman','hrdexdb_allegro_v5']
           or s['source']=='feel' and (s.get('environment')=='Lab' or s['input_dim']==5) for s in jobs):
        raise ValueError('Removed, held-out or direct-only sources cannot train an autoencoder')
    with (output/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def status(state):
            entries=[]
            for job in config['jobs']:
                p=output/job['id']/'status.json'
                entries.append(dict({'job':job['id']},**(json.loads(p.read_text()) if p.exists() else dict(stage='queued'))))
            write_json(output/'status.json',dict(updated_at=time.time(),**state,jobs=entries))
        failures=[]
        for spec in jobs:
            folder=output/spec['id']
            status(dict(stage='preparing',job=spec['id'],source=spec['source']))
            try:run_job(config,spec,folder,status)
            except Exception as error:
                failures.append(dict(job=spec['id'],error=f'{type(error).__name__}: {error}'))
                folder.mkdir(exist_ok=True);write_json(folder/'status.json',dict(stage='failed',job=spec['id'],error=failures[-1]['error']))
                traceback.print_exc()
            torch.cuda.empty_cache()
        status(dict(stage='complete' if not failures else 'complete_with_failures',failures=failures))
        if failures:raise RuntimeError('Teacher jobs failed; see per-job status and queue log')


if __name__=='__main__':main()
