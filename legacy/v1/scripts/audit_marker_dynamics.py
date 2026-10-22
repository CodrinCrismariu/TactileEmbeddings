"""Read-only dynamics audit of frozen marker-code visual models and teachers.

All validation clips of all four marker sources; task-balanced FreeTacMan
training sample. No optimization or modifications to targets/checkpoints.
"""
from collections import defaultdict
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from tactilelatent.aligned_frames import file_hash, manifest_signature, write_json
from tactilelatent.alignment_data import AlignedClipDataset
from tactilelatent.frozen_probe import seed_all
from tactilelatent.marker_autoencoder import load_marker_autoencoder
from tactilelatent.marker_code_model import model_from_state
from tactilelatent.marker_data import MarkerCache, MarkerTargetClips, marker_collate
from tactilelatent.marker_targets import MARKER_SOURCES
from tactilelatent.sensor_code_cache import CodeCache
from tactilelatent.vjepa_model import VJEPABackbone
from tactilelatent.vjepa_spatial import spatial_video_features

ROOT = Path('processing/research/marker_dynamics_v1')
RUN = Path('processing/runs/joint_vjepa21_spatial_marker_codes_32d_v1')


def sums(prediction, target, valid):
    """Additive statistics; time centering uses each marker's valid frames."""
    p, y = np.asarray(prediction, np.float64), np.asarray(target, np.float64)
    m = np.broadcast_to(valid[...,None], y.shape) & np.isfinite(y)
    y = np.nan_to_num(y)
    n = m.sum(0, keepdims=True)
    ym = (y*m).sum(0,keepdims=True)/np.maximum(n,1)
    pm = (p*m).sum(0,keepdims=True)/np.maximum(n,1)
    yc, pc = y-ym, p-pm
    cm = m & (n>=2)
    dy, dp = y[1:]-y[:-1], p[1:]-p[:-1]
    dm = m[1:] & m[:-1]
    return dict(count=float(m.sum()), absolute=float(np.abs(p-y)[m].sum()),
        target_absolute=float(np.abs(y)[m].sum()), squared=float(((p-y)**2)[m].sum()),
        centered_count=float(cm.sum()), centered_absolute=float(np.abs(pc-yc)[cm].sum()),
        centered_error2=float(((pc-yc)**2)[cm].sum()),
        centered_target2=float((yc**2)[cm].sum()), centered_prediction2=float((pc**2)[cm].sum()),
        centered_cross=float((pc*yc)[cm].sum()),
        delta_count=float(dm.sum()), delta_absolute=float(np.abs(dp-dy)[dm].sum()),
        delta_baseline_absolute=float(np.abs(dy)[dm].sum()), delta_error2=float(((dp-dy)**2)[dm].sum()),
        delta_target2=float((dy**2)[dm].sum()), delta_prediction2=float((dp**2)[dm].sum()),
        delta_cross=float((dp*dy)[dm].sum()))


def summarize(rows):
    z={k:sum(r[k] for r in rows) for k in rows[0]}
    ratio=lambda a,b:float(a/b) if b>1e-15 else None
    return dict(clips=len(rows),mae_px=ratio(z['absolute'],z['count']),
        zero_baseline_mae_px=ratio(z['target_absolute'],z['count']),
        centered_mae_px=ratio(z['centered_absolute'],z['centered_count']),
        centered_variance_skill=1-ratio(z['centered_error2'],z['centered_target2']) if z['centered_target2']>1e-15 else None,
        centered_amplitude_ratio=np.sqrt(ratio(z['centered_prediction2'],z['centered_target2'])) if z['centered_target2']>1e-15 else None,
        centered_correlation=ratio(z['centered_cross'],np.sqrt(z['centered_prediction2']*z['centered_target2'])),
        delta_mae_px=ratio(z['delta_absolute'],z['delta_count']),
        no_change_delta_mae_px=ratio(z['delta_baseline_absolute'],z['delta_count']),
        delta_variance_skill=1-ratio(z['delta_error2'],z['delta_target2']) if z['delta_target2']>1e-15 else None,
        delta_amplitude_ratio=np.sqrt(ratio(z['delta_prediction2'],z['delta_target2'])) if z['delta_target2']>1e-15 else None,
        delta_correlation=ratio(z['delta_cross'],np.sqrt(z['delta_prediction2']*z['delta_target2'])))


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    def progress(stage,**detail):
        row=dict(stage=stage,pid=os.getpid(),elapsed_seconds=time.monotonic()-started,**detail)
        write_json(ROOT/'status.json',row);print(json.dumps(row),flush=True)
    torch.set_num_threads(4);seed_all(45);device=torch.device('cuda')
    models={};states={}
    for epoch in (10,20,30):
        state=torch.load(RUN/'viewer_epochs'/f'epoch_{epoch:03d}.pt',map_location='cpu',weights_only=False)
        if state['epoch']+1!=epoch or state['config']['heldout_source']!='ftp_motiontrans':raise ValueError('Wrong checkpoint')
        if manifest_signature(state['config']['manifest'])!=state['provenance']['data']:raise ValueError('Data changed')
        for name,digest in state['provenance']['implementation'].items():
            if file_hash(Path('src/tactilelatent')/name)!=digest:raise ValueError(f'Implementation changed: {name}')
        states[epoch]=state;models[epoch]=model_from_state(state).to(device).eval().requires_grad_(False)
    cfg=states[30]['config']
    teachers={};codes={}
    registry=json.loads(Path(cfg['marker_teachers']).read_text())
    for source in MARKER_SOURCES:
        info=registry['sources'][source]
        if file_hash(info['checkpoint'])!=info['sha256']:raise ValueError('Teacher changed')
        teachers[source]=load_marker_autoencoder(info['checkpoint'],device)[0]
        codes[source]=CodeCache(cfg['code_cache'],source)
        for name,digest in codes[source].index['array_sha256'].items():
            if file_hash(codes[source].folder/name)!=digest:raise ValueError('Corrupt teacher codes')
        if codes[source].index['signature']['teacher_sha256']!=info['sha256']:raise ValueError('Wrong teacher cache')
        stats=codes[source].index['statistics'];norm=models[30].normalization[source]
        np.testing.assert_allclose(norm.mean.cpu(),stats['mean'],rtol=1e-6,atol=1e-6)
        np.testing.assert_allclose(norm.std.cpu(),stats['std'],rtol=1e-6,atol=1e-6)
    backbone=VJEPABackbone(cfg['backbone_repo'],cfg['backbone_checkpoint']).to(device).eval().requires_grad_(False)
    if backbone.provenance['checkpoint_sha256']!=states[30]['provenance']['backbone']['checkpoint_sha256']:raise ValueError('Wrong backbone')
    write_json(ROOT/'provenance.json',dict(checkpoints={str(e):file_hash(RUN/'viewer_epochs'/f'epoch_{e:03d}.pt') for e in models},
        script_sha256=file_hash(__file__),data=states[30]['provenance']['data'],teacher_registry_sha256=file_hash(cfg['marker_teachers']),
        sampling='All chronological val clips of four marker sources; eight random train clips per FreeTacMan task, seed45.',
        metrics='Within-clip, per-marker time-centered correlation/amplitude and finite-difference skill; pairwise valid masks.',
        caution='These are measured marker motions, not calibrated forces. Frame-zero references are not verified unloaded.'))
    records=[];traces=[];examples={};cache_checks=[]
    for split in ('val','train'):
        source_list=list(MARKER_SOURCES) if split=='val' else ['freetacman']
        base=AlignedClipDataset(cfg['manifest'],split=split,sources=source_list,rgb_size=384,include_targets=False)
        wrapped=MarkerTargetClips(base,MarkerCache(cfg['marker_targets']))
        for source in source_list:
            ids=list(base.indices_by_source[source])
            if split=='train':
                grouped=defaultdict(list)
                for i in ids:grouped[base.episodes[base.windows[i][0]]['episode_id'].split('/')[0]].append(i)
                rng=np.random.default_rng(45)
                ids=sorted(i for task in sorted(grouped) for i in rng.choice(grouped[task],min(8,len(grouped[task])),replace=False).tolist())
            loader=DataLoader(Subset(wrapped,ids),batch_size=2,num_workers=4,shuffle=False,pin_memory=True,collate_fn=marker_collate)
            ae=teachers[source];dim=registry['sources'][source]['code_dim']
            progress('inference',source=source,split=split,clips=len(ids))
            for step,batch in enumerate(loader):
                target=batch['marker_target'].to(device);valid=batch['marker_valid'].to(device)&torch.isfinite(target).all(-1)
                ref=batch['marker_reference'].to(device);surface=batch['marker_surface'].to(device)
                b,t,n,_=target.shape
                reference=ref[:,None].expand(-1,t,-1,-1).reshape(b*t,n,2)
                surfaces=surface[:,None].expand(-1,t,-1).reshape(b*t,n)
                raw=torch.stack([codes[source].read(ep,frames.numpy(),normalized=False)[0] for ep,frames in zip(batch['episode_id'],batch['frame_indices'])]).to(device)
                with torch.no_grad():
                    oracle=ae.decode(raw.reshape(b*t,dim),reference,surfaces).reshape(b,t,n,2)
                    mean=torch.tensor(codes[source].index['statistics']['mean'],device=device).float()[None].expand(b*t,-1)
                    constant=ae.decode(mean,reference,surfaces).reshape(b,t,n,2)
                    if step==0:
                        fresh=ae.encode(target.reshape(b*t,n,2),valid.reshape(b*t,n),reference,surfaces).reshape(b,t,-1)
                        torch.testing.assert_close(raw,fresh,rtol=5e-4,atol=5e-5)
                        cache_checks.append(dict(source=source,split=split,max_code_error=float((raw-fresh).abs().max())))
                predictions={'teacher':oracle.cpu().numpy(),'mean_code':constant.cpu().numpy()}
                with torch.no_grad(),torch.autocast('cuda',torch.bfloat16):
                    tokens=spatial_video_features(backbone,batch,device,chunk_size=2)
                    for epoch,model in models.items():
                        out=model(tokens,batch['view_mask'].to(device),source,sample_latent=False)
                        raw_pred=model.raw_code(source,out['sensor_code'])
                        with torch.autocast('cuda',enabled=False):
                            pred=ae.decode(raw_pred.float().reshape(b*t,dim),reference,surfaces).reshape(b,t,n,2)
                        predictions[f'epoch{epoch}']=pred.float().cpu().numpy()
                    del tokens,out,raw_pred,pred
                spacing=batch['marker_spacing'].numpy()[...,None]
                y=target.cpu().numpy()*spacing[:,None];v=valid.cpu().numpy()
                predictions={k:p*spacing[:,None] for k,p in predictions.items()}
                for i in range(b):
                    ep=batch['episode_id'][i];frames=batch['frame_indices'][i].numpy()
                    one=dict(source=source,split=split,episode=ep,frames=frames.tolist(),metrics={})
                    predset={k:p[i] for k,p in predictions.items()}
                    predset['epoch30_time_reversed']=predset['epoch30'][::-1]
                    for name,p in predset.items():one['metrics'][name]=sums(p,y[i],v[i])
                    records.append(one)
                    if source=='freetacman' and split=='val':
                        magnitude=lambda a:np.abs(a).sum((1,2),where=np.broadcast_to(v[i][...,None],a.shape))/np.maximum(2*v[i].sum(1),1)
                        traces.append(dict(episode=ep,frames=frames.tolist(),target=magnitude(np.nan_to_num(y[i])).tolist(),
                                           **{k:magnitude(p).tolist() for k,p in predset.items() if k!='epoch30_time_reversed'}))
                        # Retain high-motion and typical examples after selection
                        # by ground-truth motion, never by prediction quality.
                        activity=one['metrics']['teacher']['delta_target2']/max(one['metrics']['teacher']['delta_count'],1)
                        sample=dict(episode=ep,frames=frames,target=y[i],valid=v[i],reference=batch['marker_reference'][i].numpy(),
                                    surface=batch['marker_surface'][i].numpy(),spacing=batch['marker_spacing'][i].numpy(),
                                    **{k:p for k,p in predset.items() if k!='epoch30_time_reversed'})
                        if 'highest' not in examples or activity>examples['highest'][0]:examples['highest']=(activity,sample)
                        if len(traces)==1:examples['first']=(activity,sample)
                if step%50==0 or step+1==len(loader):progress('inference',source=source,split=split,batches_done=step+1,batches_total=len(loader))
            write_json(ROOT/'clip_metrics.json',records);write_json(ROOT/'freetacman_traces.json',traces)
    groups={}
    for source in MARKER_SOURCES:
        for split in ('val','train'):
            rows=[r for r in records if r['source']==source and r['split']==split]
            if not rows:continue
            activity=np.array([r['metrics']['teacher']['delta_target2']/max(r['metrics']['teacher']['delta_count'],1) for r in rows])
            high=activity>=np.quantile(activity,.75)
            groups[f'{source}/{split}']=dict(clips=len(rows),episodes=len({r['episode'] for r in rows}),
                all={name:summarize([r['metrics'][name] for r in rows]) for name in rows[0]['metrics']},
                highest_motion_quartile={name:summarize([r['metrics'][name] for r,h in zip(rows,high) if h]) for name in rows[0]['metrics']})
    write_json(ROOT/'summary.json',dict(groups=groups,cache_checks=cache_checks))
    for name,(_,example) in examples.items():
        np.savez_compressed(ROOT/f'example_{name}.npz',**example)
    # Native tactile snapshot and detections for the most dynamic example.
    from tactilelatent.aligned_frames import target_images
    from PIL import Image
    rows=json.loads(Path(cfg['manifest']).read_text())['episodes']
    example=examples['highest'][1];row=next(r for r in rows if r['source']=='freetacman' and r['episode_id']==example['episode'])
    frames=[int(example['frames'][0]),int(example['frames'][-1])]
    for sid,images in enumerate(target_images(row,frames)):
        for j,image in enumerate(images):Image.fromarray(image).save(ROOT/f'high_motion_surface{sid}_frame{frames[j]}.png')
    plot_report(groups,traces,examples,ROOT)
    progress('complete',summary=str(ROOT/'summary.json'),report=str(ROOT/'REPORT.md'))


def plot_report(groups,traces,examples,root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    free=groups['freetacman/val']
    # Episodes picked using target trajectory amplitude only.
    episodes=defaultdict(list)
    for row in traces:episodes[row['episode']].append(row)
    energy={e:float(np.std([y for r in rs for y in r['target']])) for e,rs in episodes.items()}
    ordered=sorted(energy,key=energy.get)
    chosen=[ordered[len(ordered)//4],ordered[len(ordered)//2],ordered[3*len(ordered)//4],ordered[-1]]
    fig,axes=plt.subplots(4,1,figsize=(12,11),sharex=True)
    for ax,episode in zip(axes,chosen):
        rows=sorted(episodes[episode],key=lambda r:r['frames'][0]);x=np.concatenate([r['frames'] for r in rows])
        for name,color in [('target','black'),('teacher','#3b9d54'),('epoch10','#82b5e0'),('epoch30','#d0523d')]:
            ax.plot(x,np.concatenate([r[name] for r in rows]),label=name,color=color,linewidth=1.3)
        ax.set_title(episode);ax.set_ylabel('Mean |displacement| (px)');ax.grid(alpha=.2)
    axes[0].legend(ncol=4);axes[-1].set_xlabel('Original chronological frame index')
    fig.suptitle('FreeTacMan held-out task: measured marker motion vs frozen predictions')
    fig.tight_layout();fig.savefig(root/'freetacman_episode_trajectories.png',dpi=150);plt.close(fig)
    example=examples['highest'][1];y=example['target'];v=example['valid'];surface=example['surface']
    # Per-marker signed dynamics, avoiding cancellation in a field average.
    candidate=np.where(v.all(0),np.nanvar(y,axis=0).sum(-1),-1)
    marker=int(candidate.argmax())
    fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True)
    for d,ax in enumerate(axes):
        for name,color in [('target','black'),('teacher','#3b9d54'),('epoch10','#82b5e0'),('epoch30','#d0523d')]:
            ax.plot(example['frames'],example[name][:,marker,d],marker='o',label=name,color=color)
        ax.set_ylabel(['dx (native px)','dy (native px)'][d]);ax.grid(alpha=.2)
    axes[0].legend(ncol=4);axes[-1].set_xlabel('Original frame index')
    fig.suptitle(f"Highest-motion validation clip: {example['episode']}\nMost varying fully tracked marker, surface {surface[marker]}")
    fig.tight_layout();fig.savefig(root/'freetacman_high_motion_marker.png',dpi=150);plt.close(fig)
    lines=['# Marker interaction dynamics audit', '',
           'Frozen epochs 10, 20, 30; no training. Every canonical validation clip of the four marker sources.',
           'FreeTacMan training diagnostic: eight clips per task, seed45. Chronological native marker coordinates preserved.',
           'FreeTacMan validation is exclusively the held-out SprinkleSesame task.', '',
           'All MAEs are native pixels. Dynamics compare consecutive sampled frames (stride two), within each eight-frame clip.',
           'Amplitude ratio = predicted/true temporal RMS. Skill = 1 − squared error / no-change squared error;',
           'zero skill matches a temporally constant predictor; negative skill is worse. Correlation uses time-centered marker coordinates.',
           'Teacher = frozen AE decoding the measured tactile code; mean_code = train-mean tactile code decoded at the same static geometry.', '',
           '| Source/split | Predictor | MAE px | Centered correlation | Centered amplitude ratio | Dynamics skill | Delta correlation | Delta amplitude ratio |',
           '|---|---|---:|---:|---:|---:|---:|---:|']
    fmt=lambda v:'undefined' if v is None else f'{v:.4f}'
    for group,rows in groups.items():
        for name,r in rows['all'].items():
            lines.append('| '+' | '.join([group,name]+[fmt(r[k]) for k in ['mae_px','centered_correlation','centered_amplitude_ratio','delta_variance_skill','delta_correlation','delta_amplitude_ratio']])+' |')
    lines.extend(['', '## Highest-motion quartile of FreeTacMan validation clips', '',
                  '| Predictor | Delta MAE px | No-change delta MAE px | Dynamics skill | Delta correlation | Delta amplitude ratio |',
                  '|---|---:|---:|---:|---:|---:|'])
    for name,r in free['highest_motion_quartile'].items():
        lines.append('| '+' | '.join([name]+[fmt(r[k]) for k in ['delta_mae_px','no_change_delta_mae_px','delta_variance_skill','delta_correlation','delta_amplitude_ratio']])+' |')
    lines.extend(['', '## Visualizations', '', '![Episode trajectories](freetacman_episode_trajectories.png)', '',
                  '![Signed marker trajectory](freetacman_high_motion_marker.png)', '',
                  'High-motion examples are chosen from ground-truth activity, not model errors.',
                  'Marker tracks measure visible dot movement; they do not prove force accuracy or separate load from sensor/camera motion.',
                  'The AE teacher is a reconstruction reference using tactile observations, not a deployable visual baseline.',
                  'Dynamic metrics are within clips; concatenated episode plots may also include discontinuities between independently encoded clips.'])
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
