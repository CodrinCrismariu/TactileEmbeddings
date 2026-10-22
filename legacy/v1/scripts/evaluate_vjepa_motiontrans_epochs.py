"""Deterministic V-JEPA checkpoint transfer sweep with the deeper head only."""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
from torch.utils.data import DataLoader, Subset

from tactilelatent.aligned_frames import file_hash,manifest_signature,write_json
from tactilelatent.alignment_data import AlignedClipDataset
from tactilelatent.frozen_probe import select_indices
from tactilelatent.representation_probe import fit_probe_sweep,paired_difference
from tactilelatent.vjepa_cache import VJEPAFeatureCache,CachedAlignedClips
from tactilelatent.vjepa_model import MODEL_TYPE,BACKBONE_ID,REVISION,model_from_state

ROOT=Path('processing/evaluations/motiontrans_vjepa21_deterministic_epochs_v1')
PARENT=Path('processing/runs/joint_vjepa21_pair_32d_v1/deterministic')
PREVIOUS=Path('processing/evaluations/motiontrans_vjepa21_best_v1')
DINO=Path('processing/evaluations/motiontrans_representation_screen_v1')
EPOCHS=(5,10,15,20,25)


def atomic_save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');torch.save(value,temp);temp.replace(path)


def extract():
    cache_root=Path('processing/vjepa21_clip_cache_v1')
    cache=VJEPAFeatureCache(cache_root)
    data=manifest_signature('processing/joint_alignment/manifest.json')
    if cache.index['signature']['data']!=data:raise ValueError('Cached video data changed')
    snapshots={};models={}
    for epoch in EPOCHS:
        source=PARENT/'viewer_epochs'/f'epoch_{epoch:03d}.pt'
        snapshot=ROOT/'checkpoints'/f'epoch_{epoch:03d}.pt';snapshot.parent.mkdir(parents=True,exist_ok=True)
        if not snapshot.exists():os.link(source,snapshot)
        digest=file_hash(snapshot)
        if digest!=file_hash(source):raise ValueError('Source checkpoint changed')
        state=torch.load(snapshot,map_location='cpu',weights_only=False,mmap=True)
        if (state['epoch']+1!=epoch or state.get('model_type')!=MODEL_TYPE
                or state['backbone']!=BACKBONE_ID or state['backbone_revision']!=REVISION
                or state['config']['heldout_source']!='ftp_motiontrans'
                or state['model_config']['latent_mode']!='deterministic'
                or state['config']['stochastic'] or state['config']['beta_max']!=0
                or state['model_config']['latent_dim']!=32
                or state['provenance']['data']!=data
                or 'ftp_motiontrans' in state['model_config']['numeric_shapes']
                or 'ftp_motiontrans' in state['model_config']['sensor_code_dims']
                or state['provenance']['feature_cache']['index_sha256']!=file_hash(cache_root/'index.json')
                or state['provenance']['feature_cache']['complete_sha256']!=file_hash(cache_root/'complete.json')):
            raise ValueError('Wrong epoch, latent mode, held-out policy or data/backbone provenance')
        models[epoch]=model_from_state(state).cuda().eval().requires_grad_(False)
        snapshots[str(epoch)]=dict(path=str(snapshot),sha256=digest,epoch=epoch,
            source_validation_score=sum(m['supervision_loss'] for m in state['validation'].values())/11)
    provenance=dict(epochs=list(EPOCHS),latent_mode='deterministic',latent_dim=32,data=data,checkpoints=snapshots,
        video_cache_index_sha256=file_hash(cache_root/'index.json'),
        video_cache_complete_sha256=file_hash(cache_root/'complete.json'),
        previous_epoch28_summary_sha256=file_hash(PREVIOUS/'summary.json'),
        dino_summary_sha256=file_hash(DINO/'summary.json'),
        code={p:file_hash(p) for p in ['scripts/evaluate_vjepa_motiontrans_epochs.py',
            'src/tactilelatent/representation_probe.py','src/tactilelatent/vjepa_model.py','src/tactilelatent/vjepa_cache.py']},
        head=[32,512,512,512,6],head_seeds=[45,46,47],learning_rates=[.0003,.001,.003],
        protocol='Frozen deterministic 32D latents; same deeper head only; fit-only z-score; fixed episode fit/tune split seed45; three head-fit seeds; three LRs selected by mean tune loss; max100 head epochs; patience10; refit selected durations on all training episodes.',
        note='Requested epoch sweep uses the existing development target; validation-guided epoch ranking is not a fresh final test. Encoder and V-JEPA weights stay frozen. No classifier or uncertainty evaluation.')
    pinned=ROOT/'provenance.json'
    if pinned.exists() and json.loads(pinned.read_text())!=provenance:raise ValueError('Sweep provenance changed')
    write_json(pinned,provenance)
    for split in ('train','val'):
        common_path=ROOT/'features'/f'{split}_targets.pt'
        destinations={e:ROOT/'features'/f'epoch_{e:03d}'/f'{split}.pt' for e in EPOCHS}
        if common_path.exists() and all(p.exists() for p in destinations.values()):continue
        folder=cache_root/'ftp_motiontrans'/split
        complete=json.loads((folder/'complete.json').read_text())
        if file_hash(folder/'features.npy')!=complete['features_sha256']:raise ValueError('Corrupt held-out video cache')
        aligned=AlignedClipDataset('processing/joint_alignment/manifest.json',split=split,sources=['ftp_motiontrans'],include_targets=True,rgb_size=384)
        indices=select_indices(aligned)
        loader=DataLoader(Subset(CachedAlignedClips(aligned,cache),indices),batch_size=256,num_workers=4,pin_memory=True)
        values={e:[] for e in EPOCHS};targets=[];valid=[];episodes=[]
        for batch in loader:
            if 'rgb' in batch or any(s!='ftp_motiontrans' for s in batch['source']):raise ValueError('Wrong source or live RGB input')
            with torch.no_grad(),torch.autocast('cuda',torch.bfloat16):
                x=batch['video_features'].cuda();mask=batch['view_mask'].cuda()
                for epoch,model in models.items():
                    mu,logvar=model.visual.encode(x,mask)
                    if not torch.isneginf(logvar).all():raise ValueError('Unexpected stochastic latent')
                    values[epoch].append(mu.float().cpu())
            targets.append(batch['target']);valid.append(batch['target_mask']);episodes.extend(batch['episode_id'])
        common=dict(target=torch.cat(targets),valid=torch.cat(valid),episode_id=episodes,indices=indices,provenance=provenance)
        reference=torch.load(PREVIOUS/'features'/f'{split}_targets.pt',map_location='cpu',weights_only=False)
        if (indices!=reference['indices'] or episodes!=reference['episode_id']
                or not torch.equal(common['target'],reference['target']) or not torch.equal(common['valid'],reference['valid'])):
            raise ValueError('MotionTrans targets or clip ordering differ from epoch28')
        atomic_save(common_path,common)
        for epoch in EPOCHS:atomic_save(destinations[epoch],dict(values=torch.cat(values[epoch]),indices=indices,provenance=provenance))
    del models,cache;gc.collect();torch.cuda.empty_cache()
    return provenance


def summarize(provenance):
    results={}
    for epoch in EPOCHS:
        path=ROOT/'probes'/f'epoch_{epoch:03d}'/'results.json'
        if path.exists():
            row=json.loads(path.read_text())
            if row.get('provenance')!=provenance:raise ValueError('Stale completed transfer result')
            results[str(epoch)]=row
    previous=json.loads((PREVIOUS/'summary.json').read_text())['results']['deterministic']
    dino=json.loads((DINO/'summary.json').read_text())['results']['epoch007_latent_32']
    comparisons={e:dict(vs_epoch28=paired_difference(previous,r),vs_dino_epoch7=paired_difference(dino,r)) for e,r in results.items()}
    best=min(results,key=lambda e:results[e]['validation']['mae']) if results else None
    write_json(ROOT/'summary.json',dict(results=results,comparisons=comparisons,completed_epochs=list(map(int,results)),
        requested_epochs=list(EPOCHS),best_requested_epoch=int(best) if best else None,
        epoch28_reference=previous,dino_epoch7_reference=dino,provenance=provenance))
    lines=['# MotionTrans transfer across deterministic V-JEPA epochs','',provenance['protocol'],'',
        '| Encoder epoch | Mean MAE ± head-seed SD | Episode-bootstrap 95% CI | Selected LR | Selected head epochs |',
        '|---|---:|---|---:|---|']
    for epoch in EPOCHS:
        if str(epoch) not in results:continue
        row=results[str(epoch)];m=row['validation'];lo,hi=m['episode_bootstrap_mae_ci95']
        lines.append(f"| {epoch} | {m['mae']:.6f} ± {m['mae_head_seed_std']:.6f} | [{lo:.6f}, {hi:.6f}] | {row['selected_learning_rate']:g} | {row['selected_head_epochs']} |")
    for label,row in [('28 (previously evaluated)',previous),('DINO epoch7 32D reference',dino)]:
        m=row['validation'];lo,hi=m['episode_bootstrap_mae_ci95']
        lines.append(f"| {label} | {m['mae']:.6f} ± {m['mae_head_seed_std']:.6f} | [{lo:.6f}, {hi:.6f}] | {row['selected_learning_rate']:g} | {row['selected_head_epochs']} |")
    lines+=['','Paired episode-bootstrap comparisons (candidate minus reference; negative favors the candidate):','',
        '| Encoder epoch | Reference | MAE difference | 95% paired CI |','|---|---|---:|---|']
    for epoch,refs in comparisons.items():
        for label,row in refs.items():
            lo,hi=row['paired_episode_bootstrap_ci95']
            lines.append(f"| {epoch} | {label} | {row['candidate_minus_reference_mae']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |")
    lines+=['','All conditions use full canonical MotionTrans: 18,613 training clips, 1,972 validation clips, 167 validation episodes, six normalized output channels. Fit-only feature normalization and episode-level fit/tune selection match the previous epoch-28 protocol. Only the requested deterministic encoders and the three-hidden-layer 512-unit head are evaluated.',
        '',provenance['note'],'','Reported seeds vary the head fit only, not encoder pretraining. Confidence intervals resample validation episodes conditional on the fitted heads. Any ranking by these validation results is exploratory and time-confounded.']
    (ROOT/'analysis.md').write_text('\n'.join(lines)+'\n')
    return list(map(int,results))


def worker(epochs):
    provenance=json.loads((ROOT/'provenance.json').read_text())
    torch.set_num_threads(4);device=torch.device('cuda')
    status_path=ROOT/'workers'/('_'.join(map(str,epochs))+'.json')
    for epoch in epochs:
        folder=ROOT/'probes'/f'epoch_{epoch:03d}';folder.mkdir(parents=True,exist_ok=True)
        import fcntl
        lock=(folder/'fit.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (folder/'results.json').exists():continue
        caches={}
        for split in ('train','val'):
            common=torch.load(ROOT/'features'/f'{split}_targets.pt',map_location='cpu',weights_only=False)
            values=torch.load(ROOT/'features'/f'epoch_{epoch:03d}'/f'{split}.pt',map_location='cpu',weights_only=False)
            if common['provenance']!=provenance or values['provenance']!=provenance or common['indices']!=values['indices']:
                raise ValueError('Latent/target cache provenance differs')
            caches[split]={**common,'values':values['values']}
        def progress(stage,**details):
            write_json(status_path,dict(stage=stage,pid=os.getpid(),epoch=epoch,**details))
        result=fit_probe_sweep(caches['train'],caches['val'],folder,device=device,progress=progress)
        result.update(provenance=provenance,encoder_epoch=epoch)
        write_json(folder/'results.json',result)
        progress('epoch_complete',mae=result['validation']['mae'])
        del caches;gc.collect();torch.cuda.empty_cache()
        lock.close()
    write_json(status_path,dict(stage='complete',pid=os.getpid(),epochs=epochs))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker-epochs',nargs='+',type=int)
    args=parser.parse_args()
    if args.worker_epochs:
        if any(e not in EPOCHS for e in args.worker_epochs):raise ValueError('Unrequested epoch')
        worker(args.worker_epochs);return
    ROOT.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(ROOT/'evaluation.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    began=time.monotonic();processes=[]
    def status(stage,**details):
        row=dict(stage=stage,pid=os.getpid(),elapsed_seconds=time.monotonic()-began,**details)
        write_json(ROOT/'status.json',row);print(json.dumps(row),flush=True)
    try:
        torch.set_num_threads(4)
        if not torch.cuda.is_available():raise RuntimeError('CUDA required')
        status('extracting_requested_epoch_latents',epochs=list(EPOCHS))
        provenance=extract();summarize(provenance)
        for epochs in (EPOCHS[::2],EPOCHS[1::2]):
            path=ROOT/'workers'/('_'.join(map(str,epochs))+'.log');path.parent.mkdir(parents=True,exist_ok=True)
            log=path.open('a')
            p=subprocess.Popen([sys.executable,'-u',__file__,'--worker-epochs',*map(str,epochs)],
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            processes.append((p,log,epochs))
        last=-1
        while True:
            completed=summarize(provenance)
            states=[dict(pid=p.pid,epochs=list(e),returncode=p.poll()) for p,_,e in processes]
            if len(completed)!=last or time.monotonic()-began-last_status>20:
                status('fitting_deeper_heads',completed_epochs=completed,total_conditions=5,workers=states)
                last=len(completed);last_status=time.monotonic()-began
            if all(p.poll() is not None for p,_,_ in processes):break
            time.sleep(5)
        for p,log,epochs in processes:
            log.close()
            if p.returncode:raise RuntimeError(f'Head worker {epochs} failed; inspect worker log')
        if summarize(provenance)!=list(EPOCHS):raise RuntimeError('Requested epoch evaluations incomplete')
        status('complete',summary=str(ROOT/'summary.json'),epochs=list(EPOCHS))
    except Exception as error:
        for p,_,_ in processes:
            if p.poll() is None:p.terminate()
        status('failed',error=str(error));raise


if __name__=='__main__':main()
