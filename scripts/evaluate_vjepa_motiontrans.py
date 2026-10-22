"""MotionTrans-only transfer of frozen best V-JEPA task representations.

Reuse the completed whole-clip video cache. Match the latest DINO screen:
means only, fit-only z-score, three hidden 512 layers, three head seeds and
three training-tune-selected learning rates. No encoder or classifier fitting.
"""
import gc
import json
import os
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader, Subset

from tactilelatent.aligned_frames import file_hash, manifest_signature, write_json
from tactilelatent.alignment_data import AlignedClipDataset
from tactilelatent.frozen_probe import select_indices
from tactilelatent.representation_probe import fit_probe_sweep, paired_difference
from tactilelatent.vjepa_cache import VJEPAFeatureCache, CachedAlignedClips
from tactilelatent.vjepa_model import MODEL_TYPE, BACKBONE_ID, REVISION, model_from_state

ARMS=('deterministic','stochastic')


def atomic_save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');torch.save(value,temp);temp.replace(path)


def main():
    root=Path('processing/runs/joint_vjepa21_pair_32d_v1')
    output=Path('processing/evaluations/motiontrans_vjepa21_best_v1');output.mkdir(parents=True,exist_ok=True)
    baseline=Path('processing/evaluations/motiontrans_representation_screen_v1')
    began=time.monotonic()
    def status(stage,**details):
        row=dict(stage=stage,pid=os.getpid(),elapsed_seconds=time.monotonic()-began,**details)
        write_json(output/'status.json',row);print(json.dumps(row),flush=True)
    try:
        import fcntl
        lock=(output/'evaluation.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(4);device=torch.device('cuda')
        if not torch.cuda.is_available():raise RuntimeError('CUDA required')
        status('validating_checkpoints')
        models,states,snapshots={}, {}, {}
        cache_root=Path('processing/vjepa21_clip_cache_v1')
        cache=VJEPAFeatureCache(cache_root)
        data=manifest_signature('processing/joint_alignment/manifest.json')
        if cache.index['signature']['data']!=data:raise ValueError('Stale cached video data')
        for arm in ARMS:
            original=root/arm/'best.pt'
            snapshot=output/'checkpoints'/f'{arm}.pt';snapshot.parent.mkdir(parents=True,exist_ok=True)
            if not snapshot.exists():os.link(original,snapshot)
            if file_hash(snapshot)!=file_hash(original):raise ValueError('Best source checkpoint changed')
            state=torch.load(snapshot,map_location='cpu',weights_only=False,mmap=True)
            mode='gaussian' if arm=='stochastic' else 'deterministic'
            if (state['model_type']!=MODEL_TYPE or state['backbone']!=BACKBONE_ID or state['backbone_revision']!=REVISION
                or state['config']['heldout_source']!='ftp_motiontrans' or state['model_config']['latent_dim']!=32
                or state['model_config']['latent_mode']!=mode or state['provenance']['data']!=data
                or 'ftp_motiontrans' in state['model_config']['numeric_shapes']
                or 'ftp_motiontrans' in state['model_config']['sensor_code_dims']
                or state['provenance']['feature_cache']['index_sha256']!=file_hash(cache_root/'index.json')
                or state['provenance']['feature_cache']['complete_sha256']!=file_hash(cache_root/'complete.json')):
                raise ValueError('Incorrect checkpoint, held-out source, or cache provenance')
            models[arm]=model_from_state(state).to(device).eval().requires_grad_(False)
            snapshots[arm]=dict(path=str(snapshot),sha256=file_hash(snapshot),epoch=state['epoch']+1,
                                source_validation_score=sum(v['supervision_loss'] for v in state['validation'].values())/11)
            states[arm]=state
        provenance=dict(data=data,source='ftp_motiontrans',checkpoints=snapshots,
            clip_cache_index_sha256=file_hash(cache_root/'index.json'),clip_cache_complete_sha256=file_hash(cache_root/'complete.json'),
            code={p:file_hash(p) for p in ['scripts/evaluate_vjepa_motiontrans.py','src/tactilelatent/representation_probe.py',
                                          'src/tactilelatent/vjepa_model.py','src/tactilelatent/vjepa_cache.py']},
            protocol='Source-validation-selected frozen checkpoints; latent means only; fit-only z-score; head 32→512→512→512→6; episode split seed45; head seeds45,46,47; LRs0.0003,0.001,0.003; max100 head epochs; patience10; tune-selected LR then all-training refit.',
            uncertainty='Point transfer only; no calibrated uncertainty or classifier evaluation.')
        if (output/'provenance.json').exists() and json.loads((output/'provenance.json').read_text())!=provenance:
            raise ValueError('Evaluation provenance changed')
        write_json(output/'provenance.json',provenance)
        for split in ('train','val'):
            folder=cache_root/'ftp_motiontrans'/split
            complete=json.loads((folder/'complete.json').read_text())
            if file_hash(folder/'features.npy')!=complete['features_sha256']:raise ValueError('Corrupt held-out video cache')
            common_path=output/'features'/f'{split}_targets.pt'
            destinations={arm:output/'features'/arm/f'{split}.pt' for arm in ARMS}
            if common_path.exists() and all(p.exists() for p in destinations.values()):continue
            aligned=AlignedClipDataset('processing/joint_alignment/manifest.json',split=split,sources=['ftp_motiontrans'],include_targets=True,rgb_size=384)
            indices=select_indices(aligned)
            dataset=CachedAlignedClips(aligned,cache)
            loader=DataLoader(Subset(dataset,indices),batch_size=256,num_workers=4,pin_memory=True)
            values={arm:[] for arm in ARMS};targets=[];valid=[];episodes=[]
            for step,batch in enumerate(loader):
                if 'rgb' in batch or any(s!='ftp_motiontrans' for s in batch['source']):raise ValueError('Unexpected video decoding or source')
                with torch.no_grad(),torch.autocast('cuda',torch.bfloat16):
                    encoded=batch['video_features'].to(device);mask=batch['view_mask'].to(device)
                    for arm,model in models.items():
                        mu,_=model.visual.encode(encoded,mask);values[arm].append(mu.float().cpu())
                targets.append(batch['target']);valid.append(batch['target_mask']);episodes.extend(batch['episode_id'])
                if step%20==0:status('extracting_cached_latents',split=split,batch=step,batches=len(loader))
            common=dict(target=torch.cat(targets),valid=torch.cat(valid),episode_id=episodes,indices=indices,provenance=provenance)
            previous=torch.load(baseline/'features'/f'{split}_targets.pt',map_location='cpu',weights_only=False)
            if (previous['indices']!=indices or previous['episode_id']!=episodes
                    or not torch.equal(previous['target'],common['target']) or not torch.equal(previous['valid'],common['valid'])):
                raise ValueError('Target data or clip ordering differs from DINO comparison')
            atomic_save(common_path,common)
            for arm in ARMS:atomic_save(destinations[arm],dict(values=torch.cat(values[arm]),indices=indices,provenance=provenance))
        del models,states,cache;gc.collect();torch.cuda.empty_cache()
        results={}
        def summarize():
            baselines=json.loads((baseline/'summary.json').read_text())['results']
            comparisons={}
            for arm,row in results.items():
                comparisons[arm]={name:paired_difference(baselines[name],row) for name in
                    ('epoch007_latent_32','epoch100_latent_32','dino_single_frame_768','dino_temporal_summary_2304')}
            arm_comparison=paired_difference(results['deterministic'],results['stochastic']) if len(results)==2 else None
            write_json(output/'summary.json',dict(results=results,comparisons_vs_dino=comparisons,
                stochastic_minus_deterministic=arm_comparison,provenance=provenance,
                baseline_results={name:baselines[name] for name in ('epoch007_latent_32','epoch100_latent_32','dino_single_frame_768','dino_temporal_summary_2304')},
                note='The target is a development benchmark previously used to guide experiments. These are three readout-head seeds, not three encoder-training seeds. All target data remains excluded from representation optimization.'))
            lines=['# MotionTrans transfer: frozen V-JEPA models','',provenance['protocol'],'',
                '| Representation | Encoder epoch | Mean MAE ± head-seed SD | Episode-bootstrap 95% CI |','|---|---:|---:|---|']
            for arm,row in results.items():
                m=row['validation'];lo,hi=m['episode_bootstrap_mae_ci95']
                lines.append(f"| V-JEPA {arm} | {snapshots[arm]['epoch']} | {m['mae']:.6f} ± {m['mae_head_seed_std']:.6f} | [{lo:.6f}, {hi:.6f}] |")
            for name in ('epoch007_latent_32','epoch100_latent_32','dino_single_frame_768','dino_temporal_summary_2304'):
                m=baselines[name]['validation'];lo,hi=m['episode_bootstrap_mae_ci95']
                lines.append(f"| {name} | — | {m['mae']:.6f} ± {m['mae_head_seed_std']:.6f} | [{lo:.6f}, {hi:.6f}] |")
            lines+=['','Paired validation-episode MAE differences; negative favors V-JEPA.','',
                '| V-JEPA | Reference | Difference | 95% CI |','|---|---|---:|---|']
            for arm,refs in comparisons.items():
                for name,row in refs.items():
                    lo,hi=row['paired_episode_bootstrap_ci95']
                    lines.append(f"| {arm} | {name} | {row['candidate_minus_reference_mae']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |")
            lines+=['','Every representation receives a separately fit three-hidden-layer 512-unit head. All use the same canonical 18,613 training and 1,972 validation clips, native six-channel targets, episode fit/tune partition, standardization, LR candidates and head seeds. Source-validation-selected V-JEPA checkpoints are compared with previously reported DINO references; V-JEPA epoch selection does not use target scores.',
                    '', 'Three head-fit seeds measure readout variation, not representation-training variation. Bootstrap intervals resample episodes conditional on these heads. Wider raw-DINO baselines have larger first layers. Metrics use the existing normalized torque transform, not verified SI units. No predictive-uncertainty calibration or classifier evaluation is performed.']
            (output/'analysis.md').write_text('\n'.join(lines)+'\n')
        for arm in ARMS:
            path=output/'probes'/arm/'results.json'
            if path.exists():
                result=json.loads(path.read_text())
                if result.get('provenance')!=provenance:raise ValueError('Stale probe result')
            else:
                caches={}
                for split in ('train','val'):
                    common=torch.load(output/'features'/f'{split}_targets.pt',map_location='cpu',weights_only=False)
                    features=torch.load(output/'features'/arm/f'{split}.pt',map_location='cpu',weights_only=False)
                    if common['provenance']!=provenance or features['provenance']!=provenance or common['indices']!=features['indices']:
                        raise ValueError('Stale latent cache')
                    caches[split]={**common,'values':features['values']}
                status('fitting_motiontrans_head',arm=arm)
                result=fit_probe_sweep(caches['train'],caches['val'],path.parent,device=device,
                    progress=lambda stage,**details:status(stage,arm=arm,**details))
                result['provenance']=provenance;write_json(path,result)
                del caches;gc.collect();torch.cuda.empty_cache()
            results[arm]=result;summarize()
        status('complete',summary=str(output/'summary.json'))
    except Exception as error:
        status('failed',error=str(error));raise


if __name__=='__main__':main()
