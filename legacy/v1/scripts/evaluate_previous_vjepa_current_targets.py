"""Compare the previous 8-query JEPA to the current models on matched targets.

Reuse completed current-model probes. Re-extract the old epoch-30 encoder's
last-query representation on the same compressed RGB and current-frame targets,
then fit its head with exactly the same tune/refit protocol. Historical all-frame
heads and viewer entries are preserved.
"""
import argparse
import fcntl
import gc
import json
import os
from pathlib import Path
import shutil
import time
from datetime import datetime, timezone

import torch
from torch.utils.data import DataLoader, Subset

from tactilelatent.aligned_frames import file_hash, manifest_signature, write_json
from tactilelatent.alignment_data import AlignedClipDataset
from tactilelatent.frozen_probe import seed_all, select_indices
from tactilelatent.marker_code_model import MODEL_TYPE, model_from_state
from tactilelatent.representation_probe import check_pair, fit_probe_sweep, paired_difference
from tactilelatent.storage import compatible_data
from tactilelatent.vjepa_model import BACKBONE_ID, REVISION, VJEPABackbone
from tactilelatent.vjepa_spatial import spatial_video_features
from evaluate_vjepa_current_motiontrans import digest_tensor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--current',type=Path,default=Path('processing/evaluations/motiontrans_vjepa21_current_epoch030_v1'))
    parser.add_argument('--previous',type=Path,default=Path('processing/runs/joint_vjepa21_spatial_marker_codes_32d_v1/viewer_epochs/epoch_030.pt'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    root=args.output;root.mkdir(parents=True,exist_ok=True)
    lock=(root/'evaluation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    began=time.monotonic()

    def progress(stage,**details):
        row=dict(stage=stage,pid=os.getpid(),elapsed_seconds=time.monotonic()-began,
            updated_at=time.time(),updated_utc=datetime.now(timezone.utc).isoformat(),**details)
        write_json(root/'status.json',row)
        with (root/'progress.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
        (root/'STATUS.md').write_text('# Previous versus current JEPA: matched MotionTrans status\n\n'
            +f"Updated UTC: {row['updated_utc']}\n\nStage: {stage}\n\n"
            +'Previous 32D epoch 30: re-extract last-query latents and fit a matched head.\n'
            +'Current 32D/128D epoch 30: reuse their completed current-frame probe results.\n\n'
            +'```json\n'+json.dumps(details,indent=2)+'\n```\n')
        print(json.dumps(row),flush=True)

    try:
        torch.set_num_threads(4);seed_all(45);device=torch.device('cuda')
        progress('verifying_matched_evaluation')
        current=json.loads((args.current/'summary.json').read_text())
        current_provenance=json.loads((args.current/'provenance.json').read_text())
        if current['provenance']!=current_provenance or set(current['results'])!={'current32','current128'}:
            raise ValueError('Expected completed, pinned current-model evaluation')
        snapshot=root/'checkpoints/previous32_epoch_030.pt';snapshot.parent.mkdir(parents=True,exist_ok=True)
        if not snapshot.exists():
            temporary=snapshot.with_suffix('.tmp');shutil.copyfile(args.previous,temporary);temporary.replace(snapshot)
        if file_hash(snapshot)!=file_hash(args.previous):raise ValueError('Previous checkpoint changed')
        state=torch.load(snapshot,map_location='cpu',weights_only=False)
        cfg,mc=state['config'],state['model_config']
        if (state['model_type']!=MODEL_TYPE or state['epoch']+1!=30 or mc['latent_dim']!=32
                or mc.get('prediction_mode','all_frames')!='all_frames'
                or state['backbone']!=BACKBONE_ID or state['backbone_revision']!=REVISION
                or cfg['heldout_source']!='ftp_motiontrans' or cfg.get('stochastic') or cfg['beta_max']!=0
                or mc['latent_mode']!='deterministic' or 'ftp_motiontrans' in mc['sensor_code_dims']
                or 'ftp_motiontrans' in mc['numeric_shapes']):
            raise ValueError('Unexpected previous representation or held-out policy')
        if (manifest_signature(cfg['manifest'])!=state['provenance']['data']
                and not compatible_data(cfg['manifest'],state['provenance']['data'])):
            raise ValueError('Historical data lacks approved storage-migration compatibility')
        current_checkpoint=torch.load(current_provenance['checkpoints']['current32']['path'],
            map_location='cpu',weights_only=False,mmap=True)
        manifest=current_checkpoint['config']['manifest']
        if manifest_signature(manifest)!=current_provenance['data']:
            raise ValueError('Current evaluation data changed')
        for key in ('id','revision','checkpoint_sha256','rgb_size','frames','stride','tubelet_size','rgb_preprocessing'):
            if state['provenance']['backbone'][key]!=current_provenance['backbone'][key]:
                raise ValueError('Visual backbone/input contract differs: '+key)
        if (current_provenance['head_seeds']!=[45,46,47]
                or current_provenance['learning_rates']!=[.0003,.001,.003]
                or current_provenance['head_epochs']!=100 or current_provenance['patience']!=10):
            raise ValueError('Unexpected current probe protocol')
        checkpoint=dict(path=str(snapshot),sha256=file_hash(snapshot),epoch=30,latent_dim=32)
        provenance=dict(previous_checkpoint=checkpoint,previous_training_data=state['provenance']['data'],
            evaluation_data=current_provenance['data'],current_summary_sha256=file_hash(args.current/'summary.json'),
            current_provenance_sha256=file_hash(args.current/'provenance.json'),
            sample_contract_sha256={s:file_hash(args.current/f'{s}_sample_contract.json') for s in ('train','val')},
            backbone=state['provenance']['backbone'],head=[32,512,512,512,6],head_seeds=[45,46,47],
            learning_rates=[.0003,.001,.003],epochs=100,patience=10,
            protocol='Identical current-frame targets, compressed RGB and canonical clips; previous eight-query output sliced to its last query; fit-only z-score; identical fixed episode partition, LR search, head seeds and refit',
            historical_head_export='None: preserve previous eight-target viewer heads and published evaluations',
            implementation={p:file_hash(p) for p in [__file__,'scripts/evaluate_vjepa_current_motiontrans.py',
                'src/tactilelatent/representation_probe.py','src/tactilelatent/frozen_probe.py',
                'src/tactilelatent/vjepa_spatial.py','src/tactilelatent/marker_code_model.py','src/tactilelatent/storage.py']})
        pin=root/'provenance.json'
        if pin.exists() and json.loads(pin.read_text())!=provenance:raise ValueError('Pinned comparison provenance changed')
        write_json(pin,provenance)
        result_path=root/'probes/previous32/results.json'
        if result_path.exists():
            previous=json.loads(result_path.read_text())
            if previous['provenance']!=provenance:raise ValueError('Previous matched result provenance changed')
        else:
            model=model_from_state(state).to(device).eval().requires_grad_(False)
            backbone=VJEPABackbone(cfg['backbone_repo'],cfg['backbone_checkpoint']).to(device).eval().requires_grad_(False)
            caches={}
            for split in ('train','val'):
                contract=json.loads((args.current/f'{split}_sample_contract.json').read_text())
                dataset=AlignedClipDataset(manifest,split=split,sources=['ftp_motiontrans'],
                    frames=8,stride=2,rgb_size=384,include_targets=True)
                indices=select_indices(dataset)
                if indices!=contract['indices']:raise ValueError('Canonical clip indices differ')
                loader=DataLoader(Subset(dataset,indices),batch_size=2,num_workers=4,shuffle=False,pin_memory=True)
                values=[];targets=[];valid=[];episodes=[];frames=[];split_began=time.monotonic()
                for step,batch in enumerate(loader):
                    if set(batch['source'])!={'ftp_motiontrans'}:raise ValueError('Unexpected held-out source')
                    with torch.no_grad(),torch.autocast('cuda',torch.bfloat16):
                        features=spatial_video_features(backbone,batch,device,chunk_size=2)
                        mu,logvar=model.visual.encode(features,batch['view_mask'].to(device))
                    if mu.shape!=(len(batch['rgb']),8,32) or not torch.isfinite(mu).all() or not torch.isneginf(logvar).all():
                        raise ValueError('Expected eight deterministic previous-model frame codes')
                    values.append(mu[:,-1:].float().cpu())
                    targets.append(batch['target'][:,-1:]);valid.append(batch['target_mask'][:,-1:])
                    episodes.extend(batch['episode_id']);frames.extend(batch['frame_indices'][:,-1].tolist())
                    del features,mu,logvar
                    if step%100==0 or step+1==len(loader):
                        elapsed=time.monotonic()-split_began
                        progress('extracting_previous_latents',split=split,batches_done=step+1,batches_total=len(loader),
                            split_eta_seconds=elapsed/(step+1)*(len(loader)-step-1))
                caches[split]=dict(values=torch.cat(values),target=torch.cat(targets),valid=torch.cat(valid),
                    episode_id=episodes,indices=indices)
                cache=caches[split]
                if (episodes!=contract['episode_id'] or frames!=contract['current_frame_indices']
                        or list(cache['target'].shape)!=contract['target_shape']
                        or digest_tensor(cache['target'])!=contract['target_sha256']
                        or digest_tensor(cache['valid'])!=contract['valid_sha256']):
                    raise ValueError('Matched current-frame target or chronological mapping differs')
                write_json(root/f'{split}_sample_contract.json',contract)
                progress('split_verified',split=split,clips=len(indices),episodes=len(set(episodes)))
            if any(p.grad is not None for m in (model,backbone) for p in m.parameters()):
                raise ValueError('Frozen encoder acquired gradients')
            del model,backbone,state,current_checkpoint,batch,values
            gc.collect();torch.cuda.empty_cache()
            check_pair(caches['train'],caches['val'])
            previous=fit_probe_sweep(caches['train'],caches['val'],result_path.parent,
                device=device,progress=progress)
            previous.update(provenance=provenance,checkpoint=checkpoint,encoder_epoch=30)
            write_json(result_path,previous)
        results={'previous32':previous,**current['results']}
        comparisons={name:paired_difference(previous,current['results'][name]) for name in current['results']}
        write_json(root/'summary.json',dict(results=results,comparisons_vs_previous=comparisons,provenance=provenance))
        lines=['# Matched MotionTrans: previous versus current epoch-30 JEPA','',
            'All models use identical compressed eight-frame clips, current-frame targets, episode splits and probe fitting.',
            'The previous 32D marker-code model supplies its last-query representation. Its readout is freshly fitted',
            'on the same one-target-per-clip samples as the current models; current results are reused unchanged.', '',
            '| Model | MAE | Head-fit seed SD | Change versus previous | Episode-bootstrap 95% CI |',
            '|---|---:|---:|---:|---|']
        baseline=previous['validation']['mae']
        for name,result in results.items():
            v=result['validation'];lo,hi=v['episode_bootstrap_mae_ci95']
            lines.append(f"| {name} | {v['mae']:.6f} | {v['mae_head_seed_std']:.6f} | {100*(v['mae']/baseline-1):+.2f}% | [{lo:.6f}, {hi:.6f}] |")
        lines+=['','Paired episode-bootstrap differences (candidate minus previous; negative favors the candidate):','',
            '| Model | MAE difference | 95% paired interval |','|---|---:|---|']
        for name,c in comparisons.items():
            lo,hi=c['paired_episode_bootstrap_ci95']
            lines.append(f"| {name} | {c['candidate_minus_reference_mae']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |")
        lines+=['','| Channel in existing target order | Previous 32D | Current 32D | Current 128D |',
            '|---|---:|---:|---:|']
        for index in range(6):
            label=('Finger' if index<3 else 'Thumb')+' torque component '+str(index%3)
            lines.append('| '+label+' | '+' | '.join(f"{r['validation']['per_channel_mae'][index]:.6f}" for r in results.values())+' |')
        lines+=['','Within-episode channel correlations (episode means removed):','']
        for name,r in results.items():lines.append(name+': '+str(r['validation']['within_episode_channel_correlations']))
        lines+=['','18,613 training clips/1,601 episodes; 1,972 validation clips/167 episodes; six existing normalized torque channels.',
            'All representations stay frozen. Each readout has three hidden layers of 512 units, fit-only feature standardization,',
            'a fixed episode fit/tune split, the same three head-fit seeds and three tune-selected learning rates.',
            'Historical all-eight-target scores and heads remain unchanged. These matched numbers replace any direct comparison',
            'between the new current-frame scores and those archived all-target scores.',
            'Training differs in output-query count, FreeTacMan inclusion, RGB compression and (for 128D) latent width.',
            'Head-fit variation and episode bootstrap intervals do not measure variation over visual-model training seeds.',
            'MotionTrans remains excluded from representation training but is an existing development target.']
        (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
        progress('complete',mae={name:r['validation']['mae'] for name,r in results.items()},report=str(root/'REPORT.md'))
    except Exception as error:
        progress('failed',error=str(error));raise


if __name__=='__main__':main()
