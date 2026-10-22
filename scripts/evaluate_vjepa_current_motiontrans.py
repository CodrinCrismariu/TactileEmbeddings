"""Matched MotionTrans transfer for the final current-frame 32D/128D encoders.

One causal eight-image clip produces one current-frame latent and torque target.
The frozen V-JEPA pass is shared. Only compact latents are retained in RAM;
spatial features and RGB are never cached. Readout fitting uses the established
three-hidden-layer probe, episode split, fit-only z-score and tune-only LR search.
"""
import argparse
import fcntl
import gc
import hashlib
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
from tactilelatent.vjepa_model import BACKBONE_ID, REVISION, VJEPABackbone
from tactilelatent.vjepa_spatial import spatial_video_features
from register_vjepa_transfer_viewer import register


RUNS = {32: Path('processing/runs/joint_vjepa21_spatial_current32d_v1'),
        128: Path('processing/runs/joint_vjepa21_spatial_current128d_v1')}
REFERENCE = Path('processing/evaluations/motiontrans_vjepa21_best_v1/features')


def digest_tensor(value):
    return hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    lock = (root/'evaluation.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    began = time.monotonic()
    results = {}

    def progress(stage, **details):
        row = dict(stage=stage, pid=os.getpid(), epoch=args.epoch,
            elapsed_seconds=time.monotonic()-began, updated_at=time.time(),
            updated_utc=datetime.now(timezone.utc).isoformat(), completed_conditions=list(results), **details)
        write_json(root/'status.json', row)
        with (root/'progress.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')
        lines = ['# MotionTrans evaluation status', '',
            f"Updated UTC: {row['updated_utc']}", f"Stage: {stage}",
            f"Elapsed: {row['elapsed_seconds']/60:.1f} minutes", '',
            'Requested models: deterministic current-frame 32D and 128D, encoder epoch '+str(args.epoch)+'.',
            'Head: latent -> 512 -> 512 -> 512 -> 6. Both visual encoders stay frozen.',
            'Targets: final frame of each canonical eight-frame clip.', '', '```json', json.dumps(details, indent=2), '```']
        if results:
            lines += ['', '| Completed model | MAE | Head-fit seed SD |', '|---|---:|---:|']
            for name, result in results.items():
                v = result['validation']
                lines.append(f"| {name} | {v['mae']:.6f} | {v['mae_head_seed_std']:.6f} |")
        (root/'STATUS.md').write_text('\n'.join(lines)+'\n')
        print(json.dumps(row), flush=True)

    def summarize(provenance):
        comparison = (paired_difference(results['current32'], results['current128'])
                      if {'current32', 'current128'} <= results.keys() else None)
        write_json(root/'summary.json', dict(results=results, comparison_128_minus_32=comparison, provenance=provenance))
        lines = ['# Final current-frame V-JEPA MotionTrans transfer', '',
            'Frozen deterministic epoch-'+str(args.epoch)+' encoders; separately fitted heads with three hidden layers of 512 units.',
            'Identical canonical clips and current-frame targets; fit-only feature z-score, fixed episode fit/tune split seed 45,',
            'three learning rates selected using tune episodes and three head-fit seeds, followed by all-training refit.', '',
            '| Model | MAE | Head-fit seed SD | Episode-bootstrap 95% CI | LR | Head epochs |',
            '|---|---:|---:|---|---:|---|']
        for name, result in results.items():
            v = result['validation']; lo, hi = v['episode_bootstrap_mae_ci95']
            lines.append(f"| {name} | {v['mae']:.6f} | {v['mae_head_seed_std']:.6f} | [{lo:.6f}, {hi:.6f}] | {result['selected_learning_rate']:g} | {result['selected_head_epochs']} |")
        if comparison:
            lines += ['', '128D minus 32D MAE: '+f"{comparison['candidate_minus_reference_mae']:+.6f}",
                'Paired episode-bootstrap 95% CI: '+str(comparison['paired_episode_bootstrap_ci95'])+'. Negative favors 128D.']
        lines += ['', '| Channel in existing target order | '+ ' | '.join(results)+' |',
            '|---|'+'---:|'*len(results)] if results else []
        if results:
            for index in range(6):
                label = ('Finger' if index < 3 else 'Thumb')+' torque component '+str(index % 3)
                lines.append('| '+label+' | '+' | '.join(f"{r['validation']['per_channel_mae'][index]:.6f}" for r in results.values())+' |')
            lines += ['', 'Within-episode channel correlations (episode means removed):', '']
            for name, result in results.items():
                lines.append(name+': '+str(result['validation']['within_episode_channel_correlations']))
        lines += ['', 'The complete canonical split is used: 18,613 training clips from 1,601 episodes and 1,972 validation clips from 167 episodes.',
            'Each clip contributes ONE current-frame sample. Earlier eight-output evaluations contributed eight samples per clip,',
            'so archived scores such as the previous marker-code model score are not directly matched benchmarks for this evaluation.',
            'MAE is in the existing six tanh-normalized torque channels, not raw physical torque units.',
            'Head-fit seeds do not represent multiple visual-model training seeds. Bootstrap intervals resample validation episodes',
            'conditional on the fitted readouts. MotionTrans is excluded from representation training; this is the existing development target.',
            'Only prediction heads are fitted. There is no classifier, uncertainty calibration, representation fine-tuning, or dataset resizing.',
            'Full provenance, checkpoint hashes, target hashes and episode/frame identities are saved beside this report.']
        (root/'REPORT.md').write_text('\n'.join(lines)+'\n')

    try:
        torch.set_num_threads(4)
        seed_all(45)
        device = torch.device('cuda')
        progress('verifying_checkpoints')
        states, models, checkpoints = {}, {}, {}
        for dim, run in RUNS.items():
            name = f'current{dim}'
            source = run/'viewer_epochs'/f'epoch_{args.epoch:03d}.pt'
            snapshot = root/'checkpoints'/f'{name}_epoch_{args.epoch:03d}.pt'
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            if not snapshot.exists():
                temporary = snapshot.with_suffix('.tmp')
                shutil.copyfile(source, temporary)
                temporary.replace(snapshot)
            if file_hash(snapshot) != file_hash(source):
                raise ValueError('Requested checkpoint differs from pinned snapshot')
            state = torch.load(snapshot, map_location='cpu', weights_only=False)
            cfg, mc = state['config'], state['model_config']
            data = manifest_signature(cfg['manifest'])
            if (state['model_type'] != MODEL_TYPE or state['epoch']+1 != args.epoch
                    or cfg.get('prediction_mode') != 'current_frame' or mc.get('prediction_mode') != 'current_frame'
                    or state['backbone'] != BACKBONE_ID or state['backbone_revision'] != REVISION
                    or cfg['heldout_source'] != 'ftp_motiontrans' or cfg['stochastic'] or cfg['beta_max'] != 0
                    or mc['latent_mode'] != 'deterministic' or mc['latent_dim'] != dim
                    or state['provenance']['data'] != data
                    or 'ftp_motiontrans' in mc['sensor_code_dims'] or 'ftp_motiontrans' in mc['numeric_shapes']):
                raise ValueError(f'Invalid checkpoint, architecture or held-out policy: {name}')
            for filename, digest in state['provenance']['implementation'].items():
                if file_hash(Path('src/tactilelatent')/filename) != digest:
                    raise ValueError(f'Changed training implementation: {filename}')
            models[name] = model_from_state(state).to(device).eval().requires_grad_(False)
            states[name] = state
            checkpoints[name] = dict(path=str(snapshot), sha256=file_hash(snapshot), epoch=args.epoch, latent_dim=dim)
        first = states['current32']
        cfg = first['config']
        if any(state['provenance']['data'] != first['provenance']['data']
               or state['provenance']['backbone'] != first['provenance']['backbone'] for state in states.values()):
            raise ValueError('Cannot share visual extraction across different data or backbones')
        provenance = dict(checkpoints=checkpoints, data=first['provenance']['data'], backbone=first['provenance']['backbone'],
            heads={name:[dim,512,512,512,6] for dim,name in ((32,'current32'),(128,'current128'))},
            head_seeds=[45,46,47], learning_rates=[.0003,.001,.003], head_epochs=100, patience=10,
            target_policy='One final current-frame target per canonical 8-frame stride-2 clip; seven past images; no future images',
            normalization='Existing native target transform; fit-episode-only feature z-score',
            protocol='Fixed episode fit/tune seed45; three LRs selected by mean tune loss across three head seeds; all-training refit; no validation selection',
            extraction='One shared live frozen V-JEPA pass per clip; compact final latents in RAM only; no RGB/spatial feature cache',
            reference_target_hashes={split:file_hash(REFERENCE/f'{split}_targets.pt') for split in ('train','val')},
            implementation={p:file_hash(p) for p in [__file__, 'src/tactilelatent/representation_probe.py',
                'src/tactilelatent/frozen_probe.py', 'src/tactilelatent/vjepa_spatial.py',
                'src/tactilelatent/marker_code_model.py', 'scripts/register_vjepa_transfer_viewer.py']})
        pin = root/'provenance.json'
        if pin.exists() and json.loads(pin.read_text()) != provenance:
            raise ValueError('Pinned evaluation provenance changed')
        write_json(pin, provenance)
        pending = []
        for name in models:
            path = root/'probes'/name/'results.json'
            if path.exists():
                result = json.loads(path.read_text())
                if result['provenance'] != provenance or result['checkpoint'] != checkpoints[name]:
                    raise ValueError('Completed result provenance changed')
                results[name] = result
            else:
                pending.append(name)
        summarize(provenance)
        if pending:
            backbone = VJEPABackbone(cfg['backbone_repo'],cfg['backbone_checkpoint']).to(device).eval().requires_grad_(False)
            if backbone.provenance['checkpoint_sha256'] != first['provenance']['backbone']['checkpoint_sha256']:
                raise ValueError('Backbone checkpoint changed')
            caches = {name:{} for name in pending}
            for split in ('train','val'):
                dataset = AlignedClipDataset(cfg['manifest'],split=split,sources=['ftp_motiontrans'],
                    frames=8,stride=2,rgb_size=384,include_targets=True)
                indices = select_indices(dataset)
                loader = DataLoader(Subset(dataset,indices),batch_size=2,num_workers=4,
                    shuffle=False,pin_memory=True)
                values = {name:[] for name in pending}
                target, valid, episodes, frames = [], [], [], []
                reference = torch.load(REFERENCE/f'{split}_targets.pt',map_location='cpu',weights_only=False)
                split_began = time.monotonic()
                offset = 0
                for step,batch in enumerate(loader):
                    count = len(batch['rgb'])
                    positions = slice(offset,offset+count)
                    if set(batch['source']) != {'ftp_motiontrans'}:
                        raise ValueError('Unexpected source in held-out extraction')
                    if (batch['episode_id'] != reference['episode_id'][positions]
                            or not torch.equal(batch['target'],reference['target'][positions])
                            or not torch.equal(batch['target_mask'],reference['valid'][positions])):
                        raise ValueError('Canonical target values or episode order differ from historical benchmark')
                    expected_frames = torch.stack([torch.arange(dataset.windows[i][1],dataset.windows[i][1]+15,2)
                                                   for i in indices[offset:offset+count]])
                    if not torch.equal(batch['frame_indices'],expected_frames):
                        raise ValueError('Chronological input clip mapping changed')
                    with torch.no_grad(),torch.autocast('cuda',torch.bfloat16):
                        features = spatial_video_features(backbone,batch,device,chunk_size=2)
                        for name in pending:
                            dim = checkpoints[name]['latent_dim']
                            mu, logvar = models[name].visual.encode(features,batch['view_mask'].to(device))
                            if (mu.shape != (count,1,dim) or not torch.isfinite(mu).all()
                                    or not torch.isneginf(logvar).all()):
                                raise ValueError('Expected finite deterministic current-frame latents')
                            values[name].append(mu.float().cpu())
                        del features,mu,logvar
                    target.append(batch['target'][:,-1:])
                    valid.append(batch['target_mask'][:,-1:])
                    episodes.extend(batch['episode_id'])
                    frames.extend(batch['frame_indices'][:,-1].tolist())
                    offset += count
                    if step % 100 == 0 or step+1 == len(loader):
                        elapsed = time.monotonic()-split_began
                        progress('extracting_live_latents',split=split,batches_done=step+1,batches_total=len(loader),
                            clips_done=offset,split_eta_seconds=elapsed/(step+1)*(len(loader)-step-1))
                if indices != reference['indices'] or offset != len(indices):
                    raise ValueError('Canonical clip selection changed')
                shared = dict(target=torch.cat(target),valid=torch.cat(valid),episode_id=episodes,indices=indices)
                if not torch.equal(shared['target'],reference['target'][:,-1:]):
                    raise ValueError('Current target slice differs')
                contract = dict(clips=len(indices),episodes=len(set(episodes)),target_shape=list(shared['target'].shape),
                    target_sha256=digest_tensor(shared['target']),valid_sha256=digest_tensor(shared['valid']),
                    episode_id=episodes,current_frame_indices=frames,indices=indices)
                write_json(root/f'{split}_sample_contract.json',contract)
                for name in pending:
                    caches[name][split] = dict(values=torch.cat(values[name]),**shared)
                progress('split_verified',split=split,clips=len(indices),episodes=len(set(episodes)),
                    compact_latent_bytes={name:caches[name][split]['values'].numel()*4 for name in pending})
            if any(p.grad is not None for model in [backbone,*models.values()] for p in model.parameters()):
                raise ValueError('Frozen visual encoder acquired gradients')
            del backbone,models,states,first,reference,values,batch
            gc.collect(); torch.cuda.empty_cache()
            for name in pending:
                check_pair(caches[name]['train'],caches[name]['val'])
                progress('fitting_motiontrans_heads',condition=name)
                result = fit_probe_sweep(caches[name]['train'],caches[name]['val'],root/'probes'/name,
                    device=device,progress=lambda stage,_name=name,**detail:progress(stage,condition=_name,**detail))
                result.update(provenance=provenance,checkpoint=checkpoints[name],encoder_epoch=args.epoch)
                write_json(root/'probes'/name/'results.json',result)
                results[name] = result
                summarize(provenance)
                published = register(root,name,checkpoints[name])
                write_json(root/f'{name}_viewer_registration.json',dict(heads=published))
                progress('condition_complete',condition=name,validation=result['validation']['mae'])
                del caches[name]
        for name in results:
            published = register(root,name,checkpoints[name])
            write_json(root/f'{name}_viewer_registration.json',dict(heads=published))
        summarize(provenance)
        progress('complete',summary=str(root/'summary.json'),report=str(root/'REPORT.md'))
    except Exception as error:
        progress('failed',error=str(error))
        raise


if __name__ == '__main__':
    main()
