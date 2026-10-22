"""Matched MotionTrans probes: marker-code epochs 20/30 and image-code epoch 20.

Decode each held-out clip and run frozen V-JEPA once for all three models.
Keep extracted 32D latents in RAM only; reuse the established probe protocol.
"""
import fcntl
import gc
import json
import os
from pathlib import Path
import shutil
import time

import torch
from torch.utils.data import DataLoader, Subset

from tactilelatent.aligned_frames import file_hash, manifest_signature, write_json
from tactilelatent.alignment_data import AlignedClipDataset
from tactilelatent.frozen_probe import seed_all, select_indices
from tactilelatent.marker_code_model import MODEL_TYPE as MARKER_TYPE, model_from_state as marker_model
from tactilelatent.representation_probe import check_pair, fit_probe_sweep, paired_difference
from tactilelatent.vjepa_model import BACKBONE_ID, REVISION, VJEPABackbone
from tactilelatent.vjepa_spatial import MODEL_TYPE as SPATIAL_TYPE, model_from_state as spatial_model, spatial_video_features
from register_vjepa_transfer_viewer import register


ROOT = Path('processing/evaluations/motiontrans_vjepa21_marker_epochs020_030_v1')
CONDITIONS = {
    'markers_epoch020': ('joint_vjepa21_spatial_marker_codes_32d_v1', 20, MARKER_TYPE, marker_model),
    'markers_epoch030': ('joint_vjepa21_spatial_marker_codes_32d_v1', 30, MARKER_TYPE, marker_model),
    'images_epoch020': ('joint_vjepa21_spatial_live_32d_v1', 20, SPATIAL_TYPE, spatial_model),
}


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    lock = (ROOT / 'evaluation.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = time.monotonic()
    results = {}

    def progress(stage, **details):
        row = dict(stage=stage, pid=os.getpid(), elapsed_seconds=time.monotonic()-started,
                   updated_at=time.time(), completed_conditions=list(results), **details)
        write_json(ROOT / 'status.json', row)
        print(json.dumps(row), flush=True)

    def report():
        comparisons = {}
        if {'images_epoch020', 'markers_epoch020'} <= results.keys():
            comparisons['markers20_minus_images20'] = paired_difference(results['images_epoch020'], results['markers_epoch020'])
        if {'markers_epoch020', 'markers_epoch030'} <= results.keys():
            comparisons['markers30_minus_markers20'] = paired_difference(results['markers_epoch020'], results['markers_epoch030'])
        write_json(ROOT / 'summary.json', dict(results=results, comparisons=comparisons, provenance=provenance))
        lines = ['# MotionTrans: marker versus image supervision', '',
                 'Frozen deterministic 32D representations; head `32 → 512 → 512 → 512 → 6`.',
                 'Same clips, fit-only feature standardization, episode-level fit/tune split,',
                 'three learning rates selected on tune episodes and three head-fit seeds.',
                 'MotionTrans is excluded from representation training. MAE is in the existing normalized target units.',
                 '', '| Representation | MAE | Head-seed SD | Episode-bootstrap 95% CI |',
                 '|---|---:|---:|---|']
        for name, result in results.items():
            v = result['validation']; lo, hi = v['episode_bootstrap_mae_ci95']
            lines.append(f"| {name} | {v['mae']:.6f} | {v['mae_head_seed_std']:.6f} | [{lo:.6f}, {hi:.6f}] |")
        for name, value in comparisons.items():
            lines.extend(['', f"{name}: ΔMAE={value['candidate_minus_reference_mae']:.6f}; "
                          f"paired episode 95% CI={value['paired_episode_bootstrap_ci95']}."])
        lines.extend(['', 'Negative differences favor the first model named in each comparison.',
                      'Head seeds measure readout fitting variation; these are single representation-training runs.',
                      '', 'Source-validation comparison at epoch 20 (unchanged tactile targets only):', '',
                      '| Source | Older image-code model MAE | Marker-code model MAE | Change |',
                      '|---|---:|---:|---:|'])
        for source, value in source_comparison.items():
            lines.append(f"| {source} | {value['images_mae']:.6f} | {value['markers_mae']:.6f} | {value['change_percent']:+.1f}% |")
        lines.extend(['', 'Aggregate source losses and the four changed tactile representations are not directly comparable.'])
        (ROOT / 'REPORT.md').write_text('\n'.join(lines)+'\n')

    try:
        torch.set_num_threads(4)
        seed_all(45)
        device = torch.device('cuda')
        progress('verifying_checkpoints')
        states, models, checkpoints = {}, {}, {}
        for name, (run_name, epoch, model_type, load) in CONDITIONS.items():
            source = Path('processing/runs') / run_name / 'viewer_epochs' / f'epoch_{epoch:03d}.pt'
            snapshot = ROOT / 'checkpoints' / f'{name}.pt'
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            if not snapshot.exists():
                temporary = snapshot.with_suffix('.tmp')
                shutil.copyfile(source, temporary); temporary.replace(snapshot)
            digest = file_hash(snapshot)
            if digest != file_hash(source):
                raise ValueError(f'Checkpoint changed: {name}')
            state = torch.load(snapshot, map_location='cpu', weights_only=False)
            cfg, mc = state['config'], state['model_config']
            data = manifest_signature(cfg['manifest'])
            if (state['model_type'] != model_type or state['epoch']+1 != epoch
                    or state['backbone'] != BACKBONE_ID or state['backbone_revision'] != REVISION
                    or cfg['heldout_source'] != 'ftp_motiontrans' or cfg['stochastic']
                    or cfg['beta_max'] != 0 or mc['latent_mode'] != 'deterministic'
                    or mc['latent_dim'] != 32 or state['provenance']['data'] != data
                    or 'ftp_motiontrans' in mc['sensor_code_dims'] or 'ftp_motiontrans' in mc['numeric_shapes']):
                raise ValueError(f'Invalid representation or exclusion policy: {name}')
            for filename, expected in state['provenance']['implementation'].items():
                if file_hash(Path('src/tactilelatent') / filename) != expected:
                    raise ValueError(f'Training implementation changed: {name}/{filename}')
            states[name] = state
            models[name] = load(state).to(device).eval().requires_grad_(False)
            checkpoints[name] = dict(path=str(snapshot), sha256=digest, epoch=epoch)
        first = states['markers_epoch020']
        if any(s['provenance']['data'] != first['provenance']['data']
               or s['provenance']['backbone'] != first['provenance']['backbone'] for s in states.values()):
            raise ValueError('Cannot share backbone or data across these representations')
        provenance = dict(checkpoints=checkpoints, data=first['provenance']['data'],
            backbone=first['provenance']['backbone'], head=[32,512,512,512,6],
            head_seeds=[45,46,47], learning_rates=[.0003,.001,.003],
            protocol='Established fit-only z-score, fixed episode fit/tune seed45, three LRs/tune-only selection, three head seeds, max100 epochs/patience10, all-training refit.',
            extraction='Live original RGB clips, one shared frozen full-spatial V-JEPA pass; RAM-only per-frame 32D codes.',
            implementation={p:file_hash(p) for p in [__file__, 'src/tactilelatent/representation_probe.py',
                'src/tactilelatent/frozen_probe.py', 'src/tactilelatent/vjepa_spatial.py',
                'src/tactilelatent/marker_code_model.py', 'scripts/register_vjepa_transfer_viewer.py']})
        pin = ROOT / 'provenance.json'
        if pin.exists() and json.loads(pin.read_text()) != provenance:
            raise ValueError('Evaluation provenance changed')
        write_json(pin, provenance)
        source_comparison = {}
        old = states['images_epoch020']['validation']
        new = states['markers_epoch020']['validation']
        for source in sorted(set(old) & set(new)):
            if 'normalized_mae' in old[source] and 'normalized_mae' in new[source]:
                a, b = old[source]['normalized_mae'], new[source]['normalized_mae']
                source_comparison[source] = dict(images_mae=a, markers_mae=b, change_percent=100*(b/a-1))
        write_json(ROOT / 'source_validation_epoch020.json', source_comparison)
        pending = []
        for name in CONDITIONS:
            result_path = ROOT / 'probes' / name / 'results.json'
            if result_path.exists():
                result = json.loads(result_path.read_text())
                if result['provenance'] != provenance or result['checkpoint'] != checkpoints[name]:
                    raise ValueError(f'Result provenance changed: {name}')
                results[name] = result
            else:
                pending.append(name)
        report()
        if pending:
            cfg = first['config']
            backbone = VJEPABackbone(cfg['backbone_repo'], cfg['backbone_checkpoint']).to(device).eval().requires_grad_(False)
            if backbone.provenance['checkpoint_sha256'] != first['provenance']['backbone']['checkpoint_sha256']:
                raise ValueError('Backbone hash changed')
            caches = {name:{} for name in pending}
            for split in ('train','val'):
                dataset = AlignedClipDataset(cfg['manifest'], split=split, sources=['ftp_motiontrans'],
                                            frames=8, stride=2, rgb_size=384, include_targets=True)
                indices = select_indices(dataset)
                loader = DataLoader(Subset(dataset,indices), batch_size=2, num_workers=4,
                                    shuffle=False, pin_memory=True)
                values = {name:[] for name in pending}
                targets, valid, episodes = [], [], []
                split_start = time.monotonic()
                for step, batch in enumerate(loader):
                    if set(batch['source']) != {'ftp_motiontrans'}:
                        raise ValueError('Unexpected source')
                    with torch.no_grad(), torch.autocast('cuda',torch.bfloat16):
                        tokens = spatial_video_features(backbone,batch,device,chunk_size=2)
                        for name in pending:
                            mu, logvar = models[name].visual.encode(tokens,batch['view_mask'].to(device))
                            if mu.shape != (len(batch['rgb']),8,32) or not torch.isfinite(mu).all() or not torch.isneginf(logvar).all():
                                raise ValueError(f'Invalid deterministic frame codes: {name}')
                            values[name].append(mu.float().cpu())
                        del tokens, mu, logvar
                    targets.append(batch['target']); valid.append(batch['target_mask'])
                    episodes.extend(batch['episode_id'])
                    if step % 100 == 0 or step+1 == len(loader):
                        elapsed = time.monotonic()-split_start
                        progress('extracting_live_latents', split=split, batches_done=step+1,
                                 batches_total=len(loader), split_eta_seconds=elapsed/(step+1)*(len(loader)-step-1))
                shared = dict(target=torch.cat(targets), valid=torch.cat(valid), episode_id=episodes, indices=indices)
                reference = torch.load(Path('processing/evaluations/motiontrans_vjepa21_best_v1/features') /
                                       f'{split}_targets.pt', map_location='cpu', weights_only=False)
                if (indices != reference['indices'] or episodes != reference['episode_id']
                        or not torch.equal(shared['target'],reference['target'])
                        or not torch.equal(shared['valid'],reference['valid'])):
                    raise ValueError('MotionTrans labels or chronological ordering differ from established evaluation')
                for name in pending:
                    caches[name][split] = dict(values=torch.cat(values[name]), **shared)
                progress('split_verified', split=split, clips=len(indices), episodes=len(set(episodes)))
            if any(p.grad is not None for m in [backbone,*models.values()] for p in m.parameters()):
                raise ValueError('Frozen encoder acquired gradients')
            del backbone, models, states, first, reference, values, batch
            gc.collect(); torch.cuda.empty_cache()
            for name in pending:
                check_pair(caches[name]['train'],caches[name]['val'])
                progress('fitting_motiontrans_heads', condition=name)
                result = fit_probe_sweep(caches[name]['train'],caches[name]['val'], ROOT/'probes'/name,
                    device=device, progress=lambda stage, _name=name, **detail: progress(stage, condition=_name, **detail))
                result.update(provenance=provenance, checkpoint=checkpoints[name], encoder_epoch=checkpoints[name]['epoch'])
                write_json(ROOT/'probes'/name/'results.json',result)
                results[name] = result
                report()
                published = register(ROOT,name,checkpoints[name])
                write_json(ROOT/f'{name}_viewer_registration.json',dict(heads=published))
                progress('condition_complete', condition=name, mae=result['validation']['mae'])
                del caches[name]
        report()
        progress('complete', summary=str(ROOT/'summary.json'), report=str(ROOT/'REPORT.md'))
    except Exception as error:
        progress('failed',error=str(error))
        raise


if __name__ == '__main__':
    main()
