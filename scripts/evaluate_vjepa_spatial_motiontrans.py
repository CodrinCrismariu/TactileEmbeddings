"""Evaluate one frozen spatial-attention checkpoint with the established probe.

Decode full clips live. Keep only 32D/frame latents in RAM during head fitting;
never write a spatial-token or dataset feature cache.
"""
import argparse
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
from tactilelatent.frozen_probe import select_indices, seed_all
from tactilelatent.representation_probe import fit_probe_sweep, paired_difference, check_pair
from tactilelatent.vjepa_model import VJEPABackbone, BACKBONE_ID, REVISION
from tactilelatent.vjepa_spatial import MODEL_TYPE, model_from_state, spatial_video_features
from register_vjepa_transfer_viewer import register


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epoch', type=int, default=15)
    parser.add_argument('--run', type=Path, default=Path('processing/runs/joint_vjepa21_spatial_live_32d_v1'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    lock = (root/'evaluation.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = time.monotonic()

    def progress(stage, **details):
        row = dict(stage=stage, pid=os.getpid(), encoder_epoch=args.epoch,
                   elapsed_seconds=time.monotonic()-started, updated_at=time.time(), **details)
        write_json(root/'status.json', row)
        print(json.dumps(row), flush=True)

    try:
        torch.set_num_threads(4)
        seed_all(45)
        device = torch.device('cuda')
        progress('verifying_checkpoint')
        source = args.run/'viewer_epochs'/f'epoch_{args.epoch:03d}.pt'
        checkpoint = root/'checkpoints'/source.name
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        if not checkpoint.exists():
            temporary = checkpoint.with_suffix('.tmp')
            shutil.copyfile(source, temporary)
            temporary.replace(checkpoint)
        digest = file_hash(checkpoint)
        if digest != file_hash(source):
            raise ValueError('Snapshot differs from requested checkpoint')
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        cfg, mc = state['config'], state['model_config']
        data = manifest_signature(cfg['manifest'])
        if (state['model_type'] != MODEL_TYPE or state['epoch']+1 != args.epoch
                or state['backbone'] != BACKBONE_ID or state['backbone_revision'] != REVISION
                or cfg['heldout_source'] != 'ftp_motiontrans' or cfg['stochastic']
                or cfg['beta_max'] != 0 or mc['latent_mode'] != 'deterministic'
                or mc['latent_dim'] != 32 or state['provenance']['data'] != data
                or 'ftp_motiontrans' in mc['sensor_code_dims']
                or 'ftp_motiontrans' in mc['numeric_shapes']):
            raise ValueError('Wrong model, epoch, data, or held-out policy')
        for name, expected in state['provenance']['implementation'].items():
            if file_hash(Path('src/tactilelatent')/name) != expected:
                raise ValueError(f'Training implementation changed: {name}')
        model = model_from_state(state).to(device).eval().requires_grad_(False)
        backbone = VJEPABackbone(cfg['backbone_repo'], cfg['backbone_checkpoint']).to(device).eval().requires_grad_(False)
        if backbone.provenance['checkpoint_sha256'] != state['provenance']['backbone']['checkpoint_sha256']:
            raise ValueError('Backbone hash mismatch')
        previous = Path('processing/evaluations/motiontrans_vjepa21_best_v1')
        dino = Path('processing/evaluations/motiontrans_representation_screen_v1/summary.json')
        info = dict(path=str(checkpoint), sha256=digest, epoch=args.epoch)
        provenance = dict(checkpoint=info, data=data, backbone=state['provenance']['backbone'],
            head=[32,512,512,512,6], head_seeds=[45,46,47], learning_rates=[.0003,.001,.003],
            extraction='live full spatial tokens; frozen deterministic 32D/frame; RAM latents only',
            protocol='Same canonical clips; fit-only z-score; episode fit/tune seed45; three LRs selected on tune only; three head seeds; max100 epochs/patience10; all-training refit.',
            previous_summary_sha256=file_hash(previous/'summary.json'), dino_summary_sha256=file_hash(dino),
            implementation={p:file_hash(p) for p in [__file__, 'src/tactilelatent/representation_probe.py',
                'src/tactilelatent/frozen_probe.py','src/tactilelatent/vjepa_spatial.py',
                'scripts/register_vjepa_transfer_viewer.py']})
        pinned = root/'provenance.json'
        if pinned.exists() and json.loads(pinned.read_text()) != provenance:
            raise ValueError('Evaluation provenance changed')
        write_json(pinned, provenance)
        folder = root/'probes'/f'epoch_{args.epoch:03d}'
        result_path = folder/'results.json'
        if result_path.exists():
            result = json.loads(result_path.read_text())
            if result['provenance'] != provenance:
                raise ValueError('Completed result provenance differs')
        else:
            caches = {}
            for split in ('train','val'):
                dataset = AlignedClipDataset(cfg['manifest'], split=split, sources=['ftp_motiontrans'],
                    frames=8, stride=2, rgb_size=384, include_targets=True)
                indices = select_indices(dataset)
                loader = DataLoader(Subset(dataset,indices), batch_size=2, num_workers=2,
                                    shuffle=False, pin_memory=True)
                values, targets, valid, episodes = [], [], [], []
                split_started = time.monotonic()
                for step, batch in enumerate(loader):
                    if set(batch['source']) != {'ftp_motiontrans'}:
                        raise ValueError('Unexpected source')
                    with torch.no_grad(), torch.autocast('cuda',torch.bfloat16):
                        tokens = spatial_video_features(backbone,batch,device,chunk_size=2)
                        mu, logvar = model.visual.encode(tokens,batch['view_mask'].to(device))
                    if mu.shape != (len(batch['rgb']),8,32) or not torch.isfinite(mu).all() or not torch.isneginf(logvar).all():
                        raise ValueError('Invalid deterministic per-frame latents')
                    values.append(mu.float().cpu())
                    targets.append(batch['target']); valid.append(batch['target_mask'])
                    episodes.extend(batch['episode_id'])
                    del tokens, mu, logvar
                    if step % 100 == 0 or step+1 == len(loader):
                        elapsed = time.monotonic()-split_started
                        progress('extracting_live_latents',split=split,batches_done=step+1,
                            batches_total=len(loader),split_eta_seconds=elapsed/(step+1)*(len(loader)-step-1))
                cache = dict(values=torch.cat(values),target=torch.cat(targets),valid=torch.cat(valid),
                             episode_id=episodes,indices=indices)
                reference = torch.load(previous/'features'/f'{split}_targets.pt',map_location='cpu',weights_only=False)
                if (indices != reference['indices'] or episodes != reference['episode_id']
                        or not torch.equal(cache['target'],reference['target'])
                        or not torch.equal(cache['valid'],reference['valid'])):
                    raise ValueError('Targets or chronological clip ordering differ from previous evaluation')
                caches[split] = cache
                progress('split_verified',split=split,clips=len(indices),episodes=len(set(episodes)))
            check_pair(caches['train'], caches['val'])
            if any(p.grad is not None for p in list(backbone.parameters())+list(model.parameters())):
                raise ValueError('Frozen encoder acquired gradients')
            del backbone, model, state, reference
            gc.collect(); torch.cuda.empty_cache()
            result = fit_probe_sweep(caches['train'],caches['val'],folder,device=device,progress=progress)
            result.update(provenance=provenance,encoder_epoch=args.epoch)
            write_json(result_path,result)
        old = json.loads((previous/'summary.json').read_text())['results']['deterministic']
        dino_result = json.loads(dino.read_text())['results']['epoch007_latent_32']
        write_json(root/'summary.json',dict(result=result,comparisons=dict(
            vs_pooled_vjepa_epoch28=paired_difference(old,result),
            vs_dino_epoch7=paired_difference(dino_result,result)),provenance=provenance))
        published = register(root,folder.name,info)
        write_json(root/'viewer_registration.json',dict(heads=published))
        progress('complete',validation=result['validation'],summary=str(root/'summary.json'),viewer_heads=published)
    except Exception as error:
        progress('failed',error=str(error))
        raise


if __name__ == '__main__':
    main()
