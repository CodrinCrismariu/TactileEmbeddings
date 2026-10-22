"""Matched epoch-7 continuations: control, source gates, KL-off retention, both."""
import argparse
from collections import defaultdict
import copy
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .aligned_frames import file_hash, manifest_signature, write_json
from .alignment_data import AlignedClipDataset
from .alignment_model import MODEL_TYPE, JointSensorAlignment
from .continuation_objectives import SourceStops, objective
from .invariance_validation import validate
from .joint_all_config import ALL_TARGETS
from .joint_data import SourceBatchSampler
from .model import DinoBackbone, MODEL_ID, REVISION
from .relational import HELDOUT
from .sam import SAMStep, restore_rng, rng_snapshot
from .sensor_code_cache import CodeCache, cache_signature, verify_teachers
from .train import MODEL_CONFIG_KEYS, features, kl_weight, make_scheduler
from .train_alignment import atomic_checkpoint, model_weight_hash
from .viewer_checkpoints import save_viewer_epoch


def prepare(cfg):
    """Read-only native data, pinned parent, teacher and configuration checks."""
    if (cfg['model_type'] != MODEL_TYPE or cfg['latent_dim'] != 32 or not cfg['stochastic']
            or cfg['latent_objective'] != 'vae' or cfg['heldout_source'] != HELDOUT
            or cfg.get('train_sources') is not None or cfg.get('fgw') or cfg.get('matched_bank_sampling')
            or cfg.get('private_dim', 0) or cfg.get('static_context') or cfg['parent_epochs'] != 7
            or cfg['epochs'] != 27 or cfg['optimizer_method'] != 'sam_adamw'):
        raise ValueError('Expected original Gaussian 32D model, epoch 7→27, and native source sampling')
    gate, retention = cfg['source_stopping'], cfg['retention']
    if (gate['patience'] < 1 or not math.isfinite(gate['relative_tolerance']) or gate['relative_tolerance'] < 0
            or not math.isfinite(retention['weight']) or retention['weight'] < 0
            or retention['ramp_epochs'] < 1 or not math.isfinite(retention['parent_variance'])
            or retention['parent_variance'] <= 0):
        raise ValueError('Invalid stopping or retention settings')
    if (retention['enabled'] and (retention['weight'] <= 0 or cfg['beta_max'] != 0)
            or not retention['enabled'] and (retention['weight'] != 0 or cfg['beta_max'] != .001)):
        raise ValueError('Retention arms use KL-off; controls retain historical KL')
    signature = manifest_signature(cfg['manifest'])
    audit = json.loads(Path(cfg['audit']).read_text())
    if audit.get('signature') != signature or not audit.get('passed') or not audit.get('VLA_complete_pixels'):
        raise ValueError('Passing native chronological full-pixel audit required')
    registry = json.loads(Path(cfg['teachers']).read_text())
    if HELDOUT in registry['sources']:
        raise ValueError('Held-out teacher cannot enter representation training')
    verify_teachers(registry)
    rows = json.loads(Path(cfg['manifest']).read_text())['episodes']
    sources = sorted({r['source'] for r in rows} - {HELDOUT})
    statistics, dimensions, signatures = {}, {}, {}
    for source in registry['sources']:
        expected = cache_signature(cfg['manifest'], registry, source, signature)
        cache = CodeCache(cfg['code_cache'], source, expected)
        statistics[source], dimensions[source] = cache.index['statistics'], cache.index['code_dim']
        signatures[source] = expected
    numeric = {s: ALL_TARGETS[s][1] for s in sources if s not in dimensions}
    if len(sources) != 11 or set(numeric) != {'ftp_rdp', 'ftp_rdp_bimanual'}:
        raise ValueError('Incomplete eleven-source registry')
    model_config = {k: cfg[k] for k in MODEL_CONFIG_KEYS}
    model_config.update(input_dim=768, sensor_code_dims=dimensions, numeric_shapes=numeric,
                        statistics=statistics, adapter_hidden_dims=cfg['adapter_hidden_dims'])
    provenance = dict(data=signature, teachers=registry, code_caches=signatures,
        parent_sha256=file_hash(cfg['init_checkpoint']), history_sha256=file_hash(cfg['parent_metrics']),
        code={name: file_hash(Path(__file__).with_name(name)) for name in (
            'train_continuation.py', 'continuation_objectives.py', 'invariance_validation.py',
            'alignment_data.py', 'alignment_model.py', 'model.py', 'train.py', 'train_alignment.py',
            'joint_data.py', 'sam.py', 'aligned_frames.py', 'viewer_checkpoints.py')})
    if provenance['parent_sha256'] != cfg['parent_sha256'] or provenance['history_sha256'] != cfg['history_sha256']:
        raise ValueError('Pinned parent or historical source metrics changed')
    parent = torch.load(cfg['init_checkpoint'], map_location='cpu', weights_only=False, mmap=True)
    if (parent['epoch'] + 1 != 7 or parent['model_type'] != MODEL_TYPE or parent['model_config'] != model_config
            or parent['backbone_revision'] != REVISION or parent['config']['seed'] != cfg['seed']
            or parent['config']['heldout_source'] != HELDOUT
            or any(parent['provenance'][k] != provenance[k] for k in ('data', 'teachers', 'code_caches'))
            or Path(cfg['output']).resolve() == Path(parent['config']['output']).resolve()):
        raise ValueError('Parent weights, source contract, or architecture differ')
    # Copy only source histories through epoch 7. Later validation cannot drive gates.
    history = [r for r in map(json.loads, Path(cfg['parent_metrics']).read_text().splitlines()) if r['epoch'] <= 7]
    if [r['epoch'] for r in history] != list(range(1, 8)) or history[-1]['latent']['mean_coordinate_variance_mu'] != retention['parent_variance']:
        raise ValueError('Parent source history or training-only retention scale changed')
    return sources, registry, model_config, provenance, parent, history


def replay_step(model, optimizer, sam, replay, cfg, device):
    after = rng_snapshot()
    norm = sam.perturb()
    try:
        for item in replay:
            restore_rng(item['rng'])
            with torch.autocast(device.type, torch.bfloat16, enabled=device.type == 'cuda'):
                _, loss, _, _, _ = objective(model, item, cfg)
            (loss / item['group_size']).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    finally:
        sam.restore(); restore_rng(after)
    optimizer.step(); optimizer.zero_grad(set_to_none=True); sam.steps += 1
    return norm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--until-epoch', type=int, help='Stop cleanly at a probe boundary without changing the 20-epoch schedule')
    parser.add_argument('--benchmark-updates', type=int, default=0)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    if args.benchmark_updates < 0 or args.benchmark_updates and args.resume:
        raise ValueError('Invalid bounded benchmark')
    if args.benchmark_updates:
        cfg = {**cfg, 'output': cfg['output'] + '_benchmark'}
    end = args.until_epoch or cfg['epochs']
    if not cfg['parent_epochs'] < end <= cfg['epochs']:
        raise ValueError('Probe boundary must lie in continuation budget')
    torch.set_num_threads(4)
    sources, registry, model_config, provenance, parent, history = prepare(cfg)
    if args.preflight:
        print(json.dumps(dict(preflight='passed', sources=sources, start_epoch=8, end_epoch=end,
            gates=SourceStops(sources, cfg['source_stopping'], history).rows, retention=cfg['retention'])), flush=True)
        return
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    device = torch.device(args.device)
    kwargs = dict(code_cache=cfg['code_cache'], registry=registry, data_signature=provenance['data'])
    train_set = AlignedClipDataset(cfg['manifest'], 'train', sources, include_targets=False, **kwargs)
    val_set = AlignedClipDataset(cfg['manifest'], 'val', sources, include_targets=True, **kwargs)
    sampler = SourceBatchSampler(train_set, cfg['batch_size'], seed=cfg['seed'],
        max_windows_per_source=cfg['train_windows_per_source_per_epoch'], sampling_mode='cycle')
    val_sampler = SourceBatchSampler(val_set, cfg['batch_size'], seed=cfg['seed'],
        max_windows_per_source=cfg['validation_windows_per_source'])
    loaders = dict(num_workers=cfg['workers'], pin_memory=device.type == 'cuda')
    train_loader = DataLoader(train_set, batch_sampler=sampler, **loaders)
    val_loader = DataLoader(val_set, batch_sampler=val_sampler, **loaders)
    if any(set(ds.indices_by_source) != set(sources) for ds in (train_set, val_set)):
        raise ValueError('Missing train/validation sources')
    folder = Path(cfg['output']); path = folder / 'last.pt'
    if path.exists() and not args.resume:
        raise FileExistsError('Choose a new output or explicitly resume')
    folder.mkdir(parents=True, exist_ok=True)
    backbone = DinoBackbone().to(device).eval().requires_grad_(False)
    model = JointSensorAlignment(**model_config).to(device)
    model.load_state_dict(parent['model'], strict=True); del parent
    # deepcopy consumes no RNG: dropout/sampling streams are identical across arms.
    teacher = copy.deepcopy(model.visual).eval().requires_grad_(False) if cfg['retention']['enabled'] else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    scheduler = make_scheduler(optimizer, {**cfg, 'epochs': cfg['epochs'] - cfg['parent_epochs']}, len(train_loader))
    sam = SAMStep(optimizer, cfg['sam_rho'])
    gates = SourceStops(sources, cfg['source_stopping'], history)
    start, best, updates = 7, float('inf'), 0
    if args.resume:
        saved = torch.load(path, map_location='cpu', weights_only=False)
        if saved['config'] != cfg or saved['provenance'] != provenance:
            raise ValueError('Exact-resume code/configuration/data contract changed')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler']); sam.load_state_dict(saved['sam'])
        gates.rows = saved['source_stops']
        start, best, updates = saved['epoch'] + 1, saved['best_score'], saved['updates']
        restore_rng(saved['rng']); random.setstate(saved['python_rng']); np.random.set_state(saved['numpy_rng'])
        del saved
        log = folder / 'metrics.jsonl'
        if log.exists():
            rows = [r for r in map(json.loads, log.read_text().splitlines()) if r['epoch'] <= start]
            log.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    write_json(folder / 'config.json', cfg)
    if not args.resume:
        write_json(folder / 'initialization.json', dict(parent_epoch=7, parent_sha256=provenance['parent_sha256'],
            model_sha256=model_weight_hash(model), optimizer='fresh AdamW + SAM; identical across arms',
            source_stops=gates.rows, latent_mode='gaussian', anchor_teacher='frozen epoch-7 mean; eval dropout' if teacher else None))
    began = time.monotonic()
    print(json.dumps(dict(event='indexed', start_epoch=start+1, end_epoch=end, batches_per_epoch=len(train_loader))), flush=True)
    for epoch in range(start, end):
        epoch_began = time.monotonic()
        sampler.set_epoch(epoch); model.train(); optimizer.zero_grad(set_to_none=True)
        replay, totals, counts = [], defaultdict(lambda: np.zeros(4)), defaultdict(int)
        dimension = cfg['latent_dim']
        means_sum, outer_sum = torch.zeros(dimension, dtype=torch.float64), torch.zeros(dimension, dimension, dtype=torch.float64)
        latent_count, predicted_variance = 0, 0.
        active_gates = {s: gates.stopped(s) for s in sources}
        for step, original in enumerate(train_loader):
            source = original['source'][0]
            if source == HELDOUT or any(s != source for s in original['source']):
                raise ValueError('Invalid representation-training batch')
            group_size = min(cfg['accumulation'], len(train_loader)-(step//cfg['accumulation'])*cfg['accumulation'])
            batch = {k:v.to(device) for k,v in original.items() if k in ('code_target','code_valid','target','target_mask')}
            mask = original['view_mask'].to(device).clone()
            if mask[:, :, 0].all():
                mask[:, :, 1:] &= torch.rand(len(mask),1,2,device=device) >= cfg['wrist_dropout']
            with torch.no_grad(), torch.autocast(device.type, torch.bfloat16, enabled=device.type == 'cuda'):
                encoded = features(backbone, original, device)
                anchor = teacher.encode(encoded, mask)[0].detach() if teacher else None
            item = dict(features=encoded.detach(), mask=mask, source=source, batch=batch,
                beta=kl_weight(cfg, epoch+step/len(train_loader)), stop_trunk=active_gates[source],
                anchor=anchor, continuation_epoch=epoch-7, group_size=group_size, rng=rng_snapshot())
            with torch.autocast(device.type, torch.bfloat16, enabled=device.type == 'cuda'):
                predicted, loss, task, kl, retention = objective(model, item, cfg)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite continuation loss')
            (loss/group_size).backward(); replay.append(item)
            totals[source] += [loss.item(), task.item(), kl.item(), retention.item()]; counts[source] += 1
            mu = predicted['mu'].detach().double().flatten(0,1).cpu()
            means_sum += mu.sum(0); outer_sum += mu.T@mu; latent_count += len(mu)
            predicted_variance += predicted['logvar'].detach().float().exp().sum().item()
            if step % cfg['log_every'] == 0:
                print(json.dumps(dict(event='train', epoch=epoch+1, batch=step, source=source, loss=loss.item(),
                    supervision=task.item(), kl=kl.item(), retention=retention.item(), trunk_stopped=active_gates[source])), flush=True)
            if (step+1)%cfg['accumulation'] == 0 or step+1 == len(train_loader):
                replay_step(model, optimizer, sam, replay, cfg, device); replay.clear()
                scheduler.step(); updates += 1
                write_json(folder/'status.json', dict(stage='training', epoch=epoch+1, batch=step+1,
                    batches=len(train_loader), updates=updates, stopped_sources=[s for s in sources if active_gates[s]],
                    elapsed_seconds=time.monotonic()-epoch_began))
                if args.benchmark_updates and updates >= args.benchmark_updates:
                    seconds = time.monotonic()-began
                    write_json(folder/'benchmark.json', dict(updates=updates, seconds=seconds,
                        seconds_per_update=seconds/updates, projected_training_epoch_seconds=seconds/updates*math.ceil(len(train_loader)/cfg['accumulation']),
                        peak_cuda_bytes=torch.cuda.max_memory_allocated() if device.type=='cuda' else None,
                        note='Short benchmark includes worker warmup and excludes native validation; conservative estimate'))
                    return
        write_json(folder/'status.json', dict(stage='source_validation', epoch=epoch+1, updates=updates))
        validation = validate(model, backbone, val_loader, registry, device, {})
        changes = gates.update(validation, epoch+1)
        score = float(np.mean([v['supervision_loss'] for v in validation.values()]))
        improved, best = score < best, min(best,score)
        mean = means_sum/max(latent_count,1)
        covariance = outer_sum/max(latent_count,1)-mean[:,None]*mean[None,:]
        spectrum = torch.linalg.eigvalsh(covariance).clamp_min(0)
        proportion = spectrum/spectrum.sum().clamp_min(1e-12)
        rank = (-(proportion*proportion.clamp_min(1e-12).log()).sum()).exp().item() if spectrum.sum()>0 else 0.
        record = dict(epoch=epoch+1, continuation_epoch=epoch-6, validation=validation, selection_score=score,
            seconds=time.monotonic()-began, epoch_seconds=time.monotonic()-epoch_began,
            train={s:dict(zip(('loss','supervision','kl','retention'),(v/counts[s]).tolist())) for s,v in totals.items()},
            source_stops=copy.deepcopy(gates.rows), newly_stopped=changes,
            latent=dict(effective_rank_mu=rank, mean_coordinate_variance_mu=covariance.diag().mean().item(),
                mean_predicted_variance=predicted_variance/max(dimension*latent_count,1)))
        state = dict(model_type=MODEL_TYPE, model_config=model_config, model=model.state_dict(), config=cfg,
            provenance=provenance, teachers=registry, backbone=MODEL_ID, backbone_revision=REVISION,
            epoch=epoch, best_score=best, validation=validation, optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(), sam=sam.state_dict(), source_stops=copy.deepcopy(gates.rows),
            rng=rng_snapshot(), python_rng=random.getstate(), numpy_rng=np.random.get_state(), updates=updates)
        atomic_checkpoint(path,state); save_viewer_epoch(path,state)
        if improved:
            atomic_checkpoint(folder/'best.pt',state)
        if cfg['checkpoint_every'] and (epoch+1-7)%cfg['checkpoint_every']==0:
            full = folder/f'epoch_{epoch+1:03d}.pt'
            if full.exists():
                full.unlink()
            full.hardlink_to(path)
        with (folder/'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
    write_json(folder/'status.json',dict(stage='complete' if end==cfg['epochs'] else 'probe_boundary',
        completed_epoch=end, updates=updates))


if __name__ == '__main__':
    main()
