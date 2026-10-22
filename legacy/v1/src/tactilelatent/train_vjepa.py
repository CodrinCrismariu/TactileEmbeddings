"""Matched deterministic/Gaussian V-JEPA runs sharing a frozen video forward."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .aligned_frames import file_hash, manifest_signature, write_json
from .alignment_data import AlignedClipDataset
from .alignment_model import FrozenSensorDecoders
from .joint_all_config import ALL_TARGETS
from .joint_data import SourceBatchSampler
from .sam import SAMStep, rng_snapshot, restore_rng
from .sensor_code_cache import CodeCache, cache_signature, verify_teachers
from .train import kl_weight, make_scheduler
from .train_alignment import alignment_loss, atomic_checkpoint, sam_replay, model_weight_hash
from .viewer_checkpoints import save_viewer_epoch
from .vjepa_model import (MODEL_TYPE, BACKBONE_ID, REVISION, VJEPABackbone,
                          matched_models, video_features)
from .vjepa_cached_training import bind_feature_cache, encoded_inputs, check_resume_contract


def prepare_data(cfg):
    if cfg['heldout_source'] != 'ftp_motiontrans' or cfg['latent_dim'] != 32:
        raise ValueError('This study requires 32D and MotionTrans held out')
    signature = manifest_signature(cfg['manifest'])
    audit = json.loads(Path(cfg['audit']).read_text())
    if audit.get('signature') != signature or not audit.get('passed') or not audit.get('VLA_complete_pixels'):
        raise ValueError('Native/frame-pairing audit differs or failed')
    registry = json.loads(Path(cfg['teachers']).read_text())
    verify_teachers(registry)
    rows = json.loads(Path(cfg['manifest']).read_text())['episodes']
    sources = sorted({r['source'] for r in rows} - {'ftp_motiontrans'})
    if len(sources) != 11:
        raise ValueError('Expected all eleven training sources')
    statistics, dimensions, signatures = {}, {}, {}
    for source in sources:
        if source in registry['sources']:
            expected = cache_signature(cfg['manifest'], registry, source, signature)
            cache = CodeCache(cfg['code_cache'], source, expected)
            statistics[source] = cache.index['statistics']
            dimensions[source] = cache.index['code_dim']
            signatures[source] = expected
    numeric = {s: ALL_TARGETS[s][1] for s in sources if s not in dimensions}
    if set(numeric) != {'ftp_rdp', 'ftp_rdp_bimanual'}:
        raise ValueError('Missing frozen teacher')
    kwargs = dict(code_cache=cfg['code_cache'], registry=registry, data_signature=signature,
                  rgb_size=cfg['rgb_size'], frames=8, stride=2)
    train = AlignedClipDataset(cfg['manifest'], 'train', sources, include_targets=False, **kwargs)
    val = AlignedClipDataset(cfg['manifest'], 'val', sources, include_targets=True, **kwargs)
    for dataset in (train, val):
        if set(dataset.indices_by_source) != set(sources):
            raise ValueError('Missing source split or held-out leakage')
    model_config = dict(sensor_code_dims=dimensions, numeric_shapes=numeric, statistics=statistics,
                        input_dim=768, latent_dim=cfg['latent_dim'], adapter_hidden_dims=cfg['adapter_hidden_dims'])
    provenance = dict(data=signature, teachers=registry, code_caches=signatures)
    return train, val, registry, model_config, provenance


def to_device(original, device):
    return {key: value.to(device) if torch.is_tensor(value) and key not in ('rgb', 'video_features') else value
            for key, value in original.items()}


def valid_frames(batch):
    return batch['code_valid'].bool() if 'code_valid' in batch else batch['target_mask'].flatten(2).any(-1)


class LatentStats:
    def __init__(self, dim):
        self.sum = torch.zeros(dim, dtype=torch.float64)
        self.outer = torch.zeros(dim, dim, dtype=torch.float64)
        self.count, self.variance = 0, 0.

    def add(self, prediction, valid):
        mu = prediction['mu'].detach()[valid].double().cpu()
        self.sum += mu.sum(0); self.outer += mu.T @ mu; self.count += len(mu)
        self.variance += prediction['logvar'].detach()[valid].float().exp().sum().item()

    def result(self):
        mean = self.sum / max(self.count, 1)
        cov = self.outer / max(self.count, 1) - mean[:, None] * mean[None, :]
        spectrum = torch.linalg.eigvalsh(cov).clamp_min(0)
        p = spectrum / spectrum.sum().clamp_min(1e-12)
        rank = (-(p * p.clamp_min(1e-12).log()).sum()).exp().item() if spectrum.sum() > 1e-12 else 0.
        return dict(effective_rank_mu=rank, mean_coordinate_variance_mu=cov.diag().mean().item(),
                    mean_predicted_variance=self.variance / max(self.count * len(mean), 1), valid_frames=self.count)


@torch.no_grad()
def validate_pair(models, backbone, loader, registry, device, chunk_size=2):
    for model in models.values():
        model.eval()
    decoders = FrozenSensorDecoders(registry, device)
    totals = {name: defaultdict(lambda: defaultdict(float)) for name in models}
    for original in loader:
        batch = to_device(original, device)
        source = original['source'][0]
        with torch.autocast(device.type, torch.bfloat16, enabled=device.type == 'cuda'):
            encoded = encoded_inputs(backbone, original, device, chunk_size)
            predictions = {name: model(encoded, batch['view_mask'], source, sample_latent=False)
                           for name, model in models.items()}
        # Decode in small frame chunks: native optical targets stay full size.
        target = batch['target'].float().flatten(0, 1)
        valid = batch['target_mask'].bool().flatten(0, 1)
        if 'code_valid' in batch:
            valid = valid & batch['code_valid'].flatten()[:, None, None, None]
        for name, prediction in predictions.items():
            _, task, _ = alignment_loss(prediction, batch)
            row = totals[name][source]
            count = valid_frames(batch).sum().item() if source in models[name].adapters else valid.sum().item()
            row['task_sum'] += task.item() * count; row['task_count'] += count
        for lo in range(0, len(target), 4):
            y, mask = target[lo:lo+4], valid[lo:lo+4]
            if source in next(iter(models.values())).adapters:
                first = next(iter(models.values()))
                true = first.raw_code(source, batch['code_target']).flatten(0, 1)[lo:lo+4]
                oracle = decoders.decode(source, true).float()
                oracle_abs = ((oracle - y).abs() * mask).sum().item()
            else:
                oracle_abs = 0.
            for name, prediction in predictions.items():
                if source in models[name].adapters:
                    code = models[name].raw_code(source, prediction['sensor_code']).flatten(0, 1)[lo:lo+4]
                    pred = decoders.decode(source, code).float()
                else:
                    pred = prediction['pred'].float().flatten(0, 1)[lo:lo+4]
                error = pred - y
                row = totals[name][source]
                row['absolute'] += (error.abs() * mask).sum().item()
                row['squared'] += (error.square() * mask).sum().item()
                row['cells'] += mask.sum().item()
                row['oracle_absolute'] += oracle_abs
    result = {}
    for name, sources in totals.items():
        result[name] = {}
        for source, row in sources.items():
            cells = max(row['cells'], 1)
            item = dict(supervision_loss=row['task_sum'] / max(row['task_count'], 1),
                        normalized_mae=row['absolute'] / cells, valid_cells=int(row['cells']))
            if source in models[name].adapters:
                item.update(code_mse=item['supervision_loss'], frozen_ae_mae=row['oracle_absolute']/cells)
            if ALL_TARGETS[source][0] == 'optical_rgb':
                item['psnr'] = -10 * np.log10(max(row['squared']/cells, 1e-15))
            result[name][source] = item
    return result


def make_training_state(models, cfg, batches):
    optimizers, schedulers, sams = {}, {}, {}
    for name, model in models.items():
        optimizers[name] = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
        schedulers[name] = make_scheduler(optimizers[name], cfg, batches)
        sams[name] = SAMStep(optimizers[name], cfg['sam_rho'])
    return optimizers, schedulers, sams


def train_microbatch(models, optimizers, sams, replay, encoded, mask, batch, source, beta, group_size, device):
    values = {}
    for name, model in models.items():
        weight = beta if name == 'stochastic' else 0.
        rng = rng_snapshot()
        with torch.autocast(device.type, torch.bfloat16, enabled=device.type == 'cuda'):
            prediction = model(encoded, mask, source)
            loss, task, kl = alignment_loss(prediction, batch, weight)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'{name}: nonfinite loss')
        (loss / group_size).backward()
        replay[name].append(dict(features=encoded.detach(), mask=mask, source=source,
            batch={key: value.detach() for key, value in batch.items()
                   if key in ('code_target', 'code_valid', 'target', 'target_mask')}, rng=rng, beta=weight))
        values[name] = (prediction, [loss.item(), task.item(), kl.item()])
    return values


def commit_pair(root, completed, models, optimizers, schedulers, sams, model_config, cfg, provenance,
                registry, best, validation, elapsed, loader_generator):
    # Immutable epoch files are written first; pair_state is the atomic commit.
    for name, model in models.items():
        directory = root / name; directory.mkdir(parents=True, exist_ok=True)
        mode = 'gaussian' if name == 'stochastic' else 'deterministic'
        arm_cfg = {**cfg, 'output': str(directory), 'model_type': MODEL_TYPE, 'dataset_family': 'joint_all',
                   'latent_mode': mode, 'latent_objective': 'vae' if mode == 'gaussian' else 'deterministic',
                   'stochastic': mode == 'gaussian', 'beta_max': cfg['beta_max'] if mode == 'gaussian' else 0.,
                   'width': 768}
        state = dict(model_type=MODEL_TYPE, model_config={**model_config, 'latent_mode': mode},
                     model=model.state_dict(), config=arm_cfg, provenance=provenance,
                     optimizer=optimizers[name].state_dict(), scheduler=schedulers[name].state_dict(),
                     sam=sams[name].state_dict(), epoch=completed-1, best_score=best[name],
                     backbone=BACKBONE_ID, backbone_revision=REVISION, teachers=registry,
                     validation=validation.get(name, {}))
        write_json(directory/'config.json', arm_cfg)
        epoch_path = directory/f'epoch_{completed:03d}.pt'
        atomic_checkpoint(epoch_path, state)
        atomic_checkpoint(directory/'last.pt', state)
        save_viewer_epoch(directory/'last.pt', state)
        if completed and np.mean([x['supervision_loss'] for x in validation[name].values()]) <= best[name]:
            atomic_checkpoint(directory/'best.pt', state)
        if completed == 0:
            write_json(directory/'initialization.json', dict(seed=cfg['seed'], model_sha256=model_weight_hash(model),
                parameters=sum(p.numel() for p in model.parameters()), epoch0_checkpoint=str(epoch_path)))
    pair_state = dict(completed_epochs=completed, config=cfg, provenance=provenance,
        rng=rng_snapshot(), python_rng=random.getstate(), numpy_rng=np.random.get_state(),
        loader_rng=loader_generator.get_state(), elapsed_seconds=elapsed, best=best)
    (root/'pair_epochs').mkdir(exist_ok=True)
    atomic_checkpoint(root/'pair_epochs'/f'epoch_{completed:03d}.pt', pair_state)
    atomic_checkpoint(root/'pair_state.pt', pair_state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/joint_vjepa21_pair_32d_v1.json')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after-epoch', type=int)
    parser.add_argument('--adopt-feature-cache', action='store_true', help='Explicitly migrate the stopped live-input trainer to cached clips')
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    root = Path(cfg['output']); root.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (root/'training.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.adopt_feature_cache and (not args.resume or not cfg.get('feature_cache')):
        raise ValueError('Cache adoption requires --resume and feature_cache configuration')
    if (root/'pair_state.pt').exists() and not args.resume:
        raise FileExistsError('Pair already exists; use --resume')
    torch.set_num_threads(4)
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    device = torch.device('cuda')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for production training')
    write_json(root/'status.json', dict(stage='initializing', pid=os.getpid(), total_epochs=cfg['epochs']))
    train, val, registry, model_config, provenance = prepare_data(cfg)
    if cfg.get('feature_cache'):
        train, val, backbone_info, cache_info = bind_feature_cache(cfg, train, val, provenance)
        provenance['backbone'] = backbone_info
        provenance['feature_cache'] = cache_info
    train_sampler = SourceBatchSampler(train, cfg['batch_size'], seed=cfg['seed'],
        max_windows_per_source=cfg['train_windows_per_source_per_epoch'], sampling_mode='cycle')
    val_sampler = SourceBatchSampler(val, cfg['batch_size'], seed=cfg['seed'], max_windows_per_source=cfg['validation_windows_per_source'])
    generator = torch.Generator().manual_seed(cfg['seed'])
    loader_kwargs = dict(num_workers=cfg['workers'], pin_memory=True, generator=generator)
    train_loader = DataLoader(train, batch_sampler=train_sampler, **loader_kwargs)
    val_loader = DataLoader(val, batch_sampler=val_sampler, **loader_kwargs)
    backbone = None
    if not cfg.get('feature_cache'):
        backbone = VJEPABackbone(cfg['backbone_repo'], cfg['backbone_checkpoint']).to(device).eval()
        provenance['backbone'] = backbone.provenance
    provenance['implementation'] = {name: file_hash(Path(__file__).with_name(name)) for name in
        ('train_vjepa.py', 'vjepa_model.py', 'alignment_data.py', 'train_alignment.py', 'sam.py', 'train.py', 'joint_data.py')}
    if cfg.get('feature_cache'):
        provenance['implementation']['vjepa_cached_training.py'] = file_hash(Path(__file__).with_name('vjepa_cached_training.py'))
    models = {name: model.to(device) for name, model in matched_models(model_config, cfg['seed']).items()}
    optimizers, schedulers, sams = make_training_state(models, cfg, len(train_loader))
    start, elapsed_before = 0, 0.
    best = {name: float('inf') for name in models}
    if args.resume:
        saved = torch.load(root/'pair_state.pt', map_location='cpu', weights_only=False)
        migrated = check_resume_contract(saved, cfg, provenance, args.adopt_feature_cache)
        if migrated:
            write_json(root/'cache_adoption.json', dict(completed_epochs=saved['completed_epochs'],
                parent_pair_sha256=file_hash(root/'pair_state.pt'), previous_provenance=saved['provenance'],
                new_provenance=provenance, change='frozen_live_video_to_verified_cached_video_only'))
        start, best, elapsed_before = saved['completed_epochs'], saved['best'], saved['elapsed_seconds']
        for name, model in models.items():
            state = torch.load(root/name/f'epoch_{start:03d}.pt', map_location=device, weights_only=False)
            expected_mode = 'gaussian' if name == 'stochastic' else 'deterministic'
            if (state['epoch'] + 1 != start or state['provenance'] != saved['provenance']
                    or state['model_config'] != {**model_config, 'latent_mode': expected_mode}):
                raise ValueError('Incomplete paired epoch')
            model.load_state_dict(state['model']); optimizers[name].load_state_dict(state['optimizer'])
            schedulers[name].load_state_dict(state['scheduler']); sams[name].load_state_dict(state['sam'])
            metrics = root/name/'metrics.jsonl'
            if metrics.exists():
                kept = [line for line in metrics.read_text().splitlines() if json.loads(line)['epoch'] <= start]
                metrics.write_text(''.join(line+'\n' for line in kept))
        restore_rng(saved['rng']); random.setstate(saved['python_rng']); np.random.set_state(saved['numpy_rng'])
        generator.set_state(saved['loader_rng'])
    else:
        commit_pair(root, 0, models, optimizers, schedulers, sams, model_config, cfg, provenance, registry,
                    best, {}, 0., generator)
    write_json(root/'config_pair.json', cfg)
    write_json(root/'provenance.json', provenance)
    print(json.dumps(dict(event='indexed', batches_per_epoch=len(train_loader),
        clips_per_source={s: min(len(ids), cfg['train_windows_per_source_per_epoch']) for s, ids in train.indices_by_source.items()},
        start_epoch=start+1, epochs=cfg['epochs'], trainable_parameters={n: sum(p.numel() for p in m.parameters()) for n,m in models.items()},
        input_mode='cached_clips' if cfg.get('feature_cache') else 'live_vjepa',
        frozen_parameters_loaded=0 if backbone is None else sum(p.numel() for p in backbone.parameters()))), flush=True)
    began = time.monotonic() - elapsed_before
    for epoch in range(start, min(cfg['epochs'], args.stop_after_epoch or cfg['epochs'])):
        epoch_start = time.monotonic()
        train_sampler.set_epoch(epoch)
        for name, model in models.items():
            model.train(); optimizers[name].zero_grad(set_to_none=True)
        totals = {n: defaultdict(lambda: np.zeros(3)) for n in models}
        counts = defaultdict(int); replay = {n: [] for n in models}
        stats = {n: LatentStats(cfg['latent_dim']) for n in models}
        torch.cuda.reset_peak_memory_stats()
        for step, original in enumerate(train_loader):
            source = original['source'][0]
            if len(set(original['source'])) != 1 or source == cfg['heldout_source']:
                raise ValueError('Mixed source or held-out training batch')
            batch = to_device(original, device)
            mask = batch['view_mask'].clone()
            # Dedicated deterministic stream: camera masks do not depend on latent draws.
            dropout_generator = torch.Generator(device=device).manual_seed(cfg['seed'] + epoch*1_000_000 + step)
            if mask[:, :, 0].all():
                mask[:, :, 1:] &= torch.rand(len(mask), 1, 2, device=device, generator=dropout_generator) >= cfg['wrist_dropout']
            with torch.no_grad(), torch.autocast(device.type, torch.bfloat16):
                encoded = encoded_inputs(backbone, original, device, cfg['video_chunk_size'])
            group_size = min(cfg['accumulation'], len(train_loader) - (step//cfg['accumulation'])*cfg['accumulation'])
            beta = kl_weight(cfg, epoch + step/len(train_loader))
            values = train_microbatch(models, optimizers, sams, replay, encoded, mask, batch, source, beta, group_size, device)
            for name, (prediction, losses) in values.items():
                totals[name][source] += losses
                stats[name].add(prediction, valid_frames(batch))
            counts[source] += 1
            if (step+1) % cfg['accumulation'] == 0 or step+1 == len(train_loader):
                for name, model in models.items():
                    sam_replay(model, optimizers[name], sams[name], replay[name], device)
                    replay[name].clear(); schedulers[name].step()
            if step % cfg['log_every'] == 0:
                elapsed = time.monotonic() - epoch_start
                status = dict(stage='training', pid=os.getpid(), epoch=epoch+1, completed_epochs=epoch,
                    batch=step+1, batches_per_epoch=len(train_loader), total_epochs=cfg['epochs'],
                    epoch_seconds=elapsed, seconds_per_batch=elapsed/(step+1),
                    projected_training_seconds_remaining=(len(train_loader)*(cfg['epochs']-epoch)-step-1)*elapsed/(step+1),
                    losses={n: v[1] for n,v in values.items()}, beta=beta,
                    learning_rate=schedulers['deterministic'].get_last_lr()[0],
                    peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                    input_mode='cached_clips' if cfg.get('feature_cache') else 'live_vjepa')
                write_json(root/'status.json', status)
                print(json.dumps(status), flush=True)
        write_json(root/'status.json', dict(stage='validation', pid=os.getpid(), epoch=epoch+1,
                   completed_epochs=epoch, total_epochs=cfg['epochs']))
        validation = validate_pair(models, backbone, val_loader, registry, device, cfg['video_chunk_size'])
        for name in models:
            score = float(np.mean([x['supervision_loss'] for x in validation[name].values()]))
            best[name] = min(best[name], score)
            record = dict(epoch=epoch+1, validation=validation[name], selection_score=score,
                          train={s: dict(zip(('loss','supervision','kl'), (v/counts[s]).tolist())) for s,v in totals[name].items()},
                          latent=stats[name].result(), seconds=time.monotonic()-began,
                          epoch_seconds=time.monotonic()-epoch_start, uncertainty='uncalibrated' if name=='stochastic' else 'none')
            with (root/name/'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(record)+'\n')
            print(json.dumps(dict(arm=name, **record)), flush=True)
        commit_pair(root, epoch+1, models, optimizers, schedulers, sams, model_config, cfg, provenance, registry,
                    best, validation, time.monotonic()-began, generator)
        write_json(root/'status.json', dict(stage='epoch_complete', pid=os.getpid(), completed_epochs=epoch+1,
                   total_epochs=cfg['epochs'], seconds=time.monotonic()-began))
    finished = min(cfg['epochs'], args.stop_after_epoch or cfg['epochs'])
    write_json(root/'status.json', dict(stage='complete' if finished==cfg['epochs'] else 'stopped_at_boundary',
               completed_epochs=finished, total_epochs=cfg['epochs'], seconds=time.monotonic()-began))


if __name__ == '__main__':
    main()
