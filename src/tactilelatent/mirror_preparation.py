"""Audit/adapt tactile teachers, then cache both physical mirror variants.

No RGB is duplicated, no visual features are cached, and no joint run starts.
Incomplete geometry or failed teacher quality is published as an explicit block.
"""
from collections import OrderedDict, defaultdict
import copy
import fcntl
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from .aligned_frames import file_hash, json_hash, targets, write_json
from .bimanual_mirror import SOURCES, load_policy, require_spec, transform_targets, verify_evidence
from .pretrain_egotouch_autoencoder import reconstruction_loss
from .pressure_teacher import metric_sums, summarize_metrics
from .sensor_autoencoder import make_autoencoder
from .train_alignment import atomic_checkpoint

ROOT = Path('processing/bimanual_mirror_v1')
MANIFEST = 'processing/storage_v1/manifest.json'
POLICY = 'configs/bimanual_mirror_policy_v1.json'


def progress(root, **fields):
    path = Path(root) / 'status.json'
    state = json.loads(path.read_text()) if path.exists() else {}
    state.update(fields, updated_at=time.time())
    write_json(path, state)
    with (Path(root) / 'progress.jsonl').open('a') as stream:
        stream.write(json.dumps(state, allow_nan=False) + '\n')
    print(json.dumps(state, allow_nan=False), flush=True)
    return state


def source_inputs(source, manifest=MANIFEST):
    if source not in SOURCES:
        raise ValueError('Only the five bilateral sources may enter mirror preparation')
    if source == 'baihu':
        from .data_expansion import ROOT as expansion
        publication = json.loads((expansion / 'teachers.json').read_text())
        rows = json.loads((expansion / 'manifest.json').read_text())['episodes']
        info = publication['sources'][source]
    else:
        rows = json.loads(Path(manifest).read_text())['episodes']
        path = ('processing/storage_v1/marker_teachers.json' if source == 'ftp_unit_bimanual'
                else 'processing/storage_v1/teachers.json')
        registry = json.loads(Path(path).read_text())['sources']
        info = registry.get(source)
    rows = sorted([r for r in rows if r['source'] == source], key=lambda r: (r['split'], r['episode_id']))
    if not rows or not {'train', 'val'} <= {r['split'] for r in rows}:
        raise ValueError(f'{source}: native training/validation episodes are required')
    if len({r['episode_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate episode identities across mirror splits')
    if info is not None and file_hash(info['checkpoint']) != info['sha256']:
        raise ValueError('Original frozen teacher changed')
    return rows, info


def load_model(info, device):
    if file_hash(info['checkpoint']) != info['sha256']:
        raise ValueError('Pinned mirror teacher changed')
    state = torch.load(info['checkpoint'], map_location='cpu', weights_only=False)
    if info.get('representation') == 'marker_displacement':
        if state.get('model_type') == 'expansion_marker_autoencoder_v1':
            from .expansion_models import load_teacher
            model = load_teacher(info['checkpoint'], device)[0]
        else:
            from .marker_autoencoder import load_marker_autoencoder
            model = load_marker_autoencoder(info['checkpoint'], device)[0]
    else:
        model = make_autoencoder(state['model_config']).to(device)
        model.load_state_dict(state['model'], strict=True)
    return model.eval().requires_grad_(False), state


class NativeTargets:
    def __init__(self, marker_root='processing/marker_displacement_storage_v1/targets'):
        from .marker_data import MarkerCache
        self.markers = MarkerCache(marker_root, max_loaded_episodes=1)
        self.native = OrderedDict()

    def read(self, row, indices):
        source = row['source']
        if source == 'baihu':
            from .data_expansion import marker_batch
            return marker_batch(row, indices)
        if source == 'ftp_unit_bimanual':
            return self.markers.read(row, indices)
        key = (source, row['episode_id'])
        if key not in self.native:
            self.native[key] = targets(row, np.arange(row['frames']))
            while len(self.native) > 2:
                self.native.popitem(last=False)
        self.native.move_to_end(key)
        value = torch.from_numpy(self.native[key][np.asarray(indices)].copy()).float()
        valid = torch.isfinite(value)
        return dict(target=torch.where(valid, value, 0.), target_mask=valid)


def to_device(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def predict(model, batch):
    if 'marker_target' in batch:
        y = batch['marker_target']
        n = len(y)
        reference = batch['marker_reference'][None].expand(n, -1, -1)
        surface = batch['marker_surface'][None].expand(n, -1)
        return model(y, batch['marker_valid'], reference, surface)
    return model(batch['target'], batch['target_mask'])


def encode(model, batch):
    if 'marker_target' in batch:
        y = batch['marker_target']; n = len(y)
        valid = batch['marker_valid'] & torch.isfinite(y).all(-1)
        code = model.encode(y, valid, batch['marker_reference'][None].expand(n, -1, -1),
                            batch['marker_surface'][None].expand(n, -1))
        return code, valid.any(-1)
    return model.encode(batch['target'], batch['target_mask']), batch['target_mask'].flatten(1).any(-1)


def sampled_batches(rows, split, limit, epoch=0, batch_size=128, seed=45):
    selected = [r for r in rows if r['split'] == split]
    rng = np.random.default_rng(seed + epoch if split == 'train' else seed + 100000)
    if split == 'train':
        rng.shuffle(selected)
    quota = max(1, math.ceil(limit / len(selected))); seen = 0
    for row in selected:
        n = min(quota, row['frames'], limit - seen)
        if n <= 0:
            break
        indices = (np.sort(rng.choice(row['frames'], n, replace=False)) if split == 'train'
                   else np.unique(np.linspace(0, row['frames'] - 1, n, dtype=np.int64)))
        for start in range(0, len(indices), batch_size):
            ix = indices[start:start + batch_size]
            yield row, ix
            seen += len(ix)


@torch.no_grad()
def evaluate(model, rows, spec, reader, device, limit=10000):
    model.eval(); totals = {name: defaultdict(float) for name in ('original', 'mirrored')}
    for row, indices in sampled_batches(rows, 'val', limit):
        batch = reader.read(row, indices)
        for name, value in [('original', batch), ('mirrored', transform_targets(batch, spec))]:
            value = to_device(value, device); prediction = predict(model, value)
            target = value.get('target', value.get('marker_target'))
            valid = value.get('target_mask')
            if valid is None:
                valid = value['marker_valid'][..., None].expand_as(target)
            sums = metric_sums(prediction, target, valid)
            for key, number in sums.items():
                totals[name][key] += number
    result = {}
    for name, total in totals.items():
        if not total['all_count']:
            raise ValueError('No supported native teacher-validation targets')
        result[name] = summarize_metrics(total)
        result[name]['mae'] = result[name]['all_mae']
    return result


def quality(candidate, baseline, pressure=True):
    orig = candidate['original']; mirrored = candidate['mirrored']
    tests = dict(original_preserved=orig['mae'] <= 1.1 * max(baseline['original']['mae'], 1e-12),
                 mirrored_improved=mirrored['mae'] <= .9 * max(baseline['mirrored']['mae'], 1e-12))
    if pressure:
        for name, metrics in candidate.items():
            tests[name + '_beats_zero'] = (metrics['all_mae'] < metrics['all_zero_baseline_mae']
                and metrics['contact_count'] > 0 and metrics['contact_to_zero_ratio'] < .9)
    return dict(accepted=all(tests.values()), tests=tests)


def reconstruction_objective(model, prediction, batch, original_config):
    if 'marker_target' in batch:
        n = len(batch['marker_target'])
        return model.loss(prediction, batch['marker_target'], batch['marker_valid'],
                          batch['marker_surface'][None].expand(n, -1))
    return reconstruction_loss(prediction, batch['target'], batch['target_mask'],
        original_config.get('contact_weight', 3.), original_config.get('loss_threshold', .1),
        original_config.get('mae_weight', .5))


def prepare_teacher(source, rows, info, policy, root, device, train_frames=100000, val_frames=10000):
    spec = require_spec(policy, source); folder = Path(root) / 'teachers' / source
    folder.mkdir(parents=True, exist_ok=True)
    signature = dict(source=source, original_teacher_sha256=info['sha256'],
        policy_sha256=json_hash(policy), rows_sha256=json_hash(rows),
        implementation={name: file_hash(Path(__file__).with_name(name)) for name in
                        ('bimanual_mirror.py', 'mirror_preparation.py')},
        seed=45, epochs=10, train_frames=train_frames, val_frames=val_frames,
        learning_rate=1e-4, min_learning_rate=1e-6, weight_decay=1e-4)
    receipt = folder / 'quality.json'
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if saved['signature'] != signature:
            raise ValueError('Mirror teacher provenance changed; choose a fresh preparation root')
        if saved.get('accepted') and file_hash(saved['teacher']['checkpoint']) != saved['teacher']['sha256']:
            raise ValueError('Accepted mirror teacher changed')
        return saved
    model, state = load_model(info, device); reader = NativeTargets()
    baseline_path = folder / 'baseline.json'
    if baseline_path.exists():
        document = json.loads(baseline_path.read_text())
        if document['signature'] != signature:
            raise ValueError('Mirror teacher audit provenance changed')
        baseline = document['metrics']
    else:
        progress(root, stage='auditing_teacher', source=source)
        baseline = evaluate(model, rows, spec, reader, device, val_frames)
        write_json(baseline_path, dict(signature=signature, metrics=baseline))
    pressure = spec['kind'] != 'markers'
    needs_adaptation = baseline['mirrored']['mae'] > 1.1 * max(baseline['original']['mae'], 1e-12)
    if not needs_adaptation:
        accepted = True
        if pressure:
            accepted = all(x['contact_count'] > 0 and x['contact_to_zero_ratio'] < .9
                           and x['all_mae'] < x['all_zero_baseline_mae'] for x in baseline.values())
        result = dict(signature=signature, adapted=False, accepted=accepted, baseline=baseline,
                      validation=baseline, teacher=info if accepted else None)
        write_json(receipt, result)
        return result
    model.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
    start = 0; best = float('inf'); best_validation = None
    if (folder / 'last.pt').exists():
        last = torch.load(folder / 'last.pt', map_location='cpu', weights_only=False)
        if last['mirror_signature'] != signature:
            raise ValueError('Mirror teacher resume provenance differs')
        model.load_state_dict(last['model']); optimizer.load_state_dict(last['optimizer'])
        scheduler.load_state_dict(last['scheduler']); start = last['epoch'] + 1
        best = last['best_accepted_score']; best_validation = last['best_accepted_validation']
        torch.set_rng_state(last['torch_rng'])
        if device.type == 'cuda': torch.cuda.set_rng_state_all(last['cuda_rng'])
    else:
        atomic_checkpoint(folder / 'epoch_000.pt', {**state, 'mirror_signature': signature})
    for epoch in range(start, 10):
        began = time.monotonic(); model.train(); total = 0.; count = 0
        rng = np.random.default_rng(45 + epoch)
        for row, indices in sampled_batches(rows, 'train', train_frames, epoch):
            # Each native frame is used once. Choose half of each batch without
            # duplicating examples or changing source/sample weights.
            flags = np.zeros(len(indices), bool)
            n_mirrored = len(indices)//2 + (int(rng.random()<.5) if len(indices)%2 else 0)
            flags[rng.permutation(len(indices))[:n_mirrored]] = True
            # Static marker geometry differs between variants, so split groups
            # rather than pretending that a reflected point has old geometry.
            optimizer.zero_grad(set_to_none=True); supported = 0; losses = []
            for mirrored in (False, True):
                chosen = np.nonzero(flags == mirrored)[0]
                if not len(chosen): continue
                value = reader.read(row, indices[chosen])
                if mirrored: value = transform_targets(value, spec)
                value = to_device(value, device)
                if not (value['marker_valid'].any() if 'marker_target' in value else value['target_mask'].any()): continue
                prediction = predict(model, value)
                loss = reconstruction_objective(model, prediction, value, state.get('config', {}))
                losses.append(loss * len(chosen)); supported += len(chosen)
            if not supported: continue
            loss = sum(losses) / supported
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite mirrored teacher loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step(); total += loss.item() * supported; count += supported
        scheduler.step(); validation = evaluate(model, rows, spec, reader, device, val_frames)
        decision = quality(validation, baseline, pressure)
        score = .5 * sum(validation[name]['mae'] / max(baseline[name]['mae'], 1e-12) for name in validation)
        saved = {**state, 'model': model.state_dict(), 'encoder': model.encoder.state_dict() if hasattr(model,'encoder') else None,
                 'decoder': model.decoder.state_dict(), 'epoch': epoch, 'validation': validation,
                 'mirror_signature': signature, 'mirror_adapted': True}
        if decision['accepted'] and score < best:
            best = score; best_validation = validation
            atomic_checkpoint(folder / 'best.pt', saved)
        atomic_checkpoint(folder / 'last.pt', {**saved, 'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(), 'torch_rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else [],
            'best_accepted_score': best, 'best_accepted_validation': best_validation})
        row = dict(epoch=epoch+1, train_loss=total/max(count, 1), train_frames=count,
                   validation=validation, quality=decision, seconds=time.monotonic()-began)
        with (folder / 'metrics.jsonl').open('a') as stream: stream.write(json.dumps(row)+'\n')
        progress(root, stage='adapting_teacher', source=source, epoch=epoch+1, total_epochs=10,
                 epoch_seconds=row['seconds'], validation=validation, accepted_checkpoint=best_validation is not None)
    accepted = best_validation is not None
    selected = None
    if accepted:
        selected = {**info, 'checkpoint': str((folder/'best.pt').resolve()),
                    'sha256': file_hash(folder/'best.pt'), 'mirror_adapted': True}
    result = dict(signature=signature, adapted=True, accepted=accepted, baseline=baseline,
                  validation=best_validation, teacher=selected)
    write_json(receipt, result)
    return result


def target_snapshot(rows, reader):
    paths = {}; hashes = {}
    for row in rows:
        if row['source'] == 'ftp_unit_bimanual':
            from .marker_data import episode_key
            path = reader.markers.root / row['source'] / row['split'] / episode_key(row) / 'tracks.npz'
        else:
            path = Path(row.get('marker_path', row.get('target_path', '')))
        if str(path) not in hashes: hashes[str(path)] = file_hash(path)
        paths[row['episode_id']] = hashes[str(path)]
    return json_hash(paths)


def fit_pair_statistics(totals, count):
    if not count: raise ValueError('No valid training frames for paired code statistics')
    mean = totals[0] / count
    std = np.sqrt(np.maximum(totals[1] / count - mean**2, 0.)).clip(min=1e-6)
    return dict(mean=mean.tolist(), std=std.tolist(), train_code_observations=count,
                variant_weights=[.5, .5], fit_split='train')


def cache_codes(source, rows, teacher, policy, root, device, chunk_frames=128):
    spec = require_spec(policy, source); folder = Path(root) / 'codes' / source
    folder.mkdir(parents=True, exist_ok=True); reader = NativeTargets()
    signature = dict(source=source, teacher_sha256=teacher['sha256'], policy_sha256=json_hash(policy),
        rows_sha256=json_hash(rows), target_snapshot_sha256=target_snapshot(rows, reader),
        implementation={name:file_hash(Path(__file__).with_name(name)) for name in
                        ('mirror_preparation.py','bimanual_mirror.py')},
        dtype='float32', variants=['original','mirrored'], frame_order='chronological')
    index_path = folder / 'index.json'
    if index_path.exists():
        saved = json.loads(index_path.read_text())
        if saved['signature'] != signature: raise ValueError('Mirror code provenance changed')
        for row in saved['episodes']:
            if file_hash(row['path']) != row['sha256']: raise ValueError('Corrupt mirrored code artifact')
        verification=folder/'verification.json'
        if not verification.exists() or json.loads(verification.read_text()).get('index_sha256')!=file_hash(index_path):
            verify_codes(rows,saved,policy,device)
        return saved
    marker = folder/'signature.json'
    if marker.exists() and json.loads(marker.read_text()) != signature:
        raise ValueError('Partial mirror cache signature differs')
    write_json(marker, signature)
    from .data_expansion import reserve, atomic_npz
    dimension = teacher['code_dim']
    reserve(sum(row['frames'] for row in rows) * (8 * dimension + 2))
    model, _ = load_model(teacher, device)
    records = []; totals = np.zeros((2, dimension), np.float64); count = 0; began = time.monotonic()
    with torch.inference_mode():
        for number, row in enumerate(rows):
            key = json_hash([source, row['split'], row['episode_id']])[:24]
            path = folder / (key + '.npz'); receipt = folder / (key + '.json')
            expected = dict(signature_sha256=json_hash(signature), row_sha256=json_hash(row))
            if receipt.exists():
                document = json.loads(receipt.read_text())
                if document['provenance'] != expected or file_hash(path) != document['sha256']:
                    raise ValueError('Partial mirrored episode changed')
                with np.load(path, allow_pickle=False) as f:
                    codes, valid = f['codes'].copy(), f['valid'].copy()
            else:
                codes = np.zeros((2, row['frames'], dimension), np.float32)
                valid = np.zeros((2, row['frames']), bool)
                for start in range(0, row['frames'], chunk_frames):
                    indices = np.arange(start, min(start+chunk_frames, row['frames']))
                    original = reader.read(row, indices)
                    for variant, value in enumerate([original, transform_targets(original, spec)]):
                        raw, supported = encode(model, to_device(value, device))
                        raw = raw.float().cpu().numpy()
                        if raw.shape != (len(indices),dimension) or not np.isfinite(raw).all():
                            raise ValueError('Invalid fresh mirrored teacher code')
                        codes[variant, indices] = raw; valid[variant, indices] = supported.cpu().numpy()
                if not np.array_equal(valid[0], valid[1]): raise ValueError('Mirroring changed target support')
                atomic_npz(path, codes=codes, valid=valid, frame_indices=np.arange(row['frames'],dtype=np.int64))
                write_json(receipt, dict(provenance=expected, sha256=file_hash(path)))
            if codes.shape != (2,row['frames'],dimension) or valid.shape != (2,row['frames']):
                raise ValueError('Unexpected paired cache layout')
            if row['split'] == 'train':
                values = codes[valid].astype(np.float64)
                totals[0] += values.sum(0); totals[1] += np.square(values).sum(0); count += len(values)
            records.append(dict(episode_id=row['episode_id'],split=row['split'],frames=row['frames'],
                row_sha256=json_hash(row),path=str(path.resolve()),sha256=file_hash(path)))
            if number%50 == 0 or number+1 == len(rows):
                progress(root,stage='caching_codes',source=source,episodes_done=number+1,episodes=len(rows),
                         cache_seconds=time.monotonic()-began)
    result = dict(complete=True,signature=signature,teacher=teacher,episodes=records,
                  statistics=fit_pair_statistics(totals,count))
    write_json(index_path,result)
    verify_codes(rows, result, policy, device)
    return result


def verify_codes(rows, index, policy, device):
    source = index['signature']['source']; spec = require_spec(policy,source)
    lookup = {r['episode_id']:r for r in index['episodes']}; reader=NativeTargets()
    model,_ = load_model(index['teacher'],device); checks=[]
    with torch.inference_mode():
        for split in ('train','val'):
            row = next(r for r in rows if r['split']==split)
            indices = np.unique([0,row['frames']//2,row['frames']-1])
            original = reader.read(row,indices)
            with np.load(lookup[row['episode_id']]['path'],allow_pickle=False) as f:
                np.testing.assert_array_equal(f['frame_indices'],np.arange(row['frames']))
                for variant,value in enumerate([original,transform_targets(original,spec)]):
                    fresh,valid = encode(model,to_device(value,device))
                    np.testing.assert_allclose(f['codes'][variant,indices],fresh.cpu().numpy(),rtol=2e-5,atol=2e-5)
                    np.testing.assert_array_equal(f['valid'][variant,indices],valid.cpu().numpy())
            checks.append(dict(episode_id=row['episode_id'],split=split,frame_indices=indices.tolist()))
    folder=Path(index['episodes'][0]['path']).parent
    write_json(folder/'verification.json',dict(passed=True,index_sha256=file_hash(folder/'index.json'),episodes=checks))


def run(root=ROOT,manifest=MANIFEST,policy_path=POLICY,device='cuda',train_frames=100000,val_frames=10000):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    lock=(root/'prepare.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(4);torch.manual_seed(45);np.random.seed(45);device=torch.device(device)
    policy=load_policy(policy_path);evidence=verify_evidence(policy)
    write_json(root/'policy_audit.json',dict(policy_sha256=json_hash(policy),sources=evidence))
    results={};started=time.monotonic()
    for source in SOURCES:
        if not evidence[source]['verified']:
            results[source]=dict(ready=False,stage='blocked_geometry',reason=evidence[source]['reason'])
            continue
        try:
            rows,info=source_inputs(source,manifest)
            if info is None:
                results[source]=dict(ready=True,numeric=True,policy_sha256=json_hash(policy))
            else:
                decision=prepare_teacher(source,rows,info,policy,root,device,train_frames,val_frames)
                if not decision['accepted']:
                    results[source]=dict(ready=False,stage='blocked_teacher_quality',report=str(root/'teachers'/source/'quality.json'))
                    continue
                index=cache_codes(source,rows,decision['teacher'],policy,root,device)
                path=root/'codes'/source/'index.json'
                results[source]=dict(ready=True,numeric=False,teacher=decision['teacher'],
                    code_index=str(path.resolve()),code_index_sha256=file_hash(path),
                    statistics=index['statistics'],teacher_quality=str(root/'teachers'/source/'quality.json'))
        except Exception as error:
            results[source]=dict(ready=False,stage='failed',reason=f'{type(error).__name__}: {error}')
            progress(root,stage='source_failed',source=source,error=results[source]['reason'])
        write_json(root/'ready.json',dict(complete=all(r.get('ready') for r in results.values()) and len(results)==len(SOURCES),
            policy_sha256=json_hash(policy),policy_path=str(Path(policy_path).resolve()),sources=results))
    ready=dict(complete=all(r.get('ready') for r in results.values()),policy_sha256=json_hash(policy),
               policy_path=str(Path(policy_path).resolve()),sources=results)
    write_json(root/'ready.json',ready)
    progress(root,stage='complete' if ready['complete'] else 'partial_ready_with_explicit_blocks',
             sources=results,seconds=time.monotonic()-started)
    return ready
