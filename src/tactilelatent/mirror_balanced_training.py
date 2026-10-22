"""Longer paired mirror training, with quality-preserving parity selection."""
from collections import defaultdict
from contextlib import contextmanager
import copy
import fcntl
import json
from pathlib import Path
import time

import numpy as np
import torch

from .aligned_frames import file_hash, json_hash, write_json
from .bimanual_mirror import load_policy, require_spec, verify_evidence
from . import mirror_preparation as prep
from .gel_mirror_preparation import load_gel_model, _metrics as marker_metrics
from .pressure_teacher import metric_sums, summarize_metrics
from .train_alignment import atomic_checkpoint

ROOT = Path('processing/bimanual_mirror_v3')


def progress(root, **fields):
    state = dict(updated_at=time.time(), **fields)
    write_json(Path(root)/'status.json', state)
    with (Path(root)/'progress.jsonl').open('a') as stream:
        stream.write(json.dumps(state, allow_nan=False)+'\n')
    print(json.dumps(state, allow_nan=False), flush=True)


class FramePool:
    """Native, episode-balanced targets, shuffled across episodes on the GPU."""
    def __init__(self, rows, split, limit, seed=45):
        reader = prep.NativeTargets(); parts = []; records = []
        self.groups = sorted({r.get('platform', r['source']) for r in rows if r['split']==split})
        for row, indices in prep.sampled_batches(rows, split, limit, seed=seed):
            item = reader.read(row, indices)
            parts.append((item, indices, self.groups.index(row.get('platform', row['source']))))
            records.append(dict(episode=row['episode_id'], frames=indices.tolist()))
        if not parts:
            raise ValueError('Empty mirror training/validation split')
        self.marker = 'marker_target' in parts[0][0]
        counts = [len(indices) for _, indices, _ in parts]; n = sum(counts)
        self.data = dict(episode=torch.repeat_interleave(torch.arange(len(parts)), torch.tensor(counts)),
            frame_indices=torch.from_numpy(np.concatenate([ix for _, ix, _ in parts])),
            group=torch.tensor([g for _, _, g in parts], dtype=torch.long))
        if self.marker:
            maximum = max(len(b['marker_reference']) for b, _, _ in parts)
            self.data.update(marker_target=torch.zeros(n, maximum, 2),
                marker_valid=torch.zeros(n, maximum, dtype=torch.bool),
                marker_reference=torch.zeros(len(parts), maximum, 2),
                marker_surface=torch.zeros(len(parts), maximum, dtype=torch.long),
                marker_spacing=torch.ones(len(parts), maximum),
                marker_count=torch.tensor([len(b['marker_reference']) for b, _, _ in parts]))
            start = 0
            for i, (b, indices, _) in enumerate(parts):
                end = start+len(indices); k = len(b['marker_reference'])
                self.data['marker_target'][start:end, :k] = b['marker_target']
                self.data['marker_valid'][start:end, :k] = b['marker_valid'] & torch.isfinite(b['marker_target']).all(-1)
                for key in ('marker_reference', 'marker_surface', 'marker_spacing'):
                    self.data[key][i, :k] = b[key]
                start = end
            supported = self.data['marker_valid'].any(-1)
        else:
            self.data.update(target=torch.cat([b['target'] for b, _, _ in parts]),
                             target_mask=torch.cat([b['target_mask'] for b, _, _ in parts]))
            supported = self.data['target_mask'].flatten(1).any(-1)
        dynamic = ('marker_target', 'marker_valid') if self.marker else ('target', 'target_mask')
        for key in (*dynamic, 'episode', 'frame_indices'):
            self.data[key] = self.data[key][supported]
        self.signature = dict(split=split, limit=limit, seed=seed, records_sha256=json_hash(records),
            frames=len(self), sampling='episode_balanced_native_frames_global_batch_shuffle',
            unsupported_frames=int((~supported).sum()))

    def __len__(self):
        return len(self.data['episode'])

    def to(self, device):
        self.data = {k: v.to(device) for k, v in self.data.items()}
        return self

    def batch(self, indices):
        ix = torch.as_tensor(indices, device=self.data['episode'].device, dtype=torch.long)
        episode = self.data['episode'][ix]
        keys = ('marker_target', 'marker_valid') if self.marker else ('target', 'target_mask')
        result = {key: self.data[key][ix] for key in keys}
        result.update(frame_indices=self.data['frame_indices'][ix], group=self.data['group'][episode])
        if self.marker:
            result.update({key: self.data[key][episode] for key in
                ('marker_reference', 'marker_surface', 'marker_spacing', 'marker_count')})
        return result


def reflect_batch(batch, spec):
    """Apply the pinned policy to batched geometry; leave padding absent."""
    out = dict(batch)
    if spec['kind']=='grid':
        permutation = spec['hand_permutation']
        for key in ('target', 'target_mask'):
            value = batch[key][:, permutation]
            out[key] = value.flip(-1) if spec.get('flip_width') else value
        return out
    if spec['kind']!='markers':
        raise ValueError('Only verified grid/marker autoencoders are trainable here')
    surface = batch['marker_surface']; device = surface.device
    transforms = torch.tensor(spec['surface_transforms'], device=device, dtype=batch['marker_reference'].dtype)[surface]
    matrix, offset = transforms[..., :2], transforms[..., 2]
    actual = torch.arange(surface.shape[1], device=device)[None] < batch['marker_count'][:, None]
    reference = torch.einsum('bnij,bnj->bni', matrix, batch['marker_reference'])+offset
    out['marker_reference'] = torch.where(actual[..., None], reference, 0.)
    out['marker_target'] = torch.einsum('bnij,bnj->bni', matrix, batch['marker_target'])
    mapped = torch.tensor(spec['surface_permutation'], device=device)[surface]
    out['marker_surface'] = torch.where(actual, mapped, 0)
    return out


def predict(model, batch):
    if 'marker_target' in batch:
        return model(batch['marker_target'], batch['marker_valid'],
                     batch['marker_reference'], batch['marker_surface'])
    return model(batch['target'], batch['target_mask'])


def objective(model, prediction, batch, config):
    if 'marker_target' in batch:
        return model.loss(prediction, batch['marker_target'], batch['marker_valid'], batch['marker_surface'])
    return prep.reconstruction_objective(model, prediction, batch, config)


def paired_loss(original, mirrored, worse_weight=.25):
    # Both partial derivatives remain positive: making the easier side worse
    # never reduces this objective. At .25 the worse side gets 75% weight.
    if not 0 <= worse_weight < .5:
        raise ValueError('Worse-side emphasis must preserve positive weight for both sides')
    return .5*(original+mirrored)+worse_weight*(original-mirrored).abs()


@torch.no_grad()
def evaluate(model, pool, spec, batch_size=128):
    model.eval()
    totals = {name: defaultdict(float) for name in ('original', 'mirrored')}
    groups = {name: defaultdict(lambda: defaultdict(float)) for name in totals}
    for start in range(0, len(pool), batch_size):
        original = pool.batch(np.arange(start, min(start+batch_size, len(pool))))
        for name, batch in [('original', original), ('mirrored', reflect_batch(original, spec))]:
            prediction = predict(model, batch)
            if not torch.isfinite(prediction).all():
                raise ValueError('Nonfinite mirror reconstruction')
            for group in batch['group'].unique().tolist():
                ix = batch['group']==group
                if pool.marker:
                    y = batch['marker_target'][ix]
                    mask = batch['marker_valid'][ix][..., None].expand_as(y)
                    safe = torch.where(mask, y, 0.)
                    error = (prediction[ix]-safe).abs(); spacing = batch['marker_spacing'][ix, :, None]
                    active = mask & (safe.norm(dim=-1) > .5*model.displacement_scale)[..., None]
                    sums = dict(absolute=float((error*spacing)[mask].sum()),
                        zero=float((safe.abs()*spacing)[mask].sum()), spacing_absolute=float(error[mask].sum()),
                        count=int(mask.sum()), active_absolute=float((error*spacing)[active].sum()),
                        active_zero=float((safe.abs()*spacing)[active].sum()), active_count=int(active.sum()), frames=int(ix.sum()))
                else:
                    sums = metric_sums(prediction[ix], batch['target'][ix], batch['target_mask'][ix])
                for key, value in sums.items():
                    totals[name][key] += value
                    groups[name][pool.groups[group]][key] += value
    if pool.marker:
        return {name: dict(**marker_metrics(total),
            by_platform={g: marker_metrics(v) for g, v in groups[name].items()}) for name, total in totals.items()}
    result = {name: summarize_metrics(total) for name, total in totals.items()}
    for metrics in result.values():
        metrics.update(mae=metrics['all_mae'], units='native_normalized_pressure')
    return result


def quality(metrics, reference, initial, gap=.1):
    """Guard native quality first; parity alone can never qualify a bad AE."""
    active = 'active_mae' if 'active_mae' in metrics['original'] else 'contact_mae'
    tests = {}; ratios = {}; scores = []
    def inspect(values, ref, start, prefix):
        for key in ('mae', active):
            a, b = values['original'][key], values['mirrored'][key]
            scale = max(ref['original'][key], 1e-12)
            tests[prefix+key+'_original_preserved'] = a <= (1+gap)*scale
            tests[prefix+key+'_worst_not_worse'] = max(a,b) <= max(start[v][key] for v in start)+1e-12
            ratios[prefix+key] = max(a,b)/max(min(a,b), 1e-12)
            scores.append(max(a,b)/scale)
    inspect(metrics, reference, initial, '')
    platforms = metrics['original'].get('by_platform', {})
    for group in platforms:
        inspect({v:metrics[v]['by_platform'][group] for v in metrics},
                {v:reference[v]['by_platform'][group] for v in reference},
                {v:initial[v]['by_platform'][group] for v in initial}, group+'/')
    for variant, value in metrics.items():
        tests[variant+'_beats_zero'] = value['mae'] < value['all_zero_baseline_mae']
        zero_key = 'active_zero_baseline_mae' if active=='active_mae' else 'contact_zero_baseline_mae'
        tests[variant+'_active_beats_zero'] = value[active] < .9*value[zero_key]
    accepted = all(tests.values())
    return dict(accepted=accepted, parity_achieved=accepted and all(v<=1+gap for v in ratios.values()),
        score=float(np.mean(scores)), ratios=ratios, tests=tests)


@contextmanager
def cache_runtime():
    from .mirror_cache_verification import verify_codes
    previous = prep.load_model, prep.verify_codes
    original = prep.load_model
    def load(info, device):
        return load_gel_model(info, device) if info.get('representation')=='marker_displacement' else original(info, device)
    prep.load_model, prep.verify_codes = load, verify_codes
    try:
        yield
    finally:
        prep.load_model, prep.verify_codes = previous


def train_source(source, rows, info, parent, policy, root, config, device):
    folder = root/'teachers'/source; folder.mkdir(parents=True, exist_ok=True)
    spec = require_spec(policy, source)
    signature = dict(policy_sha256=json_hash(policy), source=source, parent=parent,
        original_teacher_sha256=info['sha256'], rows_sha256=json_hash(rows),
        native_targets_sha256=prep.target_snapshot(rows, prep.NativeTargets()),
        config_sha256=json_hash(config), implementation_sha256=file_hash(__file__), epochs=config['epochs'])
    contract = folder/'signature.json'
    if contract.exists() and json.loads(contract.read_text())!=signature:
        raise ValueError('Mirror continuation changed; choose a fresh preparation root')
    write_json(contract, signature)
    receipt=folder/'quality.json'
    if receipt.exists():
        previous=json.loads(receipt.read_text())
        if previous.get('training_complete'):
            if previous['signature']!=signature:raise ValueError('Completed mirror continuation changed')
            teacher=previous.get('teacher')
            if teacher and file_hash(teacher['checkpoint'])!=teacher['sha256']:
                raise ValueError('Completed mirror teacher changed')
            return previous
    progress(root, stage='loading_validation', source=source, total_epochs=config['epochs'])
    validation_pool = FramePool(rows, 'val', config['val_frames']).to(device)
    with cache_runtime():
        original_model, _ = prep.load_model(info, device)
        reference = evaluate(original_model, validation_pool, spec, config['batch_size']); del original_model
        selected = dict(info, checkpoint=parent['checkpoint'], sha256=parent['sha256'])
        model, state = prep.load_model(selected, device)
    initial = evaluate(model, validation_pool, spec, config['batch_size'])
    write_json(folder/'baseline.json', dict(reference=reference, initial=initial, validation_pool=validation_pool.signature))
    model.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=config['min_learning_rate'])
    start = 0; best = None; best_key = (2, float('inf')); best_epoch = 0
    if (folder/'last.pt').exists():
        saved = torch.load(folder/'last.pt', map_location='cpu', weights_only=False)
        if saved['mirror_signature']!=signature:raise ValueError('Mirror continuation resume changed')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler']); start=saved['continuation_epoch']
        best=saved['best_quality']; best_key=tuple(saved['best_key']); best_epoch=saved['best_epoch']
        torch.set_rng_state(saved['torch_rng'])
        if device.type=='cuda':torch.cuda.set_rng_state_all(saved['cuda_rng'])
    base = {k:v for k,v in state.items() if k not in
            ('optimizer','scheduler','torch_rng','cuda_rng','best_accepted_score','best_accepted_validation')}
    pool = None; completed_epochs=start
    for epoch in range(start, config['epochs']):
        began = time.monotonic(); cycle=epoch//config['resample_every']
        if pool is None or epoch % config['resample_every']==0:
            del pool
            pool=FramePool(rows, 'train', config['train_frames'], config['seed']+cycle).to(device)
            write_json(folder/f'pool_{cycle:03d}.json', pool.signature)
        model.train(); order=np.random.default_rng(config['seed']+epoch).permutation(len(pool))
        total=0.; count=0
        for offset in range(0, len(order), config['batch_size']):
            batch=pool.batch(order[offset:offset+config['batch_size']]); mirrored=reflect_batch(batch,spec)
            optimizer.zero_grad(set_to_none=True)
            losses=[objective(model,predict(model,value),value,state.get('config',{})) for value in (batch,mirrored)]
            loss=paired_loss(*losses,config['worse_weight'])
            if not torch.isfinite(loss):raise ValueError('Nonfinite paired mirror objective')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step(); total+=float(loss.detach())*len(batch['frame_indices']); count+=len(batch['frame_indices'])
        scheduler.step(); metrics=evaluate(model,validation_pool,spec,config['batch_size'])
        decision=quality(metrics,reference,initial,config['parity_gap'])
        key=(0 if decision['parity_achieved'] else 1, decision['score'])
        saved={**base,'model':model.state_dict(),'encoder':model.encoder.state_dict() if hasattr(model,'encoder') else None,
            'decoder':model.decoder.state_dict(),'mirror_signature':signature,'mirror_adapted':True,
            'epoch':epoch,'continuation_epoch':epoch+1,'validation':metrics}
        if decision['accepted'] and (key[0]<best_key[0] or
                (key[0]==best_key[0] and key[1]<best_key[1]*(1-config['min_delta']))):
            path=folder/f'accepted_epoch_{epoch+1:03d}.pt'; atomic_checkpoint(path,saved)
            teacher={**info,'checkpoint':str(path.resolve()),'sha256':file_hash(path),
                'model_config':state['model_config'],'original_checkpoint':parent['original_checkpoint'],'mirror_adapted':True}
            best=dict(accepted=True,adapted=True,parity_achieved=decision['parity_achieved'],
                signature=signature,teacher=teacher,baseline=reference,initial=initial,validation=metrics,
                decision=decision,best_epoch=epoch+1,completed_epochs=epoch+1)
            best_key=key; best_epoch=epoch+1
        if best:
            best['completed_epochs']=epoch+1
            write_json(folder/'quality.json',best)
        atomic_checkpoint(folder/'last.pt',{**saved,'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
            'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else [],
            'best_quality':best,'best_key':best_key,'best_epoch':best_epoch})
        elapsed=time.monotonic()-began
        record=dict(epoch=epoch+1,train_loss=total/max(count,1),validation=metrics,quality=decision,
                    best_epoch=best_epoch,seconds=elapsed,training_frames=count,pool=pool.signature)
        with (folder/'metrics.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
        progress(root,stage='training',source=source,epoch=epoch+1,total_epochs=config['epochs'],
            epoch_seconds=elapsed,original_mae=metrics['original']['mae'],mirrored_mae=metrics['mirrored']['mae'],
            ratios=decision['ratios'],best_epoch=best_epoch,parity_achieved=bool(best and best['parity_achieved']))
        completed_epochs=epoch+1
        if (epoch+1>=config['min_epochs'] and best and best['parity_achieved']
                and epoch+1-best_epoch>=config['patience']):
            break
    del model, optimizer, pool, validation_pool
    if device.type=='cuda':torch.cuda.empty_cache()
    if best:
        best.update(training_complete=True,completed_epochs=completed_epochs)
        write_json(folder/'quality.json',best)
    else:
        best=dict(accepted=False,adapted=True,parity_achieved=False,signature=signature,teacher=None,
                  completed_epochs=completed_epochs,training_complete=True,baseline=reference,initial=initial,validation=None)
        write_json(folder/'quality.json',best)
    return best


def run(config_path='configs/mirror_balanced_training_v3.json'):
    config=json.loads(Path(config_path).read_text());root=Path(config['output']);root.mkdir(parents=True,exist_ok=True)
    lock=(root/'prepare.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(4);torch.manual_seed(config['seed']);device=torch.device('cuda')
    policy=load_policy(config['policy']);parents=json.loads((root/'parents.json').read_text())
    implementation={str(Path(__file__).relative_to(Path.cwd())):file_hash(__file__),
        'src/tactilelatent/gel_mirror_preparation.py':file_hash(Path(__file__).with_name('gel_mirror_preparation.py'))}
    audit=verify_evidence(policy);write_json(root/'policy_audit.json',dict(policy_sha256=json_hash(policy),sources=audit))
    results={s:dict(ready=False,stage='queued_training') for s in config['sources']}
    results['ftp_rdp_bimanual']=dict(ready=True,numeric=True,policy_sha256=json_hash(policy))
    results['opengraph']=dict(ready=False,stage='blocked_geometry',reason=policy['sources']['opengraph']['reason'])
    def publish():
        done=all(results[s].get('ready') for s in config['sources'])
        ready=dict(complete=False,complete_for_verified_sources=done,policy_path=str(Path(config['policy']).resolve()),
            policy_sha256=json_hash(policy),implementation=implementation,sources=results)
        write_json(root/'ready.json',ready)
        if done:
            joint=json.loads(Path('configs/joint_vjepa21_spatial_expanded_32d_mirror_v2.json').read_text())
            joint['output']='processing/runs/joint_vjepa21_spatial_expanded32d_mirror_v3'
            joint['mirror_augmentation']['preparation']=str(root)
            write_json(root/'available_joint_config.json',joint)
        return ready
    publish()
    for source in config['sources']:
        try:
            rows,info=prep.source_inputs(source)
            best=train_source(source,rows,info,parents[source],policy,root,config,device)
            if best['accepted'] and best['parity_achieved']:
                progress(root,stage='caching_codes',source=source)
                with cache_runtime():index=prep.cache_codes(source,rows,best['teacher'],policy,root,device)
                path=root/'codes'/source/'index.json'
                results[source]=dict(ready=True,numeric=False,teacher=best['teacher'],statistics=index['statistics'],
                    code_index=str(path.resolve()),code_index_sha256=file_hash(path),
                    teacher_quality=str((root/'teachers'/source/'quality.json').resolve()),parity_achieved=True)
            else:
                results[source]=dict(ready=False,stage='parity_target_not_met' if best['accepted'] else 'blocked_teacher_quality',
                    quality=str(root/'teachers'/source/'quality.json'))
        except Exception as error:
            results[source]=dict(ready=False,stage='failed',reason=f'{type(error).__name__}: {error}')
            progress(root,stage='source_failed',source=source,error=results[source]['reason'])
        publish()
    ready=publish();progress(root,stage='complete' if ready['complete_for_verified_sources'] else 'complete_with_explicit_blocks',
                            sources=results,production_joint_run_started=False)
    return ready
