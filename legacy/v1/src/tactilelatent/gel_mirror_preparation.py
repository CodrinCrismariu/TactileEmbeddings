"""Prepare user-reviewed gel mirrors without rewriting accepted v1 artifacts.

The pinned native transform/cache builder is reused. This extension supplies
the current Baihu loader and evaluates signed marker errors in native pixels.
"""
from collections import defaultdict
from contextlib import contextmanager
import copy
import fcntl
import json
from pathlib import Path

import torch

from .aligned_frames import file_hash, json_hash, write_json
from .bimanual_mirror import load_policy, require_spec, transform_targets, verify_evidence
from . import mirror_preparation as preparation

ROOT = Path('processing/bimanual_mirror_v2')
POLICY = 'configs/bimanual_mirror_policy_v2.json'
GEL_SOURCES = ('ftp_unit_bimanual', 'baihu')


def load_gel_model(info, device):
    if file_hash(info['checkpoint']) != info['sha256']:
        raise ValueError('Pinned gel teacher changed')
    state = torch.load(info['checkpoint'], map_location='cpu', weights_only=False)
    if state.get('model_type', '').startswith('expansion_marker_autoencoder_'):
        from .expansion_models import load_teacher
        return load_teacher(info['checkpoint'], device)
    from .marker_autoencoder import load_marker_autoencoder
    return load_marker_autoencoder(info['checkpoint'], device)


def _metrics(total):
    count = total['count']
    if not count:
        raise ValueError('No valid native marker coordinates in mirror audit')
    return dict(mae=total['absolute']/count, all_mae=total['absolute']/count,
        all_zero_baseline_mae=total['zero']/count,
        mae_spacing=total['spacing_absolute']/count, valid_coordinates=int(count),
        active_mae=total['active_absolute']/max(total['active_count'], 1),
        active_zero_baseline_mae=total['active_zero']/max(total['active_count'], 1),
        active_coordinates=int(total['active_count']), frames=int(total['frames']),
        units='native_pixels')


@torch.no_grad()
def evaluate_marker_pair(model, rows, spec, reader, device, limit=10000):
    """Audit equal original/mirrored frames, including each Baihu platform."""
    model.eval()
    totals = {name: defaultdict(float) for name in ('original', 'mirrored')}
    platforms = {name: defaultdict(lambda: defaultdict(float)) for name in totals}
    for row, indices in preparation.sampled_batches(rows, 'val', limit):
        original = reader.read(row, indices)
        for name, item in [('original', original), ('mirrored', transform_targets(original, spec))]:
            item = preparation.to_device(item, device)
            prediction = preparation.predict(model, item)
            y = item['marker_target']
            mask = (item['marker_valid'] & torch.isfinite(y).all(-1))[..., None].expand_as(y)
            if not torch.isfinite(prediction).all():
                raise ValueError('Nonfinite gel reconstruction during mirror audit')
            spacing = item['marker_spacing'][None, :, None]
            safe = torch.where(mask, y, 0.)
            error = (prediction-safe).abs()
            active = mask & (safe.norm(dim=-1) > .5*model.displacement_scale)[..., None]
            values = dict(absolute=float((error*spacing)[mask].sum()),
                zero=float((safe.abs()*spacing)[mask].sum()),
                spacing_absolute=float(error[mask].sum()), count=int(mask.sum()),
                active_absolute=float((error*spacing)[active].sum()),
                active_zero=float((safe.abs()*spacing)[active].sum()),
                active_count=int(active.sum()), frames=int(mask.flatten(1).any(-1).sum()))
            for key, value in values.items():
                totals[name][key] += value
                platforms[name][row.get('platform', row['source'])][key] += value
    return {name: dict(**_metrics(total),
                      by_platform={key: _metrics(value) for key, value in platforms[name].items()})
            for name, total in totals.items()}


@contextmanager
def gel_runtime():
    """Scope loader/audit extensions to new gel artifacts, never alter v1."""
    from .mirror_cache_verification import verify_codes
    previous = preparation.load_model, preparation.evaluate, preparation.verify_codes
    preparation.load_model = load_gel_model
    preparation.evaluate = evaluate_marker_pair
    preparation.verify_codes = verify_codes
    try:
        yield
    finally:
        preparation.load_model, preparation.evaluate, preparation.verify_codes = previous


def inherited_sources(policy, parent='processing/bimanual_mirror_v1'):
    """Retain actual historical receipts; never relabel their policy hashes."""
    from .mirror_joint import read_preparation
    parent = Path(parent)
    ready = json.loads((parent/'ready.json').read_text())
    old_policy = load_policy(ready['policy_path'])
    sources = ('egotouch', 'ftp_rdp_bimanual')
    cfg = dict(heldout_source='ftp_motiontrans', prediction_mode='current_frame',
        mirror_augmentation=dict(enabled=True, sources=list(sources), probability=.5,
            policy=ready['policy_path'], preparation=str(parent)))
    read_preparation(cfg)
    result = {}
    for source in sources:
        if policy['sources'][source] != old_policy['sources'][source]:
            raise ValueError('Cannot reuse an unchanged-source cache for a different mapping')
        result[source] = dict(copy.deepcopy(ready['sources'][source]),
            artifact_policy_path=ready['policy_path'],
            artifact_policy_sha256=ready['policy_sha256'])
    return result


def run(root=ROOT, policy_path=POLICY, manifest=preparation.MANIFEST, device='cuda',
        train_frames=100000, val_frames=10000):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    lock = (root/'prepare.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.set_num_threads(4); torch.manual_seed(45)
    device = torch.device(device)
    policy = load_policy(policy_path)
    extension = {str(Path(__file__).relative_to(Path.cwd())): file_hash(__file__)}
    receipt = root/'extension.json'
    contract = dict(policy_sha256=json_hash(policy), implementation=extension,
                    pinned_builder_sha256=file_hash(preparation.__file__),
                    pinned_transform_sha256=file_hash(Path(__file__).with_name('bimanual_mirror.py')))
    if receipt.exists() and json.loads(receipt.read_text()) != contract:
        raise ValueError('Gel preparation provenance changed; use a fresh output directory')
    write_json(receipt, contract)
    audit = verify_evidence(policy)
    write_json(root/'policy_audit.json', dict(policy_sha256=json_hash(policy), sources=audit))
    results = inherited_sources(policy)
    results['opengraph'] = dict(ready=False, stage='blocked_geometry',
                               reason=policy['sources']['opengraph']['reason'])
    for source in GEL_SOURCES:
        results[source] = dict(ready=False, stage='pending_teacher_audit')

    def publish():
        verified = [s for s, spec in policy['sources'].items() if spec['status']=='verified']
        ready = dict(complete=all(entry['ready'] for entry in results.values()),
            complete_for_verified_sources=all(results[s]['ready'] for s in verified),
            policy_path=str(Path(policy_path).resolve()), policy_sha256=json_hash(policy),
            implementation=extension, sources=results)
        write_json(root/'ready.json', ready)
        if ready['complete_for_verified_sources']:
            config = json.loads(Path('configs/joint_vjepa21_spatial_expanded_32d_mirror_v2.json').read_text())
            write_json(root/'available_joint_config.json', config)
        return ready

    publish()
    for source in GEL_SOURCES:
        try:
            require_spec(policy, source)
            rows, info = preparation.source_inputs(source, manifest)
            model, state = load_gel_model(info, device)
            info = dict(info, model_config=state['model_config'],
                        original_checkpoint=info['checkpoint'])
            del model, state
            with gel_runtime():
                decision = preparation.prepare_teacher(source, rows, info, policy, root,
                                                       device, train_frames, val_frames)
                if decision['accepted']:
                    index = preparation.cache_codes(source, rows, decision['teacher'], policy, root, device)
                    path = root/'codes'/source/'index.json'
                    results[source] = dict(ready=True, numeric=False, teacher=decision['teacher'],
                        code_index=str(path.resolve()), code_index_sha256=file_hash(path),
                        statistics=index['statistics'],
                        teacher_quality=str((root/'teachers'/source/'quality.json').resolve()))
                else:
                    results[source] = dict(ready=False, stage='blocked_teacher_quality',
                        report=str(root/'teachers'/source/'quality.json'))
        except Exception as error:
            results[source] = dict(ready=False, stage='failed', reason=f'{type(error).__name__}: {error}')
            preparation.progress(root, stage='source_failed', source=source,
                                 error=results[source]['reason'])
        publish()
    ready = publish()
    preparation.progress(root,
        stage='complete_verified_sources' if ready['complete_for_verified_sources'] else 'partial_ready_with_explicit_blocks',
        sources=results, production_joint_run_started=False)
    return ready
