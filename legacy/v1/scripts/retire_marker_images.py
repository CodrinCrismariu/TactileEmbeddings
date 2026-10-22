"""Retire only verified FTP marker-source tactile pixels, preserving all else.

Stage and checksum replacements before atomic archive/index replacement. The
plan and per-file receipts make an interrupted commit recoverable. No teacher,
marker, code array, scene camera, or unrelated sensor target is removed.
"""
import argparse
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tarfile
import time
import zipfile

from tactilelatent.expansion_runtime import configure
configure()

import numpy as np
import torch

from tactilelatent.aligned_frames import file_hash, json_hash, write_json as atomic_json
from tactilelatent.marker_storage import ROOT, SOURCES

STORAGE = Path('processing/storage_v1')
MANIFEST = STORAGE / 'manifest.json'
TARGETS = Path('processing/marker_displacement_storage_v1/targets')
CODES = Path('processing/sensor_code_cache_storage_v1')
TACTILE_KEYS = {'left_tactile_data_gripper', 'right_tactile_data_gripper'}


def write_json(path, value):
    atomic_json(path, value)
    path=Path(path)
    with path.open('rb') as stream:os.fsync(stream.fileno())
    descriptor=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(descriptor)
    finally:os.close(descriptor)


def verify_retained(result):
    for path,digest in result['retained_artifacts'].items():
        if file_hash(path)!=digest:
            raise ValueError(f'Retained marker artifact changed: {path}')


def stat(path):
    value = Path(path).stat()
    return [value.st_size, value.st_mtime_ns]


def progress(stage, **values):
    record = dict(stage=stage, updated_at=time.time(), **values)
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / 'progress.jsonl').open('a') as stream:
        stream.write(json.dumps(record) + '\n'); stream.flush(); os.fsync(stream.fileno())
    write_json(ROOT / 'status.json', record)
    print(json.dumps(record), flush=True)


def tactile_member(name):
    return '/data/' in name and name.split('/data/', 1)[1].split('/')[0] in TACTILE_KEYS


def retained_tar_hashes(path):
    result = {}; members = {}
    with tarfile.open(path, 'r:') as source:
        for member in source:
            if member.isfile():
                data = source.extractfile(member).read()
                result[member.name] = hashlib.sha256(data).hexdigest()
                members[member.name] = [member.offset_data, member.size]
    return result, members


def stage_tar(path, temporary):
    """The only removed keys are the two named tactile image arrays."""
    hashes = {}; removed = {}; count = 0
    with tarfile.open(path, 'r:') as source, tarfile.open(temporary, 'w:') as output:
        for member in source:
            if tactile_member(member.name):
                removed[member.name] = member.size
                continue
            if member.isfile():
                data = source.extractfile(member).read()
                hashes[member.name] = hashlib.sha256(data).hexdigest()
                output.addfile(member, io.BytesIO(data)); count += 1
            else:
                output.addfile(member)
    after, members = retained_tar_hashes(temporary)
    if not removed or after != hashes or any(tactile_member(name) for name in members):
        raise ValueError('Tactile-only TAR replacement failed retained-payload verification')
    return dict(retained_sha256=hashes, removed_members=removed,
                retained_members=count), members


def stage_zip(path, temporary):
    hashes = {}; removed = {}
    with zipfile.ZipFile(path) as source, zipfile.ZipFile(temporary, 'w', allowZip64=True) as output:
        for member in source.infolist():
            if '/gelsight/' in member.filename:
                removed[member.filename] = member.compress_size
                continue
            data = source.read(member)
            hashes[member.filename] = hashlib.sha256(data).hexdigest()
            output.writestr(member, data)
    with zipfile.ZipFile(temporary) as archive:
        after = {m.filename: hashlib.sha256(archive.read(m)).hexdigest() for m in archive.infolist()}
    if not removed or after != hashes or any('/gelsight/' in name for name in after):
        raise ValueError('Tactile-only ZIP replacement failed retained-payload verification')
    return dict(retained_sha256=hashes, removed_members=removed, retained_members=len(hashes))


def pin_implementation():
    pin = STORAGE / 'implementation.json'
    value = json.loads(pin.read_text())
    backup = ROOT / 'previous_implementation.json'
    if not backup.exists():
        write_json(backup, value)
    for name in ('marker_storage.py', 'native_optical.py', 'marker_viewer.py', 'train_vjepa_markers.py'):
        value['files'][f'src/tactilelatent/{name}'] = ''
    for path in value['files']:
        value['files'][path] = file_hash(path)
    write_json(pin, value)


def verify_consumers():
    from tactilelatent.marker_autoencoder import load_marker_autoencoder
    from tactilelatent.marker_data import MarkerCache
    from tactilelatent.marker_code_cache import verify_samples
    from tactilelatent.marker_viewer import payload
    from tactilelatent.sensor_code_cache import CodeCache
    from tactilelatent.train_vjepa_markers import prepare
    from tactilelatent.marker_code_model import MarkerCodeVJEPA
    from tactilelatent.marker_data import marker_collate
    from tactilelatent.train_alignment import alignment_loss

    teachers = json.loads((STORAGE / 'teachers.json').read_text())
    rows = json.loads(MANIFEST.read_text())['episodes']
    cache = MarkerCache(TARGETS, max_loaded_episodes=1)
    records = {}; sources = {}
    for source in SOURCES:
        selected = [r for r in rows if r['source'] == source]
        shapes = {tuple(r['target_shape']) for r in selected}
        if len(shapes) != 1:
            raise ValueError('Marker source has inconsistent native sensor geometry')
        for row in selected:
            ids = np.array(sorted({0, row['frames']//2, row['frames']-1}))
            batch = cache.read(row, ids)
            if batch['marker_target'].shape[0] != len(ids):
                raise ValueError('Missing marker observations')
            from tactilelatent.marker_data import episode_key
            folder = TARGETS / source / row['split'] / episode_key(row)
            info = json.loads((folder / 'complete.json').read_text())
            if info['surfaces'] != row['target_shape'][0]//3:
                raise ValueError('Marker surface identities differ from tactile source')
            if [3*info['surfaces'], *info['sensors'][0]['shape'][:2]] != row['target_shape']:
                raise ValueError('Native marker geometry changed')
            records[str(folder/'tracks.npz')] = info['tracks_sha256']
            records[str(folder/'complete.json')] = file_hash(folder/'complete.json')
        verify_samples(str(MANIFEST), teachers, source, str(TARGETS), str(CODES), 'cpu')
        code_cache = CodeCache(CODES, source)
        for name, expected in code_cache.index['array_sha256'].items():
            path = CODES/source/name
            if file_hash(path) != expected:
                raise ValueError('Cached marker code array changed')
            records[str(path)] = expected
        # A disposable teacher copy verifies a real masked training update.
        model, _ = load_marker_autoencoder(teachers['sources'][source]['checkpoint'])
        model.train().requires_grad_(True)
        row = selected[len(selected)//2]; b = cache.read(row, np.array([row['frames']//2]))
        ref = b['marker_reference'][None]; surface = b['marker_surface'][None]
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        prediction = model(b['marker_target'], b['marker_valid'], ref, surface)
        loss = model.loss(prediction, b['marker_target'], b['marker_valid'], surface)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        rendering = payload(dict(manifest=str(MANIFEST), marker_targets=str(TARGETS)), row,
                            row['frames']//2, prediction.detach()[0].numpy())
        json.dumps(rendering, allow_nan=False)
        if len(rendering['optical_panels']) != len(info['sensors']):
            raise ValueError('Geometry-only viewer lost a surface')
        sources[source] = dict(episodes=len(selected), target_shape=list(next(iter(shapes))),
                               code_dim=code_cache.index['code_dim'], teacher_update_loss=float(loss.detach()))
        progress('verified_source', source=source, **sources[source])
    cfg = json.loads(Path('configs/joint_vjepa21_spatial_current_32d.json').read_text())
    train, val, registry, mc, provenance = prepare(cfg)
    # Exercise actual current-frame RGB/code batches through the same attention
    # and adapter implementation, with a small disposable visual configuration.
    tiny = {**mc, 'width':16, 'heads':4, 'layers':1, 'feedforward':32, 'grid_size':2, 'dropout':0.}
    network = MarkerCodeVJEPA(**tiny)
    optimizer = torch.optim.AdamW(network.parameters(), lr=1e-5)
    updates = {}
    for source in SOURCES:
        b = marker_collate([train[train.indices_by_source[source][0]]])
        features = torch.zeros(1,4,3,4,768)
        output = network(features, b['view_mask'], source, sample_latent=False)
        loss, _, _ = alignment_loss(output, b, 0.)
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        updates[source] = dict(loss=float(loss.detach()), rgb_frames=b['rgb'].shape[1],
            code_frames=b['code_target'].shape[1], sensor_frame_indices=b['sensor_frame_indices'].tolist())
    return dict(sources=sources, retained_artifacts=records,
        joint_preparation=dict(sources=sorted(train.indices_by_source),train_clips=len(train),validation_clips=len(val)),
        cached_joint_updates=updates, data=provenance['data'])


def preflight():
    if (ROOT/'preflight.json').exists():
        return json.loads((ROOT/'preflight.json').read_text())
    progress('verifying_marker_consumers')
    pin_implementation()
    # Capture audited logical signatures without rewriting immutable manifests,
    # marker indexes, cached codes, teacher weights or checkpoint provenance.
    ledger_path = STORAGE/'registry.json'
    ledger = json.loads(ledger_path.read_text())
    write_json(ROOT/'previous_registry.json', ledger)
    for name in ('audit.json', 'preparation.json'):
        value = json.loads((STORAGE/name).read_text())
        signature = value.get('signature', value.get('data'))
        if signature not in ledger['legacy_signatures']:
            ledger['legacy_signatures'].append(signature)
    write_json(ledger_path, ledger)
    result = verify_consumers()
    write_json(ROOT/'preflight.json', result)
    rows = json.loads(MANIFEST.read_text())['episodes']
    jobs = []
    for source in SOURCES:
        row = next(r for r in rows if r['source']==source)
        path=Path(row['tar_path']); index=Path(row['tar_index'])
        jobs.append(dict(kind='tar',path=str(path),index=str(index),source=source,
            old_stat=stat(path),old_index_stat=stat(index)))
    for path in sorted(Path('datasets/AllenBi21__VLA-Touch/raw_data').glob('*.zip')):
        jobs.append(dict(kind='zip',path=str(path),source='ftp_vla_touch',old_stat=stat(path)))
    write_json(ROOT/'plan.json',dict(jobs=jobs,created_at=time.time(),verified_episodes=sum(s['episodes'] for s in result['sources'].values())))
    progress('preflight_complete',episodes=sum(s['episodes'] for s in result['sources'].values()),archives=len(jobs))
    return result


def stage(job):
    path=Path(job['path']);key=hashlib.sha256(str(path).encode()).hexdigest()[:16]
    if path.is_symlink() or (job.get('index') and Path(job['index']).is_symlink()):
        raise ValueError('Refusing a symlinked source archive or index')
    receipt=ROOT/'archives'/f'{key}.json'
    if receipt.exists():
        return receipt,json.loads(receipt.read_text())
    if stat(path)!=job['old_stat']:
        raise ValueError(f'Original archive changed before staging: {path}')
    if shutil.disk_usage(path).free < 100*2**30 + path.stat().st_size:
        raise RuntimeError('Staging would violate the 100 GiB free-space reserve')
    temporary=path.with_name(path.name+'.markers.partial')
    if temporary.is_symlink():
        raise ValueError('Refusing a symlinked staging path')
    progress('staging_archive',path=str(path),old_bytes=path.stat().st_size)
    old_hash=file_hash(path)
    if job['kind']=='tar':
        detail,members=stage_tar(path,temporary)
        index=Path(job['index']);new_index=json.loads(index.read_text())
        if stat(index)!=job['old_index_stat']:
            raise ValueError('Original tar index changed before staging')
        temporary_index=index.with_name(index.name+'.markers.partial')
        new_index.update(members=members,signature=[str(path.resolve()),*stat(temporary)])
        write_json(temporary_index,new_index)
        detail.update(temporary_index=str(temporary_index),old_index_sha256=file_hash(index),
            new_index_sha256=file_hash(temporary_index),new_index_stat=stat(temporary_index))
    else:
        detail=stage_zip(path,temporary)
    if stat(path)!=job['old_stat'] or temporary.stat().st_size>=path.stat().st_size:
        raise ValueError('Source changed or replacement did not reclaim space')
    with temporary.open('rb') as stream:os.fsync(stream.fileno())
    value=dict(**job,**detail,temporary=str(temporary),old_sha256=old_hash,
        new_stat=stat(temporary),new_sha256=file_hash(temporary),stage='staged')
    write_json(receipt,value)
    return receipt,value


def commit(receipt, value):
    path=Path(value['path']);temporary=Path(value['temporary'])
    # The journal is durable before each destructive replacement. A crash
    # between archive and index replacement is safely completed on re-entry.
    progress('commit_intent',path=str(path),receipt=str(receipt))
    current=file_hash(path)
    if current==value['old_sha256']:
        if stat(path)!=value['old_stat'] or file_hash(temporary)!=value['new_sha256']:
            raise ValueError('Changed source or staged replacement before commit')
        os.replace(temporary,path)
    elif current!=value['new_sha256']:
        raise ValueError('Archive is neither the original nor the verified replacement')
    if value['kind']=='tar':
        index=Path(value['index']);index_hash=file_hash(index)
        if index_hash==value['old_index_sha256']:
            staged=Path(value['temporary_index'])
            if file_hash(staged)!=value['new_index_sha256']:
                raise ValueError('Staged tar index changed')
            os.replace(staged,index)
        elif index_hash!=value['new_index_sha256']:
            raise ValueError('Tar index is neither its original nor verified replacement')
    ledger_path=STORAGE/'registry.json';ledger=json.loads(ledger_path.read_text())
    for p in ([path,Path(value['index'])] if value['kind']=='tar' else [path]):
        key=str(p.resolve());previous=ledger['replacements'].get(key)
        if not previous or previous.get('marker_retirement_receipt')!=str(receipt):
            ledger['replacements'][key]=dict(kind='marker_tactile_retirement',deleted=False,
                new_stat=stat(p),marker_retirement_receipt=str(receipt),
                sha256=file_hash(p),prior_replacement=previous)
    write_json(ledger_path,ledger)
    value.update(stage='committed',committed_at=time.time(),saved_bytes=value['old_stat'][0]-stat(path)[0])
    write_json(receipt,value)
    progress('archive_committed',path=str(path),saved_bytes=value['saved_bytes'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply',action='store_true')
    args=parser.parse_args();ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'migration.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(2)
        result=preflight()
        if not args.apply:return
        verify_retained(result)
        # Publishing geometry mode first makes every subsequent consumer fail
        # closed on optical pixels, even if interrupted between archive commits.
        write_json(ROOT/'retirement.json',dict(active=True,sources=result['sources'],
            marker_artifacts_sha256=json_hash(result['retained_artifacts']),
            preflight_sha256=file_hash(ROOT/'preflight.json'),started_at=time.time()))
        for job in json.loads((ROOT/'plan.json').read_text())['jobs']:
            receipt,value=stage(job)
            verify_retained(result)
            commit(receipt,value)
        verify_retained(result)
        progress('verifying_after_retirement')
        after=verify_consumers();write_json(ROOT/'verification.json',after)
        if after['retained_artifacts']!=result['retained_artifacts']:
            raise ValueError('Marker/code identity changed during retirement')
        receipts=[json.loads(p.read_text()) for p in (ROOT/'archives').glob('*.json')]
        saved=sum(r['saved_bytes'] for r in receipts)
        free=shutil.disk_usage(ROOT).free
        progress('complete',archives=len(receipts),episodes=sum(s['episodes'] for s in result['sources'].values()),
            saved_bytes=saved,free_bytes=free,retained_marker_and_code_checks_passed=True)


if __name__=='__main__':
    try:
        main()
    except BaseException as error:
        progress('failed',error=str(error))
        raise
