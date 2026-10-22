"""Verify preserved targets and prepare the next source-filtered joint run.

No training is launched. Tactile arrays are hard-linked from verified caches;
only provenance metadata changes to describe the new visual storage revision.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from tactilelatent.aligned_frames import targets,manifest_signature,file_hash,write_json
from tactilelatent.marker_ae_data import snapshot
from tactilelatent.marker_code_cache import marker_signature
from tactilelatent.sensor_code_cache import cache_signature,train_statistics

ROOT=Path('processing/storage_v1')


def main():
    status=json.loads((ROOT/'status.json').read_text())
    if status['stage']!='complete':raise ValueError('Visual conversion has not completed')
    manifest=ROOT/'manifest.json';payload=json.loads(manifest.read_text())
    rows={(r['source'],r['episode_id']):r for r in payload['episodes']}
    for check in json.loads((ROOT/'native_target_checks.json').read_text()):
        value=targets(rows[(check['source'],check['episode_id'])],np.array(check['indices']))
        if hashlib.sha256(value.tobytes()).hexdigest()!=check['target_sha256']:raise ValueError(f'Native targets changed: {check["source"]}')
    signature=manifest_signature(manifest)
    index=json.loads(Path('processing/marker_displacement_v2/targets/index.json').read_text())
    index['episodes']=[r for r in index['episodes'] if r['source']!='freetacman'];index['statistics'].pop('freetacman',None)
    index['provenance']['data']=signature
    target_root=Path('processing/marker_displacement_storage_v1/targets');target_root.mkdir(parents=True,exist_ok=True)
    for source in ('ftp_unit','ftp_unit_bimanual','ftp_vla_touch'):
        link=target_root/source
        if not link.exists():link.symlink_to(Path('processing/marker_displacement_v2/targets',source).resolve(),target_is_directory=True)
    write_json(target_root/'index.json',index)
    native=json.loads(Path('processing/joint_alignment/teachers.json').read_text())
    markers=json.loads(Path('processing/joint_alignment/marker_teachers_v1.json').read_text())
    markers['sources'].pop('freetacman',None)
    for key in list(markers):
        if key.startswith('freetacman'):markers.pop(key)
    markers['selection']='Retained frozen marker teachers after FreeTacMan retirement; no retraining or target changes'
    write_json(ROOT/'marker_teachers.json',markers)
    native['sources']={s:v for s,v in native['sources'].items() if s not in ('freetacman',*markers['sources'])}
    combined={**native,'sources':{**native['sources'],**markers['sources']}}
    write_json(ROOT/'teachers.json',combined)
    normalization=json.loads(Path('configs/vjepa_live_teacher_normalization.json').read_text())
    normalization['data']=signature;normalization['statistics'].pop('freetacman',None)
    normalization['teacher_hashes']={s:v['sha256'] for s,v in native['sources'].items()}
    write_json(ROOT/'normalization.json',normalization)
    cache_root=Path('processing/sensor_code_cache_storage_v1')
    for source,info in combined['sources'].items():
        old=Path('processing/sensor_code_cache_markers_v1')/source
        previous=json.loads((old/'index.json').read_text())
        if previous['signature']['teacher_sha256']!=info['sha256']:raise ValueError('Teacher cache differs')
        expected_rows=[r for r in payload['episodes'] if r['source']==source]
        if previous['episodes']!=expected_rows:raise ValueError(f'Cached episode/frame order differs: {source}')
        folder=cache_root/source;folder.mkdir(parents=True,exist_ok=True)
        hashes=previous.get('array_sha256')
        if hashes is None:
            # The original native cache predates per-array checksums. Validate
            # its shape, train-only statistics and episode order, then record
            # hashes now without claiming these were pre-migration hashes.
            codes=np.load(old/'codes.npy',mmap_mode='r');valid=np.load(old/'valid.npy',mmap_mode='r')
            if (codes.shape!=(previous['frames'],previous['code_dim']) or codes.dtype!=np.float32
                    or valid.shape!=(previous['frames'],) or valid.dtype!=np.bool_):
                raise ValueError(f'Invalid legacy code arrays: {source}')
            statistics=train_statistics(codes,valid,previous['episodes'],previous['offsets'])
            for name in ('mean','std'):
                if not np.allclose(statistics[name],previous['statistics'][name],rtol=1e-7,atol=1e-8):
                    raise ValueError(f'Legacy cached training statistics changed: {source}')
            hashes={name:file_hash(old/name) for name in ('codes.npy','valid.npy')}
            previous['array_checksum_origin']='Recorded during RGB migration; original cache had no per-array hashes'
        for name,digest in hashes.items():
            if file_hash(old/name)!=digest:raise ValueError('Cached tactile code arrays changed')
            if not (folder/name).exists():os.link((old/name).resolve(),folder/name)
            elif not os.path.samefile(old/name,folder/name):raise ValueError('Migration cache array is not the original verified file')
        previous['array_sha256']=hashes
        if source in markers['sources']:
            pinned=snapshot(manifest,target_root,source)
            previous['signature']=marker_signature(manifest,combined,source,pinned,signature)
        else:previous['signature']=cache_signature(manifest,combined,source,signature)
        write_json(folder/'index.json',previous)
    write_json(cache_root/'registry.json',dict(complete=True,teachers=combined,data_signature=signature,
        marker_target_index_sha256=file_hash(target_root/'index.json'),
        cache_migration='Same pinned teacher outputs and bitwise-preserved targets; arrays hard-linked; FreeTacMan excluded'))
    old_audit=json.loads(Path('processing/joint_alignment/audit.json').read_text())
    old_audit.update(signature=signature,passed=True,storage_migration=str(ROOT/'registry.json'),native_target_checks=str(ROOT/'native_target_checks.json'))
    write_json(ROOT/'audit.json',old_audit)
    config_path=Path('configs/joint_vjepa21_spatial_storage_32d.json');cfg=json.loads(config_path.read_text())
    cfg.update(audit=str(ROOT/'audit.json'),teachers=str(ROOT/'teachers.json'),normalization=str(ROOT/'normalization.json'),
               marker_targets=str(target_root),marker_teachers=str(ROOT/'marker_teachers.json'))
    write_json(config_path,cfg)
    from tactilelatent.train_vjepa_markers import prepare
    train,val,registry,mc,provenance=prepare(cfg)
    if 'freetacman' in train.indices_by_source or len(train.indices_by_source)!=10:raise ValueError('Incorrect active source set')
    write_json(ROOT/'preparation.json',dict(passed=True,sources=sorted(train.indices_by_source),
        heldout_source='ftp_motiontrans',latent_dim=32,train_clips=len(train),validation_clips=len(val),
        data=signature,training_launched=False,completed_at=time.time()))
    subprocess.run(['systemctl','--user','restart','tactilelatent-viewer.service'],check=True)
    print('Native targets verified; 10-source joint configuration and tactile caches prepared. No training launched.',flush=True)


if __name__=='__main__':
    write_json(ROOT/'finalization_status.json',dict(stage='running',pid=os.getpid(),started_at=time.time()))
    try:
        main()
    except BaseException as error:
        write_json(ROOT/'finalization_status.json',dict(stage='failed',error=str(error),updated_at=time.time()))
        raise
    write_json(ROOT/'finalization_status.json',dict(stage='complete',updated_at=time.time()))
