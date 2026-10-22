"""One configurable AE on a fixed snapshot of completed FreeTacMan tracks."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import time

import torch

from tactilelatent.aligned_frames import file_hash,manifest_signature,write_json
from tactilelatent.marker_data import episode_key
from tactilelatent.marker_ae_data import snapshot,MarkerAEFrames
from tactilelatent.train_marker_autoencoders import train_trial,pin_teacher


def freeze_available(manifest,targets,output):
    selection_path=output/'available_manifest.json'
    if not selection_path.exists():
        payload=json.loads(Path(manifest).read_text());all_rows=[r for r in payload['episodes'] if r['source']=='freetacman']
        rows=[r for r in all_rows if (Path(targets)/r['source']/r['split']/episode_key(r)/'complete.json').exists()]
        if not any(r['split']=='train' for r in rows) or not any(r['split']=='val' for r in rows):
            raise ValueError('Need processed training and validation episodes')
        write_json(selection_path,{**payload,'episodes':rows})
        write_json(output/'selection.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
            original_manifest=manifest,original_data=manifest_signature(manifest),
            selection_manifest_sha256=file_hash(selection_path),selected_episodes=len(rows),total_episodes=len(all_rows),
            selected_by_split=dict(Counter(r['split'] for r in rows)),
            selected_frames_by_split={s:sum(r['frames'] for r in rows if r['split']==s) for s in ('train','val')},
            missing_episode_ids=[r['episode_id'] for r in all_rows if r not in rows],
            policy='Completed episode tracks at snapshot time; retain canonical train/validation split; no incremental changes during fitting'))
    pinned_path=output/'pinned_snapshot.json'
    if pinned_path.exists():
        frozen=json.loads(pinned_path.read_text());verified=snapshot(str(selection_path),targets,'freetacman')
        if (verified['source_snapshot_sha256']!=frozen['source_snapshot_sha256']
                or manifest_signature(manifest)!=frozen['selection']['original_data']):
            raise ValueError('Frozen available-frame snapshot changed')
        return frozen
    pinned=snapshot(str(selection_path),targets,'freetacman')
    pinned['selection']=json.loads((output/'selection.json').read_text());write_json(pinned_path,pinned)
    return pinned


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/marker_freetacman_available_16d.json');args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text());output=Path(cfg['output']);output.mkdir(parents=True,exist_ok=True)
    codes=cfg['sources'].get('freetacman',[])
    if set(cfg['sources'])!={'freetacman'} or len(codes)!=1:raise ValueError('Specify exactly one FreeTacMan code dimension')
    code=int(codes[0]);publish_parent=cfg.get('publish_parent',True)
    parent=Path('processing/runs/marker_autoencoders_v1')
    import fcntl
    lock=(output/'training.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if publish_parent:
        parent_lock=(parent/'training.lock').open('w');fcntl.flock(parent_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(4);torch.set_float32_matmul_precision('high');began=time.monotonic()
    def report(stage,**details):
        row=dict(stage=stage,pid=os.getpid(),source='freetacman',code_dim=code,scope='processed_episode_snapshot',
                 run=str(output),elapsed_seconds=time.monotonic()-began,updated_at=time.time())
        row.update(details);write_json(output/'status.json',row)
        if publish_parent:write_json(parent/'status.json',row)
        print(json.dumps(row),flush=True)
    try:
        plan=output/'plan.json'
        if plan.exists() and json.loads(plan.read_text())!=cfg:raise ValueError('Partial-data AE configuration changed')
        write_json(plan,cfg);report('freezing_available_marker_targets')
        if cfg.get('snapshot_from'):
            origin=Path(cfg['snapshot_from'])
            if origin.resolve()==output.resolve():raise ValueError('Snapshot origin must be a different run')
            for name in ('available_manifest.json','selection.json','pinned_snapshot.json'):
                source=origin/name;destination=output/name
                if destination.exists():
                    if file_hash(destination)!=file_hash(source):raise ValueError(f'Copied snapshot changed: {name}')
                else:shutil.copyfile(source,destination)
        pinned=freeze_available(cfg['manifest'],cfg['marker_targets'],output)
        report('loading_marker_vectors',selection=pinned['selection'])
        train=MarkerAEFrames(pinned,'train',cfg['frame_stride'],cfg['max_train_frames'],cfg['seed'])
        val=MarkerAEFrames(pinned,'val',cfg['frame_stride'],0,cfg['seed'])
        report('data_ready',train_frames=len(train),validation_frames=len(val),max_markers=train.shape[1],selected_episodes=len(pinned['episodes']))
        train_trial(cfg,'freetacman',code,pinned,train,val,output,cfg['final_epochs'],report)
        info=pin_teacher(output,'freetacman',code,pinned,train,cfg)
        info.update(training_scope='processed_episode_snapshot',selection=pinned['selection'],
                    selection_manifest=str(output/'available_manifest.json'),original_canonical_data=pinned['selection']['original_data'])
        local=json.loads((output/'teachers.json').read_text());local['sources']['freetacman']=info;write_json(output/'teachers.json',local)
        registry_path=output/'teachers.json'
        if publish_parent:
            registry_path=parent/'teachers.json';registry=json.loads(registry_path.read_text())
            if not {'ftp_unit','ftp_unit_bimanual','ftp_vla_touch'}<=set(registry['sources']):raise ValueError('Completed earlier teachers are missing')
            registry['sources']['freetacman']=info;registry['complete']=len(registry['sources'])==4
            registry['freetacman_handover']=str(output/'plan.json');write_json(registry_path,registry)
        report('complete',validation=info['validation'],teachers=str(registry_path),selected_episodes=len(pinned['episodes']))
    except Exception as error:
        report('failed',error=str(error));raise


if __name__=='__main__':main()
