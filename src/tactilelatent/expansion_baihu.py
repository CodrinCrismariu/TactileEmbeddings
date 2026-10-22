"""Pilot, convert, verify artifact-only consumers, then retire Baihu media."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from .aligned_frames import file_hash,json_hash,write_json
from .data_expansion import (ROOT,BAIHU,CAMERAS,SURFACES,discover_baihu,
    extract_baihu,marker_arrays,marker_batch,encode_rgb,video_images,
    video_info,safe_delete,report,download)
from .expansion_models import ExpansionMarkerAutoencoder
from .expansion_render import marker_payload

BAIHU_REV='d7dfecfb34b95fd1b72a050174f0e96916cc290b'


def upgrade_initial_pilot(row):
    """Reacquire exact original tactile bytes for the initial reference-gated
    pilot, then replace its derived targets with the temporal tracker.
    This happens before any production teacher/code cache is trained.
    """
    path=Path(row['marker_path']);complete=path.with_name('complete.json')
    if not complete.exists():return
    old=json.loads(complete.read_text())
    if old.get('association')=='temporal-centroid-v1':return
    accepted=path.with_name('accepted.json')
    for source in old['source_files']:
        p=Path(source['path'])
        if not p.exists():
            root=BAIHU.parents[1]
            download('yxma/tactile-video-pretrain-nc',BAIHU_REV,str(p.relative_to(root)),root)
        if file_hash(p)!=source['sha256']:raise ValueError('Reacquired pilot tactile bytes differ from original')
    backup=path.with_name('tracks.reference_pilot.npz');metadata=path.with_name('complete.reference_pilot.json')
    shutil.copyfile(path,backup);shutil.copyfile(complete,metadata)
    complete.unlink()
    try:
        extract_baihu(row)
        if accepted.exists():accepted.unlink()
        backup.unlink()
    except Exception:
        shutil.copyfile(backup,path);shutil.copyfile(metadata,complete)
        raise


def consumer_preflight(row):
    """Disposable teacher update + disk code round-trip + grid rendering.

    Probe codes are never published as frozen-teacher training targets.
    Every operation reads only tracks and static geometry.
    """
    torch.set_num_threads(1);torch.manual_seed(45)
    indices=np.unique([0,row['frames']//2,row['frames']-1])
    batch=marker_batch(row,indices);d,info=marker_arrays(row)
    y=batch['marker_target'];valid=batch['marker_valid'];n=len(y)
    ref=batch['marker_reference'][None].expand(n,-1,-1)
    surface=batch['marker_surface'][None].expand(n,-1)
    count=info['loss_statistics']['coordinate_count']
    scale=max((info['loss_statistics']['square_sum']/max(count,1))**.5,1e-6)
    ae=ExpansionMarkerAutoencoder(latent_dim=32,width=128,queries=8,heads=4,displacement_scale=scale)
    optimizer=torch.optim.AdamW(ae.parameters(),lr=.001)
    pred=ae(y,valid,ref,surface);loss=ae.loss(pred,y,valid,surface)
    if not torch.isfinite(loss):raise ValueError('Nonfinite four-surface AE preflight')
    optimizer.zero_grad();loss.backward();optimizer.step()
    ae.eval()
    with torch.no_grad():code=ae.encode(y,valid,ref,surface);pred=ae.decode(code,ref,surface)
    probe=Path(row['marker_path']).with_name('probe_codes.npy')
    np.save(probe,code.numpy());torch.testing.assert_close(torch.from_numpy(np.load(probe)),code,rtol=0,atol=0);probe.unlink()
    payload=marker_payload(row,d,int(indices[-1]),pred[-1].numpy()*batch['marker_spacing'].numpy()[:,None])
    json.dumps(payload,allow_nan=False)
    if len(payload['optical_panels'])!=4 or code.shape!=(n,32):raise ValueError('Four-surface consumer preflight failed')
    return dict(passed=True,frames=indices.tolist(),loss=float(loss.detach()),code_shape=list(code.shape),
        training_reads='marker artifacts only',viewer_background='geometry only',
        probe_teacher='disposable; not a trained or published teacher')


def process_episode(row):
    torch.set_num_threads(1);began=time.monotonic()
    upgrade_initial_pilot(row)
    receipt=Path(row['marker_path']).with_name('accepted.json')
    if receipt.exists():
        accepted=json.loads(receipt.read_text());marker_arrays(row)
        if accepted['row_sha256']!=json_hash(row):raise ValueError('Accepted Baihu episode changed')
        for info in accepted['rgb'].values():
            if file_hash(info['path'])!=info['sha256']:raise ValueError('Accepted RGB changed')
        # Finish any interrupted deletion using the previously verified receipt.
        if accepted['marker_sha256']!=file_hash(row['marker_path']):raise ValueError('Accepted marker artifact changed')
        reclaimed=safe_delete(accepted['retire'],row['original_dir'],ROOT/'baihu_deletions.jsonl','accepted artifact-only Baihu episode',
            expected={r['path']:r['sha256'] for r in accepted['originals']})
        return dict(episode_id=row['episode_id'],resumed=True,reclaimed_bytes=reclaimed,seconds=time.monotonic()-began)
    info=extract_baihu(row);rgb={};folder=Path(row['original_dir'])
    for slot,path in row['views'].items():
        original=folder/(CAMERAS[slot]+'.mp4')
        if Path(path).with_suffix('.receipt.json').exists():
            rgb[slot]=encode_rgb(iter(()),path,row['frames'])
            continue
        source=video_info(original)
        if source['frames']!=row['frames'] or source['fps']!=30:raise ValueError('Baihu RGB frame contract differs from tactile')
        rgb[slot]=encode_rgb(video_images(original),path,row['frames'])
    preflight=consumer_preflight(row)
    retire=[folder/(s+'.mp4') for s in SURFACES]
    retire += [folder/(CAMERAS[s]+'.mp4') for s in row['views']]
    retire += [folder/n for n in ('bimanual.mp4','head_depth.mkv','head_ir_left.mp4','head_ir_right.mp4')]
    accepted=dict(row_sha256=json_hash(row),marker_sha256=info['tracks_sha256'],rgb=rgb,
        preflight=preflight,retire=[str(p) for p in retire if p.exists()],
        originals=[dict(path=str(p),bytes=p.stat().st_size,sha256=file_hash(p)) for p in retire if p.exists()])
    write_json(receipt,accepted)
    reclaimed=safe_delete(accepted['retire'],folder,ROOT/'baihu_deletions.jsonl','accepted artifact-only Baihu episode',
        expected={r['path']:r['sha256'] for r in accepted['originals']})
    # Reopen AFTER removal too: consumer success must not rely on original paths.
    marker_arrays(row)
    json.dumps(marker_payload(row,marker_arrays(row)[0],row['frames']-1),allow_nan=False)
    return dict(episode_id=row['episode_id'],reclaimed_bytes=reclaimed,
        frames=row['frames'],valid_fraction=info['valid_fraction'],seconds=time.monotonic()-began)


def run(workers=2,pilot_only=False):
    rows=discover_baihu();pilot=[]
    for task in sorted({r['task'] for r in rows}):pilot.extend([r for r in rows if r['task']==task][:2])
    keys={r['episode_id'] for r in pilot};stages=[('pilot',pilot)]
    if not pilot_only:stages.append(('full',[r for r in rows if r['episode_id'] not in keys]))
    ready=[];failures=[];began=time.monotonic();reclaimed=0;work_seconds=0.;work_frames=0
    def publish_progress(stage):
        ready_ids={r['episode_id'] for r in ready}
        remaining_frames=sum(r['frames'] for r in rows if r['episode_id'] not in ready_ids)
        write_json(ROOT/'baihu_available.json',dict(episodes=ready,complete=len(ready)==len(rows),failures=failures))
        report('baihu',stage=stage,completed_episodes=len(ready),episodes=len(rows),failures=failures,
            reclaimed_bytes=reclaimed,elapsed_seconds=time.monotonic()-began,
            eta_seconds=work_seconds/work_frames/workers*remaining_frames if work_frames and not failures else None,
            workers=workers,new_frames_processed=work_frames)
    for stage,subset in stages:
        publish_progress(stage)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures={pool.submit(process_episode,row):row for row in subset}
            for future in as_completed(futures):
                row=futures[future]
                try:
                    result=future.result();reclaimed+=result['reclaimed_bytes'];ready.append(row)
                    if not result.get('resumed'):
                        work_seconds+=result['seconds'];work_frames+=row['frames']
                except Exception as e:
                    failures.append(dict(episode_id=row['episode_id'],error=f'{type(e).__name__}: {e}'))
                publish_progress(stage)
        if failures:
            publish_progress('blocked_by_failed_episodes')
            raise RuntimeError(f'{len(failures)} Baihu episodes failed; their source imagery is retained')
    if not pilot_only:
        # Task-level montages/assets duplicate tactile imagery and are explicitly
        # retired only once all canonical episodes have accepted marker artifacts.
        assets=BAIHU.parents[1]/'assets'
        if assets.is_dir():
            reclaimed+=safe_delete([p for p in assets.rglob('*') if p.is_file() and p.suffix.lower() in ('.mp4','.png','.jpg')],
                assets,ROOT/'baihu_deletions.jsonl','all canonical Baihu episodes verified; retire tactile previews')
        # Early release sample_* folders contain short preview copies rather
        # than complete canonical episodes. Retire only those with a uniquely
        # identified, fully accepted canonical counterpart.
        unmatched=[]
        for folder in BAIHU.glob('sample_*'):
            if not folder.is_dir():continue
            prefix=folder.name.rsplit('_',1)[-1]
            matches=[r for r in rows if r['resource_id'].startswith(prefix)]
            if len(matches)!=1:
                unmatched.append(str(folder));continue
            files=[p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in ('.mp4','.mkv','.png','.jpg','.jpeg')]
            reclaimed+=safe_delete(files,folder,ROOT/'baihu_deletions.jsonl','preview of fully accepted canonical Baihu episode '+matches[0]['resource_id'])
        write_json(ROOT/'baihu_preview_cleanup.json',dict(unmatched_preview_directories=unmatched,complete=not unmatched))
    publish_progress('pilot_complete' if pilot_only else 'complete')
