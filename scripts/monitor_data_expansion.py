#!/usr/bin/env python3
"""Continuously publish a readable status page and aggregate JSON snapshot."""
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import time

from tactilelatent.aligned_frames import write_json

ROOT=Path('processing/data_expansion_v1')


def snapshot():
    jobs={}
    for job in ('baihu','egopressure','feel','teachers','publish'):
        p=ROOT/f'{job}_status.json'
        if p.exists():jobs[job]=json.loads(p.read_text())
    # The repaired pressure teacher has its own service and completion index.
    # Its final cache publication supersedes the older worker's last heartbeat.
    index=ROOT/'codes/egopressure_balanced_v2/index.json'
    quality=ROOT/'teachers/egopressure_balanced_v2/quality.json'
    selection=ROOT/'source_teacher_overrides.json'
    if selection.exists():
        choice=json.loads(selection.read_text()).get('egopressure',{})
        index=Path(choice.get('code_index',index));quality=Path(choice.get('quality',quality))
    if index.exists() and quality.exists():
        codes=json.loads(index.read_text());accepted=json.loads(quality.read_text())
        if (codes.get('complete') and accepted.get('accepted') and
                codes.get('teacher_sha256')==accepted.get('teacher_sha256')):
            n=len(codes['episodes'])
            jobs['egopressure_teacher']=dict(stage='complete',completed_episodes=n,episodes=n,
                source='balanced v2 teacher; task-only code selection' if codes.get('selection_version')
                       else 'balanced v2 teacher and sensor-code cache')
            if jobs.get('teachers',{}).get('source')=='egopressure':
                jobs['teachers']=dict(stage='partial_complete',completed_episodes=n,episodes=n,
                    source='EgoPressure complete; Baihu teacher publication pending',
                    supersedes_worker_heartbeat=jobs['teachers'].get('updated_at'))
    retired={}
    p=ROOT/'baihu_deletions.jsonl'
    if p.exists():
        for line in p.read_text().splitlines():
            try:
                r=json.loads(line)
                if r['event']=='deleted':retired[r['path']]=r['bytes']
            except (ValueError,KeyError):continue
    now=time.time();data=dict(updated_at=now,free_GiB=shutil.disk_usage(ROOT).free/2**30,
        reserve_GiB=100,baihu_original_bytes_retired=sum(retired.values()),jobs=jobs,
        joint_training_started=False)
    write_json(ROOT/'status.json',data)
    lines=['# Dataset expansion status','',f'Updated: {datetime.now(timezone.utc).isoformat()}','',
        f'Free space: **{data["free_GiB"]:.1f} GiB**; reserve: **100 GiB**.',
        f'Baihu original media retired: **{sum(retired.values())/2**30:.2f} GiB** (unique original files; derived artifacts consume additional space).','',
        '| Job | Stage | Progress | Details |','| --- | --- | --- | --- |']
    for job,r in jobs.items():
        count=r.get('completed_episodes');total=r.get('episodes',r.get('total_episodes'))
        progress=f'{count}/{total} episodes' if count is not None and total is not None else str(r.get('epoch','—'))
        details=r.get('sequence',r.get('source',r.get('reason','')))
        if r.get('failures'):details=f'{len(r["failures"])} episode failures; originals retained'
        if r.get('eta_seconds') is not None:details+=f' · estimated {r["eta_seconds"]/3600:.1f} hours remaining'
        if r.get('cuda_available') is not None:details+=f' · CUDA available: {r["cuda_available"]}'
        lines.append(f'| {job} | {r.get("stage","unknown")} | {progress} | {details} |')
    lines+=['','No joint visual-model training is launched by this pipeline.',
        'EgoPressure and FEEL downloads are resumable. Only verified episode media is deleted.',
        'The original FEEL P12 job is a pilot. Its full expansion and the other active acquisitions are tracked in the separate queue below. TacQuad and T-Dex were removed by the user. MotionTrans stays held out; FreeTacMan stays excluded.','',
        '[Additional downloads and processing: T-Rex, FEEL, HRDexDB](../data_queue_v2/STATUS.md).','',
        'Logs: `*_progress.jsonl`, `egopressure.log`, `feel.log`; deletion records: `baihu_deletions.jsonl`.','']
    tmp=ROOT/'STATUS.tmp';tmp.write_text('\n'.join(lines));tmp.replace(ROOT/'STATUS.md')


if __name__=='__main__':
    ROOT.mkdir(parents=True,exist_ok=True)
    while True:
        snapshot();time.sleep(15)
