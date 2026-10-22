#!/usr/bin/env python3
"""SSH-independent progress log for the supervised mirror preparation job."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import time

from tactilelatent.aligned_frames import write_json


def read(path):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return {}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('processing/bimanual_mirror_v1'))
    p.add_argument('--interval',type=float,default=30.)
    a=p.parse_args();a.root.mkdir(parents=True,exist_ok=True)
    lock=(a.root/'monitor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        result=subprocess.run(['systemctl','--user','show','tactilelatent-bimanual-mirror-v1.service',
            '-p','ActiveState','-p','SubState','-p','MainPID','-p','Result'],capture_output=True,text=True)
        service=dict(line.split('=',1) for line in result.stdout.splitlines() if '=' in line)
        status=read(a.root/'status.json');ready=read(a.root/'ready.json')
        verification=read(a.root/'verification/report.json')
        snapshot=dict(updated_at=time.time(),service=service,
            progress={k:status[k] for k in ('stage','source','epoch','total_epochs','episodes_done','episodes',
                'accepted_checkpoint','cache_seconds') if k in status},
            ready_sources=[s for s,r in ready.get('sources',{}).items() if r.get('ready')],
            blocked={s:r.get('reason',r.get('stage')) for s,r in ready.get('sources',{}).items() if not r.get('ready')},
            verification_passed=verification.get('passed',False),production_joint_run_started=False)
        pressure=read(Path('processing/data_expansion_v1/teachers/egopressure_balanced_v2/status.json'))
        snapshot['egopressure_uv_teacher']={k:pressure[k] for k in ('stage','epoch','total_epochs') if k in pressure}
        write_json(a.root/'monitor_status.json',snapshot)
        print(json.dumps(snapshot,allow_nan=False),flush=True)
        if service.get('ActiveState')=='failed' or (service.get('ActiveState')=='inactive' and not verification.get('passed')):
            snapshot['result']='failed_or_interrupted'
            log=subprocess.run(['journalctl','--user','-u','tactilelatent-bimanual-mirror-v1.service',
                '-n','30','--no-pager'],capture_output=True,text=True)
            (a.root/'failure.log').write_text(log.stdout+log.stderr)
            write_json(a.root/'background_result.json',snapshot)
            return
        if verification.get('passed') and service.get('SubState')=='exited':
            snapshot['result']='complete' if ready.get('complete') else 'completed_verified_sources_with_explicit_blocks'
            write_json(a.root/'background_result.json',snapshot)
            print(json.dumps(dict(final_result=snapshot['result'],ready_sources=snapshot['ready_sources'],
                blocked=snapshot['blocked'],joint_training_started=False)),flush=True)
            return
        time.sleep(max(a.interval,1.))


if __name__=='__main__':main()
