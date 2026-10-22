"""Run the requested fresh 128D experiment after RGB migration is verified."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from tactilelatent.aligned_frames import manifest_signature,write_json

CONFIG=Path('configs/joint_vjepa21_spatial_current_128d.json')


def main():
    cfg=json.loads(CONFIG.read_text());root=Path(cfg['output']);root.mkdir(parents=True,exist_ok=True)
    lock=(root/'launcher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def report(stage,**details):
        row=dict(stage=stage,pid=os.getpid(),updated_at=time.time(),config=str(CONFIG),**details)
        write_json(root/'launch_status.json',row);print(json.dumps(row),flush=True)
    try:
        if (root/'last.pt').exists():raise FileExistsError('Fresh run already has a checkpoint; do not overwrite or implicitly resume')
        report('waiting_for_verified_rgb_migration',latent_dim=128,epochs=30,history_frames=7)
        ready=Path(cfg['storage_preparation']);migration=Path('processing/storage_v1')
        while not ready.exists():
            for name in ('status.json','finalization_status.json'):
                path=migration/name
                if path.exists() and json.loads(path.read_text()).get('stage')=='failed':
                    raise RuntimeError(f'Prerequisite failed; see {path}')
            time.sleep(10)
        prepared=json.loads(ready.read_text());status=json.loads((migration/'status.json').read_text())
        if (not prepared.get('passed') or status['stage']!='complete'
                or prepared['heldout_source']!='ftp_motiontrans' or len(prepared['sources'])!=10
                or {'freetacman','ftp_motiontrans'}&set(prepared['sources'])
                or prepared['data']!=manifest_signature(cfg['manifest'])):
            raise ValueError('Expected the verified ten-source manifest without FreeTacMan or MotionTrans')
        if cfg!=json.loads(CONFIG.read_text()):raise ValueError('Queued training configuration changed')
        command=['bash','scripts/run_vjepa_markers.sh','--config',str(CONFIG)]
        report('gpu_preflight',sources=prepared['sources'])
        with (root/'preflight.log').open('a') as log:
            subprocess.run(command+['--preflight-only'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
        checked=json.loads((root/'preflight.json').read_text())
        if not checked.get('passed') or checked['config']!=cfg:raise ValueError('Production preflight failed')
        if any(row['latent_shape'][1:]!=[1,128] for row in checked['rows']):raise ValueError('Expected a single 128D current latent')
        report('starting_training',latent_dim=128,epochs=30,preflight_seconds=checked['seconds'])
        log=(root/'nohup.log').open('a');os.dup2(log.fileno(),1);os.dup2(log.fileno(),2)
        (root/'launcher.pid').write_text(str(os.getpid())+'\n')
        os.execvp(command[0],command)
    except BaseException as error:
        report('failed',error=str(error));raise


if __name__=='__main__':main()
