"""Launch a fresh deterministic current-frame run with a configured latent size.

The existing 128D experiment is untouched. Each run has its own preflight,
checkpoints and process lock, so a second size can train alongside it.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from tactilelatent.aligned_frames import manifest_signature,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args();cfg=json.loads(args.config.read_text())
    root=Path(cfg['output']);root.mkdir(parents=True,exist_ok=True)
    lock=(root/'launcher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def report(stage,**details):
        row=dict(stage=stage,pid=os.getpid(),updated_at=time.time(),config=str(args.config),
                 latent_dim=cfg['latent_dim'],total_epochs=cfg['epochs'],**details)
        write_json(root/'launch_status.json',row);print(json.dumps(row),flush=True)
    try:
        if (root/'last.pt').exists():raise FileExistsError('Fresh run already has a checkpoint; do not implicitly resume')
        if (cfg.get('prediction_mode')!='current_frame' or cfg.get('latent_mode')!='deterministic'
                or cfg.get('stochastic') or cfg['beta_max']!=0):
            raise ValueError('Expected deterministic current-frame training without KL')
        ready=json.loads(Path(cfg['storage_preparation']).read_text())
        status=json.loads(Path('processing/storage_v1/status.json').read_text())
        if (not ready.get('passed') or status['stage']!='complete'
                or ready['data']!=manifest_signature(cfg['manifest'])
                or ready['heldout_source']!='ftp_motiontrans' or len(ready['sources'])!=10
                or {'freetacman','ftp_motiontrans'}&set(ready['sources'])):
            raise ValueError('Expected verified ten-source data without FreeTacMan or MotionTrans')
        command=['bash','scripts/run_vjepa_markers.sh','--config',str(args.config)]
        report('gpu_preflight',sources=ready['sources'])
        with (root/'preflight.log').open('a') as log:
            subprocess.run(command+['--preflight-only'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
        checked=json.loads((root/'preflight.json').read_text())
        if not checked.get('passed') or checked['config']!=cfg:
            raise ValueError('Production GPU preflight did not pass')
        if any(row['latent_shape'][1:]!=[1,cfg['latent_dim']] for row in checked['rows']):
            raise ValueError('Expected a single current-frame latent of the configured size')
        report('starting_training',preflight_seconds=checked['seconds'])
        log=(root/'nohup.log').open('a');os.dup2(log.fileno(),1);os.dup2(log.fileno(),2)
        (root/'launcher.pid').write_text(str(os.getpid())+'\n')
        os.execvp(command[0],command)
    except BaseException as error:
        report('failed',error=str(error));raise


if __name__=='__main__':main()
