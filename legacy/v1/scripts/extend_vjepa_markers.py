"""Resume the marker-code trainer at its next committed epoch, with a longer LR tail."""
import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time

import torch

from tactilelatent.aligned_frames import file_hash,write_json
from tactilelatent.train_alignment import extension_config
from tactilelatent.train_vjepa_markers import prepare,check_epoch_extension
from stop_spatial_vjepa_after_epoch import committed_record,identity


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',required=True);args=parser.parse_args()
    request_path=Path(args.request);request=json.loads(request_path.read_text())
    cfg=request['original_config'];root=Path(cfg['output']);pid=request['trainer_pid']
    import fcntl
    lock=(root/'extension.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def report(stage,**details):
        row=dict(stage=stage,controller_pid=os.getpid(),previous_trainer_pid=pid,
            target_epochs=request['target_epochs'],after_epoch=request['after_epoch'],updated_at=time.time(),**details)
        write_json(root/'extension_status.json',row);print(json.dumps(row),flush=True)
    try:
        if file_hash(Path(__file__))!=request['controller_sha256']:raise ValueError('Extension controller changed')
        report('waiting_for_complete_epoch_checkpoint')
        while committed_record(root,request['after_epoch']) is None:
            if identity(pid)!=request['trainer_start_ticks']:raise RuntimeError('Original trainer exited before handover')
            time.sleep(.5)
        # Freeze the original group at a fully saved boundary before inspecting
        # or modifying any shared training artifacts.
        if identity(pid)!=request['trainer_start_ticks'] or os.getpgid(pid)!=pid:
            raise RuntimeError('Original trainer identity/process group changed')
        os.killpg(pid,signal.SIGSTOP)
        try:
            state=torch.load(root/'last.pt',map_location='cpu',weights_only=False)
            extended=extension_config(cfg,state,request['target_epochs'])
            request['lr_extension']=extended['lr_extension']
            torch.set_num_threads(4)
            _,_,_,mc,provenance=prepare(extended)
            provenance['backbone']=state['provenance']['backbone']
            check_epoch_extension(state,extended,mc,provenance,request)
            if not (root/f"viewer_epochs/epoch_{request['after_epoch']:03d}.pt").exists():
                raise ValueError('Complete epoch viewer snapshot is missing')
            if file_hash(root/'preflight.json')!=request['original_preflight_sha256']:
                raise ValueError('Original preflight changed')
            if json.loads(Path(request['config_path']).read_text())!=cfg:raise ValueError('Original config changed')
            write_json(request_path,request)
            backup=root/'before_epoch_extension'/f"epoch_{request['after_epoch']:03d}.pt"
            if not backup.exists():backup.hardlink_to(root/f"epoch_{request['after_epoch']:03d}.pt")
            del state
        except BaseException:
            os.killpg(pid,signal.SIGCONT)
            raise
        report('handover',anchor_lr=extended['lr_extension']['anchor_lr'])
        os.killpg(pid,signal.SIGTERM);os.killpg(pid,signal.SIGCONT)
        for _ in range(100):
            path=Path(f'/proc/{pid}/status')
            if identity(pid)!=request['trainer_start_ticks'] or (path.exists() and 'State:\tZ' in path.read_text()):break
            time.sleep(.1)
        else:raise RuntimeError('Original trainer did not exit; refusing concurrent resume')
        write_json(request['config_path'],extended);write_json(root/'config.json',extended)
        (root/'launcher.pid').write_text(str(os.getpid())+'\n')
        report('resuming',completed_epochs=request['after_epoch'],new_trainer_pid=os.getpid())
        command=[sys.executable,'-u','-m','tactilelatent.train_vjepa_markers','--config',request['config_path'],
                 '--resume','--extension-plan',str(request_path)]
        log=(root/'nohup.log').open('a');os.dup2(log.fileno(),1);os.dup2(log.fileno(),2)
        os.execv(sys.executable,command)
    except BaseException as error:
        report('failed',error=str(error));raise


if __name__=='__main__':main()
