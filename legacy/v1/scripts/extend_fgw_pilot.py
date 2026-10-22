"""Extend both FGW placements after the active worker saves its next epoch."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import torch

from tactilelatent.aligned_frames import file_hash, write_json
from tactilelatent.fgw_pilot import joint_config, signature
from tactilelatent.train_alignment import extension_config, check_resume_contract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--after-epoch', type=int, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    args = parser.parse_args()
    root = Path(args.root)
    previous = json.loads((root / 'status.json').read_text())
    controller, worker = previous['controller_pid'], previous['worker_pid']
    status_path = root / 'extension_status.json'
    output = root / 'fgw_visual_partial_seed45'

    def status(stage, **fields):
        write_json(status_path, {'stage': stage, 'target_epochs': args.epochs,
            'after_epoch': args.after_epoch, 'previous_controller': controller,
            'previous_worker': worker, 'extension_pid': os.getpid(), **fields})

    try:
        controller_cmd = Path(f'/proc/{controller}/cmdline').read_bytes()
        worker_cmd = Path(f'/proc/{worker}/cmdline').read_bytes()
        if (b'tactilelatent.fgw_pilot' not in controller_cmd or
                b'tactilelatent.train_alignment' not in worker_cmd or
                os.getpgid(controller) != controller or os.getpgid(worker) != controller):
            raise ValueError('Unexpected active controller or worker; refusing handover')
        os.kill(controller, signal.SIGSTOP)  # Worker continues to the checkpoint boundary.
        status('waiting_for_checkpoint')
        while True:
            state = torch.load(output / 'last.pt', map_location='cpu', weights_only=False, mmap=True)
            completed = state['epoch'] + 1
            viewer = output / 'viewer_epochs' / f'epoch_{completed:03d}.pt'
            if completed >= args.after_epoch and viewer.exists():
                break
            del state
            if not Path(f'/proc/{worker}').exists():
                raise RuntimeError('Worker exited before the requested checkpoint')
            time.sleep(10)

        cfg = json.loads((root / 'configs/visual.json').read_text())
        extended = extension_config(cfg, state, args.epochs)
        provenance = {**state['provenance'], 'trainer_sha256': file_hash(
            Path(__file__).parents[1] / 'src/tactilelatent/train_alignment.py')}
        check_resume_contract(state, extended, provenance, extending=True)
        spec = json.loads((root / 'pilot.json').read_text())
        if spec['implementation_sha256'] != signature():
            raise ValueError('Implementation changed after the extension was queued')
        if spec['completed_epoch'] != args.epochs:
            raise ValueError('Pilot endpoint differs from the requested extension')
        checkpoint = root / 'epoch30_extension_backup' / 'checkpoint_before_extension.pt'
        if not checkpoint.exists():
            checkpoint.hardlink_to(viewer)
        del state

        status('handover', completed_epochs=completed)
        os.killpg(controller, signal.SIGTERM)
        os.kill(controller, signal.SIGCONT)  # Deliver pending TERM to the stopped controller.
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            result = subprocess.run(['ps', '-eo', 'pgid=,stat='], capture_output=True, text=True, check=True)
            alive = [line for line in result.stdout.splitlines()
                     if line.split()[0] == str(controller) and not line.split()[1].startswith('Z')]
            if not alive:
                break
            time.sleep(1)
        else:
            raise RuntimeError('Original training process group did not exit')

        write_json(root / 'configs/visual.json', extended)
        status('resumed', completed_epochs=completed, controller_pid=os.getpid(),
               anchor_lr=extended['lr_extension']['anchor_lr'])
        (root / 'controller.pid').write_text(str(os.getpid()) + '\n')
        log = (root / 'controller.log').open('a')
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        os.execv(sys.executable, [sys.executable, '-u', '-m', 'tactilelatent.fgw_pilot',
                                'run', '--root', str(root)])
    except BaseException as error:
        status('failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
