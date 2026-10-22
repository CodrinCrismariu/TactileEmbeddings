"""Hand over the running alignment job after its next complete checkpoint."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from tactilelatent.aligned_frames import file_hash, write_json
from tactilelatent.train_alignment import extension_config, check_resume_contract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--service', required=True)
    parser.add_argument('--pid', required=True, type=int)
    parser.add_argument('--after-epoch', required=True, type=int)
    parser.add_argument('--epochs', required=True, type=int)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    output = Path(cfg['output'])
    status_path = output / 'extension_status.json'
    def status(stage, **fields):
        write_json(status_path, {'stage': stage, 'target_epochs': args.epochs,
            'after_epoch': args.after_epoch, 'previous_pid': args.pid, **fields})
    status('waiting_for_checkpoint')
    print(json.dumps({'event': 'extension_queued', 'target_epochs': args.epochs,
                      'after_epoch': args.after_epoch}), flush=True)
    try:
        while True:
            records = []
            for line in (output / 'metrics.jsonl').read_text().splitlines():
                try:
                    records.append(json.loads(line))
                except ValueError:
                    continue
            completed = records[-1]['epoch'] if records else 0
            if completed >= args.after_epoch:
                state = torch.load(output / 'last.pt', map_location='cpu', weights_only=False, mmap=True)
                if state['epoch'] + 1 != completed:
                    time.sleep(1)
                    continue
                extended = extension_config(cfg, state, args.epochs)
                # Verify the permitted migration before stopping the original job.
                provenance = {**state['provenance'], 'trainer_sha256': file_hash(
                    Path(__file__).parents[1] / 'src/tactilelatent/train_alignment.py')}
                check_resume_contract(state, extended, provenance, extending=True)
                if not (output / 'viewer_epochs' / f'epoch_{completed:03d}.pt').exists():
                    time.sleep(1)
                    continue
                del state
                current = int(subprocess.check_output(['systemctl', '--user', 'show',
                    args.service, '--property=MainPID', '--value'], text=True).strip() or 0)
                if current not in (0, args.pid):
                    raise RuntimeError('Original training service changed process; refusing handover')
                if current:
                    subprocess.run(['systemctl', '--user', 'stop', args.service], check=True)
                command = [sys.executable, '-u', 'scripts/train_joint_alignment.py',
                           '--config', args.config, '--resume', '--extend-epochs', str(args.epochs),
                           '--device', 'cuda']
                (output / 'trainer.pid').write_text(str(os.getpid()) + '\n')
                status('handover', completed_epochs=completed, resumed_pid=os.getpid(), command=command)
                print(json.dumps({'event': 'extension_handover', 'completed_epochs': completed,
                                  'target_epochs': args.epochs, 'command': command}), flush=True)
                os.execv(sys.executable, command)
            if not Path(f'/proc/{args.pid}').exists():
                raise RuntimeError('Original trainer exited before the requested checkpoint boundary')
            time.sleep(2)
    except Exception as error:
        status('failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
