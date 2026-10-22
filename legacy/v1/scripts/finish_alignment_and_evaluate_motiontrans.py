"""Stop the alignment service after a saved epoch and evaluate held-out MotionTrans."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from tactilelatent.aligned_frames import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--service', required=True)
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--epoch', type=int, required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run, output = Path(args.run), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    status_path = run / 'stop_status.json'
    stopped = False

    def status(stage, **fields):
        record = {'stage': stage, 'stop_after_epoch': args.epoch,
                  'training_stopped': stopped, 'output': str(output), **fields}
        write_json(status_path, record)
        print(json.dumps(record), flush=True)

    status('waiting_for_checkpoint')
    try:
        while True:
            records = []
            for line in (run / 'metrics.jsonl').read_text().splitlines():
                try:
                    records.append(json.loads(line))
                except ValueError:
                    continue
            checkpoint = run / f'epoch_{args.epoch:03d}.pt'
            if any(row['epoch'] == args.epoch for row in records) and checkpoint.exists():
                state = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
                if (state['epoch'] + 1 != args.epoch
                        or state.get('model_type') != 'joint_sensor_alignment_v1'
                        or state['config']['heldout_source'] != 'ftp_motiontrans'):
                    raise ValueError('Checkpoint is not the requested completed MotionTrans-holdout epoch')
                manifest = state['config']['manifest']
                del state
                current = int(subprocess.check_output(['systemctl', '--user', 'show',
                    args.service, '--property=MainPID', '--value'], text=True).strip() or 0)
                if current not in (0, args.pid):
                    raise RuntimeError('Training service changed process; refusing to stop another job')
                if current:
                    subprocess.run(['systemctl', '--user', 'stop', args.service], check=True)
                stopped = True
                snapshot = output / 'checkpoint.pt'
                if snapshot.exists():
                    if not os.path.samefile(checkpoint, snapshot):
                        raise FileExistsError(f'Different evaluation snapshot already exists: {snapshot}')
                else:
                    os.link(checkpoint, snapshot)
                status('encoder_stopped', checkpoint=str(snapshot))
                if not torch.cuda.is_available():
                    raise RuntimeError('CUDA is unavailable for MotionTrans evaluation; epoch checkpoint is safely saved')
                commands = {}
                for name, widths in (('shallow', [512]), ('deep', [512, 512])):
                    destination = output.with_name(output.name + '_' + name)
                    destination.mkdir(parents=True, exist_ok=True)
                    commands[name] = [sys.executable, '-u', '-m', 'tactilelatent.evaluate_holdout_decoder',
                        '--checkpoint', str(snapshot), '--manifest', manifest,
                        '--source', 'ftp_motiontrans', '--output', str(destination),
                        '--cache-dir', str(output / 'features'),
                        '--allow-intermediate', '--hidden-dims', *map(str, widths), '--samples', '64',
                        '--epochs', '100', '--patience', '10', '--workers', '4']
                write_json(output / 'launch.json', {'commands': commands,
                    'completed_encoder_epoch': args.epoch, 'manifest': manifest,
                    'controller_pid': os.getpid()})
                results = {}
                for name, command in commands.items():
                    destination = output.with_name(output.name + '_' + name)
                    with (destination / 'evaluation.log').open('a') as log:
                        evaluator = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                                     stdin=subprocess.DEVNULL)
                        status('evaluating', head=name, evaluator_pid=evaluator.pid, command=command)
                        code = evaluator.wait()
                    if code:
                        raise RuntimeError(f'{name} MotionTrans head evaluation exited with code {code}; see {destination}/evaluation.log')
                    results[name] = json.loads((destination / 'results.json').read_text())
                write_json(output / 'comparison.json', {'encoder_epoch': args.epoch,
                    'same_encoder_features': True, 'source': 'ftp_motiontrans',
                    'heads': {name: {k: result[k] for k in ('head', 'head_parameters',
                        'selected_epochs', 'train_clips', 'validation')}
                        for name, result in results.items()}})
                status('complete', results=str(output / 'comparison.json'))
                return
            if not Path(f'/proc/{args.pid}').exists():
                raise RuntimeError('Trainer exited before saving the requested epoch')
            time.sleep(2)
    except Exception as error:
        status('failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
