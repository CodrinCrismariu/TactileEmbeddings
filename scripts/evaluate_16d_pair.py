"""Evaluate the matched 16D pair on MotionTrans at one completed epoch."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from tactilelatent.aligned_frames import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='processing/runs/joint_16d_pair_v1')
    parser.add_argument('--epoch', type=int, default=10)
    args = parser.parse_args()
    pair = Path(args.root)
    root = pair / 'evaluations' / f'epoch_{args.epoch:03d}'
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    began = time.monotonic()
    report_name = f'motiontrans_epoch_{args.epoch:03d}_evaluation'
    results = {}
    try:
        for name in ['no_fgw', 'visual_fgw']:
            run = pair / name
            command = [sys.executable, '-u', 'scripts/evaluate_alignment_checkpoint_sweep.py',
                       '--run', str(run), '--epochs', str(args.epoch), '--report-name', report_name,
                       '--head-epochs', '100', '--patience', '10', '--samples', '64',
                       '--seed', '45', '--workers', '2']
            with (root / f'{name}.log').open('a') as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                           stdin=subprocess.DEVNULL)
                write_json(root / 'status.json', {'stage': 'running', 'run': name,
                    'epoch': args.epoch, 'worker_pid': process.pid, 'controller_pid': os.getpid(),
                    'elapsed_seconds': time.monotonic() - began, 'command': command,
                    'completed_runs': list(results)})
                try:
                    code = process.wait()
                except BaseException:
                    process.terminate(); process.wait(); raise
            if code:
                raise RuntimeError(f'{name} evaluation exited {code}; see {root}/{name}.log')
            summary = json.loads((run / 'evaluations' / report_name / 'summary.json').read_text())
            if summary['completed_evaluations'] != 2 or summary['failures']:
                raise ValueError('Both transfer heads must complete without failures')
            results[name] = summary['results']
            write_json(root / 'summary.json', {'epoch': args.epoch, 'results': results})
        comparisons = {}
        for head in ['shallow', 'deep']:
            a = next(r for r in results['no_fgw'] if r['head'] == head)
            b = next(r for r in results['visual_fgw'] if r['head'] == head)
            if (a['latent_dim'] != 16 or b['latent_dim'] != 16 or
                    a['train_clips'] != b['train_clips'] or a['validation_clips'] != b['validation_clips']):
                raise ValueError('Paired evaluations have different dimensions or clip counts')
            comparisons[head] = {'no_fgw_mae': a['deterministic_mae'], 'fgw_mae': b['deterministic_mae'],
                'fgw_change_percent': 100 * (b['deterministic_mae'] / a['deterministic_mae'] - 1),
                'no_fgw_coverage90': a['coverage_90'], 'fgw_coverage90': b['coverage_90']}
        write_json(root / 'summary.json', {'epoch': args.epoch, 'latent_dim': 16,
            'results': results, 'comparisons': comparisons,
            'note': 'Matched one-seed comparison at the same completed epoch; uncertainty coverage is uncalibrated. Training continues independently.'})
        write_json(root / 'status.json', {'stage': 'complete', 'epoch': args.epoch,
            'elapsed_seconds': time.monotonic() - began, 'summary': str(root / 'summary.json')})
    except BaseException as error:
        previous = json.loads((root / 'status.json').read_text()) if (root / 'status.json').exists() else {}
        write_json(root / 'status.json', {**previous, 'stage': 'failed', 'error': str(error)})
        raise


if __name__ == '__main__':
    main()
