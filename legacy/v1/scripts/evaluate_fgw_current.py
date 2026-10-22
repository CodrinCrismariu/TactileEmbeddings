"""Run requested transfer and identity probes against an immutable FGW snapshot."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

from tactilelatent.aligned_frames import file_hash, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True)
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text())
    root = Path(request['root']); run = root.parent.parent
    lock = (root / 'evaluation.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if file_hash(request['checkpoint']) != request['checkpoint_sha256']:
        raise ValueError('Frozen evaluation snapshot changed')
    epoch, began = request['checkpoint_epoch'], time.monotonic()
    jobs = []
    for name, widths in request['motiontrans_heads'].items():
        destination = run / 'evaluations' / f'motiontrans_epoch_{epoch:03d}_{name}'
        command = [sys.executable, '-u', '-m', 'tactilelatent.evaluate_holdout_decoder',
            '--checkpoint', request['checkpoint'], '--manifest', request['manifest'],
            '--source', 'ftp_motiontrans', '--output', str(destination),
            '--cache-dir', str(root / 'motiontrans_features'), '--allow-intermediate',
            '--hidden-dims', *map(str, widths), '--seed', str(request['seed']),
            '--samples', str(request['samples']), '--epochs', '100', '--patience', '10', '--workers', '2']
        jobs.append((f'motiontrans_{name}', destination, command))
    destination = run / 'evaluations' / f'dataset_identity_epoch_{epoch:03d}'
    command = [sys.executable, '-u', '-m', 'tactilelatent.evaluate_dataset_classifier',
        '--checkpoint', request['checkpoint'], '--manifest', request['manifest'],
        '--heldout-source', 'ftp_motiontrans', '--output', str(destination), '--allow-intermediate',
        '--seed', str(request['seed']), '--samples', str(request['samples']),
        '--train-clips-per-source', str(request['classifier_fit_clips_per_source']),
        '--tune-clips-per-source', str(request['classifier_tune_clips_per_source']),
        '--epochs', '100', '--patience', '10', '--workers', '2']
    jobs.append(('dataset_identity', destination, command))
    results = {}
    try:
        for name, destination, command in jobs:
            path = destination / 'results.json'
            if not path.exists():
                destination.mkdir(parents=True, exist_ok=True)
                with (destination / 'evaluation.log').open('a') as log:
                    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                               stdin=subprocess.DEVNULL)
                    status = {'stage': 'running', 'job': name, 'checkpoint_epoch': epoch,
                              'worker_pid': process.pid, 'command': command,
                              'log': str(destination / 'evaluation.log'),
                              'elapsed_seconds': time.monotonic() - began}
                    write_json(root / 'status.json', status); print(json.dumps(status), flush=True)
                    try:
                        code = process.wait()
                    except BaseException:
                        process.terminate(); process.wait(); raise
                if code:
                    raise RuntimeError(f'{name} failed with code {code}; see {destination}/evaluation.log')
            result = json.loads(path.read_text())
            if result['checkpoint_sha256'] != request['checkpoint_sha256']:
                raise ValueError('Evaluation used another checkpoint')
            results[name] = result
            write_json(root / 'summary.json', {'checkpoint_epoch': epoch, 'results': results})
        reference = Path('processing/runs/joint_alignment_holdout_motiontrans/evaluations')
        comparisons = {}
        for name in request['motiontrans_heads']:
            baseline = json.loads((reference / f'motiontrans_epoch_010_{name}' / 'results.json').read_text())
            current = results[f'motiontrans_{name}']
            old, new = baseline['validation']['deterministic_mae'], current['validation']['deterministic_mae']
            comparisons[name] = {'parent_epoch10_mae': old, 'fgw_mae': new,
                                  'change_percent': 100 * (new / old - 1)}
        write_json(root / 'summary.json', {'checkpoint_epoch': epoch, 'results': results,
            'MotionTrans_parent_comparison': comparisons,
            'note': 'Exploratory continuation comparison; no matched no-FGW continuation control. Dataset identity accuracy is diagnostic, not the transfer objective.'})
        write_json(root / 'status.json', {'stage': 'complete', 'checkpoint_epoch': epoch,
            'elapsed_seconds': time.monotonic() - began, 'summary': str(root / 'summary.json')})
    except BaseException as error:
        before = json.loads((root / 'status.json').read_text()) if (root / 'status.json').exists() else {}
        write_json(root / 'status.json', {**before, 'stage': 'failed', 'error': str(error)})
        raise


if __name__ == '__main__':
    main()
