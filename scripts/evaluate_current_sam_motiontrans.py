"""Snapshot a completed SAM epoch and evaluate without interrupting its trainer."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
from tactilelatent.sam import check_storage_budget


def main():
    run = Path('processing/runs/joint_all_sam_holdout_motiontrans_fresh_8k_50ep')
    check_storage_budget(Path.cwd(), additional=1_000_000_000)
    root = run / 'evaluations'
    root.mkdir(exist_ok=True)
    snapshot = root / f'snapshot_{time.time_ns()}.pt'
    # Atomic trainer replacements of last.pt cannot change this inode.
    # No copy or checkpoint-content modification is needed.
    os.link(run / 'last.pt', snapshot)
    state = torch.load(snapshot, map_location='cpu', weights_only=False)
    epoch = state['epoch'] + 1
    if state['config']['heldout_source'] != 'ftp_motiontrans':
        raise ValueError('Wrong SAM holdout')
    output = root / f'motiontrans_epoch_{epoch:03d}_{time.time_ns()}'
    output.mkdir()
    destination = output / 'checkpoint.pt'
    snapshot.rename(destination)
    del state
    command = [sys.executable, '-u', '-m', 'tactilelatent.evaluate_holdout_decoder',
               '--source', 'ftp_motiontrans', '--checkpoint', str(destination),
               '--output', str(output), '--allow-intermediate', '--samples', '64',
               '--hidden-dims', '512', '--workers', '2']
    status_path = output / 'status.json'
    def status(**fields):
        temporary = status_path.with_suffix('.tmp')
        temporary.write_text(json.dumps({'completed_checkpoint_epoch': epoch,
                                        'output': str(output), **fields}, indent=2))
        temporary.replace(status_path)
    (root / 'current_motiontrans_evaluation.json').write_text(json.dumps({
        'output': str(output), 'snapshot': str(destination), 'completed_checkpoint_epoch': epoch,
        'command': command}, indent=2))
    env = {**os.environ, 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4'}
    with (output / 'evaluation.log').open('a') as log:
        evaluator = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, env=env)
        status(stage='evaluating', evaluator_pid=evaluator.pid, command=command)
        print(json.dumps({'event': 'started', 'epoch': epoch, 'output': str(output),
                          'evaluator_pid': evaluator.pid}), flush=True)
        code = evaluator.wait()
    status(stage='complete' if code == 0 else 'failed', returncode=code)
    if code == 0:
        current = json.loads((output / 'results.json').read_text())
        reference = json.loads((Path('processing/runs/joint_all_holdout_motiontrans_fresh_8k_50ep')
                                / 'evaluations/motiontrans/results.json').read_text())
        comparison = {'note': 'Intermediate SAM epoch versus FINAL epoch-50 non-SAM baseline; not an equal-epoch causal comparison.',
                      'sam_completed_epoch': epoch, 'reference_completed_epoch': 50,
                      'same_decoder': '32-512-6', 'sample_count': 64,
                      'sam': current['validation'], 'reference': reference['validation']}
        (output / 'baseline_comparison.json').write_text(json.dumps(comparison, indent=2))
        print(json.dumps({'event': 'complete', 'epoch': epoch,
                          'deterministic_mae': current['validation']['deterministic_mae'],
                          'ensemble_mean_mae': current['validation']['ensemble_mean_mae']}), flush=True)
    sys.exit(code)


if __name__ == '__main__':
    main()
