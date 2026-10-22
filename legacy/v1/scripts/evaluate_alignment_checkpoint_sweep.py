"""Evaluate requested saved alignment epochs on held-out MotionTrans.

Each frozen encoder gets shallow (512) and deep (512, 512) transfer heads.
Completed evaluations are checked and reused; both heads share that epoch's
feature cache. Original joint checkpoints and weights are never modified.
"""
import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
import numpy as np

from tactilelatent.aligned_frames import manifest_signature, write_json
from tactilelatent.frozen_probe import head, sha256


HEADS = {'shallow': [512], 'deep': [512, 512],
         'deeper': [512, 512, 512], 'deepest': [512, 512, 512, 512]}


def choose_checkpoint(run, epoch, kind='auto'):
    """Make weights-only selection explicit when a full snapshot also exists."""
    run = Path(run)
    viewer = run / 'viewer_epochs' / f'epoch_{epoch:03d}.pt'
    if kind == 'viewer':
        return viewer
    if kind != 'auto':
        raise ValueError('Unknown checkpoint kind')
    full = run / f'epoch_{epoch:03d}.pt'
    return full if full.exists() else viewer


def completed_result(destination, snapshot, digest, manifest_digest, epoch, name, args, latent_dim=32):
    results_path = destination / 'results.json'
    if not results_path.exists():
        return None
    result = json.loads(results_path.read_text())
    saved = torch.load(destination / 'decoder.pt', weights_only=False, map_location='cpu', mmap=True)
    expected = {'checkpoint_sha256': digest, 'manifest_sha256': manifest_digest,
        'checkpoint_epoch': epoch - 1, 'source': 'ftp_motiontrans', 'frozen_encoder': True,
        'head': '-'.join(map(str, [latent_dim, *HEADS[name], 6])), 'seed': args.seed,
        'target_shape': [6, 1, 1]}
    if any(result.get(key) != value for key, value in expected.items()):
        raise ValueError(f'Incompatible completed evaluation: {results_path}')
    cfg = saved['config']
    if (saved['checkpoint_sha256'] != digest
            or Path(cfg['checkpoint']).resolve() != snapshot.resolve()
            or cfg['hidden_dims'] != HEADS[name]
            or cfg['epochs'] != args.head_epochs or cfg['patience'] != args.patience
            or cfg['samples'] != args.samples or cfg['seed'] != args.seed
            or result['validation']['sample_count'] != args.samples
            or saved['selected_epochs'] != result['selected_epochs']):
        raise ValueError(f'Completed head settings differ: {destination}')
    with torch.device('meta'):
        network = head(latent_dim, 6, hidden_dims=HEADS[name])
    expected_weights = network.state_dict()
    if (set(expected_weights) != set(saved['head']) or any(
            value.shape != saved['head'][key].shape for key, value in expected_weights.items())):
        raise ValueError(f'Completed head structure differs: {destination}')
    return result


def report_row(result, name, destination):
    metrics = result['validation']
    group = metrics['regions']['all']
    return {'encoder_epoch': result['checkpoint_epoch'] + 1, 'head': name,
        'latent_dim': int(result['head'].split('-')[0]),
        'architecture': result['head'], 'parameters': result['head_parameters'],
        'selected_head_epochs': result['selected_epochs'],
        'deterministic_mae': metrics['deterministic_mae'],
        'deterministic_mse': metrics['deterministic_mse'],
        'ensemble_mean_mae': metrics['ensemble_mean_mae'],
        'baseline_mae': metrics['mean_target_baseline_mae'],
        'crps': group['crps'], 'mean_predictive_std': group['mean_predictive_std'],
        'std_error_pearson': group['std_error_pearson'],
        'coverage_90': group['intervals']['0.9']['empirical_coverage'],
        'interval_width_90': group['intervals']['0.9']['mean_width'],
        'coverage_95': group['intervals']['0.95']['empirical_coverage'],
        'train_clips': result['train_clips'], 'fit_clips': result['fit_clips'],
        'tune_clips': result['tune_clips'], 'validation_clips': metrics['validation_clips'],
        'results': str(destination / 'results.json')}


def save_summary(root, records, epochs, failures, heads=None):
    heads = heads or {name: HEADS[name] for name in ('shallow', 'deep')}
    rows = sorted(records.values(), key=lambda row: (row['encoder_epoch'], row['head']))
    per_head = {}
    for name in heads:
        selected = [row for row in rows if row['head'] == name]
        if selected:
            best = min(selected, key=lambda row: row['deterministic_mae'])
            ranked = [row for row in selected if row.get('effective_rank_mu') is not None]
            rank = np.array([row['effective_rank_mu'] for row in ranked])
            error = np.array([row['deterministic_mae'] for row in ranked])
            correlation = float(np.corrcoef(rank, error)[0, 1]) if len(rank) > 2 and rank.std() > 0 and error.std() > 0 else None
            per_head[name] = {'best_validation_mae': best, 'rank_vs_mae_pearson': correlation,
                              'completed_checkpoints': len(selected)}
    summary = {'source': 'ftp_motiontrans', 'encoder_epochs': epochs,
        'heads': heads, 'completed_evaluations': len(rows), 'total_evaluations': len(epochs) * len(heads),
        'metric_space': 'normalized torque: 0.5 + 0.5*tanh(raw/1000); not verified SI units',
        'selection_note': 'Head duration selected on training-episode tune split, then refit on all training episodes. Cross-checkpoint rankings below use validation and are not a separate test result.',
        'results': rows, 'failures': failures,
        'per_head': per_head,
        'rank_note': 'Effective rank is the logged training-epoch covariance statistic, not a causal control or fixed-checkpoint MotionTrans rank.',
        'best_validation_mae': min(rows, key=lambda row: row['deterministic_mae']) if rows else None}
    write_json(root / 'summary.json', summary)
    if rows:
        temporary = root / 'summary.csv.tmp'
        with temporary.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(root / 'summary.csv')
    return summary


def plot_summary(root, records):
    # The system Python provides matplotlib; keep the training environment intact.
    subprocess.run(['/usr/bin/python3', str(Path(__file__).with_name(
        'plot_motiontrans_checkpoint_sweep.py')), str(root / 'summary.json')], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='processing/runs/joint_alignment_holdout_motiontrans')
    parser.add_argument('--epochs', type=int, nargs='+', help='Defaults to all saved epochs divisible by 10')
    parser.add_argument('--report-name', default='motiontrans_checkpoint_sweep',
                        help='Separate report folder within evaluations; individual head results remain reusable')
    parser.add_argument('--head-epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--samples', type=int, default=64)
    parser.add_argument('--seed', type=int, default=45)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--checkpoint-kind', choices=('auto', 'viewer'), default='auto',
                        help='Explicitly use viewer weights when sharing a precomputed snapshot/cache')
    parser.add_argument('--heads', nargs='+', choices=tuple(HEADS), default=['shallow', 'deep'],
                        help='Head presets; default retains the original one/two hidden layers')
    args = parser.parse_args()
    if len(set(args.heads)) != len(args.heads):
        parser.error('Select each head once')
    selected_heads = {name: HEADS[name] for name in args.heads}
    run = Path(args.run)
    if not args.report_name or Path(args.report_name).name != args.report_name:
        parser.error('--report-name must be a folder name')
    root = run / 'evaluations' / args.report_name
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / '.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    began, records, failures = time.monotonic(), {}, []
    epochs = sorted(set(args.epochs or [int(p.stem.split('_')[-1])
        for p in run.glob('epoch_*.pt') if int(p.stem.split('_')[-1]) % 10 == 0]))
    if not epochs or any(epoch < 1 for epoch in epochs):
        raise ValueError('Choose positive saved epochs')
    ranks = {}
    metrics_path = run / 'metrics.jsonl'
    if metrics_path.exists():
        for line in metrics_path.read_text().splitlines():
            row = json.loads(line)
            ranks[row['epoch']] = row.get('latent', {}).get('effective_rank_mu')

    def status(stage, **fields):
        write_json(root / 'status.json', {'stage': stage, 'encoder_epochs': epochs,
            'completed_evaluations': len(records), 'total_evaluations': len(epochs) * len(selected_heads),
            'elapsed_seconds': time.monotonic() - began, 'failures': failures, **fields})
        print(json.dumps({'event': 'sweep_status', 'stage': stage,
            'completed_evaluations': len(records), **fields}), flush=True)

    status('preflight')
    try:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; use libraries matching the loaded kernel driver')
        jobs = []
        for epoch in epochs:
            checkpoint = choose_checkpoint(run, epoch, args.checkpoint_kind)
            state = torch.load(checkpoint, weights_only=False, map_location='cpu', mmap=True)
            manifest = state['config']['manifest']
            latent_dim = state['config']['latent_dim']
            if (state['epoch'] + 1 != epoch or state.get('model_type') != 'joint_sensor_alignment_v1'
                    or state['config']['heldout_source'] != 'ftp_motiontrans'
                    or latent_dim < 1 or state['model_config']['latent_dim'] != latent_dim
                    or state['provenance']['data'] != manifest_signature(manifest)):
                raise ValueError(f'Wrong checkpoint or incompatible canonical data: {checkpoint}')
            del state
            base = run / 'evaluations' / f'motiontrans_epoch_{epoch:03d}'
            base.mkdir(exist_ok=True)
            snapshot = base / 'checkpoint.pt'
            if not snapshot.exists():
                os.link(checkpoint, snapshot)
            digest, manifest_digest = sha256(checkpoint), sha256(manifest)
            if not os.path.samefile(snapshot, checkpoint) and sha256(snapshot) != digest:
                raise ValueError(f'Evaluation snapshot differs: {snapshot}')
            for name, widths in selected_heads.items():
                destination = base.with_name(base.name + '_' + name)
                destination.mkdir(exist_ok=True)
                result = completed_result(destination, snapshot, digest, manifest_digest, epoch, name, args, latent_dim)
                key = (epoch, name)
                if result is not None:
                    records[key] = report_row(result, name, destination)
                    records[key]['effective_rank_mu'] = ranks.get(epoch)
                    print(json.dumps({'event': 'reuse', 'epoch': epoch, 'head': name}), flush=True)
                command = [sys.executable, '-u', '-m', 'tactilelatent.evaluate_holdout_decoder',
                    '--checkpoint', str(snapshot), '--manifest', manifest, '--source', 'ftp_motiontrans',
                    '--output', str(destination), '--cache-dir', str(base / 'features'),
                    '--allow-intermediate', '--hidden-dims', *map(str, widths),
                    '--samples', str(args.samples), '--seed', str(args.seed),
                    '--epochs', str(args.head_epochs), '--patience', str(args.patience),
                    '--workers', str(args.workers)]
                jobs.append((key, destination, snapshot, digest, manifest_digest, latent_dim, command))
        write_json(root / 'launch.json', {'args': vars(args), 'controller_pid': os.getpid(),
            'commands': [command for key, *_, command in jobs if key not in records]})
        save_summary(root, records, epochs, failures, selected_heads)
        for key, destination, snapshot, digest, manifest_digest, latent_dim, command in jobs:
            if key in records:
                continue
            epoch, name = key
            try:
                with (destination / 'evaluation.log').open('a') as log:
                    evaluator = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                                 stdin=subprocess.DEVNULL)
                    status('evaluating', encoder_epoch=epoch, head=name,
                           evaluator_pid=evaluator.pid, log=str(destination / 'evaluation.log'))
                    code = evaluator.wait()
                if code:
                    raise RuntimeError(f'Evaluator exited with code {code}; see {destination}/evaluation.log')
                result = completed_result(destination, snapshot, digest, manifest_digest, epoch, name, args, latent_dim)
                if result is None:
                    raise RuntimeError('Evaluator finished without results')
                records[key] = report_row(result, name, destination)
                records[key]['effective_rank_mu'] = ranks.get(epoch)
            except Exception as error:
                failures.append({'encoder_epoch': epoch, 'head': name, 'error': str(error)})
            save_summary(root, records, epochs, failures, selected_heads)
        if records:
            plot_summary(root, records)
        status('complete' if not failures else 'failed', summary=str(root / 'summary.json'))
        if failures:
            raise RuntimeError(f'{len(failures)} evaluations failed; completed evaluations are preserved')
    except Exception as error:
        status('failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
