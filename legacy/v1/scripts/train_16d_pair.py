"""Run one matched 16D control and one visual-FGW model concurrently."""
import argparse
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from tactilelatent.aligned_frames import file_hash, write_json
from tactilelatent.fgw_mixer import MixerConfig


ROOT = 'processing/runs/joint_16d_pair_v1'


def prepare(root):
    root = Path(root)
    if (root / 'plan.json').exists():
        raise FileExistsError('Pair already prepared; use run to resume it')
    root.mkdir(parents=True, exist_ok=True)
    base = json.loads(Path('configs/joint_alignment_holdout_motiontrans.json').read_text())
    base.update(latent_dim=16, epochs=30, seed=45, init_checkpoint=None, init_joint_checkpoint=None,
                matched_bank_sampling={'bank_size': 256, 'fraction': .25},
                preserve_refresh_rng=True, log_initial_weight_hash=True)
    configurations = {}
    for name in ['no_fgw', 'visual_fgw']:
        cfg = {**base, 'output': str(root / name)}
        if name == 'visual_fgw':
            cfg['fgw'] = {**asdict(MixerConfig(variant='partial_stochastic_fgw')), 'placement': 'visual'}
        path = root / 'configs' / f'{name}.json'
        write_json(path, cfg)
        configurations[name] = str(path)
    write_json(root / 'plan.json', {'latent_dim': 16, 'epochs': 30, 'seed': 45,
        'initialization': 'fresh identical weights, verified by model hash',
        'sampling': base['matched_bank_sampling'], 'heldout_source': 'ftp_motiontrans',
        'runs': configurations, 'fgw_placement': 'visual',
        'notes': 'Two runs only; no additional seeds, screening, teacher training or automatic evaluations. Native targets and existing sensor-specific AE codes are retained.'})


def run(root):
    root = Path(root)
    lock = (root / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = json.loads((root / 'plan.json').read_text())
    if shutil.disk_usage(root).free < 15 * 2**30:
        raise RuntimeError('Pair needs 15 GiB free for checkpoints')
    configurations = {name: json.loads(Path(path).read_text()) for name, path in plan['runs'].items()}
    stripped = [{k: v for k, v in cfg.items() if k not in ('output', 'fgw')} for cfg in configurations.values()]
    if stripped[0] != stripped[1] or any(cfg['latent_dim'] != 16 or cfg['epochs'] != 30 for cfg in configurations.values()):
        raise ValueError('Pair architecture or training settings differ')
    processes, streams, began = {}, [], time.monotonic()
    status_path = root / 'status.json'
    try:
        for name, path in plan['runs'].items():
            output = Path(configurations[name]['output'])
            command = [sys.executable, '-u', '-m', 'tactilelatent.train_alignment', '--config', path]
            if (output / 'last.pt').exists():
                command.append('--resume')
            log_path = root / 'logs' / f'{name}.log'; log_path.parent.mkdir(exist_ok=True)
            stream = log_path.open('a'); streams.append(stream)
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            processes[name] = process
            write_json(root / f'{name}.pid', {'pid': process.pid, 'command': command})
            print(json.dumps({'event': 'started', 'run': name, 'pid': process.pid, 'command': command}), flush=True)
        verified_initialization = False
        while True:
            states = {}
            for name, process in processes.items():
                output = Path(configurations[name]['output'])
                code = process.poll()
                if code not in (None, 0):
                    raise RuntimeError(f'{name} exited {code}; see {root}/logs/{name}.log')
                metrics = output / 'metrics.jsonl'
                completed = json.loads(metrics.read_text().splitlines()[-1])['epoch'] if metrics.exists() else 0
                if code == 0 and completed != 30:
                    raise RuntimeError(f'{name} finished before epoch 30')
                states[name] = {'pid': process.pid, 'stage': 'complete' if code == 0 else 'training',
                                'completed_epoch': completed, 'target_epoch': 30,
                                'output': str(output), 'log': str(root / 'logs' / f'{name}.log')}
            paths = [Path(cfg['output']) / 'initialization.json' for cfg in configurations.values()]
            if not verified_initialization and all(p.exists() for p in paths):
                records = [json.loads(p.read_text()) for p in paths]
                if records[0] != records[1]:
                    raise ValueError('Initial model weights differ between paired runs')
                verified_initialization = True
                write_json(root / 'matched_initialization.json', records[0])
                print(json.dumps({'event': 'identical_initialization', **records[0]}), flush=True)
            complete = all(s['stage'] == 'complete' for s in states.values())
            write_json(status_path, {'stage': 'complete' if complete else 'running',
                'controller_pid': os.getpid(), 'elapsed_seconds': time.monotonic() - began,
                'identical_initialization_verified': verified_initialization, 'runs': states})
            if complete:
                break
            time.sleep(10)
    except BaseException as error:
        for process in processes.values():
            if process.poll() is None:
                process.terminate()
        for process in processes.values():
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
        write_json(status_path, {'stage': 'failed', 'error': str(error), 'controller_pid': os.getpid()})
        raise
    finally:
        for stream in streams:
            stream.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run'))
    parser.add_argument('--root', default=ROOT)
    args = parser.parse_args()
    (prepare if args.action == 'prepare' else run)(args.root)


if __name__ == '__main__':
    main()
