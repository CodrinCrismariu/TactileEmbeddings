"""Detached supervisor for the new SAM experiment; never touches old runs."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from tactilelatent.sam import check_storage_budget


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    config = Path('configs/joint_all_sam_holdout_motiontrans_fresh_8k_50ep.json')
    cfg = json.loads(config.read_text())
    output = Path(cfg['output'])
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'last.pt').exists() and not args.resume:
        raise FileExistsError('Existing SAM checkpoint: pass --resume explicitly')
    used = check_storage_budget(Path.cwd(), additional=6_000_000_000)
    command = [sys.executable, '-u', '-m', 'tactilelatent.train_joint', '--config', str(config)]
    if args.resume:
        command.append('--resume')
    status_path = output / 'status.json'
    def status(data):
        temp = status_path.with_suffix('.tmp')
        temp.write_text(json.dumps(data, indent=2))
        temp.replace(status_path)
    with (output / 'nohup.log').open('a') as log, (output / 'tensorboard_bridge.log').open('a') as bridge_log:
        trainer = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        bridge = subprocess.Popen([sys.executable, '-u', 'scripts/tensorboard_joint_live.py',
                                   str(output), '--pid', str(trainer.pid)],
                                  stdout=bridge_log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        status({'stage': 'training', 'trainer_pid': trainer.pid, 'bridge_pid': bridge.pid,
                'epochs': cfg['epochs'], 'optimizer': cfg['optimizer_method'], 'sam_rho': cfg['sam_rho'],
                'storage_bytes_at_launch': used, 'command': command})
        code = trainer.wait()
        status({'stage': 'complete' if code == 0 else 'failed', 'returncode': code,
                'trainer_pid': trainer.pid, 'epochs': cfg['epochs']})
        bridge.wait(timeout=30)
    sys.exit(code)


if __name__ == '__main__':
    main()
