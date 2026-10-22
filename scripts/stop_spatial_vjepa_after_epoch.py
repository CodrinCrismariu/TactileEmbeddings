"""Stop a running live trainer after its complete checkpoint/validation log.

External controller keeps the pinned trainer and its LR horizon unchanged.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import time

from tactilelatent.aligned_frames import write_json


def identity(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split()[21]
    except FileNotFoundError:
        return None


def committed_record(root, epoch):
    path = root/'nohup.log'
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size-131072))
        tail = stream.read().decode(errors='replace')
    for line in reversed(tail.splitlines()):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get('epoch') == epoch and 'validation' in row and 'selection_score' in row:
            # Printed only after epoch, last, viewer and best saves return.
            return row
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    root = args.run
    request = json.loads((root/'stop_request.json').read_text())
    pid, ticks, epoch = request['trainer_pid'], request['trainer_start_ticks'], request['stop_after_epoch']
    import fcntl
    lock = (root/'stop_controller.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def report(stage, **details):
        row = dict(stage=stage, controller_pid=os.getpid(), trainer_pid=pid,
                   stop_after_epoch=epoch, updated_at=time.time(), **details)
        write_json(root/'stop_controller_status.json', row)
        print(json.dumps(row), flush=True)

    report('waiting_for_complete_epoch_checkpoint')
    last_report = 0
    try:
        while True:
            if identity(pid) != ticks:
                raise RuntimeError('Expected trainer exited before the stopping boundary')
            record = committed_record(root, epoch)
            if record is not None:
                if os.getpgid(pid) != pid:
                    raise RuntimeError('Trainer does not have its own process group')
                os.killpg(pid, signal.SIGSTOP)
                # Freeze the group before inspecting files or sending termination.
                os.killpg(pid, signal.SIGTERM)
                os.killpg(pid, signal.SIGCONT)
                for _ in range(100):
                    if identity(pid) != ticks or 'State:\tZ' in Path(f'/proc/{pid}/status').read_text():
                        break
                    time.sleep(.1)
                else:
                    os.killpg(pid, signal.SIGKILL)
                import torch
                for name in (f'epoch_{epoch:03d}.pt', 'last.pt', f'viewer_epochs/epoch_{epoch:03d}.pt'):
                    state = torch.load(root/name, map_location='cpu', weights_only=False, mmap=True)
                    if state['epoch']+1 != epoch:
                        raise RuntimeError(f'Unexpected final epoch in {name}')
                    del state
                stopped = dict(stage='stopped_by_user', completed_epochs=epoch,total_epochs=epoch,
                    original_schedule_epochs=request['original_schedule_epochs'],pid=None,
                    selection_score=record['selection_score'], stopped_at=time.time(),
                    reason='User requested stopping after epoch 20; original LR schedule preserved',
                    checkpoint=str(root/f'epoch_{epoch:03d}.pt'))
                write_json(root/'stop_status.json', stopped)
                write_json(root/'status.json', stopped)
                report('complete', checkpoint=stopped['checkpoint'])
                return
            if time.monotonic()-last_report > 30:
                report('waiting_for_complete_epoch_checkpoint')
                last_report = time.monotonic()
            time.sleep(.1)
    except Exception as error:
        report('failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
