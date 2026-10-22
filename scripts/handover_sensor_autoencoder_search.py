#!/usr/bin/env python3
"""Switch a search plan after its live trial finishes, preserving all checkpoints.

Stops only the coordinator, keeps its training child alive, and waits for that
trial to finish before changing plan provenance and exec-ing the coordinator.
The advisory sequence lock prevents a second coordinator during the handover.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def running(pid, birth):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return fields[0] != 'Z' and fields[19] == birth
    except FileNotFoundError:
        return False


def validate_reduction(old, new):
    """Reject changes to retained trial parameters, data or training budgets."""
    for key in ('common', 'screen_epochs', 'final_epochs', 'bootstrap_epochs', 'repeat_seeds', 'egotouch'):
        if old[key] != new[key]:
            raise ValueError(f'Handover must preserve {key}')
    if old['sources'].keys() != new['sources'].keys():
        raise ValueError('Handover must preserve sources')
    for source, spec in new['sources'].items():
        previous = old['sources'][source]
        for key in ('target_kind', 'target_shape'):
            if previous[key] != spec[key]:
                raise ValueError(f'Handover must preserve {source}/{key}')
        names = [trial['name'] for trial in spec['trials']]
        if len(names) != 4 or len(set(names)) != 4 or spec['bootstrap'] not in names:
            raise ValueError(f'Expected four unique candidates including bootstrap for {source}')
        lookup = {trial['name']: trial for trial in previous['trials']}
        for trial in spec['trials']:
            if trial['name'] in lookup and trial != lookup[trial['name']]:
                raise ValueError(f'Retained candidate changed: {source}/{trial["name"]}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--coordinator-pid', type=int)
    parser.add_argument('--trial-pid', type=int)
    parser.add_argument('--resume-wait', action='store_true',
                        help='Resume a recorded handover after the old coordinator has exited')
    parser.add_argument('--code-changes', type=Path,
                        help='Verified old/new hashes and archived old code for authorized implementation changes')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    old = json.loads((args.output / 'plan.json').read_text())
    new = json.loads(args.plan.read_text())
    validate_reduction(old, new)
    provenance = json.loads((args.output / 'provenance.json').read_text())
    changes = json.loads(args.code_changes.read_text()) if args.code_changes else {}
    changes = {Path(name).resolve(): record for name, record in changes.items()}
    updated_provenance = dict(provenance)
    plan_keys = [key for key in provenance if key.startswith('configs/sensor_autoencoder_search_')]
    if len(plan_keys) != 1:
        raise ValueError('Expected one original plan provenance entry')
    for filename, digest in provenance.items():
        record = changes.get(Path(filename).resolve())
        expected = record['new_sha256'] if record else digest
        if record and (record['old_sha256'] != digest
                       or hashlib.sha256(Path(record['archived_copy']).read_bytes()).hexdigest() != digest):
            raise ValueError(f'Old implementation archive differs: {filename}')
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Original provenance changed: {filename}')
        updated_provenance[filename] = expected
    if any(args.output.glob('*_finalists.json')) or any(args.output.glob('*_repeat.json')):
        raise ValueError('This bootstrap handover requires no frozen promotion decisions')
    state_path = args.output / 'four_candidate_migration.json'
    if args.resume_wait:
        pending = json.loads(state_path.read_text())
        if (pending['status'] != 'waiting_for_live_trial' or pending['new_plan'] != str(args.plan)
                or pending['new_plan_sha256'] != hashlib.sha256(args.plan.read_bytes()).hexdigest()):
            raise ValueError('Recorded pending handover differs')
        args.trial_pid = pending['live_trial_pid']
        child_path = Path(f'/proc/{args.trial_pid}/cmdline')
        if child_path.exists() and child_path.read_bytes():
            child_cmd = child_path.read_bytes().split(b'\0')
            if (not any(b'pretrain_sensor_autoencoder.py' in value for value in child_cmd)
                    or child_cmd[child_cmd.index(b'--output') + 1].decode() != pending['live_trial_output']):
                raise ValueError('Recorded trainer PID now belongs to another process')
            birth = Path(f'/proc/{args.trial_pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
        else:
            birth = ''
    else:
        if args.coordinator_pid is None or args.trial_pid is None:
            parser.error('Initial handover needs coordinator and trial PIDs')
        parent_cmd = Path(f'/proc/{args.coordinator_pid}/cmdline').read_bytes().split(b'\0')
        child_cmd = Path(f'/proc/{args.trial_pid}/cmdline').read_bytes().split(b'\0')
        if not any(b'train_sensor_autoencoders.py' in value for value in parent_cmd):
            raise ValueError('Coordinator PID does not identify expected search')
        if not any(b'pretrain_sensor_autoencoder.py' in value for value in child_cmd):
            raise ValueError('Trial PID does not identify expected trainer')
        if str(args.output).encode() not in parent_cmd:
            raise ValueError('Coordinator output does not match')
        child_output = child_cmd[child_cmd.index(b'--output') + 1].decode()
        if not Path(child_output).is_relative_to(args.output / 'trials'):
            raise ValueError('Trial output is outside expected search')
        parent_fields = Path(f'/proc/{args.coordinator_pid}/stat').read_text().rsplit(')', 1)[1].split()
        child_fields = Path(f'/proc/{args.trial_pid}/stat').read_text().rsplit(')', 1)[1].split()
        if int(child_fields[1]) != args.coordinator_pid:
            raise ValueError('Trial is not a child of the coordinator')
        birth = child_fields[19]
        revision = args.output / 'provenance_revisions' / 'before_four_candidates'
        revision.mkdir(parents=True, exist_ok=False)
        for name in ('plan.json', 'provenance.json'):
            shutil.copy2(args.output / name, revision / name)
        shutil.copy2(plan_keys[0], revision / Path(plan_keys[0]).name)
        if args.code_changes:
            shutil.copy2(args.code_changes, revision / 'code_changes.json')
        pending = {'status': 'waiting_for_live_trial', 'new_plan': str(args.plan),
                   'old_coordinator_pid': args.coordinator_pid, 'live_trial_pid': args.trial_pid,
                   'live_trial_output': child_output, 'archive': str(revision),
                   'new_plan_sha256': hashlib.sha256(args.plan.read_bytes()).hexdigest(),
                   'old_configurations': sum(len(s['trials']) for s in old['sources'].values()),
                   'new_configurations': sum(len(s['trials']) for s in new['sources'].values()),
                   'new_total_epochs': sum(new['screen_epochs'] * len(s['trials'])
                       + (new['final_epochs'] - new['screen_epochs']) * len({t['sensor_code_dim'] for t in s['trials']})
                       + new['final_epochs'] * len(new['repeat_seeds']) for s in new['sources'].values()),
                   'native_pixels_and_retained_trial_parameters_unchanged': True}
        pending['implementation_changes'] = str(args.code_changes) if args.code_changes else None
        write_json(state_path, pending)
        # Terminate the supervisor alone. Its CUDA trainer and loader workers continue.
        os.kill(args.coordinator_pid, signal.SIGTERM)
        while running(args.coordinator_pid, parent_fields[19]):
            time.sleep(.2)
    lock = (args.output / 'sequence.lock').open('a')
    # Process state can become Z just before the kernel releases its flock.
    deadline = time.monotonic() + 20
    while True:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(.1)
    print(json.dumps(pending), flush=True)
    while running(args.trial_pid, birth):
        time.sleep(10)
    # Refuse to overwrite provenance if files changed during the wait.
    if hashlib.sha256(args.plan.read_bytes()).hexdigest() != pending['new_plan_sha256']:
        raise ValueError('Pending plan changed')
    for filename, digest in updated_provenance.items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Provenance changed during handover: {filename}')
    for filename in plan_keys:
        del updated_provenance[filename]
    updated_provenance[str(args.plan)] = pending['new_plan_sha256']
    write_json(args.output / 'plan.json', new)
    write_json(args.output / 'provenance.json', updated_provenance)
    pending.update(status='applied', applied_at_unix=time.time())
    write_json(state_path, pending)
    print(json.dumps(pending), flush=True)
    # Python closes the lock FD on exec; the coordinator acquires it itself.
    os.execv(sys.executable, [sys.executable, '-u', 'scripts/train_sensor_autoencoders.py',
                             '--plan', str(args.plan), '--output', str(args.output)])


if __name__ == '__main__':
    main()
