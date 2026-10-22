"""Protect a live trainer while changing the coordinator's search plan."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import threading
import time


def test_handover_leaves_live_trainer_alive_and_preserves_checkpoints(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / 'scripts/handover_sensor_autoencoder_search.py'
    spec = importlib.util.spec_from_file_location('handover', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, '__file__', str(tmp_path / 'scripts/handover.py'))
    monkeypatch.chdir(tmp_path)
    output = tmp_path / 'runs'
    output.mkdir()
    scripts = tmp_path / 'scripts'
    scripts.mkdir()
    configs = tmp_path / 'configs'
    configs.mkdir()
    trial = output / 'trials/test/medium'
    trial.mkdir(parents=True)
    marker = trial / 'last.pt'
    marker.write_bytes(b'preserved checkpoint')
    names = ['small', 'compact', 'medium', 'large', 'lr_variant']
    old = {'common': {}, 'screen_epochs': 30, 'final_epochs': 100, 'bootstrap_epochs': 3,
           'repeat_seeds': [46, 47], 'egotouch': 'preserved',
           'sources': {'test': {'target_kind': 'force_tanh', 'target_shape': [1, 1, 1],
               'bootstrap': 'medium', 'trials': [{'name': name, 'sensor_code_dim': 1} for name in names]}}}
    new = json.loads(json.dumps(old))
    new['sources']['test']['trials'] = new['sources']['test']['trials'][:4]
    old_file = configs / 'sensor_autoencoder_search_test_old.json'
    old_file.write_text(json.dumps(old))
    new_file = configs / 'sensor_autoencoder_search_test_new.json'
    new_file.write_text(json.dumps(new))
    (output / 'plan.json').write_text(json.dumps(old))
    original_key = str(old_file.relative_to(tmp_path))
    code = tmp_path / 'model.py'
    code.write_text('original code')
    backup = tmp_path / 'model_original.py'
    backup.write_bytes(code.read_bytes())
    old_hash = hashlib.sha256(code.read_bytes()).hexdigest()
    code.write_text('authorized new code')
    new_hash = hashlib.sha256(code.read_bytes()).hexdigest()
    changes = tmp_path / 'changes.json'
    changes.write_text(json.dumps({str(code): {'old_sha256': old_hash,
        'new_sha256': new_hash, 'archived_copy': str(backup)}}))
    (output / 'provenance.json').write_text(json.dumps({original_key: hashlib.sha256(old_file.read_bytes()).hexdigest(), str(code): old_hash}))
    child_script = scripts / 'pretrain_sensor_autoencoder.py'
    child_script.write_text('''import sys,time
from pathlib import Path
root=Path(sys.argv[sys.argv.index('--output')+1])
while not (root/'finish').exists():time.sleep(.02)
(root/'finished').write_text('training finished after handover')
''')
    parent_script = scripts / 'train_sensor_autoencoders.py'
    parent_script.write_text('''import subprocess,sys,time,fcntl
from pathlib import Path
output=Path(sys.argv[sys.argv.index('--output')+1]);lock=(output/'sequence.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX)
child=subprocess.Popen([sys.executable,'scripts/pretrain_sensor_autoencoder.py','--output',str(output/'trials/test/medium')])
(output/'child.pid').write_text(str(child.pid))
child.wait()
while True:time.sleep(1)
''')
    parent = subprocess.Popen([sys.executable, str(parent_script), '--output', str(output)], cwd=tmp_path)
    child_pid = None
    observed = []
    try:
        deadline = time.monotonic() + 5
        while not (output / 'child.pid').exists():
            assert time.monotonic() < deadline
            time.sleep(.02)
        child_pid = int((output / 'child.pid').read_text())
        def permit_finish():
            while parent.poll() is None:
                time.sleep(.02)
            assert Path(f'/proc/{child_pid}').exists()
            observed.append('trainer survived coordinator termination')
            (trial / 'finish').touch()
        waiter = threading.Thread(target=permit_finish)
        waiter.start()
        executed = []
        monkeypatch.setattr(module.os, 'execv', lambda executable, arguments: executed.append(arguments))
        original_flock = module.fcntl.flock
        lock_calls = []
        def delayed_lock(descriptor, operation):
            lock_calls.append(operation)
            if len(lock_calls) == 1:
                raise BlockingIOError('simulated coordinator exit/lock-release race')
            return original_flock(descriptor, operation)
        monkeypatch.setattr(module.fcntl, 'flock', delayed_lock)
        monkeypatch.setattr(sys, 'argv', ['handover', '--plan', str(new_file), '--output', str(output),
            '--coordinator-pid', str(parent.pid), '--trial-pid', str(child_pid), '--code-changes', str(changes)])
        module.main()
        waiter.join(timeout=5)
        assert observed and (trial / 'finished').exists()
        assert len(lock_calls) >= 2
        assert marker.read_bytes() == b'preserved checkpoint'
        assert json.loads((output / 'plan.json').read_text()) == new
        assert json.loads((output / 'provenance.json').read_text())[str(code)] == new_hash
        assert json.loads((output / 'four_candidate_migration.json').read_text())['status'] == 'applied'
        assert json.loads((output / 'provenance_revisions/before_four_candidates/plan.json').read_text()) == old
        assert executed[0][-4:] == ['--plan', str(new_file), '--output', str(output)]
    finally:
        if parent.poll() is None:
            parent.terminate()
        parent.wait(timeout=5)
        (trial / 'finish').touch()


def test_removed_starter_cannot_win_reduced_search(tmp_path):
    script = Path(__file__).resolve().parents[1] / 'scripts/train_sensor_autoencoders.py'
    spec = importlib.util.spec_from_file_location('search', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / 'plan.json').write_text(json.dumps({'sources': {'ftp_rdp_bimanual': {
        'trials': [{'name': 'affine_code2'}]}}, 'repeat_seeds': [46, 47]}))
    for name, score, code in [('medium_code32', .0001, 32), ('affine_code2', .01, 2)]:
        trial = tmp_path / 'trials/ftp_rdp_bimanual' / name
        trial.mkdir(parents=True)
        (trial / 'config.json').write_text(json.dumps({'source': 'ftp_rdp_bimanual', 'latent_dim': code,
            'architecture': 'mlp', 'seed': 45}))
        (trial / 'metrics.jsonl').write_text(json.dumps({'epoch': 3, 'seconds': 1}) + '\n')
        (trial / 'best_metrics.json').write_text(json.dumps({'epoch': 3, 'selection_score': score,
            'val': {'mae': score, 'contact_mae': None, 'contact_f1': None, 'psnr': None}}))
    result = module.results(tmp_path)
    assert [row['name'] for row in result] == ['affine_code2']
    assert json.loads((tmp_path / 'best_by_source.json').read_text())['ftp_rdp_bimanual']['sensor_code_dim'] == 2
    assert (tmp_path / 'trials/ftp_rdp_bimanual/medium_code32/best_metrics.json').exists()


def test_single_winner_uses_equal_screening_budget_not_old_extended_scores(tmp_path):
    script = Path(__file__).resolve().parents[1] / 'scripts/train_sensor_autoencoders.py'
    spec = importlib.util.spec_from_file_location('search', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    names = ['compact128_code8', 'compact128_code16']
    (tmp_path / 'plan.json').write_text(json.dumps({'sources': {'opengraph': {
        'trials': [{'name': name} for name in names]}}, 'repeat_seeds': []}))
    for name, code, screen_score, extra_score in [(names[0], 8, .1, .001), (names[1], 16, .05, None)]:
        path = tmp_path / 'trials/opengraph' / name
        path.mkdir(parents=True)
        history = [{'epoch': 30, 'seconds': 1, 'selection_score': screen_score}]
        if extra_score is not None:
            history.append({'epoch': 100, 'seconds': 1, 'selection_score': extra_score})
        (path / 'metrics.jsonl').write_text('\n'.join(json.dumps(h) for h in history) + '\n')
        (path / 'config.json').write_text(json.dumps({'source': 'opengraph', 'latent_dim': code,
            'architecture': 'mlp', 'seed': 45}))
        best = min(history, key=lambda h: h['selection_score'])
        (path / 'best_metrics.json').write_text(json.dumps({'epoch': best['epoch'], 'selection_score': best['selection_score'],
            'val': {'mae': best['selection_score'], 'contact_mae': None, 'contact_f1': None, 'psnr': None}}))
    screened = module.screening_results(tmp_path, 'opengraph', names, 30)
    assert module.select_finalists(screened, 'best_one') == [names[1]]
    assert module.select_finalists(screened, 'per_code') == names
