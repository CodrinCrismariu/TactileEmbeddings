"""Scalar monitoring must stay faithful to logs and independent of training."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from threading import Thread
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from tactilelatent import training_dashboard as module
from tactilelatent.training_dashboard import TrainingDashboard, make_training_handler, normalize_history


def write_run(root, name, rows, config=None, status=None):
    parent = root / 'processing/runs' / name
    parent.mkdir(parents=True)
    (parent / 'config.json').write_text(json.dumps(config or {}))
    (parent / 'metrics.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    if status is not None:(parent / 'status.json').write_text(json.dumps(status))
    return parent


def find(curves, source, split, metric):
    return next(c for c in curves if (c['source'], c['split'], c['metric']) == (source, split, metric))


def test_joint_losses_preserve_zeros_native_scales_and_macro_mean():
    rows = [dict(epoch=1, train={'egotouch': {'supervision': 0., 'loss': .01}, 'trex': {'supervision': 2.}},
                 validation={'egotouch': {'supervision_loss': .2, 'code_mse': .2, 'normalized_mae': .03},
                             'trex': {'supervision_loss': 4., 'native_mae': 12., 'valid_cells': 9999}})]
    curves, completed, offset = normalize_history(rows, {}, Path('joint'))
    assert completed == 1 and offset == 0
    assert find(curves, 'egotouch', 'train', 'supervision_loss')['points'] == [(1, 0.)]
    assert find(curves, '__macro__', 'validation', 'supervision_loss')['points'] == [(1, 2.1)]
    assert find(curves, 'trex', 'validation', 'native_mae')['points'] == [(1, 12.)]
    assert not any(c['metric'] == 'valid_cells' for c in curves)
    assert len([c for c in curves if c['source'] == 'egotouch' and c['split'] == 'validation' and c['metric'] == 'supervision_loss']) == 1


def test_old_zero_based_logs_do_not_fabricate_validation_loss():
    rows = [dict(epoch=0, train_loss_by_source={'ftp_motiontrans': .8},
                 validation={'ftp_motiontrans': {'normalized_mae': .1}}),
            dict(epoch=1, train_loss_by_source={'ftp_motiontrans': .7},
                 validation={'ftp_motiontrans': {'normalized_mae': .09}})]
    curves, completed, offset = normalize_history(rows, {}, Path('motiontrans'))
    assert (completed, offset) == (2, 1)
    assert find(curves, 'ftp_motiontrans', 'train', 'loss')['points'] == [(1, .8), (2, .7)]
    assert not any(c['split'] == 'validation' and c['metric'] == 'loss' for c in curves)


def test_autoencoder_and_mirror_histories_keep_original_and_mirrored_validation_distinct():
    rows = [dict(epoch=80, train_loss=.02,
                 validation={'original': {'mae': .1, 'active_mae': .3}, 'mirrored': {'mae': .11, 'active_mae': .31}})]
    curves, completed, _ = normalize_history(rows, {'source': 'ftp_unit_bimanual'}, Path('teacher'))
    assert completed == 80
    assert find(curves, 'ftp_unit_bimanual', 'validation_original', 'mae')['points'] == [(80, .1)]
    assert find(curves, 'ftp_unit_bimanual', 'validation_mirrored', 'mae')['points'] == [(80, .11)]
    assert find(curves, 'ftp_unit_bimanual', 'train', 'loss')['points'] == [(80, .02)]
    ae, _, _ = normalize_history([dict(epoch=1, train={'loss': .1, 'mae': .2}, val={'loss': .3, 'mae': .4})],
                                 {'source': 'egopressure'}, Path('teacher'))
    assert find(ae, 'egopressure', 'validation', 'loss')['points'] == [(1, .3)]


def test_single_egotouch_validation_model_does_not_show_baseline_as_dataset():
    curves, _, _ = normalize_history([dict(epoch=0, train_loss=.3, validation={
        'model': {'mae': .1}, 'zero_baseline': {'mae': .2}})], {}, Path('egotouch_dinov2_long50/metrics.jsonl'))
    assert {c['source'] for c in curves} == {'egotouch'}
    assert find(curves, 'egotouch', 'validation', 'mae')['points'] == [(1, .1)]


def test_missing_nonfinite_and_duplicate_epochs_are_not_zero_filled():
    rows = [dict(epoch=1, source='trex', train_loss=.4, val={'mae': float('nan')}),
            dict(epoch=2, source='trex', train_loss=.2, val={'mae': None}),
            dict(epoch=2, source='trex', train_loss=.1, val={'mae': .3})]
    curves, _, _ = normalize_history(rows, {}, Path('teacher'))
    assert find(curves, 'trex', 'train', 'loss')['points'] == [(1, .4), (2, .1)]
    assert find(curves, 'trex', 'validation', 'mae')['points'] == [(2, .3)]
    json.dumps(curves, allow_nan=False)


@pytest.fixture
def dashboard(tmp_path):
    return TrainingDashboard(tmp_path, [tmp_path / 'processing/runs'], tmp_path / 'monitor')


def test_discovery_filters_smokes_and_reads_partial_appends_without_loading_weights(dashboard, tmp_path):
    parent = write_run(tmp_path, 'production', [dict(epoch=1, source='egotouch', train_loss=.4, val={'loss': .5})])
    write_run(tmp_path, 'tiny_smoke', [dict(epoch=1, source='egotouch', train_loss=1)])
    with (parent / 'metrics.jsonl').open('a') as f:f.write('{"epoch":2, "source":"egotouch", "train_loss":0.2')
    catalog = dashboard.catalog()
    assert len(catalog['runs']) == 1 and catalog['runs'][0]['completed_epochs'] == 1
    identifier = catalog['runs'][0]['id']
    assert dashboard.history([identifier])['runs'][0]['completed_epochs'] == 1
    with (parent / 'metrics.jsonl').open('a') as f:f.write(', "val":{"loss":0.3}}\n')
    current = dashboard.history([identifier])['runs'][0]
    assert current['completed_epochs'] == 2
    assert find(current['curves'], 'egotouch', 'validation', 'loss')['points'] == [(1, .5), (2, .3)]
    assert not list(tmp_path.glob('**/*.pt'))


def test_live_collector_deduplicates_and_stops_sampling_dead_processes(dashboard, tmp_path, monkeypatch):
    active = [True]
    monkeypatch.setattr(module, 'pid_active', lambda *args: active[0])
    status = dict(stage='training', pid=123, epoch=2, batch=26, batches_per_epoch=100, source='egotouch', loss=0.)
    parent = write_run(tmp_path, 'joint', [dict(epoch=1, train={'egotouch': {'supervision': .4}})], status=status)
    dashboard.sample_once(); dashboard.sample_once()
    identifier = dashboard.catalog()['runs'][0]['id']
    assert len(dashboard.history([identifier])['runs'][0]['live_samples']) == 1
    status.update(batch=51, loss=.3); (parent / 'status.json').write_text(json.dumps(status))
    dashboard.sample_once()
    assert len(dashboard.history([identifier])['runs'][0]['live_samples']) == 2
    active[0] = False
    status.update(batch=76)
    assert dashboard.catalog()['runs'][0]['state'] == 'inactive'
    dashboard.sample_once()
    assert len(dashboard.history([identifier])['runs'][0]['live_samples']) == 2


def test_active_first_epoch_is_visible_before_any_epoch_metrics(dashboard, tmp_path, monkeypatch):
    monkeypatch.setattr(module, 'pid_active', lambda *args: True)
    parent=write_run(tmp_path, 'fresh_joint', [], {'epochs':30},
        dict(stage='training',pid=123,epoch=1,batch=1,source='trex',loss=.3))
    (parent/'metrics.jsonl').unlink()
    run=dashboard.catalog()['runs'][0]
    assert run['state']=='running' and run['completed_epochs']==0 and run['sources']==['trex']
    dashboard.sample_once()
    assert len(dashboard.history([run['id']])['runs'][0]['live_samples'])==1


def test_pid_reuse_cannot_mark_a_stale_status_live():
    import os
    assert not module.pid_active(os.getpid(),Path.cwd().resolve(),reported_at=1)


def test_unknown_run_ids_cannot_read_arbitrary_paths(dashboard, tmp_path):
    write_run(tmp_path, 'joint', [dict(epoch=1, train_loss=.3)])
    dashboard.catalog()
    with pytest.raises(ValueError, match='Unknown'):dashboard.history(['../../secrets'])
    with pytest.raises(ValueError, match='eight'):dashboard.history(['x'] * 9)


def test_all_seventeen_source_curves_are_available(dashboard, tmp_path):
    sources = ['baihu', 'egopressure', 'egotactile', 'egotouch', 'feel_kitchen', 'feel_lab',
        'ftp_rdp', 'ftp_rdp_bimanual', 'ftp_unit', 'ftp_unit_bimanual', 'ftp_vitamin', 'ftp_vla_touch',
        'hrdexdb_allegro_v5', 'hrdexdb_inspire_f1', 'opengraph', 'opentouch', 'trex']
    write_run(tmp_path, 'joint_all', [dict(epoch=1, train={s: {'supervision': .2} for s in sources},
        validation={s: {'supervision_loss': .3} for s in sources})])
    run = dashboard.catalog()['runs'][0]
    assert set(run['sources']) == set(sources)
    history = dashboard.history([run['id']])['runs'][0]
    assert len([c for c in history['curves'] if c['metric'] == 'supervision_loss']) == 36


def test_http_dashboard_proxy_paths_unknown_runs_and_existing_route_fallback(dashboard, tmp_path):
    write_run(tmp_path, 'joint', [dict(epoch=1, source='egotouch', train_loss=.3, val={'loss': .4})])
    class Original(BaseHTTPRequestHandler):
        def do_GET(self):
            body=b'existing viewer'; self.send_response(200); self.send_header('Content-Length', str(len(body)))
            self.end_headers(); self.wfile.write(body)
        def log_message(self, *args):pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), make_training_handler(Original, dashboard))
    thread = Thread(target=server.serve_forever, daemon=True); thread.start()
    base = 'http://127.0.0.1:' + str(server.server_port)
    try:
        for route in ('/?page=training', '/training', '/proxy/8080/?page=training'):
            assert b'Training monitor' in urlopen(base + route).read()
        assert b'ScalarChart' in urlopen(base + '/proxy/8080/training-dashboard.js').read()
        catalog = json.load(urlopen(base + '/proxy/8080/api/training/catalog'))
        identifier = catalog['runs'][0]['id']
        assert json.load(urlopen(base + '/api/training/history?runs=' + identifier))['runs'][0]['curves']
        with pytest.raises(HTTPError) as error:urlopen(base + '/api/training/history?runs=../../etc/passwd')
        assert error.value.code == 400
        assert urlopen(base + '/api/catalog').read() == b'existing viewer'
    finally:
        server.shutdown(); server.server_close(); dashboard.close(); thread.join(timeout=3)
