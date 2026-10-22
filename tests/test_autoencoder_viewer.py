import json
from pathlib import Path

import numpy as np
import pytest
import torch
from http.server import ThreadingHTTPServer
from threading import Thread
from urllib.request import urlopen

from tactilelatent.autoencoder_viewer import Viewer, make_handler
from tactilelatent.tactile_autoencoder import TactileAutoencoder


@pytest.fixture(autouse=True)
def isolated_storage_receipts(tmp_path, monkeypatch):
    # These fixtures create their own optical pixels and must not inherit the
    # training machine's real dataset-retirement state.
    import tactilelatent.marker_storage as storage
    monkeypatch.setattr(storage, 'RECEIPT', tmp_path/'retirement.json')


@pytest.fixture
def viewer(tmp_path):
    array = np.zeros((3, 21, 21), dtype=np.float32)
    array[:, 0, 0] = np.nan
    array[1, 5, 5] = .8
    pressure = tmp_path / "pressure.npz"
    np.savez(pressure, left_pressure_grid=array, right_pressure_grid=array)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"episodes": [{"source": "egotouch", "episode_id": "test/video",
        "split": "val", "frames": 3, "views": {"0": "unused.mp4"}, "target_path": str(pressure)}]}))
    root = tmp_path / "models"
    for name, score, value in [("worse", .2, 2.), ("best", .1, 0.)]:
        run = root / name
        run.mkdir(parents=True)
        network = TactileAutoencoder(channels=[2], latent_dim=4)
        for param in network.parameters():
            param.data.zero_()
        network.decoder[-2].bias.data.fill_(value)
        torch.save({"model": network.state_dict(), "model_config": network.config}, run / "best.pt")
        (run / "config.json").write_text(json.dumps({"model": network.config}))
        (run / "best_metrics.json").write_text(json.dumps({"epoch": 3, "selection_score": score, "val": {"mae": score}}))
    return Viewer(manifest, [root])


def test_model_selection_and_masked_frame_metrics(viewer):
    assert viewer.catalog()["models"][0]["name"] == "best"
    first = viewer.frame(0, 0, 0)
    assert len(first["original"]) == 882
    assert first["original"][0] is None
    assert first["reconstruction"][0] is None
    assert first["error"][0] is None
    assert first["mae"] == pytest.approx(.5)
    assert first["contact_mae"] is None
    assert first["contact_f1"] == 0
    second = viewer.frame(0, 1, 0)
    assert second["contact_mae"] == pytest.approx(.3)
    assert second["original"][5 * 21 + 5] == pytest.approx(.8)
    assert viewer.frame(0, 0, 1)["mae"] > first["mae"]
    json.dumps(second, allow_nan=False)


def test_reject_unknown_selections(viewer):
    for args in [(100, 0, 0), (0, -1, 0), (0, 3, 0), (0, 0, 100)]:
        with pytest.raises(ValueError):
            viewer.frame(*args)
    with pytest.raises(ValueError, match="Camera"):
        viewer.video(0, 0, "9")


def test_original_only_and_wrong_sensor_model(viewer, tmp_path):
    path = tmp_path / 'glove.npy'
    np.save(path, np.full((3,2,80), 1024., dtype=np.float32))
    viewer.episodes.append({'source':'opengraph','episode_id':'viewer_test','split':'val','frames':3,
        'views':{'0':'unused.mp4'},'target_path':str(path),'target_start':0,'video_start':0,
        'target_kind':'taxel_vector','target_shape':[2,1,80],'target_scale':4095.})
    frame = viewer.frame(1, 0, -1)
    assert frame['target_shape'] == [2,1,80]
    assert frame['original'][0] == pytest.approx(1024/4095)
    assert all(value is None for value in frame['reconstruction'])
    assert frame['mae'] is None
    with pytest.raises(ValueError, match='sensor dataset'):
        viewer.frame(1, 0, 0)


def test_catalog_refresh_keeps_ids_and_inference_reloads_updated_weights(viewer):
    original_ids = {m['name']:m['id'] for m in viewer.catalog()['models']}
    model = viewer.models[0]
    checkpoint = torch.load(model['path'], weights_only=False)
    key = next(k for k in checkpoint['model'] if k.startswith('decoder.') and k.endswith('bias')
               and checkpoint['model'][k].numel()==2)
    checkpoint['model'][key].fill_(2.)
    viewer.frame(0,0,0)
    torch.save(checkpoint,model['path'])
    assert viewer.frame(0,0,0)['mae'] > .8
    assert {m['name']:m['id'] for m in viewer.catalog()['models']} == original_ids


def test_catalog_hides_superseded_smoke_and_removed_models(viewer):
    root = Path(viewer.checkpoint_roots[0])
    for name, source, extra in [('trial_smoke', 'egotouch', {}),
                                ('old_optical', 'ftp_vla_touch', {}),
                                ('obsolete_numeric', 'ftp_rdp', {})]:
        folder = root / name
        folder.mkdir()
        (folder / 'config.json').write_text(json.dumps({'source': source,
            'model': {'architecture': 'conv', 'latent_dim': 4}, **extra}))
        (folder / 'best.pt').write_bytes(b'unused')
    catalog = viewer.catalog()
    assert {m['name'] for m in catalog['models']} == {'best', 'worse'}
    original_ids = {m['name']: m['id'] for m in catalog['models']}
    (root / 'worse' / 'best.pt').unlink()
    assert {m['name']: m['id'] for m in viewer.catalog()['models']} == {'best': original_ids['best']}


def test_html_asset_is_available():
    import tactilelatent.autoencoder_viewer as module
    html = Path(module.__file__).with_name("autoencoder_viewer.html").read_text()
    assert 'id="frameSlider"' in html
    assert 'id="model"' in html


def test_opentouch_layout_preserves_official_channels():
    from tactilelatent.opentouch_layout import hand_layout, mapped_mask
    layout = hand_layout()
    mask = mapped_mask()
    assert layout['mapped_channels'] == mask.sum() == 169
    assert layout['excluded_channels'] == 87
    assert len({n['index'] for n in layout['nodes']}) == 169
    assert not mask[0, 0, 11]
    first = next(n for n in layout['nodes'] if n['id'] == '0-0')
    assert first == {'id': '0-0', 'index': 0, 'row': 0, 'column': 0,
                     'x': 229.1, 'y': 195.6}
    # If the upstream audit asset is present, compare every coordinate/exclusion.
    official = Path('processing/audits/opentouch_format/handLayoutNewest_meshid.json')
    if official.exists():
        source = json.loads(official.read_text())
        assert {n['id'] for n in layout['nodes']} == set(source['positions']) - set(source['erasedNodes'])
        for node in layout['nodes']:
            assert (node['x'], node['y']) == (source['positions'][node['id']]['x'], source['positions'][node['id']]['y'])


def test_opentouch_mapped_metrics_preserve_native_model_input(viewer, tmp_path):
    import h5py
    from tactilelatent.opentouch_layout import mapped_mask
    mask = mapped_mask()
    raw = np.full((3, 16, 16), 3072., dtype=np.float32)
    raw[:, mask[0]] = 768.
    path = tmp_path / 'opentouch.h5'
    with h5py.File(path, 'w') as h5:
        h5.create_dataset('data/demo_00/right_pressure', data=raw)
    viewer.episodes.append({'source': 'opentouch', 'episode_id': 'mapped/test',
        'split': 'val', 'frames': 3, 'views': {'0': str(path)}, 'hdf5_demo': 'demo_00',
        'target_scale': 3072., 'target_kind': 'pressure_grid', 'target_shape': [1, 16, 16]})
    prediction = np.zeros((1, 16, 16), dtype=np.float32)
    result = viewer.frame(1, 0, -1, prediction=prediction)
    assert result['target_shape'] == [1, 16, 16]
    assert result['metric_channels'] == 169
    assert result['mae'] == result['hand_mae'][0] == pytest.approx(.25)
    assert result['contact_mae'] is result['contact_f1'] is None
    for index, valid in enumerate(mask.flat):
        assert result['original'][index] == (.25 if valid else None)
        assert result['reconstruction'][index] == (0. if valid else None)
        assert result['error'][index] == (.25 if valid else None)
    assert viewer.catalog()['datasets'][-1]['hand_layout']['mapped_channels'] == 169

    class CaptureInput(torch.nn.Module):
        def forward(self, target, valid):
            assert target.shape == valid.shape == (1, 1, 16, 16)
            assert valid.sum() == 256  # Display exclusions never change AE inputs.
            np.testing.assert_array_equal(target[0].numpy(), raw[0][None] / 3072)
            return torch.zeros_like(target)

    checkpoint = tmp_path / 'fake.pt'
    checkpoint.touch()
    index = len(viewer.models)
    viewer.models.append({'source': 'opentouch', 'path': str(checkpoint)})
    viewer.model_cache[index] = (checkpoint.stat().st_mtime_ns, CaptureInput())
    assert viewer.frame(1, 0, index)['mae'] == pytest.approx(.25)
    json.dumps(result, allow_nan=False)


def test_viewer_works_under_browser_ide_proxy_prefix(viewer):
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(viewer))
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    base = f"http://127.0.0.1:{server.server_port}/proxy/{server.server_port}"
    try:
        with urlopen(base + "/") as response:
            assert 'id="frameSlider"' in response.read().decode()
        with urlopen(base + "/api/catalog") as response:
            assert json.load(response)["models"][0]["name"] == "best"
        with urlopen(base + "/api/frame?episode=0&frame=1&model=0") as response:
            assert json.load(response)["contact_mae"] == pytest.approx(.3)
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


def test_native_optical_preserves_source_pixels_and_reference(viewer, monkeypatch):
    import base64
    from io import BytesIO
    from PIL import Image
    import tactilelatent.native_optical as module

    images = np.zeros((3, 1, 12, 18, 3), dtype=np.uint8)
    images[0] = 30  # Frame zero need not be a no-contact image.
    images[1] = 35
    images[2, 0, 4, 7] = 200
    monkeypatch.setattr(module, 'ftp_group', lambda row: {'data/sensor': images})
    viewer.episodes.append({'source': 'ftp_vla_touch', 'episode_id': 'native', 'frames': 2,
                            'target_start': 1, 'target_keys': ['sensor']})
    result = viewer.optical(1, 1, gain=8)
    panel = result['panels'][0]
    original = np.asarray(Image.open(BytesIO(base64.b64decode(panel['original'].split(',')[1]))))
    change = np.asarray(Image.open(BytesIO(base64.b64decode(panel['change'].split(',')[1]))))
    np.testing.assert_array_equal(original, images[2, 0])
    np.testing.assert_array_equal(change, np.clip(np.abs(images[2, 0].astype(float) - 35) * 8, 0, 255))
    assert (panel['width'], panel['height']) == (18, 12)
    assert panel['change_mae'] == pytest.approx(np.abs(images[2, 0].astype(float) - 35).mean() / 255)
    assert viewer.optical(1, 0)['panels'][0]['identical_to_reference']
    for gain in [0, 33, float('nan')]:
        with pytest.raises(ValueError, match='gain'):
            viewer.optical(1, 0, gain)
    with pytest.raises(ValueError, match='optical sensor'):
        viewer.optical(0, 0)


def test_native_video_frame_selection_preserves_resolution(viewer, tmp_path):
    import av
    path = tmp_path / 'tactile.mkv'
    with av.open(str(path), 'w') as output:
        stream = output.add_stream('ffv1', rate=15)
        stream.width, stream.height, stream.pix_fmt = 18, 12, 'bgr0'
        for value in [10, 40, 90, 160]:
            image = np.full((12, 18, 3), value, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(image, format='rgb24')
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    row = {'source': 'freetacman', 'tactile_videos': [str(path), str(path)]}
    for index, expected in enumerate([10, 40, 90, 160]):
        images = viewer._native_optical(row, index)
        assert len(images) == 2
        for image in images:
            assert image.shape == (12, 18, 3)
            np.testing.assert_array_equal(image, expected)


def test_raw_tactile_order_restores_contact_sequence_without_reordering_scene(tmp_path, monkeypatch):
    import base64
    from io import BytesIO
    from PIL import Image
    import tactilelatent.native_optical as optical
    import tactilelatent.autoencoder_viewer as module

    # Stored sequence is contact, no contact, intermediate. Scene stays chronological.
    images = np.stack([np.full((12, 18, 3), value, np.uint8) for value in (200, 20, 80)])[:, None]
    monkeypatch.setattr(optical, 'ftp_group', lambda row: {'data/sensor': images})
    scene_requests = []
    def scene_frames(row, indices, camera, size=480):
        scene_requests.append(list(indices))
        return torch.zeros((1, 3, 12, 18))
    monkeypatch.setattr(module, 'scene_frames', scene_frames)
    row = {'source': 'ftp_vla_touch', 'episode_id': 'test', 'frames': 3, 'split': 'val',
           'target_kind': 'optical_rgb', 'target_keys': ['sensor'], 'target_start': 0,
           'zarr_name': 'test.zarr', 'tar_path': 'unused.tar', 'views': {'0': 'camera'}}
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'episodes': [row]}))
    order_file = tmp_path / 'order.json'
    record = {'chronological_to_stored_index': [1, 2, 0], 'exact_pixel_multiset_match': True,
              **{key: row[key] for key in ('target_start', 'zarr_name', 'tar_path')}}
    order_file.write_text(json.dumps({'source': 'ftp_vla_touch', 'episodes': {'test': record}}))
    viewer = Viewer(manifest, [tmp_path / 'empty'], frame_order=order_file)
    for frame, value in enumerate((20, 80, 200)):
        result = viewer.frame(0, frame, -1)
        image = np.asarray(Image.open(BytesIO(base64.b64decode(result['optical_panels'][0]['original'].split(',')[1]))))
        np.testing.assert_array_equal(image, value)
        assert result['stored_sensor_frame'] == [1, 2, 0][frame]
    change = viewer.optical(0, 2, gain=1)
    assert change['stored_reference_frame'] == 1
    assert change['panels'][0]['change_mae'] == pytest.approx(180 / 255)
    viewer.video(0, 2, '0')
    assert scene_requests == [[2]]
    assert viewer.catalog()['episodes'][0]['tactile_chronology']
    record['chronological_to_stored_index'] = [1, 1, 0]
    order_file.write_text(json.dumps({'source': 'ftp_vla_touch', 'episodes': {'test': record}}))
    with pytest.raises(ValueError, match='lookup'):
        Viewer(manifest, [tmp_path / 'empty'], frame_order=order_file)
