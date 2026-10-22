import json
from http.server import ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from threading import Thread
from urllib.error import HTTPError
from urllib.request import urlopen

import numpy as np
from PIL import Image
import pytest
import torch

from tactilelatent.aligned_frames import file_hash, json_hash
from tactilelatent.expansion_viewer import ExpandedViewer
from tactilelatent.mirror_viewer import MirroredViewer, make_mirror_handler
from tactilelatent.sensor_autoencoder import SensorAutoencoder


@pytest.fixture
def viewer(tmp_path, monkeypatch):
    import tactilelatent.expansion_viewer as expansion
    monkeypatch.setattr(expansion, 'ROOT', tmp_path / 'expansion')
    y = np.zeros((3, 21, 21), np.float32)
    y[:, 2, 3] = .9
    y[:, 1, 1] = np.nan
    right = np.zeros_like(y)
    right[:, 7, 8] = .4
    pressure = tmp_path / 'pressure.npz'
    np.savez(pressure, left_pressure_grid=y, right_pressure_grid=right)
    row = dict(source='egotouch', episode_id='asymmetric', split='val', frames=3,
               views={'0': 'ego.mp4', '1': 'left.mp4'}, target_path=str(pressure))
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(dict(episodes=[row])))
    models = tmp_path / 'models'
    models.mkdir()
    return MirroredViewer(manifest, [models], frame_order=None,
                          acquisition_teachers=tmp_path/'acquisition-teachers',
                          mirror_policy='configs/bimanual_mirror_policy_v1.json',
                          mirror_preparation=tmp_path / 'mirrors')


def test_mirrored_measurements_native_shape_masks_and_blocked_sources(viewer):
    native = viewer.frame(0, 1, -1)
    mirrored = viewer.mirror_frame(0, 1, -1)
    y = np.asarray(native['original'], object).reshape(2, 21, 21)
    np.testing.assert_array_equal(np.asarray(mirrored['original'], object).reshape(2, 21, 21),
                                  y[[1, 0], :, ::-1])
    assert mirrored['target_shape'] == [2, 21, 21]
    assert mirrored['frame'] == mirrored['stored_sensor_frame'] == 1
    assert mirrored['original'][21 * 21 + 21 + 19] is None
    assert mirrored['mae'] is None
    json.dumps(mirrored, allow_nan=False)
    viewer.episodes.append(dict(source='opengraph', episode_id='blocked', frames=3))
    with pytest.raises(ValueError, match='blocked'):
        viewer.mirror_frame(1, 1, -1)


def test_mirror_runs_teacher_on_transformed_target_instead_of_flipping_prediction(viewer, tmp_path):
    seen = []

    class NonEquivariantAE(torch.nn.Module):
        def forward(self, y, valid):
            seen.append((y.clone(), valid.clone()))
            return .7 * y + torch.linspace(0, .1, 21)[None, None, None, :]

    path = tmp_path / 'fake.pt'
    path.touch()
    viewer.models.append(dict(source='egotouch', path=str(path)))
    viewer.model_cache[0] = (path.stat().st_mtime_ns, NonEquivariantAE())
    first = viewer.frame(0, 1, 0)
    second = viewer.mirror_frame(0, 1, 0)
    torch.testing.assert_close(seen[1][0], seen[0][0][:, [1, 0]].flip(-1))
    torch.testing.assert_close(seen[1][1], seen[0][1][:, [1, 0]].flip(-1))
    p = np.asarray(first['reconstruction'], dtype=float).reshape(2, 21, 21)[[1, 0], :, ::-1]
    q = np.asarray(second['reconstruction'], dtype=float).reshape(2, 21, 21)
    assert not np.allclose(p, q, equal_nan=True)
    assert second['mae'] is not None


def test_wrist_role_swap_and_lossless_reflected_preview(viewer, monkeypatch):
    calls = []
    image = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)

    def video(self, episode, frame, camera):
        calls.append(camera)
        f = BytesIO()
        Image.fromarray(image).save(f, 'PNG')
        return f.getvalue()

    monkeypatch.setattr(ExpandedViewer, 'video', video)
    mirrored = np.asarray(Image.open(BytesIO(viewer.mirror_video(0, 1, '2'))))
    assert calls == ['1']
    np.testing.assert_array_equal(mirrored, image[:, ::-1])
    with pytest.raises(ValueError, match='missing'):
        viewer.mirror_video(0, 1, '1')
    catalog = viewer.catalog()
    assert catalog['episodes'][0]['mirror_cameras'] == ['0', '2']
    assert catalog['mirror_sources']['egotouch']['enabled']
    assert not catalog['mirror_sources']['ftp_unit_bimanual']['enabled']


def test_signed_direct_force_swap_has_no_autoencoder(viewer, monkeypatch):
    from tactilelatent.sensor_data import SENSORS
    assert 'ftp_rdp_bimanual' in SENSORS
    row = dict(source='ftp_rdp_bimanual', episode_id='force', frames=3,
               target_transform_scale=30., views={'0': 'unused'})
    viewer.episodes.append(row)
    monkeypatch.setattr(viewer, '_cached_target', lambda row, frame:
                        np.array([.2, .8], np.float32).reshape(2, 1, 1))
    result = viewer.mirror_frame(1, 1, -1)
    np.testing.assert_allclose(result['original'], [.8, .2])
    assert result['raw_original'][0] > 0 and result['raw_original'][1] < 0
    assert result['contact_mae'] is None and result['mae'] is None
    assert all(v is None for v in result['reconstruction'])


def test_accepted_mirrored_teacher_is_listed_once_with_stable_id(viewer):
    folder = viewer.mirror_root / 'teachers/egotouch'
    folder.mkdir(parents=True)
    path = folder / 'best.pt'
    network = SensorAutoencoder([2, 21, 21], latent_dim=4, channels=[2])
    torch.save(dict(model=network.state_dict(), model_config=network.config), path)
    teacher = dict(checkpoint=str(path), sha256=file_hash(path), model_config=network.config,
                   original_checkpoint='parent/run/best.pt')
    quality = dict(accepted=True, adapted=True, teacher=teacher,
                   signature=dict(policy_sha256=json_hash(viewer.mirror_policy), epochs=10),
                   validation=dict(original=dict(mae=.1), mirrored=dict(mae=.2)))
    (folder / 'quality.json').write_text(json.dumps(quality))
    first = viewer.catalog()['models']
    second = viewer.catalog()['models']
    assert len(first) == len(second) == 1
    assert first[0]['id'] == second[0]['id']
    assert first[0]['mirror_adapted'] and first[0]['mirrored_validation_mae'] == .2
    assert viewer.mirror_frame(0, 1, first[0]['id'])['mae'] is not None
    quality['accepted'] = False
    (folder / 'quality.json').write_text(json.dumps(quality))
    assert not viewer.catalog()['models']


def test_proxy_prefix_mirror_api_and_invalid_selection(viewer):
    server = ThreadingHTTPServer(('127.0.0.1', 0), make_mirror_handler(viewer))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{server.server_port}/proxy/{server.server_port}'
    try:
        with urlopen(base + '/?page=mirror') as response:
            assert b'Mirror quality check' in response.read()
        with urlopen(base + '/api/mirror/frame?episode=0&frame=1&model=-1') as response:
            assert json.load(response)['mirror']['hand_permutation'] == [1, 0]
        with pytest.raises(HTTPError) as error:
            urlopen(base + '/api/mirror/frame?episode=0&frame=1&model=999')
        assert error.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_unverified_marker_preview_fresh_teacher_and_native_pixel_geometry(viewer,tmp_path,monkeypatch):
    from tactilelatent.mirror_preparation import NativeTargets
    import tactilelatent.marker_viewer as native_markers
    row=dict(source='ftp_unit_bimanual',episode_id='two-surface',frames=3,split='val',
             target_shape=[6,240,320],views={'0':'unused'})
    viewer.episodes.append(row)
    monkeypatch.setattr(native_markers,'canonical_rows',lambda path:{('ftp_unit_bimanual','two-surface'):row})
    batch=dict(marker_reference=torch.tensor([[.1,.2],[.2,.3],[.9,.7]]),
        marker_target=torch.tensor([[[.1,.2],[float('nan'),float('nan')],[.3,-.4]]]),
        marker_valid=torch.tensor([[True,False,True]]),marker_surface=torch.tensor([0,0,1]),
        marker_spacing=torch.tensor([10.,10.,20.]))
    monkeypatch.setattr(NativeTargets,'read',lambda self,row,indices:batch)
    seen=[]
    class Teacher(torch.nn.Module):
        def forward(self,y,valid,reference,surface):
            seen.append((y.clone(),reference.clone(),surface.clone()))
            return .5*torch.nan_to_num(y)+reference
    path=tmp_path/'marker.pt';path.touch()
    viewer.models.append(dict(source='ftp_unit_bimanual',path=str(path),representation='marker_displacement'))
    viewer.model_cache[0]=(path.stat().st_mtime_ns,Teacher())
    with pytest.raises(ValueError,match='blocked'):
        viewer.mirror_frame(1,1,0)
    result=viewer.mirror_frame(1,1,0,candidate='x')
    assert not result['mirror']['verified'] and not result['mirror']['training_enabled']
    np.testing.assert_allclose(result['marker_displacement']['reference_px'][0],[.9*319,.2*239],rtol=1e-6)
    np.testing.assert_allclose(result['marker_displacement']['measured_px'][0],[-1.,2.])
    assert result['marker_displacement']['measured_px'][1] is None
    torch.testing.assert_close(seen[0][0][0,0],torch.tensor([-.1,.2]))
    torch.testing.assert_close(seen[0][1][0,0],torch.tensor([.9,.2]))
    assert seen[0][2].tolist()==[[1,1,0]]
    json.dumps(result,allow_nan=False)


def test_accepted_gel_teacher_has_native_input_metadata_and_both_frames_run(viewer,monkeypatch):
    from tactilelatent.bimanual_mirror import load_policy
    from tactilelatent.marker_autoencoder import MarkerAutoencoder
    from tactilelatent.mirror_preparation import NativeTargets
    import tactilelatent.marker_viewer as markers
    viewer.mirror_policy=load_policy('configs/bimanual_mirror_policy_v2.json')
    viewer.mirror_history=[(viewer.mirror_root,viewer.mirror_policy)]
    folder=viewer.mirror_root/'teachers/ftp_unit_bimanual';folder.mkdir(parents=True)
    path=folder/'best.pt'
    network=MarkerAutoencoder(latent_dim=4,width=16,heads=2,queries=2)
    torch.save(dict(model_type='marker_autoencoder_v1',model_config=network.config,
                    model=network.state_dict()),path)
    teacher=dict(checkpoint=str(path),sha256=file_hash(path),model_config=network.config,
                 original_checkpoint='historical/marker_teacher/best.pt')
    quality=dict(accepted=True,adapted=True,teacher=teacher,
        signature=dict(policy_sha256=json_hash(viewer.mirror_policy),epochs=10),
        validation=dict(original=dict(mae=.1),mirrored=dict(mae=.2)))
    (folder/'quality.json').write_text(json.dumps(quality))
    models=viewer.catalog()['models'];assert len(models)==1
    assert models[0]['mirror_adapted'] and models[0]['representation']=='marker_displacement'
    row=dict(source='ftp_unit_bimanual',episode_id='native',frames=1,target_shape=[6,240,320],views={'0':'unused'})
    viewer.episodes.append(row)
    batch=dict(marker_target=torch.tensor([[[.1,.2],[-.3,.4]]]),
        marker_valid=torch.ones(1,2,dtype=torch.bool),marker_reference=torch.tensor([[.2,.3],[.7,.8]]),
        marker_surface=torch.tensor([0,1]),marker_spacing=torch.tensor([10.,20.]))
    configurations=[]
    def inputs(cfg,row,frame):
        configurations.append(cfg)
        return None,batch,None
    monkeypatch.setattr(markers,'inputs',inputs)
    monkeypatch.setattr(markers,'payload',lambda cfg,row,frame,pred:dict(prediction=pred.tolist()))
    monkeypatch.setattr(markers,'canonical_rows',lambda path:{('ftp_unit_bimanual','native'):row})
    monkeypatch.setattr(NativeTargets,'read',lambda self,row,indices:batch)
    assert viewer.frame(1,0,models[0]['id'])['prediction']
    assert configurations==[dict(manifest='processing/storage_v1/manifest.json',
                                  marker_targets='processing/marker_displacement_storage_v1/targets')]
    reflected=viewer.mirror_frame(1,0,models[0]['id'])
    assert reflected['mirror']['verified'] and not reflected['mirror']['training_enabled']
    assert reflected['mirror']['surface_permutation']==[1,0]
    assert reflected['mae'] is not None
    with pytest.raises(ValueError,match='speculative'):
        viewer.mirror_frame(1,0,models[0]['id'],candidate='x')
