import json
from pathlib import Path
from threading import Thread
from http.server import ThreadingHTTPServer
from urllib.request import urlopen

import numpy as np
import pytest
import torch
from torch import nn

from tactilelatent.autoencoder_viewer import Viewer, make_handler
from tactilelatent.joint_model import JointTactileVAE
from tactilelatent.joint_viewer import JointViewer, clip_indices
from tactilelatent.model import MODEL_ID, REVISION
from tactilelatent.viewer_checkpoints import save_viewer_epoch


@pytest.fixture
def visual_viewer(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    array = np.linspace(0, 1, 12 * 21 * 21, dtype=np.float32).reshape(12, 21, 21)
    pressure = tmp_path / 'pressure.npz'
    np.savez(pressure, left_pressure_grid=array, right_pressure_grid=array)
    row = {'source': 'egotouch', 'episode_id': 'test/video', 'split': 'val', 'frames': 12,
           'views': {'0': 'ego.mp4', '2': 'wrist.mp4'}, 'target_path': str(pressure)}
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'episodes': [row]}))
    viewer = Viewer(manifest, [tmp_path / 'ae'], frame_order=None)
    root = tmp_path / 'joint'
    root.mkdir()
    cfg = {'dataset_family': 'joint_all', 'latent_objective': 'vae', 'train_sources': ['egotouch'],
           'width': 16, 'heads': 4, 'temporal_layers': 1, 'cross_view_layers': 1,
           'latent_dim': 4, 'dropout': 0., 'stochastic': True, 'epochs': 10}
    network = JointTactileVAE(targets={'egotouch': ('pressure_grid', [2, 21, 21], 1.)},
                            **{key: cfg[key] for key in ('width', 'heads', 'temporal_layers',
                            'cross_view_layers', 'latent_dim', 'dropout', 'stochastic')})
    state = {'config': cfg, 'model': network.state_dict(), 'epoch': 0,
             'backbone': MODEL_ID, 'backbone_revision': REVISION, 'optimizer': {'test': torch.ones(50)}}
    (root / 'config.json').write_text(json.dumps(cfg))
    (root / 'metrics.jsonl').write_text(json.dumps({'epoch': 0}) + '\n')
    torch.save(state, root / 'last.pt')
    save_viewer_epoch(root / 'last.pt', state)
    requests = []
    def scenes(row, indices):
        requests.append((list(indices), sorted(row['views'])))
        rgb = torch.zeros(len(indices), 3, 3, 224, 224)
        masks = torch.zeros(len(indices), 3, dtype=torch.bool)
        for slot in row['views']:
            rgb[:, int(slot)] = torch.as_tensor(indices).view(-1, 1, 1, 1) / 12
            masks[:, int(slot)] = True
        return rgb, masks
    class Backbone(nn.Module):
        def forward(self, rgb):
            return rgb.mean((1, 2, 3))[:, None, None].expand(-1, 2, 768)
    import tactilelatent.joint_viewer as module
    monkeypatch.setattr(module, 'scene_rgb', scenes)
    monkeypatch.setattr(module, 'DinoBackbone', Backbone)
    joint = JointViewer(viewer, [tmp_path], 'cpu')
    model = next(item for item in joint.catalog()['models'] if item['name'] == 'joint')
    return viewer, joint, model, state, root, requests


def test_visual_inference_exact_frame_input_cameras_and_cache(visual_viewer):
    viewer, joint, model, _, _, requests = visual_viewer
    first = joint.frame(0, 5, model['id'], 'last.pt', 'ego')
    assert first['frame'] == 5
    assert first['original'] == viewer.frame(0, 5, -1)['original']
    assert first['joint']['clip_frames'][first['joint']['selected_position']] == 5
    assert first['joint']['epoch'] == 1
    assert requests[-1][1] == ['0']
    before = len(requests)
    repeated = joint.frame(0, 5, model['id'], 'last.pt', 'ego')
    assert len(requests) == before
    assert first['reconstruction'] == repeated['reconstruction']
    all_views = joint.frame(0, 5, model['id'], 'last.pt', 'all')
    assert requests[-1][1] == ['0', '2']
    assert all_views['joint']['camera_slots'] == ['0', '2']
    assert all_views['reconstruction'] != first['reconstruction']
    json.dumps(all_views, allow_nan=False)


def test_saved_epoch_is_immutable_and_latest_reloads(visual_viewer):
    _, joint, model, state, root, _ = visual_viewer
    original = joint.frame(0, 1, model['id'], 'viewer_epochs/epoch_001.pt')
    state['model']['visual.decoder.3.bias'].add_(1.)
    state['epoch'] = 1
    temp = root / 'last.tmp'
    torch.save(state, temp)
    temp.replace(root / 'last.pt')
    save_viewer_epoch(root / 'last.pt', state)
    (root / 'metrics.jsonl').write_text('{"epoch":0}\n{"epoch":1}\n{"epoch":')
    refreshed = next(item for item in joint.catalog()['models'] if item['id'] == model['id'])
    assert refreshed['completed_epochs'] == 2
    assert {item['epoch'] for item in refreshed['snapshots']} == {1, 2}
    latest = joint.frame(0, 1, model['id'], 'last.pt')
    assert latest['joint']['epoch'] == 2
    assert latest['reconstruction'] != original['reconstruction']
    assert joint.frame(0, 1, model['id'], 'viewer_epochs/epoch_001.pt')['reconstruction'] == original['reconstruction']
    snapshot = torch.load(root / 'viewer_epochs/epoch_002.pt', weights_only=False)
    assert snapshot['inference_only'] and 'optimizer' not in snapshot
    assert 'torch_rng' not in snapshot


def test_feature_cache_retains_clip_hits_while_loading_misses(visual_viewer):
    _, joint, model, _, _, requests = visual_viewer
    reference = joint.frame(0, 5, model['id'])
    needed_key = (0, 1, 'ego')
    needed = joint.features[needed_key]
    joint.features.clear()
    joint.features[needed_key] = needed  # A required hit is the oldest cache entry.
    for index in range(63):
        joint.features[(999, index, 'ego')] = needed
    assert len(joint.features) == 64
    # New frames must not evict the old required hit before assembling the clip.
    joint.predictions.clear()
    again = joint.frame(0, 5, model['id'])
    assert again['reconstruction'] == reference['reconstruction']
    assert len(joint.features) == 64
    assert needed_key in joint.features
    assert 1 not in requests[-1][0]  # The hit was reused, not re-encoded.


def test_invalid_and_heldout_selections_and_native_shape(visual_viewer):
    viewer, joint, model, _, _, _ = visual_viewer
    for args, match in [(('missing', 'last.pt', 'ego'), 'Unknown'),
                        ((model['id'], '../last.pt', 'ego'), 'not saved'),
                        ((model['id'], 'epoch_009.pt', 'ego'), 'not saved'),
                        ((model['id'], 'last.pt', 'tactile'), 'scene cameras')]:
        with pytest.raises(ValueError, match=match):
            joint.frame(0, 1, *args)
    viewer.episodes[0]['views'] = {'2': 'wrist.mp4'}
    with pytest.raises(ValueError, match='no ego'):
        joint.frame(0, 1, model['id'])
    viewer.episodes[0]['source'] = 'ftp_motiontrans'
    with pytest.raises(ValueError, match='held out'):
        joint.frame(0, 1, model['id'], views='all')
    viewer.episodes[0].update(source='ftp_vla_touch', target_shape=[3, 240, 320],
                               target_kind='optical_rgb', target_keys=['sensor'])
    joint.runs[model['id']]['source_shapes']['ftp_vla_touch'] = [3, 32, 32]
    import tactilelatent.native_optical as native
    from unittest.mock import patch
    with patch.object(native, 'native_shape', return_value=(3, 240, 320)):
        with pytest.raises(ValueError, match='reduced-resolution'):
            joint.frame(0, 1, model['id'], views='all')


def test_short_temporal_clip_and_boundary_indices():
    for length in (1, 2, 8, 15, 101):
        for frame in (0, length - 1):
            indices, position = clip_indices(frame, length)
            assert len(indices) == 8 and indices[position] == frame
            assert (indices >= 0).all() and (indices < length).all()


def test_attention_endpoint_epoch_camera_masks_and_exact_input_image(visual_viewer,monkeypatch):
    from io import BytesIO
    import base64
    from PIL import Image
    from tactilelatent.vjepa_spatial import JointSpatialVJEPA,MODEL_TYPE as SPATIAL_TYPE
    viewer,joint,run,_,_,_=visual_viewer
    with pytest.raises(ValueError,match='spatial-attention'):
        joint.attention(0,5,run['id'])
    record=joint.runs[run['id']]
    record['config'].update(model_type=SPATIAL_TYPE,prediction_mode='current_frame')
    network=JointSpatialVJEPA(sensor_code_dims={'egotouch':3},numeric_shapes={},
        statistics={'egotouch':dict(mean=[0.]*3,std=[1.]*3)},input_dim=12,width=16,
        heads=4,layers=2,feedforward=32,grid_size=2,prediction_mode='current_frame').eval()
    monkeypatch.setattr(joint,'_load',lambda *args:((network,None,18,record['source_shapes']),(18,1,2)))
    def features(episode,indices,mode,cfg):
        mask=torch.zeros(1,8,3,dtype=torch.bool);mask[:,:,0]=True
        if mode=='all':mask[:,:,2]=True
        return torch.randn(1,4,3,4,12),mask
    monkeypatch.setattr(joint,'_video_features',features)
    def scenes(row,indices,size=384):
        rgb=torch.zeros(len(indices),3,3,size,size)
        rgb[:,0]=torch.as_tensor(indices).view(-1,1,1,1)/12
        return rgb,torch.ones(len(indices),3,dtype=torch.bool)
    import tactilelatent.joint_viewer as module
    monkeypatch.setattr(module,'scene_rgb',scenes)
    result=joint.attention(0,5,run['id'],expected_epoch=18)
    a=np.frombuffer(base64.b64decode(result['weights_b64']),dtype='<f4').reshape(result['shape'])
    assert result['shape']==[2,4,4,3,2,2] and result['camera_slots']==['0']
    assert result['clip_frames'][-1]==5 and max(result['clip_frames'])==5
    assert (a[:,:,:,1:]==0).all()
    with pytest.raises(ValueError,match='Checkpoint updated'):
        joint.attention(0,5,run['id'],expected_epoch=19)
    all_cameras=joint.attention(0,5,run['id'],views='all')
    assert all_cameras['camera_slots']==['0','2']
    with pytest.raises(ValueError,match='not an input'):
        joint.attention_image(0,5,run['id'],'last','ego','2',7)
    with pytest.raises(ValueError,match='History slot'):
        joint.attention_image(0,5,run['id'],'last','all','0',8)
    image=Image.open(BytesIO(joint.attention_image(0,5,run['id'],'last','ego','0',7)))
    assert image.size==(384,384) and abs(np.asarray(image).mean()-255*5/12)<2
    server=ThreadingHTTPServer(('127.0.0.1',0),make_handler(viewer,joint))
    thread=Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        url=f'http://127.0.0.1:{server.server_port}/api/joint/attention?episode=0&frame=5&model={run["id"]}&expected_epoch=18'
        with urlopen(url) as response:assert json.load(response)['epoch']==18
    finally:server.shutdown();server.server_close();thread.join()


def test_ego_only_selection_masks_extra_cameras_from_converted_storage(visual_viewer,monkeypatch):
    _,joint,_,_,_,_=visual_viewer
    import tactilelatent.joint_viewer as module
    from tactilelatent.vjepa_spatial import MODEL_TYPE as SPATIAL_TYPE
    joint.video_backbone=object()
    monkeypatch.setattr(module,'scene_rgb',lambda row,indices,size:(
        torch.ones(len(indices),3,3,size,size),torch.ones(len(indices),3,dtype=torch.bool)))
    def extract(backbone,batch,device,chunk_size):
        assert batch['view_mask'][:,:,0].all() and not batch['view_mask'][:,:,1:].any()
        assert (batch['rgb'][:,:,1:]==0).all()
        return torch.zeros(1,4,3,4,12)
    monkeypatch.setattr(module,'spatial_video_features',extract)
    joint._video_features(0,np.arange(8),'ego',dict(model_type=SPATIAL_TYPE))


@pytest.mark.parametrize('encoder_type', ['legacy', 'alignment'])
def test_frozen_encoder_transfer_pairs_and_inference(visual_viewer, monkeypatch, encoder_type):
    from tactilelatent.frozen_probe import head, sha256
    viewer, joint, _, state, root, _ = visual_viewer
    state['config']['heldout_source'] = 'ftp_motiontrans'
    if encoder_type == 'alignment':
        from tactilelatent.alignment_model import JointSensorAlignment, MODEL_TYPE
        import tactilelatent.joint_viewer as module
        teacher = root / 'teacher.pt'
        teacher.write_bytes(b'pinned teacher')
        registry = {'sources': {'egotouch': {'checkpoint': str(teacher),
            'sha256': sha256(teacher), 'target_shape': [2, 21, 21]}}}
        registry_path = root / 'teachers.json'
        registry_path.write_text(json.dumps(registry))
        state['config'].update(model_type=MODEL_TYPE, teachers=str(registry_path),
                               manifest='test-native-manifest')
        cfg = state['config']
        model_cfg = {key: cfg[key] for key in ('width', 'heads', 'temporal_layers',
                    'cross_view_layers', 'latent_dim', 'dropout', 'stochastic')}
        model_cfg.update(sensor_code_dims={'egotouch': 4}, numeric_shapes={},
                         statistics={'egotouch': {'mean': [0.] * 4, 'std': [1.] * 4}},
                         adapter_hidden_dims=[8])
        network = JointSensorAlignment(**model_cfg)
        state.update(model_type=MODEL_TYPE, model_config=model_cfg,
                     model=network.state_dict(), teachers=registry,
                     provenance={'data': {'test': True}})
        monkeypatch.setattr(module, 'manifest_signature', lambda _: {'test': True})
        torch.save(state, root / 'last.pt')
        for path in (root / 'viewer_epochs').glob('*.pt'):
            path.unlink()
    (root / 'config.json').write_text(json.dumps(state['config']))
    viewer.episodes.append({'source': 'ftp_motiontrans', 'episode_id': 'transfer/test',
        'split': 'val', 'frames': 12, 'views': {'0': 'ego.mp4'}, 'target_transform_scale': 100.})
    monkeypatch.setattr(viewer, '_cached_target', lambda row, frame: np.full((6, 1, 1), .5, np.float32))
    for epoch, bias in [(3, 0.), (10, 2.)]:
        folder = root / 'evaluations' / f'motiontrans_epoch_{epoch:03d}'
        folder.mkdir(parents=True)
        state['epoch'] = epoch - 1
        torch.save(state, folder / 'checkpoint.pt')
        decoder = head(4, 6, hidden_dims=[8])
        for param in decoder.parameters():
            param.data.zero_()
        decoder[3].bias.data.fill_(bias)
        digest = sha256(folder / 'checkpoint.pt')
        torch.save({'head': decoder.state_dict(), 'config': {'checkpoint': str(folder / 'checkpoint.pt'),
                    'source': 'ftp_motiontrans', 'hidden_dims': [8]},
                    'checkpoint_sha256': digest, 'selected_epochs': 7}, folder / 'decoder.pt')
        (folder / 'results.json').write_text(json.dumps({'source': 'ftp_motiontrans',
            'frozen_encoder': True, 'checkpoint_sha256': digest, 'checkpoint_epoch': epoch - 1,
            'target_shape': [6, 1, 1]}))
    transfer = next(m for m in joint.catalog()['models'] if m['kind'].startswith('Frozen encoder'))
    assert transfer['source_shapes'] == {'ftp_motiontrans': [6, 1, 1]}
    assert [s['epoch'] for s in transfer['snapshots']] == [10, 3]
    assert 'transfer_heads' not in transfer
    predictions = []
    for snapshot in transfer['snapshots']:
        result = joint.frame(1, 5, transfer['id'], snapshot['id'])
        expected = .5 if snapshot['epoch'] == 3 else torch.tensor(2.).sigmoid().item()
        assert result['reconstruction'] == pytest.approx([expected] * 6)
        assert result['joint']['epoch'] == snapshot['epoch']
        assert result['joint']['decoder_epochs'] == 7
        assert result['joint']['frozen_encoder']
        assert result['joint']['decoder_widths'] == [4, 8, 6]
        predictions.append(result['reconstruction'])
    assert predictions[0] != predictions[1]
    # Updating/replacing an encoder must never silently reuse its fitted decoder.
    stale = root / 'evaluations' / 'motiontrans_epoch_010' / 'checkpoint.pt'
    state['epoch'] = 8
    torch.save(state, stale)
    refreshed = next(m for m in joint.catalog()['models'] if m['id'] == transfer['id'])
    assert [s['epoch'] for s in refreshed['snapshots']] == [3]
    with pytest.raises(ValueError, match='not saved'):
        joint.frame(1, 5, transfer['id'], transfer['snapshots'][0]['id'])


def test_catalog_filters_unusable_snapshots(visual_viewer):
    _, joint, model, state, root, _ = visual_viewer
    for name in ('training_smoke', 'pending', 'wrong_backbone', 'wrong_shapes', 'broken'):
        folder = root.parent / name
        folder.mkdir()
        (folder / 'config.json').write_text(json.dumps(state['config']))
        if name == 'pending':
            continue
        invalid = {**state}
        if name == 'wrong_backbone':
            invalid['backbone'] = 'obsolete'
        elif name == 'wrong_shapes':
            invalid['model'] = {**state['model'], 'visual.project.weight': torch.zeros(1)}
        if name == 'broken':
            (folder / 'last.pt').write_bytes(b'not a checkpoint')
        else:
            torch.save(invalid, folder / 'last.pt')
    assert {item['id'] for item in joint.catalog()['models']} == {model['id']}
    (root / 'epoch_099.pt').write_bytes(b'not a checkpoint')
    refreshed = joint.catalog()['models'][0]
    assert 'epoch_099.pt' not in {snapshot['id'] for snapshot in refreshed['snapshots']}


def test_alignment_catalog_shares_provenance_checks_and_rejects_wrong_shapes(visual_viewer, monkeypatch):
    import hashlib
    import tactilelatent.joint_viewer as module
    from tactilelatent.alignment_model import JointSensorAlignment, MODEL_TYPE
    _, joint, _, state, root, _ = visual_viewer
    teacher = root / 'teacher.pt'
    teacher.write_bytes(b'pinned teacher')
    registry = {'sources': {'egotouch': {'checkpoint': str(teacher),
        'sha256': hashlib.sha256(teacher.read_bytes()).hexdigest(), 'target_shape': [2, 21, 21]}}}
    registry_path = root / 'teachers.json'
    registry_path.write_text(json.dumps(registry))
    cfg = {**state['config'], 'model_type': MODEL_TYPE, 'teachers': str(registry_path),
           'manifest': 'test-native-manifest'}
    model_cfg = {key: cfg[key] for key in ('width', 'heads', 'temporal_layers',
                                         'cross_view_layers', 'latent_dim', 'dropout', 'stochastic')}
    model_cfg.update(sensor_code_dims={'egotouch': 4}, numeric_shapes={},
                     statistics={'egotouch': {'mean': [0.] * 4, 'std': [1.] * 4}},
                     adapter_hidden_dims=[8])
    network = JointSensorAlignment(**model_cfg)
    state = {**state, 'config': cfg, 'model_type': MODEL_TYPE, 'model_config': model_cfg,
             'model': network.state_dict(), 'teachers': registry, 'provenance': {'data': {'test': True}}}
    (root / 'config.json').write_text(json.dumps(cfg))
    for path in (root / 'viewer_epochs').glob('*.pt'):
        path.unlink()
    torch.save(state, root / 'last.pt')
    state['epoch'] = 1
    torch.save(state, root / 'epoch_002.pt')
    calls = []
    def signature(path):
        calls.append(path)
        return {'test': True}
    monkeypatch.setattr(module, 'manifest_signature', signature)
    model = joint.catalog()['models'][0]
    assert {s['epoch'] for s in model['snapshots']} == {1, 2}
    assert calls == ['test-native-manifest']  # One fresh source scan across epochs.
    bad = {**state, 'model': {**state['model'], 'visual.project.weight': torch.zeros(1)}}
    torch.save(bad, root / 'epoch_002.pt')
    assert {s['id'] for s in joint.catalog()['models'][0]['snapshots']} == {'last.pt'}


def test_joint_http_page_and_proxy_routes(visual_viewer):
    viewer, joint, model, _, _, _ = visual_viewer
    server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(viewer, joint))
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    base = f'http://127.0.0.1:{server.server_port}/proxy/{server.server_port}/'
    try:
        with urlopen(base + '?page=joint') as response:
            html = response.read().decode()
            assert 'id="epoch"' in html and 'id="modelB"' in html
        with urlopen(base + 'joint/') as response:
            assert response.status == 200
        with urlopen(base + 'api/joint/catalog') as response:
            assert model['id'] in {item['id'] for item in json.load(response)['models']}
        with urlopen(base + f'api/joint/frame?episode=0&frame=2&model={model["id"]}&epoch=last.pt') as response:
            assert json.load(response)['joint']['epoch'] == 1
    finally:
        server.shutdown(); server.server_close(); worker.join()


@pytest.mark.parametrize('mode', ['deterministic', 'gaussian'])
def test_vjepa_catalog_clip_cache_and_native_inference(visual_viewer, monkeypatch, mode):
    import tactilelatent.joint_viewer as module
    from tactilelatent.vjepa_model import JointVJEPAAlignment, MODEL_TYPE, BACKBONE_ID, REVISION, CHECKPOINT_SHA256
    from tactilelatent.sensor_autoencoder import SensorAutoencoder
    from tactilelatent.aligned_frames import file_hash
    viewer,joint,_,old,root,_=visual_viewer
    teacher=SensorAutoencoder([2,21,21],latent_dim=3,channels=[4])
    teacher_path=root/'teacher.pt'
    torch.save(dict(model_config=teacher.config,model=teacher.state_dict()),teacher_path)
    registry={'sources':{'egotouch':dict(checkpoint=str(teacher_path),sha256=file_hash(teacher_path),target_shape=[2,21,21])}}
    registry_path=root/'teachers.json';registry_path.write_text(json.dumps(registry))
    cfg={**old['config'],'model_type':MODEL_TYPE,'latent_objective':'deterministic' if mode=='deterministic' else 'vae',
         'latent_mode':mode,'stochastic':mode=='gaussian','teachers':str(registry_path),'manifest':'native-test',
         'backbone_repo':'repo','backbone_checkpoint':'vjepa-test','video_chunk_size':2,'latent_dim':32}
    mc=dict(sensor_code_dims={'egotouch':3},numeric_shapes={},statistics={'egotouch':{'mean':[0.]*3,'std':[1.]*3}},latent_mode=mode)
    network=JointVJEPAAlignment(**mc)
    state=dict(config=cfg,model_type=MODEL_TYPE,model_config=mc,model=network.state_dict(),epoch=0,
        backbone=BACKBONE_ID,backbone_revision=REVISION,teachers=registry,
        provenance={'data':{'test':True},'backbone':{'checkpoint_sha256':CHECKPOINT_SHA256}})
    torch.save(state,root/'last.pt');(root/'config.json').write_text(json.dumps(cfg))
    for path in (root/'viewer_epochs').glob('*.pt'):path.unlink()
    monkeypatch.setattr(module,'manifest_signature',lambda _: {'test':True})
    original_hash=joint._hash
    monkeypatch.setattr(joint,'_hash',lambda path: CHECKPOINT_SHA256 if path=='vjepa-test' else original_hash(path))
    calls=[]
    def scenes(row,indices,size):
        assert size==384
        calls.append(tuple(map(int,indices)))
        rgb=torch.zeros(8,3,3,384,384);mask=torch.zeros(8,3,dtype=torch.bool)
        for slot in row['views']:
            rgb[:,int(slot)]=torch.as_tensor(indices).view(8,1,1,1)/12
            mask[:,int(slot)]=True
        return rgb,mask
    class Backbone(nn.Module):
        dim=768
        def __init__(self,*args):super().__init__()
        def forward(self,x):
            return x[:,0,:,0,0].reshape(len(x),4,2).mean(-1).unsqueeze(-1).expand(-1,-1,768)
    monkeypatch.setattr(module,'scene_rgb',scenes);monkeypatch.setattr(module,'VJEPABackbone',Backbone)
    catalog=joint.catalog();model=next(m for m in catalog['models'] if m['config'].get('model_type')==MODEL_TYPE)
    assert model['name'].startswith('V-JEPA')
    result=joint.frame(0,5,model['id'])
    assert result['original']==viewer.frame(0,5,-1)['original']
    assert result['joint']['clip_frames'][3]==5
    assert result['joint']['uncertainty']==('none' if mode=='deterministic' else 'uncalibrated')
    assert len(calls)==1
    joint.predictions.clear();joint.frame(0,5,model['id']);assert len(calls)==1
    joint.frame(0,6,model['id']);assert len(calls)==2
    assert calls[0]!=calls[1]
