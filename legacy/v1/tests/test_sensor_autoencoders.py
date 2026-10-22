import json

import numpy as np
import pytest
import torch

from tactilelatent.joint_all_config import ALL_TARGETS
from tactilelatent.joint_data import JointClipDataset
from tactilelatent.sensor_autoencoder import SensorAutoencoder, make_autoencoder
from tactilelatent.sensor_cache import build_cache, CachedSensorFrames
from tactilelatent.sensor_data import load_targets, scene_frames
from tactilelatent.pretrain_sensor_autoencoder import main
from tactilelatent.native_optical import native_shape, native_images, NativeOpticalFrames


@pytest.mark.parametrize('source', list(ALL_TARGETS))
def test_native_shape_mask_and_frozen_decoder_gradient(source):
    _, shape, _ = ALL_TARGETS[source]
    if ALL_TARGETS[source][0] == 'optical_rgb':
        row = next(r for r in json.load(open('processing/joint_all/manifest.json'))['episodes'] if r['source'] == source)
        shape = native_shape(row)
    architecture = 'conv' if min(shape[1:]) > 1 else 'mlp'
    model = SensorAutoencoder(shape, architecture, latent_dim=2, channels=[4, 8], hidden_dims=[16])
    target = torch.rand(2, *shape)
    mask = torch.ones_like(target, dtype=torch.bool)
    mask.reshape(-1)[0] = False
    code = model.encode(target, mask)
    changed = target.clone(); changed.reshape(-1)[0] = float('nan')
    assert torch.equal(code, model.encode(changed, mask))
    prediction = model.decode(code)
    assert prediction.shape == target.shape
    assert torch.isfinite(prediction).all()
    model.decoder.requires_grad_(False)
    code = torch.randn(2, 2, requires_grad=True)
    model.decode(code).sum().backward()
    assert code.grad.abs().sum() > 0
    loaded = make_autoencoder(model.config); loaded.load_state_dict(model.state_dict())
    assert torch.allclose(model(target, mask), loaded(target, mask))


@pytest.mark.parametrize('source', list(ALL_TARGETS))
def test_target_only_loader_matches_existing_joint_contract(source):
    dataset = JointClipDataset('processing/joint_all/manifest.json', 'train', [source], frames=2, stride=2)
    expected = dataset[0]
    native = load_targets(dataset.episodes[0], [0, 2])
    if ALL_TARGETS[source][0] == 'optical_rgb':
        row = dataset.episodes[0]
        assert list(native.shape[1:]) == native_shape(row)
        raw = native_images(row, [0, 2])
        expected_pixels = np.concatenate([image.transpose(0,3,1,2) for image in raw],axis=1).astype(np.float32)/255.
        np.testing.assert_array_equal(native, expected_pixels)
        with pytest.raises(ValueError, match='full-resolution'):
            build_cache('processing/joint_all/manifest.json', source)
        return
    valid = np.isfinite(native)
    assert np.array_equal(valid, expected['target_mask'].numpy())
    np.testing.assert_allclose(np.where(valid,native,0.), expected['target'].numpy(), rtol=0, atol=0)
    # All storage types also expose aligned RGB frames to the viewer.
    camera = next(iter(dataset.episodes[0]['views']))
    image = scene_frames(dataset.episodes[0], [0], camera, size=48)
    assert image.shape == (1,3,48,48)


def tiny_manifest(tmp_path):
    data = (np.arange(8*2*80,dtype=np.float32).reshape(8,2,80) % 4095)
    data[0,0,0] = np.nan
    array = tmp_path/'targets.npy';np.save(array,data)
    rows = []
    for split,start in [('train',0),('val',4)]:
        rows.append({'source':'opengraph','episode_id':split,'split':split,'frames':4,
                     'target_path':str(array),'target_start':start,'video_start':0,'views':{'0':'unused.mp4'},
                     'target_shape':[2,1,80],'target_scale':4095.,'target_kind':'taxel_vector'})
    path=tmp_path/'manifest.json';path.write_text(json.dumps({'schema_version':1,'episodes':rows}))
    return path


def test_cache_alignment_invalid_values_and_resume_training(tmp_path):
    manifest=tiny_manifest(tmp_path);cache=tmp_path/'cache'
    index=build_cache(manifest,'opengraph',cache,stride=2)
    train=CachedSensorFrames(cache,'opengraph','train');val=CachedSensorFrames(cache,'opengraph','val')
    assert len(train)==len(val)==2
    assert not train[0]['valid'][0,0,0]
    assert val[0]['target'][0,0,0] == pytest.approx(640/4095)
    assert build_cache(manifest,'opengraph',cache,stride=2)==index
    base=['--source','opengraph','--manifest',str(manifest),'--cache',str(cache),'--architecture','mlp',
          '--sensor-code-dim','2','--hidden-dims','16','--device','cpu','--workers','0','--batch-size','2']
    direct=tmp_path/'direct';resumed=tmp_path/'resumed'
    main(base+['--output',str(direct),'--epochs','2'])
    main(base+['--output',str(resumed),'--epochs','1'])
    main(base+['--output',str(resumed),'--epochs','2','--resume'])
    a=torch.load(direct/'last.pt',weights_only=False);b=torch.load(resumed/'last.pt',weights_only=False)
    assert all(torch.equal(a['model'][key],b['model'][key]) for key in a['model'])
    assert a['scheduler']==b['scheduler']


def test_native_batch_loader_keeps_order_and_shape():
    dataset = NativeOpticalFrames('processing/joint_all/manifest.json','ftp_vla_touch','train',stride=2)
    batch = dataset.__getitems__([2,0,1])
    expected = load_targets(dataset.episodes[0],[0,2,4])
    for actual, index in zip(batch,[2,0,1]):
        assert tuple(actual['target'].shape)==(3,240,320)
        np.testing.assert_array_equal(actual['target'].numpy(),expected[index])


def test_freetacman_episode_camera_roles_and_corrected_native_validation():
    from tactilelatent.prepare_joint import freetacman_camera_roles
    original = json.load(open('processing/joint_all/manifest.json'))['episodes']
    corrected = json.load(open('processing/joint_all/manifest_sensor_roles_corrected_v2.json'))['episodes']
    assert len(original) == len(corrected)
    changed = 0
    for old, new in zip(original, corrected):
        assert old['episode_id'] == new['episode_id'] and old['split'] == new['split']
        if old['source'] != 'freetacman':
            assert old == new
            continue
        roles = freetacman_camera_roles(new['episode_id'])
        assert new['views']['1'].endswith(f"_camera{roles['scene']}.mp4")
        assert new['tactile_videos'][0].endswith(f"_camera{roles['surface1']}.mp4")
        assert new['tactile_videos'][1].endswith('_camera2.mp4')
        changed += old != new
    assert changed == 53
    assert freetacman_camera_roles('SprinkleSesame/SprinkleSesame_1')['scene'] == 1
    assert freetacman_camera_roles('SprinkleSesame/SprinkleSesame_54')['scene'] == 3
    assert freetacman_camera_roles('Slot/Slot_164')['scene'] == 3
    dataset = NativeOpticalFrames('processing/joint_all/manifest_sensor_roles_corrected_v2.json','freetacman','val')
    row = dataset.episodes[0]
    assert row['episode_id'] == 'SprinkleSesame/SprinkleSesame_1'
    sample = dataset[0]['target'].numpy()
    assert sample.shape == (6,480,640)
    import av
    for panel, path in enumerate(row['tactile_videos']):
        with av.open(path) as container:
            raw = next(container.decode(video=0)).to_ndarray(format='rgb24')
        np.testing.assert_array_equal(sample[panel*3:(panel+1)*3],raw.transpose(2,0,1).astype(np.float32)/255.)


@pytest.mark.parametrize('cells,code', [(1, 1), (2, 1), (2, 2), (6, 2), (6, 4)])
def test_affine_numeric_decoder_is_minimal_and_supports_frozen_adapter(cells, code):
    model = SensorAutoencoder([cells, 1, 1], 'mlp', latent_dim=code, hidden_dims=[])
    assert sum(isinstance(layer, torch.nn.Linear) for layer in model.decoder) == 1
    assert sum(parameter.numel() for parameter in model.decoder.parameters()) == cells * (code + 1)
    target = torch.rand(3, cells, 1, 1)
    mask = torch.ones_like(target, dtype=torch.bool)
    mask[0, 0] = False
    altered = target.clone(); altered[0, 0] = float('nan')
    assert torch.equal(model.encode(target, mask), model.encode(altered, mask))
    prediction = model(target, mask)
    assert prediction.shape == target.shape and ((prediction > 0) & (prediction < 1)).all()
    model.decoder.requires_grad_(False)
    latent = torch.randn(3, code, requires_grad=True)
    model.decode(latent).sum().backward()
    assert latent.grad.abs().sum() > 0
    loaded = make_autoencoder(model.config)
    loaded.load_state_dict(model.state_dict())
    assert torch.equal(loaded(target, mask), prediction)


def test_affine_cli_resume_preserves_training_state(tmp_path):
    manifest = tiny_manifest(tmp_path)
    base = ['--source', 'opengraph', '--manifest', str(manifest), '--cache', str(tmp_path/'cache'),
            '--architecture', 'mlp', '--sensor-code-dim', '2', '--hidden-dims',
            '--device', 'cpu', '--workers', '0', '--batch-size', '2']
    main(base + ['--output', str(tmp_path/'affine'), '--epochs', '1'])
    main(base + ['--output', str(tmp_path/'affine'), '--epochs', '2', '--resume'])
    main(base + ['--output', str(tmp_path/'direct'), '--epochs', '2'])
    saved = torch.load(tmp_path/'affine/last.pt', weights_only=False)
    direct = torch.load(tmp_path/'direct/last.pt', weights_only=False)
    assert saved['epoch'] == 1 and saved['config']['model']['hidden_dims'] == []
    assert all(torch.equal(saved['model'][key], direct['model'][key]) for key in saved['model'])
