"""No episodic fitting, hidden clipping, native-shape changes or mask loss."""
from pathlib import Path
import json

import numpy as np
import pytest

from tactilelatent.aligned_frames import file_hash
from tactilelatent.zero_reference import FixedReferences, SCHEMA, save_registry


def registry(tmp_path):
    path = tmp_path/'references.json'
    entry = dict(source='sensor', revision='release', target_shape=[2, 3],
        channel_order=['a','b','direction','c','d','direction2'],
        offset=[2,-3,0,4,5,0], display_limits=[100]*6)
    save_registry(path, dict(schema=SCHEMA, fit_split='train', episode_refitting=False,
        numeric={'sensor':entry}, optical={}))
    return path, entry


def test_fixed_transform_has_same_offset_for_every_observation_and_preserves_direction_and_mask(tmp_path):
    path, entry = registry(tmp_path)
    raw = np.array([[[2.001,-3.002,-1],[3,6,270]], [[12,-13,12],[4,7,-1]]])
    original = raw.copy(); valid = np.ones_like(raw,dtype=bool);valid[1,0,1]=False
    refs = FixedReferences(path)
    corrected,mask = refs.transform('sensor',raw,valid,entry['channel_order'],'release')
    np.testing.assert_allclose(corrected[0],[[.001,-.002,-1],[-1,1,270]])
    np.testing.assert_allclose(corrected[:,1,2],raw[:,1,2])
    assert np.isnan(corrected[1,0,1]) and not mask[1,0,1]
    assert corrected[1,0,0] == 10
    np.testing.assert_array_equal(raw,original)
    for i in range(2):
        one,one_mask=refs.transform('sensor',raw[i],valid[i],entry['channel_order'],'release')
        np.testing.assert_allclose(one,corrected[i],equal_nan=True)
        np.testing.assert_array_equal(one_mask,mask[i])


def test_release_layout_channel_order_and_masks_must_match(tmp_path):
    path,e=registry(tmp_path);refs=FixedReferences(path)
    assert refs.numeric('sensor',[2,3],e['channel_order'],'other') is None
    assert refs.numeric('sensor',[6],e['channel_order'],'release') is None
    assert refs.numeric('sensor',[2,3],list(reversed(e['channel_order'])),'release') is None
    for raw,valid,revision in [(np.ones(6),np.ones(6),'release'),
            (np.ones((2,3)),np.ones(6),'release'),(np.ones((2,3)),np.ones((2,3)),'other')]:
        with pytest.raises(ValueError):refs.transform('sensor',raw,valid,e['channel_order'],revision)


def test_different_sensor_layouts_share_no_offsets(tmp_path):
    path,e=registry(tmp_path)
    value=json.loads(path.read_text());value['numeric']['other_layout']=dict(e,
        target_shape=[1],channel_order=['only'],offset=[42],display_limits=[100])
    save_registry(path,value);refs=FixedReferences(path)
    one,_=refs.transform('sensor',np.array([43]),np.array([True]),['only'],'release')
    assert one.tolist()==[1]
    assert refs.numeric('sensor',[2,3],e['channel_order'],'release')['offset']==e['offset']


def test_registry_tampering_and_nontraining_fit_are_rejected(tmp_path):
    path,_=registry(tmp_path);value=json.loads(path.read_text())
    value['numeric']['sensor']['offset'][0]=123;path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='hash'):FixedReferences(path).catalog()
    value['fit_split']='val';save_registry(path,value)
    with pytest.raises(ValueError,match='training'):FixedReferences(path).catalog()
    value['fit_split']='train';value['episode_refitting']=True;save_registry(path,value)
    with pytest.raises(ValueError,match='frozen'):FixedReferences(path).catalog()


def test_optical_reference_is_signed_native_shape_and_frozen_across_images(tmp_path):
    path,_=registry(tmp_path);template=np.array([[[255,10,0],[3,40,80]]],np.float32)
    np.savez(tmp_path/'template.npz',reference=template)
    payload=json.loads(path.read_text());payload['optical']={'gel:1x2x3:uint8':dict(
        sensor='gel',revision='release',file='template.npz',sha256=file_hash(tmp_path/'template.npz'))}
    save_registry(path,payload);refs=FixedReferences(path)
    first=refs.optical('gel',template.astype(np.uint8),'release')['residual']
    assert first.shape==(1,2,3) and np.count_nonzero(first)==0
    second=refs.optical('gel',np.zeros((1,2,3),np.uint8),'release')['residual']
    assert second[0,0,0]==-255 and second[0,1,2]==-80
    assert refs.optical('gel',np.zeros((2,2,3),np.uint8),'release') is None
    assert refs.optical('gel',np.zeros((1,2,3),np.uint8),'wrong') is None


def test_current_references_have_only_training_fits_and_preserve_hrdex_directions():
    path=Path('processing/zero_reference_v1/references.json')
    if not path.exists():pytest.skip('Local acquisition calibration artifact is absent')
    refs=FixedReferences(path);catalog=refs.catalog()
    assert catalog['episode_refitting'] is False
    selections=json.loads(path.with_name('reviewed_numeric_frames.json').read_text())
    for key,e in catalog['numeric'].items():
        selected={s['episode']:s for s in selections[key]}
        assert all(selected[name]['split']=='train' for name in e['reference_fit_episodes'])
        assert e['source'] not in ('ftp_motiontrans','freetacman')
        assert not e['deadband']
    f1=catalog['numeric']['hrdexdb_inspire_f1']
    assert f1['offset'][2::3]==[0]*8
    assert f1['unloaded_envelope'][2::3]==[None]*8
    for e in catalog['optical'].values():
        selected=json.loads(path.with_name('optical_candidates.json').read_text())
        key=e['sensor']+':'+ 'x'.join(map(str,e['target_shape']))+':'+e['native_dtype']
        assert not (set(e['fit_objects']) & {s['object'] for s in selected[key] if s['split']=='val'})
