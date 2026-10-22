import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import zipfile

import numpy as np
import pytest
import torch


@pytest.fixture
def migration(tmp_path, monkeypatch):
    # Runtime configuration normally re-execs for CUDA; archive tests need none.
    import tactilelatent.expansion_runtime as runtime
    monkeypatch.setattr(runtime, 'configure', lambda:None)
    spec=importlib.util.spec_from_file_location('marker_retirement',Path('scripts/retire_marker_images.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(module,'ROOT',tmp_path/'receipts')
    module.ROOT.mkdir()
    return module


def make_tar(path, entries):
    with tarfile.open(path,'w:') as archive:
        for name,value in entries.items():
            member=tarfile.TarInfo(name);member.size=len(value)
            archive.addfile(member,io.BytesIO(value))


def test_tactile_only_repacking_preserves_cameras_metadata_and_other_sensors(migration,tmp_path):
    entries={'Unit/task.zarr/data/right_tactile_data_gripper/0':b'dots'*1000,
        'Unit/task.zarr/data/right_tactile_data_gripper/.zarray':b'image geometry',
        'Unit/task.zarr/data/right_arm_joints/0':b'joints',
        'Unit/task.zarr/data/camera_main_rgb/0':b'RGB untouched',
        'Unit/task.zarr/meta/episode_ends/0':b'chronology',
        'Unit/task.zarr/data/right_tactile_type_gripper/0':b'sensor identity'}
    old=tmp_path/'old.tar';new=tmp_path/'new.tar';make_tar(old,entries)
    detail,index=migration.stage_tar(old,new)
    expected={k:hashlib.sha256(v).hexdigest() for k,v in entries.items() if '/right_tactile_data_gripper/' not in k}
    assert detail['retained_sha256']==expected
    assert set(index)==set(expected)
    with new.open('rb') as stream:
        for name,(offset,length) in index.items():
            stream.seek(offset);assert stream.read(length)==entries[name]
    zold=tmp_path/'old.zip';znew=tmp_path/'new.zip'
    with zipfile.ZipFile(zold,'w') as archive:
        archive.writestr('episode/gelsight/gel_0.jpg',b'tactile')
        archive.writestr('episode/camera1/rgb_0.jpg',b'scene')
        archive.writestr('episode/metadata.json',b'valid')
    migration.stage_zip(zold,znew)
    with zipfile.ZipFile(znew) as archive:
        assert archive.namelist()==['episode/camera1/rgb_0.jpg','episode/metadata.json']
        assert archive.read('episode/camera1/rgb_0.jpg')==b'scene'


def test_interrupted_archive_commit_finishes_index_and_registry(migration,tmp_path,monkeypatch):
    storage=tmp_path/'storage';storage.mkdir();monkeypatch.setattr(migration,'STORAGE',storage)
    (storage/'registry.json').write_text(json.dumps({'replacements':{}}))
    old=tmp_path/'data.tar';staged=tmp_path/'staged.tar';index=tmp_path/'index.json';staged_index=tmp_path/'new.json'
    old.write_bytes(b'original');staged.write_bytes(b'kept');index.write_text('old offsets');staged_index.write_text('new offsets')
    digest=migration.file_hash
    value=dict(kind='tar',path=str(old),temporary=str(staged),old_sha256=digest(old),new_sha256=digest(staged),
        old_stat=migration.stat(old),index=str(index),temporary_index=str(staged_index),
        old_index_sha256=digest(index),new_index_sha256=digest(staged_index))
    staged.replace(old)  # A process died after archive replacement, before index.
    receipt=migration.ROOT/'archive.json'
    migration.commit(receipt,value)
    assert old.read_bytes()==b'kept' and index.read_text()=='new offsets'
    registry=json.loads((storage/'registry.json').read_text())
    assert registry['replacements'][str(old.resolve())]['new_stat']==migration.stat(old)
    migration.commit(receipt,value)  # Completed commits are idempotent.
    assert json.loads(receipt.read_text())['saved_bytes']==4


def test_staging_rejects_symlink(migration,tmp_path):
    original=tmp_path/'original';original.write_bytes(b'preserve')
    link=tmp_path/'link';link.symlink_to(original)
    with pytest.raises(ValueError,match='symlink'):
        migration.stage(dict(path=str(link)))
    assert original.read_bytes()==b'preserve'


def test_marker_viewer_uses_geometry_without_any_optical_reads(tmp_path,monkeypatch):
    import tactilelatent.marker_viewer as viewer
    import tactilelatent.aligned_frames as frames
    import tactilelatent.native_optical as native
    row=dict(source='ftp_unit_bimanual',split='val',episode_id='test',frames=3,target_shape=[6,240,320])
    root=tmp_path/'targets';folder=root/row['source']/row['split']/viewer.episode_key(row);folder.mkdir(parents=True)
    (folder/'complete.json').write_text(json.dumps({'sensors':[{'shape':[240,320,3]}]*2}))
    b={'marker_spacing':torch.tensor([20.,20.]),'marker_surface':torch.tensor([0,1]),
       'marker_target':torch.tensor([[[.5,0.],[float('nan'),float('nan')]]]),'marker_valid':torch.tensor([[True,False]])}
    monkeypatch.setattr(viewer,'inputs',lambda *args:(row,b,np.array([[20.,30.],[40.,50.]],np.float32)))
    def forbidden(*args,**kwargs):raise AssertionError('Decoded retired tactile imagery')
    monkeypatch.setattr(frames,'target_images',forbidden);monkeypatch.setattr(native,'native_images',forbidden)
    result=viewer.payload(dict(marker_targets=str(root)),row,1,np.zeros((2,2),np.float32))
    assert len(result['optical_panels'])==2
    assert result['marker_displacement']['arrow_gain']==1
    assert result['marker_displacement']['measured_px']==[[10.,0.],None]
    assert result['mae']==5.
    json.dumps(result,allow_nan=False)


def test_native_shape_and_image_options_follow_explicit_retirement(tmp_path,monkeypatch):
    import tactilelatent.marker_storage as storage
    import tactilelatent.native_optical as native
    from tactilelatent.autoencoder_viewer import Viewer
    from tactilelatent.joint_viewer import source_shapes
    receipt=tmp_path/'retirement.json';monkeypatch.setattr(storage,'RECEIPT',receipt)
    receipt.write_text(json.dumps({'active':True,'sources':{'ftp_unit':{'target_shape':[3,240,320]}}}))
    row=dict(source='ftp_unit',target_shape=[3,240,320])
    assert native.native_shape(row)==[3,240,320]
    with pytest.raises(ValueError,match='retired'):
        native.native_images(row,[0])
    root=tmp_path/'models';model=root/'old';model.mkdir(parents=True)
    (model/'config.json').write_text(json.dumps({'source':'ftp_unit','native_optical':True,
        'model':{'architecture':'conv'}}));(model/'best.pt').write_bytes(b'weights retained')
    instance=Viewer.__new__(Viewer);instance.checkpoint_roots=[root]
    assert instance._scan_models()==[]
    cfg=dict(dataset_family='joint_all',train_sources=['ftp_unit','egotouch'])
    assert 'ftp_unit' not in source_shapes(cfg)
    assert 'ftp_unit' in source_shapes(cfg,include_retired_images=True)


def test_legacy_joint_structure_keeps_retired_heads_when_loading_weights(tmp_path,monkeypatch):
    import tactilelatent.marker_storage as storage
    from tactilelatent.joint_viewer import JointViewer,source_shapes
    from tactilelatent.joint_model import JointTactileVAE
    from tactilelatent.train import MODEL_CONFIG_KEYS
    from tactilelatent.joint_all_config import ALL_TARGETS
    from tactilelatent.model import MODEL_ID,REVISION
    receipt=tmp_path/'retirement.json';monkeypatch.setattr(storage,'RECEIPT',receipt)
    receipt.write_text(json.dumps({'active':True,'sources':{'ftp_unit':{'target_shape':[3,240,320]}}}))
    cfg=dict(dataset_family='joint_all',train_sources=['egotouch','ftp_unit'],latent_objective='vae',
        width=16,heads=4,temporal_layers=1,cross_view_layers=1,latent_dim=4,dropout=0.,stochastic=True)
    targets={s:(ALL_TARGETS[s][0],shape,ALL_TARGETS[s][2]) for s,shape in source_shapes(cfg,True).items()}
    network=JointTactileVAE(targets=targets,**{k:cfg[k] for k in MODEL_CONFIG_KEYS})
    checkpoint=tmp_path/'legacy.pt'
    torch.save(dict(config=cfg,model=network.state_dict(),epoch=0,backbone=MODEL_ID,backbone_revision=REVISION),checkpoint)
    viewer=JointViewer.__new__(JointViewer);viewer.snapshot_checks={}
    assert viewer._inspect_snapshot(checkpoint) is not None
    assert 'ftp_unit' not in source_shapes(cfg)
