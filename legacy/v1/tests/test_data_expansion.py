import copy
import json
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from tactilelatent import data_expansion as de
from tactilelatent import expansion_baihu as eb
from tactilelatent.aligned_frames import file_hash,write_json
from tactilelatent.expansion_models import ExpansionMarkerAutoencoder,load_teacher
from tactilelatent.expansion_render import marker_payload
from tactilelatent.marker_autoencoder import MarkerAutoencoder
from tactilelatent.expansion_tracker import BaihuMarkerTracker


def test_pressure_explicit_ids_missing_zero_and_magnitude(tmp_path):
    p=tmp_path/'pressure.parquet'
    zero=np.zeros((224,224),np.float32).ravel().tolist()
    one=np.ones((224,224),np.float32).ravel().tolist()
    pq.write_table(pa.Table.from_pylist([
        {'frame':11,'pressure_map':one,'pressure_map_range':[0.,7.]},
        {'frame':4,'pressure_map':zero,'pressure_map_range':[0.,0.]},
        {'frame':13,'pressure_map':None,'pressure_map_range':None}]),p,row_group_size=1)
    row=dict(target_path=str(p),source_frame_ids=[4,11,12,13])
    value=de.pressure_counts(row,[0,1,2,3])
    assert value.shape==(4,1,224,224)
    assert np.all(value[0]==0) and np.all(value[1]==7)
    assert np.isnan(value[2:]).all()
    p=np.array([0.,1.,7.,1e6],np.float32)
    np.testing.assert_allclose(de.denormalize_pressure(de.normalize_pressure(p,100.),100.),p,rtol=.0003)


def test_four_surfaces_and_legacy_checkpoint(tmp_path):
    torch.set_num_threads(1);torch.manual_seed(45)
    ae=ExpansionMarkerAutoencoder(latent_dim=32,width=32,queries=2,heads=4,displacement_scale=.1)
    y=torch.randn(3,16,2)*.02;valid=torch.ones(3,16,dtype=torch.bool)
    ref=torch.rand(3,16,2);surface=torch.arange(16)[None].expand(3,-1)%4
    pred=ae(y,valid,ref,surface);loss=ae.loss(pred,y,valid,surface);loss.backward()
    assert torch.isfinite(loss) and ae.to_code[-1].weight.grad.abs().sum()>0
    state=dict(model_type='expansion_marker_autoencoder_v1',model_config=ae.config,model=copy.deepcopy(ae.state_dict()))
    path=tmp_path/'four.pt';torch.save(state,path);restored,_=load_teacher(path)
    torch.testing.assert_close(restored(y,valid,ref,surface),pred)
    legacy=MarkerAutoencoder(latent_dim=4,width=32,queries=2)
    assert legacy.surface.num_embeddings==2 and 'surfaces' not in legacy.config


def test_four_surface_validation_broadcasts_marker_masks(monkeypatch):
    from tactilelatent import expansion_training as training
    torch.set_num_threads(1)
    y=torch.randn(3,11,2)*.02
    valid=torch.ones(3,11,dtype=torch.bool);valid[0,2]=False;y[0,2]=float('nan')
    batch=dict(target=y,valid=valid,reference=torch.rand(3,11,2),
               surface=torch.arange(11)[None].expand(3,-1)%4,spacing=torch.ones(11)*20)
    monkeypatch.setattr(training,'batches',lambda *args,**kwargs:iter([batch]))
    model=ExpansionMarkerAutoencoder(width=32,queries=2,displacement_scale=.1)
    metrics=training.evaluate(model,[], 'baihu','cpu')
    assert metrics['frames']==3 and metrics['valid_coordinates']==64
    assert np.isfinite(metrics['mae']) and metrics['units']=='native_pixels'


def test_baihu_cache_verification_replays_producer_batch_context(tmp_path,monkeypatch):
    from tactilelatent import expansion_training as training
    from tactilelatent.aligned_frames import json_hash
    target=tmp_path/'native.npz';target.write_bytes(b'pinned native target')
    checkpoint=tmp_path/'teacher.pt';checkpoint.write_bytes(b'pinned teacher')
    rows=[dict(source='baihu',episode_id=split,split=split,frames=130,marker_path=str(target))
          for split in ('train','val')]
    data=dict(displacement_px=np.arange(130*4*2,dtype=np.float32).reshape(130,4,2)*.001,
              valid=np.ones((130,4),bool),spacing_px=np.ones(4,np.float32),
              reference_uv=np.zeros((4,2),np.float32),surface_id=np.arange(4,dtype=np.int64))
    class BatchSensitiveEncoder:
        def encode(self,y,valid,reference,surface):
            return y[:,0]+len(y)  # Exposes an incorrect isolated-frame replay.
    state=dict(model_config=dict(latent_dim=2),config=dict(target_snapshot_sha256=
        json_hash({r['episode_id']:file_hash(target) for r in rows})))
    monkeypatch.setattr(training,'load_teacher',lambda *args:(BatchSensitiveEncoder(),state))
    monkeypatch.setattr(training,'marker_arrays',lambda row:(data,{}))
    monkeypatch.setattr(training,'report',lambda *args,**kwargs:None)
    result=training.build_codes('baihu',rows,checkpoint,device='cpu',destination=tmp_path/'codes')
    assert result['complete'] and result['frames']==260
    verification=json.loads((tmp_path/'codes/verification.json').read_text())
    assert verification['passed'] and all(c['max_absolute_error']==0 for c in verification['checks'])


def test_safe_delete_rejects_escape_and_directory(tmp_path):
    allowed=tmp_path/'data';allowed.mkdir();external=tmp_path/'outside';external.write_text('keep')
    with pytest.raises(ValueError):de.safe_delete([external],allowed,tmp_path/'log','test')
    link=allowed/'link';link.symlink_to(external)
    with pytest.raises(ValueError):de.safe_delete([link],allowed,tmp_path/'log','test')
    with pytest.raises(ValueError):de.safe_delete([allowed],allowed,tmp_path/'log','test')
    assert external.read_text()=='keep'
    with pytest.raises(ValueError):de.safe_delete([external],tmp_path,tmp_path/'log','test',expected={str(external):'wrong'})
    assert external.exists()


def test_temporal_tracker_retains_cumulative_shear_beyond_reference_gate():
    yy,xx=np.mgrid[:240,:240]
    def image(shift):
        a=np.full((240,240,3),170,np.uint8)
        for y in range(35,176,20):
            for x in range(35,176,20):a[(xx-x-shift)**2+(yy-y)**2<12]=20
        return a
    tracker=BaihuMarkerTracker(image(0))
    for shift in np.arange(0,26,.5):r=tracker.update(image(shift))
    assert r['valid'].all()
    np.testing.assert_allclose(r['displacement_px'][:,0],25.5,atol=.3)
    assert tracker.metadata()['total_displacement_cap'] is None


def _video(path,images):
    with av.open(str(path),'w') as out:
        s=out.add_stream('libx264',rate=30);s.width=s.height=240;s.pix_fmt='yuv420p';s.options={'crf':'18'}
        for image in images:
            for packet in s.encode(av.VideoFrame.from_ndarray(image,format='rgb24')):out.mux(packet)
        for packet in s.encode():out.mux(packet)


def test_complete_episode_consumers_survive_actual_media_removal(tmp_path,monkeypatch):
    monkeypatch.setattr(de,'ROOT',tmp_path/'output');monkeypatch.setattr(de,'RESERVE',0)
    monkeypatch.setattr(eb,'ROOT',tmp_path/'output')
    folder=tmp_path/'source';folder.mkdir()
    images=[]
    yy,xx=np.mgrid[:240,:240]
    for shift in range(4):
        image=np.full((240,240,3),170,np.uint8)
        for y in range(30,211,20):
            for x in range(30,211,20):image[(xx-x-shift*.3)**2+(yy-y)**2<12]=20
        images.append(image)
    for s in de.SURFACES:_video(folder/(s+'.mp4'),images)
    _video(folder/'cam_hand_left.mp4',images)
    (folder/'head_depth.mkv').write_bytes(b'not used by the training pipeline')
    (folder/'metadata.json').write_text('{}')
    row=dict(source='baihu',episode_id='task/PIKA/ep',split='train',frames=4,
        views={'1':str(tmp_path/'rgb.mp4')},original_dir=str(folder),
        marker_path=str(tmp_path/'tracks'/'tracks.npz'),native_sensor_shape=[240,240],target_shape=[12,240,240])
    result=eb.process_episode(row)
    assert result['reclaimed_bytes']>0 and not list(folder.glob('*.mp4'))
    assert not (folder/'head_depth.mkv').exists() and (folder/'metadata.json').exists()
    data,info=de.marker_arrays(row)
    assert len(marker_payload(row,data,3)['optical_panels'])==4
    assert min(info['surface_valid_fraction'])>.95
    assert eb.process_episode(row)['resumed']
    b=de.marker_batch(row,[0,3]);assert b['marker_surface'].max()==3


def test_failed_marker_extraction_retains_every_original(tmp_path,monkeypatch):
    monkeypatch.setattr(de,'ROOT',tmp_path/'output');monkeypatch.setattr(de,'RESERVE',0)
    folder=tmp_path/'source';folder.mkdir()
    for s in de.SURFACES:_video(folder/(s+'.mp4'),[np.zeros((240,240,3),np.uint8)]*2)
    row=dict(source='baihu',episode_id='bad',split='train',frames=2,views={},original_dir=str(folder),
        marker_path=str(tmp_path/'tracks.npz'))
    with pytest.raises(ValueError):eb.process_episode(row)
    assert len(list(folder.glob('*.mp4')))==4


def test_expanded_clip_uses_eight_rgb_frames_one_current_code_and_missing_camera_mask(tmp_path,monkeypatch):
    from tactilelatent.expansion_joint import ExpandedCurrentClips
    from tactilelatent.vjepa_spatial import JointSpatialVJEPA
    from tactilelatent.aligned_frames import json_hash
    monkeypatch.setattr(de,'ROOT',tmp_path/'output');monkeypatch.setattr(de,'RESERVE',0)
    video=tmp_path/'scene.mp4';_video(video,[np.full((240,240,3),40+i,np.uint8) for i in range(15)])
    # One complete four-surface marker artifact, independent of native images.
    ref=np.array([[40.,40.],[80.,80.],[120.,120.],[160.,160.]],np.float32)
    marker=tmp_path/'marker'/'tracks.npz'
    baihu=dict(source='baihu',episode_id='b',split='train',frames=15,views={'1':str(video)},
        marker_path=str(marker),native_sensor_shape=[240,240])
    de.atomic_npz(marker,reference_px=ref,reference_uv=ref/239.,surface_id=np.arange(4),spacing_px=np.ones(4,np.float32)*20,
        displacement_px=np.zeros((15,4,2),np.float32),valid=np.ones((15,4),bool),frame_indices=np.arange(15))
    write_json(marker.with_name('complete.json'),dict(row_sha256=json_hash(baihu),tracks_sha256=file_hash(marker)))
    normalization=tmp_path/'normalization.json';write_json(normalization,dict(scale_counts=10.))
    pressure=tmp_path/'pressure.parquet'
    pq.write_table(pa.Table.from_pylist([dict(frame=i+1,pressure_map=np.zeros(224*224,np.float32).tolist(),
        pressure_map_range=[0.,0.]) for i in range(15)]),pressure,row_group_size=5)
    ego=dict(source='egopressure',episode_id='e',split='train',frames=15,views={'0':str(video)},
        source_frame_ids=list(range(1,16)),target_path=str(pressure),pressure_normalization=str(normalization))
    indices={}
    for row,dim in [(baihu,32),(ego,64)]:
        path=tmp_path/(row['source']+'_codes.npz')
        de.atomic_npz(path,codes=np.arange(15*dim,dtype=np.float32).reshape(15,dim),valid=np.ones(15,bool),frame_indices=np.arange(15))
        indices[row['source']]=dict(episodes=[dict(episode_id=row['episode_id'],path=str(path),sha256=file_hash(path))],
            statistics=dict(mean=np.zeros(dim).tolist(),std=np.ones(dim).tolist()))
    class Base:
        episodes=[];windows=[];indices_by_source={}
        def __len__(self):return 0
    dataset=ExpandedCurrentClips(Base(),[baihu,ego],indices,'train')
    assert len(dataset)==2
    statistics={s:i['statistics'] for s,i in indices.items()}
    model=JointSpatialVJEPA(sensor_code_dims={'baihu':32,'egopressure':64},numeric_shapes={},statistics=statistics,
        input_dim=8,latent_dim=32,width=16,heads=2,layers=2,feedforward=32,grid_size=2,prediction_mode='current_frame')
    for i,row in enumerate([baihu,ego]):
        item=dataset[i]
        assert item['frame_indices'].tolist()==[14] and item['history_frame_indices'].tolist()==list(range(0,15,2))
        assert item['rgb'].shape==(8,3,3,384,384)
        assert item['view_mask'][:,int(next(iter(row['views'])))].all()
        assert item['view_mask'].sum()==8
        model.zero_grad();out=model(torch.randn(1,4,3,4,8),item['view_mask'][None],row['source'])
        loss=(out['sensor_code']-item['code_target'][None]).square().mean();loss.backward()
        assert out['mu'].shape==(1,1,32) and torch.isfinite(loss)
        assert model.visual.bottleneck[-1].weight.grad.abs().sum()>0


def test_teacher_checkpoint_resume_and_frozen_code_cache_round_trip(tmp_path,monkeypatch):
    from tactilelatent import expansion_training as training
    root=tmp_path/'output';root.mkdir()
    monkeypatch.setattr(de,'ROOT',root);monkeypatch.setattr(de,'RESERVE',0);monkeypatch.setattr(training,'ROOT',root)
    norm=root/'pressure_normalization.json';write_json(norm,dict(scale_counts=10.))
    rows=[]
    for split,magnitude in [('train',1.),('val',7.)]:
        path=tmp_path/(split+'.parquet')
        pq.write_table(pa.Table.from_pylist([dict(frame=i+1,pressure_map=np.ones(224*224,np.float32).tolist(),
            pressure_map_range=[0.,magnitude]) for i in range(2)]),path,row_group_size=1)
        rows.append(dict(source='egopressure',episode_id=split,split=split,frames=2,
            source_frame_ids=[1,2],target_path=str(path),pressure_normalization=str(norm),hand_side='left',views={'0':'unused.mp4'}))
    checkpoint=training.train_teacher('egopressure',rows,device='cpu',epochs=2)
    last=root/'teachers/egopressure/last.pt';before=file_hash(last)
    assert training.train_teacher('egopressure',rows,device='cpu',epochs=2)==checkpoint
    assert file_hash(last)==before
    index=training.build_codes('egopressure',rows,checkpoint,device='cpu')
    assert index['complete'] and index['code_dim']==64
    train_record=next(r for r in index['episodes'] if r['split']=='train')
    with np.load(train_record['path']) as f:
        expected=f['codes'][f['valid']].mean(0)
    np.testing.assert_allclose(index['statistics']['mean'],expected,atol=1e-6)
    assert training.build_codes('egopressure',rows,checkpoint,device='cpu')['statistics']==index['statistics']
    from tactilelatent import expansion_viewer
    monkeypatch.setattr(expansion_viewer,'ROOT',root)
    write_json(root/'egopressure_manifest.json',dict(episodes=rows,complete=True))
    old=tmp_path/'old_manifest.json';write_json(old,dict(episodes=[]))
    viewer=expansion_viewer.ExpandedViewer(str(old),[str(tmp_path/'old_models')],frame_order=None)
    catalog=viewer.catalog()
    assert [d['id'] for d in catalog['datasets']]==['egopressure']
    assert len(catalog['models'])==1 and catalog['models'][0]['completed_epochs']==2
    payload=viewer.frame(0,0,catalog['models'][0]['id'])
    assert payload['target_shape']==[1,224,224] and payload['hand_side']=='left'
    assert np.isfinite(payload['mae']) and np.isfinite(payload['pressure_mae_counts'])
