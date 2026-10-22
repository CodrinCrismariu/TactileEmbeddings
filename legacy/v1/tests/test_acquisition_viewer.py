"""Acquisition display must preserve native values, masks, and camera pairing."""
from io import BytesIO
import json
from pathlib import Path
from threading import Thread
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import urlopen
import zipfile
import sqlite3

import numpy as np
from PIL import Image
import pytest
import torch

from tactilelatent.aligned_frames import file_hash
from tactilelatent.acquisition_viewer import AcquisitionViewer, FIRST_ID
from tactilelatent.mirror_viewer import MirroredViewer, make_mirror_handler
from tactilelatent.tacquad_viewer import TacQuadBrowser
from tactilelatent.zero_reference import FixedReferences, SCHEMA, save_registry


@pytest.fixture
def viewer(tmp_path, monkeypatch):
    import tactilelatent.expansion_viewer as expansion
    monkeypatch.setattr(expansion, 'ROOT', tmp_path/'expansion')
    old = tmp_path/'old.json'
    pressure = tmp_path/'old.npz'
    np.savez(pressure, left_pressure_grid=np.zeros((2,21,21),np.float32),
             right_pressure_grid=np.zeros((2,21,21),np.float32))
    old.write_text(json.dumps(dict(episodes=[dict(source='egotouch', episode_id='historical',
        split='val', frames=2, views={'0':'old.mp4'}, target_path=str(pressure))])))
    models = tmp_path/'models';models.mkdir()
    return MirroredViewer(old, [models], frame_order=None, acquisition_root=tmp_path/'queue',acquisition_teachers=tmp_path/'numeric-teachers',
        mirror_policy='configs/bimanual_mirror_policy_v1.json', mirror_preparation=tmp_path/'mirrors')


def add_source(viewer, source, values, *, identity='native', mapped=None, channel_order=None, **extra):
    folder=viewer.acquisition_root/source;folder.mkdir(parents=True,exist_ok=True)
    path=folder/(identity+'.npz');frames=len(values)
    valid=np.isfinite(values);valid.reshape(frames,-1)[0,-1]=False
    arrays=dict(values=values,valid=valid,timestamps=100+np.arange(frames)/30,
                frame_indices=np.arange(frames))
    if mapped is not None:arrays['rgb_frame_indices']=np.asarray(mapped)
    np.savez(path,**arrays)
    target=dict(path=str(path),sha256=file_hash(path),shape=list(values.shape),
                channel_order=channel_order or [f'channel{i}' for i in range(np.prod(values.shape[1:]))])
    view=dict(path='shard.mp4',role='fixed_scene',frame_offset=1000,frames=frames)
    if mapped is not None:view.update(frame_indices_artifact=str(path),frame_indices_key='rgb_frame_indices',frames=500)
    row=dict(episode_id=identity,frames=frames,split='val',views={'0':view},targets=target,**extra)
    manifest=folder/'manifest.json'
    rows=json.loads(manifest.read_text())['episodes'] if manifest.exists() else []
    manifest.write_text(json.dumps(dict(schema='acquisition-native-v2',episodes=rows+[row],revision='pinned')))
    return row


def index_for(viewer, source):
    return next(row['id'] for row in viewer.catalog()['episodes'] if row['source']==source)


def test_raw_signed_zero_and_missing_values_are_not_pressure_normalized(viewer):
    x=np.arange(24,dtype=np.int32).reshape(2,12);x[0,:3]=[-42,0,65535]
    add_source(viewer,'feel',x)
    index=index_for(viewer,'feel');data=viewer.frame(index,0,-1)
    assert data['target_shape']==[12] and data['original'][:3]==[-42,0,65535]
    assert data['original'][-1] is None and data['acquisition']['valid_count']==11
    assert data['reconstruction'] is data['error'] is data['mae'] is None
    assert data['acquisition']['native_dtype']=='int32'
    assert data['acquisition']['normalization'].startswith('none')
    json.dumps(data,allow_nan=False)
    with pytest.raises(ValueError,match='No trained'):viewer.frame(index,0,0)


def test_trex_episode_offsets_and_hrdex_explicit_missing_frame_maps(viewer,monkeypatch):
    decoded=[]
    def frames(path,indices,size):
        decoded.append((str(path),indices,size));return torch.zeros((1,3,4,4))
    monkeypatch.setattr('tactilelatent.acquisition_viewer.video_frames',frames)
    add_source(viewer,'trex',np.ones((3,60),np.float32))
    index=index_for(viewer,'trex');viewer.video(index,2,'0')
    assert decoded[-1][1]==[1002]
    add_source(viewer,'hrdexdb',np.ones((3,4),np.int32),mapped=[11,-1,17],embodiment='allegro_v5')
    index=index_for(viewer,'hrdexdb_allegro_v5');viewer.video(index,2,'0')
    assert decoded[-1][1]==[17]
    assert viewer.frame(index,1,-1)['acquisition']['camera_frame_indices']=={'0':None}
    with pytest.raises(ValueError,match='No recorded RGB'):viewer.video(index,1,'0')
    assert len(decoded)==2
    with pytest.raises(ValueError,match='Camera'):viewer.metadata(index,'2')
    assert viewer.metadata(index,'0')['camera_role']=='fixed_scene'


def test_retired_tdex_stays_hidden_and_two_hrdex_morphologies_stay_separate(viewer):
    x=np.arange(2*15*16*3,dtype=np.float32).reshape(2,15,16,3)
    add_source(viewer,'tdex',x)
    add_source(viewer,'hrdexdb',np.ones((3,4),np.int32),identity='allegro',embodiment='allegro_v5')
    add_source(viewer,'hrdexdb',np.ones((3,24),np.int32),identity='inspire',embodiment='inspire_f1')
    catalog=viewer.catalog();datasets={d['id']:d for d in catalog['datasets']}
    assert 'tdex' not in datasets
    assert not any(e['source']=='tdex' for e in catalog['episodes'])
    assert 'tdex' not in catalog['acquisition_status']
    assert datasets['hrdexdb_allegro_v5']['target_shape']==[4]
    assert datasets['hrdexdb_inspire_f1']['target_shape']==[24]
    assert not any(m['source'] in ('tdex','hrdexdb_allegro_v5','hrdexdb_inspire_f1') for m in catalog['models'])


def test_new_episodes_refresh_without_shifting_historical_or_existing_ids(viewer):
    add_source(viewer,'feel',np.ones((3,12)),identity='first')
    first=index_for(viewer,'feel')
    assert first>=FIRST_ID and viewer.catalog()['episodes'][0]['id']==0
    add_source(viewer,'feel',np.ones((3,12))*2,identity='second')
    add_source(viewer,'trex',np.ones((3,60)))
    second=viewer.catalog();ids={e['name']:e['id'] for e in second['episodes']}
    assert ids['first']==first and ids['historical']==0 and ids['second']!=first
    assert viewer.frame(0,0,-1)['target_shape']==[2,21,21]
    assert viewer.frame(ids['second'],2,-1)['original']==[2]*12
    with pytest.raises(ValueError):viewer.frame(first,3,-1)


def test_series_uses_native_time_and_does_not_join_missing_values(viewer):
    x=np.arange(60,dtype=float).reshape(5,12);x[2,0]=np.nan
    add_source(viewer,'feel',x)
    index=index_for(viewer,'feel');series=viewer.acquisition_series(index,2,0,2)
    assert series['values']==[0,12,None,36,48]
    np.testing.assert_allclose(series['seconds'],np.arange(5)/30)
    assert series['frame_indices']==[0,1,2,3,4]
    for channel,radius in [(-1,2),(12,2),(0,1001)]:
        with pytest.raises(ValueError):viewer.acquisition_series(index,2,channel,radius)


def test_saved_numeric_teacher_reconstructs_native_values_and_masks_in_frame_and_trace(viewer):
    from tactilelatent.numeric_autoencoder import NumericAutoencoder,MODEL_TYPE
    x=np.arange(36,dtype=np.float32).reshape(3,12);x[1,3]=np.nan
    add_source(viewer,'feel',x)
    folder=viewer.acquisition_teacher_root/'feel_kitchen';folder.mkdir(parents=True)
    model=NumericAutoencoder(12,6,[16,8],[1.]*12,[True]*12)
    config=dict(source='feel',model=model.config,spec=dict(input_dim=12,latent_dim=6))
    (folder/'config.json').write_text(json.dumps(config))
    (folder/'best_metrics.json').write_text(json.dumps(dict(epoch=1,selection_score=.1,
        val=dict(mae=1.,score=.1,per_quantity_mae={'all_native':1.}))))
    torch.save(dict(model_type=MODEL_TYPE,model_config=model.config,model=model.state_dict(),epoch=1,
        provenance=dict(manifest_sha256=file_hash(viewer.acquisition_root/'feel/manifest.json'))),folder/'best.pt')
    catalog=viewer.catalog();entry=next(m for m in catalog['models'] if m.get('native_numeric_teacher'))
    assert next(d for d in catalog['datasets'] if d['id']=='feel')['reconstruction_available']
    index=index_for(viewer,'feel');frame=viewer.frame(index,1,entry['id'])
    valid=np.isfinite(x[1]);expected=model(torch.from_numpy(x[1:2]),torch.from_numpy(valid[None])).detach().numpy()[0]
    np.testing.assert_array_equal(np.array(frame['original'],dtype=float)[valid],x[1,valid])
    np.testing.assert_allclose(np.array(frame['reconstruction'],dtype=float)[valid],expected[valid])
    assert frame['original'][3] is None and frame['reconstruction'][3] is None and frame['error'][3] is None
    assert frame['acquisition']['reconstruction']['epoch']==1
    trace=viewer.acquisition_series(index,1,3,model=entry['id'])
    assert trace['reconstruction'][1] is None and len(trace['reconstruction'])==3


def test_wrench_display_uses_shared_force_and_moment_ranges_without_modifying_readings(viewer):
    values=np.zeros((3,60),np.float32)
    values[:,0]=[-.01,.02,-.03]
    values[:,3]=[.0002,-.0005,.0001]
    values[:,30]=[0,20,-5]
    values[:,33]=[0,.4,-.1]
    add_source(viewer,'trex',values)
    frame=viewer.frame(index_for(viewer,'trex'),1,-1)
    expected=np.where(np.arange(60)%6<3,20,.4)
    np.testing.assert_allclose(frame['acquisition']['shared_wrench_scales'],expected)
    assert frame['acquisition']['display_scales'][0]==pytest.approx(.03)
    np.testing.assert_array_equal(frame['original'],values[1])


def test_hrdex_shared_scales_keep_small_signals_small_and_direction_separate(viewer):
    allegro=np.array([[0,0,0,0],[26,132,462,1126],[2,100,300,800]],np.int32)
    add_source(viewer,'hrdexdb',allegro,identity='allegro',embodiment='allegro_v5')
    frame=viewer.frame(index_for(viewer,'hrdexdb_allegro_v5'),1,-1)
    assert frame['acquisition']['shared_display_scales']==[1126]*4
    np.testing.assert_array_equal(frame['original'],allegro[1])

    inspire=np.zeros((3,24),np.float64)
    inspire[:,2::3]=-1
    inspire[1,0:3]=[2,3,8]
    inspire[1,12:15]=[45,83,304]
    inspire[2,6:9]=[10,30,120]
    add_source(viewer,'hrdexdb',inspire,identity='inspire',embodiment='inspire_f1')
    frame=viewer.frame(index_for(viewer,'hrdexdb_inspire_f1'),1,-1)
    np.testing.assert_array_equal(frame['acquisition']['shared_display_scales'],[45,83,304]*8)
    np.testing.assert_array_equal(frame['original'],inspire[1])
    assert frame['original'][5]==-1 and frame['original'][3]==0


def test_shared_ranges_include_short_peaks_in_long_hrdex_episodes(viewer):
    values=np.zeros((4097,4),np.int32)
    values[:,0]=1
    values[1,3]=1000  # Frame 1 is absent from the old 2048-point range sample.
    add_source(viewer,'hrdexdb',values,embodiment='allegro_v5')
    frame=viewer.frame(index_for(viewer,'hrdexdb_allegro_v5'),1,-1)
    assert frame['acquisition']['shared_display_scales']==[1000]*4
    np.testing.assert_array_equal(frame['original'],values[1])


def test_reference_metadata_is_fixed_across_episodes_and_raw_api_values_stay_native(viewer,tmp_path):
    first=add_source(viewer,'feel',np.ones((3,12))*10,identity='first')
    add_source(viewer,'feel',np.ones((3,12))*30,identity='second')
    path=tmp_path/'references.json'
    save_registry(path,dict(schema=SCHEMA,fit_split='train',episode_refitting=False,optical={},
        numeric={'feel':dict(source='feel',revision='pinned',target_shape=[12],
            channel_order=first['targets']['channel_order'],offset=[7]*12,display_limits=[100]*12)}))
    viewer.zero_references=FixedReferences(path)
    episodes=[e for e in viewer.catalog()['episodes'] if e['source']=='feel']
    a,b=[viewer.frame(e['id'],1,-1) for e in episodes]
    assert a['original']==[10]*12 and b['original']==[30]*12
    assert a['acquisition']['zero_reference']==b['acquisition']['zero_reference']
    assert a['acquisition']['zero_reference']['offset']==[7]*12


def test_feel_lab_five_channel_layout_does_not_invent_second_hand_channels(viewer):
    add_source(viewer,'feel',np.ones((3,5)))
    frame=viewer.frame(index_for(viewer,'feel'),1,-1)
    assert frame['acquisition']['groups'][0]['indices']==list(range(5))
    assert len(frame['acquisition']['groups'])==1


def test_feel_validation_lab_p11_owns_right_hand_without_changing_values_or_native_ids(viewer):
    from tactilelatent.feel_hand_layout import FEEL_REVISION
    x=np.arange(15,dtype=np.float32).reshape(3,5)
    native_names=[f'released_hand_sensor{i}' for i in range(5)]
    add_source(viewer,'feel',x,identity='Lab_P11',environment='Lab',participant=11,
        channel_order=native_names)
    manifest=viewer.acquisition_root/'feel/manifest.json'
    metadata=json.loads(manifest.read_text());metadata['revision']=FEEL_REVISION
    manifest.write_text(json.dumps(metadata))
    index=index_for(viewer,'feel');frame=viewer.frame(index,1,-1)
    assert frame['labels']==[f'right_sensor{i}' for i in range(5)]
    assert frame['acquisition']['native_channel_order']==native_names
    assert frame['acquisition']['sensor_layout']['channel_hands']==['right']*5
    assert frame['acquisition']['sensor_layout']['unmeasured_hands']==['left']
    assert frame['acquisition']['groups']==[dict(label='Right hand · released sensor indices',indices=list(range(5)),hand='right')]
    assert frame['original']==x[1].tolist()
    assert frame['acquisition']['display_policy']['detail_upper']==2.5
    assert frame['acquisition']['shared_display_scales']==[14]*5
    assert viewer.acquisition_series(index,1,0)['label']=='right_sensor0'
    assert viewer.acquisition_series(index,1,0)['native_label']=='released_hand_sensor0'
    episode=next(e for e in viewer.catalog()['episodes'] if e['id']==index)
    assert episode['split']=='val' and episode['sensor_layout']['channel_hands']==['right']*5


def test_feel_shared_episode_range_keeps_both_hands_comparable_and_preserves_short_peaks(viewer):
    values=np.zeros((4097,12),np.float32)
    values[:,0]=.1;values[1,11]=8
    add_source(viewer,'feel',values)
    frame=viewer.frame(index_for(viewer,'feel'),1,-1)
    assert frame['acquisition']['shared_display_scales']==[8]*12
    assert frame['original'][11]==8
    assert frame['original'][0]==pytest.approx(.1)


def test_proxy_prefixed_assets_and_native_series_api(viewer):
    add_source(viewer,'feel',np.ones((3,12)))
    index=index_for(viewer,'feel')
    server=ThreadingHTTPServer(('127.0.0.1',0),make_mirror_handler(viewer));thread=Thread(target=server.serve_forever,daemon=True);thread.start()
    base=f'http://127.0.0.1:{server.server_port}/proxy/{server.server_port}'
    try:
        for route,expected in [('/acquisition-viewer.js',b'window.NativeSensor'),
                               (f'/api/frame?episode={index}&frame=1&model=-1',b'native_numeric'),
                               (f'/api/acquisition/series?episode={index}&frame=1&channel=0',b'native_values')]:
            with urlopen(base+route) as r:assert expected in r.read()
        for route in ['/tacquad','/api/tacquad/catalog','/api/tacquad/member?contact=0&stream=digit&index=0']:
            with pytest.raises(HTTPError) as removed:urlopen(base+route)
            assert removed.value.code==410
        assert viewer.catalog()['tacquad_available'] is False
        with pytest.raises(HTTPError) as e:urlopen(base+f'/api/acquisition/series?episode={index}&frame=1&channel=999')
        assert e.value.code==400
    finally:server.shutdown();server.server_close();thread.join()


@pytest.fixture
def benchmark(tmp_path):
    image=BytesIO();Image.fromarray(np.array([[0,65535]],np.uint16)).save(image,'PNG');native=image.getvalue()
    array=BytesIO();np.save(array,np.array([[0.,-4.,np.nan]],np.float64))
    payload=BytesIO()
    with zipfile.ZipFile(payload,'w') as archive:
        archive.writestr('data_fine/object/gelsight/10.png',native)
        archive.writestr('data_fine/object/gelsight/2.png',native)
        archive.writestr('data_fine/object/img_gelsight/100.png',native)
        archive.writestr('data_fine/object/tac3d/allforce_2.npy',array.getvalue())
        archive.writestr('data_fine/object/tac3d/markerpos_2.npy',array.getvalue())
    blob=payload.getvalue();parts=[]
    for i,(start,end) in enumerate(zip([0,len(blob)//2],[len(blob)//2,len(blob)])):
        path=tmp_path/f'part{i}';path.write_bytes(blob[start:end]);parts.append(dict(path=str(path)))
    db=tmp_path/'members.sqlite'
    with sqlite3.connect(db) as c:
        c.execute('create table members(name text primary key)')
        with zipfile.ZipFile(BytesIO(blob)) as z:c.executemany('insert into members values(?)',[(n,) for n in z.namelist()])
    contacts=tmp_path/'contacts.jsonl';contacts.write_text(json.dumps(dict(csv_member='contact_fine.csv',folder='object',text='native contact',
        alignment_level='fine',intervals={'gelsight_mini':[2,10],'digit':None},interval_valid={'gelsight_mini':True,'digit':False}))+'\n')
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps(dict(schema='tacquad-archive-benchmark-v1',
        archive_parts=parts,member_index=str(db),contact_intervals=str(contacts))))
    browser=TacQuadBrowser(manifest)
    yield browser,native
    if browser.reader:browser.reader.close()


def test_tacquad_keeps_native_pixels_independent_timelines_and_payload_kinds(benchmark):
    browser,native=benchmark
    assert not browser.catalog()['framewise_alignment_verified']
    streams={s['id']:s for s in browser.streams(0)['streams']}
    assert streams['gelsight']['frame_names']==['2.png','10.png']
    assert streams['img_gelsight']['frame_names']==['100.png']
    assert 'tac3d/allforce' in streams and 'tac3d/markerpos' in streams
    payload,mime=browser.member(0,'gelsight',0)
    assert payload==native and mime=='image/png'
    np.testing.assert_array_equal(np.array(Image.open(BytesIO(payload))),[[0,65535]])
    result=browser.array(0,'tac3d/allforce',0)
    assert result['shape']==[1,3] and result['dtype']=='float64' and result['values']==[0,-4,None]
    with pytest.raises(ValueError):browser.member(0,'gelsight',2)
    with pytest.raises(ValueError):browser.member(0,'../../outside',0)
    with pytest.raises(ValueError):browser.streams(-1)
