import io
import json
import pickle
from contextlib import contextmanager
import hashlib
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np
import av
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from tactilelatent import acquisition_queue as aq
from tactilelatent.acquisition_archives import MultipartFile, RemoteRangeFile, TacQuadReader, safe_member
from tactilelatent.acquisition_data import clip_indices
from tactilelatent.acquisition_pilots import PrimitiveNumpyUnpickler, native_numpy, select_hrdex
from tactilelatent.acquisition_sources import (nearest_indices, parse_feel_csvs,
    convert_video, process_trex, split_for, target_artifact, trex_video_mapping)


def test_split_zip_reads_across_arbitrary_part_boundaries(tmp_path):
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,'w',compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr('native/one.bin',bytes(range(256))*10)
        z.writestr('native/two.txt','exact native payload')
    payload=buf.getvalue();parts=[]
    for i,start in enumerate(range(0,len(payload),43)):
        p=tmp_path/str(i);p.write_bytes(payload[start:start+43]);parts.append(p)
    with MultipartFile(parts) as f,zipfile.ZipFile(f) as z:
        assert z.read('native/one.bin')==bytes(range(256))*10
        assert z.read('native/two.txt')==b'exact native payload'
    with MultipartFile(parts) as f:
        f.seek(-5,2);assert f.read(500)==payload[-5:]
        f.seek(9);assert f.read(100)==payload[9:109]


def test_native_tactile_reader_preserves_sixteen_bit_shape_and_values(tmp_path):
    native=np.array([[0,300,60000],[1000,2000,65535]],np.uint16)
    picture=io.BytesIO();Image.fromarray(native).save(picture,format='PNG')
    payload=io.BytesIO()
    with zipfile.ZipFile(payload,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('data_fine/object/sensor/0.png',picture.getvalue())
    paths=[]
    for i,start in enumerate(range(0,len(payload.getvalue()),71)):
        path=tmp_path/f'part{i}';path.write_bytes(payload.getvalue()[start:start+71]);paths.append(path)
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps(dict(
        schema='tacquad-archive-benchmark-v1',archive_parts=[dict(path=str(p)) for p in paths])))
    with TacQuadReader(manifest) as reader:
        actual=reader.image('data_fine/object/sensor/0.png')
    np.testing.assert_array_equal(actual,native)
    assert actual.shape==(2,3) and actual.max()==65535


def test_chroma_fallback_preserves_gate_frames_and_crf(tmp_path,monkeypatch):
    from tactilelatent import data_expansion as de
    monkeypatch.setattr(de,'ROOT',tmp_path/'progress');monkeypatch.setattr(de,'RESERVE',0)
    raw=tmp_path/'source.mp4';output=tmp_path/'converted.mp4'
    image=np.zeros((384,384,3),np.uint8);image[:,::2,0]=255;image[:,1::2,2]=255
    with av.open(str(raw),'w') as out:
        stream=out.add_stream('libx264',rate=30);stream.width=stream.height=384;stream.pix_fmt='yuv444p'
        stream.options={'crf':'0','preset':'fast'}
        for i in range(4):
            for packet in stream.encode(av.VideoFrame.from_ndarray(np.roll(image,i,axis=1),format='rgb24')):out.mux(packet)
        for packet in stream.encode():out.mux(packet)
    class Job:
        deleted=False
        def check_space(self,*a,**kw):pass
        def report(self,**kw):pass
        def retire_rgb(self,source,replacement,receipt):
            assert replacement.exists() and max(receipt['sample_rgb_mae_255'])<=8
            self.deleted=True
    job=Job();receipt=convert_video(job,raw,output)
    assert job.deleted and receipt['pixel_format']=='yuv444p' and receipt['crf']==18
    assert receipt['frames']==4 and receipt['max_allowed_sample_rgb_mae_255']==8
    assert max(receipt['fidelity_retry']['sample_rgb_mae_255'])>8
    with av.open(str(output)) as c:
        frames=list(c.decode(video=0));assert len(frames)==4
        assert all((f.width,f.height)==(384,384) for f in frames)


def test_fidelity_failure_does_not_publish_or_retire_source(tmp_path,monkeypatch):
    from tactilelatent import acquisition_sources as sources
    raw=tmp_path/'source.mp4';raw.write_bytes(b'unchanged source')
    monkeypatch.setattr(sources,'video_info',lambda p:dict(frames=4,fps=30.))
    calls=[]
    def reject(*args,**kwargs):
        calls.append(kwargs.get('pixel_format','yuv420p'))
        raise sources.RGBFidelityError([10.,11.,12.],[0,2,3])
    monkeypatch.setattr(sources,'encode_rgb',reject)
    class Job:
        def check_space(self,*a,**kw):pass
        def report(self,**kw):pass
        def retire_rgb(self,*a):pytest.fail('Failed conversion must retain original')
    output=tmp_path/'out.mp4'
    with pytest.raises(sources.RGBFidelityError):sources.convert_video(Job(),raw,output)
    assert raw.read_bytes()==b'unchanged source' and not output.exists()
    assert calls==['yuv420p','yuv444p']


def test_alignment_masks_outside_support_and_sensor_dropouts():
    sensor=np.r_[np.arange(10)*.01,np.arange(40,60)*.01]
    frames=np.array([-.1,0,.034,.2,.405,.8])
    ix,valid,error,tolerance=nearest_indices(frames,sensor)
    assert valid.tolist()==[False,True,True,False,True,False]
    assert ix[1]==0 and abs(error[2])<.005 and tolerance<.03
    with pytest.raises(ValueError):nearest_indices([1,0],[0,1])


def test_feel_lab_schema_differs_from_kitchen_and_zero_is_valid(tmp_path):
    lab=tmp_path/'lab.csv';lab.write_text('frame_idx,sensor0,sensor1,sensor4,raw_sensor0\n0,0,,4,999\n1,2,3,5,999\n')
    values,ids,channels=parse_feel_csvs([('released_hand',lab)])
    assert channels==['released_hand_sensor0','released_hand_sensor1','released_hand_sensor4']
    assert values.shape==(2,3) and values[0,0]==0 and np.isnan(values[0,1])
    bad=tmp_path/'bad.csv';bad.write_text('frame_idx,sensor0\n0,0\n2,2\n')
    with pytest.raises(ValueError):parse_feel_csvs([('right',bad)])


def test_native_target_roundtrip_missing_signed_values_and_history_map(tmp_path):
    class Job:
        root=tmp_path
        def check_space(self,*args,**kwargs):pass
    x=np.array([[0.,-3.],[np.nan,4.],[2.,5.],[3.,6.]],np.float32)
    target=target_artifact(Job(),'test',x,np.arange(4,dtype=float),['a','b'],rgb_frame_indices=np.array([0,2,1,3]))
    assert target['negative_count']==1 and target['measured_zero_count']==1
    with np.load(target['path']) as f:
        assert f['valid'][0].all() and not f['valid'][1,0]
        np.testing.assert_array_equal(f['values'],x)
    row=dict(frames=4,targets=target,views={'0':dict(frame_indices_artifact=target['path'],frame_indices_key='rgb_frame_indices')})
    with pytest.raises(ValueError,match='camera'):clip_indices(row,3,frames=4,stride=1)
    with pytest.raises(IndexError):clip_indices(row,0,frames=2,stride=1)


def test_trex_offsets_are_episode_frame_exact():
    r={'length':179}
    for key,value in [('chunk_index',0),('file_index',3),('from_timestamp',160/30),('to_timestamp',339/30)]:
        r['videos/observation.images.head_left/'+key]=value
    path,offset=trex_video_mapping(r,'head_left',30)
    assert path.endswith('chunk-000/file-003.mp4') and offset==160
    r['videos/observation.images.head_left/to_timestamp']+=.01
    with pytest.raises(ValueError):trex_video_mapping(r,'head_left',30)


def test_storage_pauses_before_allocating_and_retains_staging(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path);monkeypatch.setattr(aq,'ROOT',Path('processing/queue'))
    cfg=dict(project_cap_bytes=1000,experiment_reserve_bytes=100,physical_reserve_bytes=0,staging_limit_bytes=80)
    job=aq.Job(dict(id='test',budget_bytes=300),cfg)
    monkeypatch.setattr(aq,'allocated',lambda ps: 850 if ps==[Path('datasets'),Path('processing')] else 0)
    with pytest.raises(aq.StoragePause,match='project cap'):job.check_space(51,force=True)
    with pytest.raises(aq.StoragePause):job.check_space(81,staging=True,force=True)
    job.check_space(20,staging=True,force=True)


def test_untrusted_pickle_cannot_execute_and_native_dict_npy_loads(tmp_path):
    class Malicious:
        def __reduce__(self):return (eval,('1+1',))
    with pytest.raises(pickle.UnpicklingError):PrimitiveNumpyUnpickler(io.BytesIO(pickle.dumps(Malicious()))).load()
    p=tmp_path/'native.npy';np.save(p,np.array([dict(a=0.,b=-1.),dict(a=2.,b=3.)],dtype=object))
    loaded=native_numpy(p)
    assert loaded[0]['b']==-1.
    for name in ['../escape','/absolute','x/../../bad','x\\bad']:
        with pytest.raises(ValueError):safe_member(name)


def test_object_split_shared_across_hrdex_embodiments():
    rows=[dict(embodiment=e,episode_id=e+'/'+obj+'/'+str(i),object_id=obj,
               has_tactile=True,has_videos=True,has_raw=True)
          for e in ['a','b'] for obj in ['one','two','three'] for i in range(3)]
    selected=select_hrdex(rows,dict(embodiments=['a','b'],max_per_embodiment=3))
    assert len(selected)==6
    assert {r['object_id'] for r in selected if r['embodiment']=='a'}=={'one','two','three'}


def test_trex_episode_crossing_numeric_shard_boundary_is_not_truncated(tmp_path):
    class Job:
        id='trex'
        spec=dict(target='observation.tactile_force',camera_keys=[],target_units='native')
        raw=tmp_path/'raw'
        root=tmp_path/'processed'
        def report(self,**kw):pass
        def check_space(self,*a,**kw):pass
        def download(self,p,**kw):return self.raw/p
        def inventory(self):return dict(files=[dict(path='data/000.parquet'),dict(path='data/001.parquet')])
    job=Job();(job.raw/'meta/episodes/chunk-000').mkdir(parents=True);(job.raw/'data').mkdir()
    (job.raw/'meta/info.json').write_text(json.dumps(dict(fps=1,total_frames=6)))
    pq.write_table(pa.Table.from_pylist([
        dict(episode_index=0,length=3,dataset_from_index=0,dataset_to_index=3),
        dict(episode_index=1,length=3,dataset_from_index=3,dataset_to_index=6)]),
        job.raw/'meta/episodes/chunk-000/file-000.parquet')
    rows=[{'observation.tactile_force':[float(i-3)]*60,'timestamp':float(i%3),
           'frame_index':i%3,'episode_index':i//3,'index':i} for i in range(6)]
    pq.write_table(pa.Table.from_pylist(rows[:4]),job.raw/'data/000.parquet')
    pq.write_table(pa.Table.from_pylist(rows[4:]),job.raw/'data/001.parquet')
    process_trex(job)
    with np.load(job.root/'targets/episode_000001.npz') as data:
        np.testing.assert_array_equal(data['frame_indices'],[0,1,2])
        np.testing.assert_array_equal(data['global_indices'],[3,4,5])
        np.testing.assert_array_equal(data['values'][:,0],[0,1,2])


@pytest.mark.parametrize('server_honors_range',[True,False])
def test_interrupted_download_resume_or_safe_restart(tmp_path,monkeypatch,server_honors_range):
    import httpx
    import huggingface_hub
    monkeypatch.chdir(tmp_path);monkeypatch.setattr(aq,'ROOT',Path('processing/queue'))
    cfg=dict(project_cap_bytes=10**12,experiment_reserve_bytes=0,physical_reserve_bytes=0,staging_limit_bytes=10**9)
    job=aq.Job(dict(id='test',repo='test/test',revision='pinned',budget_bytes=10**9),cfg)
    monkeypatch.setattr(job,'check_space',lambda *a,**k:None)
    payload=b'native signed data';digest=hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(huggingface_hub,'get_hf_file_metadata',lambda u:SimpleNamespace(size=len(payload),etag=digest))
    monkeypatch.setattr(huggingface_hub,'hf_hub_url',lambda *a,**k:'https://test.invalid/pinned')
    filename='episode/values.bin';part=job.raw/(filename+'.partial');part.parent.mkdir(parents=True);part.write_bytes(payload[:3])
    calls=[]
    @contextmanager
    def stream(method,url,**kwargs):
        calls.append(kwargs['headers'])
        assert kwargs['headers']['Range']=='bytes=3-'
        r=httpx.Response(206 if server_honors_range else 200,
            headers={'Content-Range':f'bytes 3-{len(payload)-1}/{len(payload)}'} if server_honors_range else {},
            content=payload[3:] if server_honors_range else payload,
            request=httpx.Request('GET',url))
        yield r
    monkeypatch.setattr(httpx,'stream',stream)
    result=job.download(filename)
    assert result.read_bytes()==payload and not part.exists()
    assert job.download(filename)==result and len(calls)==1


def test_verified_rgb_cleanup_recovers_interruption_and_records_deletion(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path);monkeypatch.setattr(aq,'ROOT',Path('processing/queue'))
    job=aq.Job(dict(id='feel',revision='pinned',budget_bytes=10**6),{})
    original=job.staging/'Kitchen/P09/rgb.tar';original.parent.mkdir(parents=True)
    original.write_bytes(b'checked original RGB archive')
    replacement=job.root/'rgb/P09.mp4';replacement.parent.mkdir()
    replacement.write_bytes(b'checked compressed replacement')
    name='Kitchen/P09/rgb.tar';receipts=job.root/'downloads';receipts.mkdir()
    (receipts/(aq.json_hash(name)+'.json')).write_text(json.dumps(dict(
        revision='pinned',bytes=original.stat().st_size,sha256=aq.file_hash(original))))
    receipt=dict(path=str(replacement),frames=3,sha256=aq.file_hash(replacement))
    assert job.recover_rgb_cleanup(name,replacement,receipt,expected_frames=3)
    assert not original.exists() and replacement.exists()
    events=[json.loads(s) for s in (job.root/'deletions.jsonl').read_text().splitlines()]
    assert [e['event'] for e in events]==['delete_intent','deleted']
    assert not job.recover_rgb_cleanup(name,replacement,receipt,expected_frames=3)


@pytest.mark.parametrize('corruption',['original','replacement','frames','revision'])
def test_cleanup_retains_original_when_recovery_evidence_differs(tmp_path,monkeypatch,corruption):
    monkeypatch.chdir(tmp_path);monkeypatch.setattr(aq,'ROOT',Path('processing/queue'))
    job=aq.Job(dict(id='feel',revision='pinned',budget_bytes=10**6),{})
    name='P09/rgb.tar';original=job.staging/name;original.parent.mkdir()
    original.write_bytes(b'original');replacement=job.root/'out.mp4';replacement.write_bytes(b'converted')
    receipt=dict(path=str(replacement),frames=3,sha256=aq.file_hash(replacement))
    saved=dict(revision='pinned',bytes=original.stat().st_size,sha256=aq.file_hash(original))
    if corruption=='original':original.write_bytes(b'bad data')
    elif corruption=='replacement':replacement.write_bytes(b'bad video')
    elif corruption=='frames':receipt['frames']=2
    else:saved['revision']='another release'
    evidence=job.root/'downloads'/(aq.json_hash(name)+'.json');evidence.parent.mkdir()
    evidence.write_text(json.dumps(saved))
    with pytest.raises(ValueError):job.recover_rgb_cleanup(name,replacement,receipt,expected_frames=3)
    assert original.exists() and not (job.root/'deletions.jsonl').exists()


def test_metadata_dns_failure_retries_before_download(tmp_path,monkeypatch):
    import httpx
    import huggingface_hub
    monkeypatch.chdir(tmp_path);monkeypatch.setattr(aq,'ROOT',Path('processing/queue'))
    job=aq.Job(dict(id='test',repo='test/test',revision='pinned',budget_bytes=10**9),{})
    monkeypatch.setattr(job,'check_space',lambda *a,**kw:None)
    monkeypatch.setattr(aq.time,'sleep',lambda s:None)
    payload=b'complete native payload';calls=[]
    def metadata(url):
        calls.append(url)
        if len(calls)<3:raise httpx.ConnectError('temporary DNS failure')
        return SimpleNamespace(size=len(payload),etag=hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(huggingface_hub,'get_hf_file_metadata',metadata)
    @contextmanager
    def stream(method,url,**kwargs):
        yield httpx.Response(200,content=payload,request=httpx.Request('GET',url))
    monkeypatch.setattr(httpx,'stream',stream)
    result=job.download('values.bin')
    assert result.read_bytes()==payload and len(calls)==3
    assert job.state['stage']=='downloaded' and job.state['error'] is None
    assert job.state['file_bytes_done']==len(payload)


def test_independent_workers_retry_without_waiting_for_long_job(tmp_path,monkeypatch):
    monkeypatch.setattr(aq,'ROOT',tmp_path)
    cfg=dict(jobs=[dict(id=s) for s in ['trex','feel','tdex']],max_parallel_jobs=3,
             retry_initial_seconds=60,retry_max_seconds=900,storage_retry_seconds=300)
    for spec in cfg['jobs']:
        p=tmp_path/spec['id'];p.mkdir();(p/'status.json').write_text(json.dumps(dict(stage='failed')))
    started=[]
    class Process:
        code=None
        def poll(self):return self.code
    def popen(args,**kw):
        process=Process();started.append((args[-1],process));return process
    monkeypatch.setattr(aq.subprocess,'Popen',popen)
    q=aq.QueueSupervisor(cfg);assert not q.step(now=0)
    assert [s for s,p in started]==['trex','feel','tdex']
    # T-Rex keeps running while FEEL pauses locally and T-Dex loses DNS.
    started[1][1].code=1;started[2][1].code=1
    (tmp_path/'feel/status.json').write_text(json.dumps(dict(stage='paused_storage',storage_scope='source')))
    q.step(now=1);q.step(now=60)
    assert len(started)==3 and q.retry_at==dict(feel=301,tdex=61)
    q.step(now=61)
    assert [s for s,p in started]==['trex','feel','tdex','tdex']
    assert q.running['trex'] is started[0][1]
    q.step(now=301)
    assert started[-1][0]=='feel' and len(q.running)==3


def test_global_storage_pause_defers_new_starts(tmp_path,monkeypatch):
    monkeypatch.setattr(aq,'ROOT',tmp_path)
    for source in ['one','two']:
        (tmp_path/source).mkdir();(tmp_path/source/'status.json').write_text('{"stage":"queued"}')
    cfg=dict(jobs=[dict(id='one'),dict(id='two')],max_parallel_jobs=1,storage_retry_seconds=300)
    started=[]
    class Process:
        code=None
        def poll(self):return self.code
    def popen(*a,**kw):
        process=Process();started.append(process);return process
    monkeypatch.setattr(aq.subprocess,'Popen',popen)
    q=aq.QueueSupervisor(cfg);q.step(now=0);started[0].code=1
    (tmp_path/'one/status.json').write_text('{"stage":"paused_storage","storage_scope":"global"}')
    q.step(now=1);q.step(now=299)
    assert len(started)==1 and q.global_retry_at==301


def test_feel_resume_cleans_verified_archive_before_next_participant(tmp_path,monkeypatch):
    from tactilelatent import acquisition_sources as sources
    from tactilelatent.data_expansion import encode_rgb
    monkeypatch.chdir(tmp_path);monkeypatch.setattr(aq,'ROOT',Path('processing/queue'))
    job=aq.Job(dict(id='feel',revision='pinned',budget_bytes=10**9),{})
    job.spec['reuse_p12']=False
    monkeypatch.setattr(job,'check_space',lambda *a,**k:None)
    monkeypatch.setattr(sources,'target_artifact',lambda *a,**k:dict(valid_fraction=[1.]))
    monkeypatch.setattr(sources,'publish',lambda *a,**k:None)
    original=job.staging/'Kitchen/P09/rgb.tar';original.parent.mkdir(parents=True)
    original.write_bytes(b'archive with verified published mapping')
    replacement=job.root/'rgb/Kitchen_P09.mp4'
    receipt=encode_rgb([np.zeros((32,32,3),np.uint8)]*3,replacement,3)
    replacement.with_suffix('.mapping.json').write_text(json.dumps(dict(
        frames=3,chronological=True,original_image_index_base=0,aligned_csv_index_base=0)))
    evidence=job.root/'downloads'/(aq.json_hash('Kitchen/P09/rgb.tar')+'.json');evidence.parent.mkdir()
    evidence.write_text(json.dumps(dict(revision='pinned',bytes=original.stat().st_size,sha256=aq.file_hash(original))))
    csvpath=job.raw/'Kitchen/P09/force_left_aligned.csv';csvpath.parent.mkdir(parents=True)
    csvpath.write_text('frame_idx,sensor0\n0,0\n1,2\n2,3\n')
    monkeypatch.setattr(job,'inventory',lambda:dict(files=[dict(path='Kitchen/P09/rgb.tar'),
        dict(path='Kitchen/P09/force_left_aligned.csv'),dict(path='Kitchen/P10/rgb.tar'),
        dict(path='Kitchen/P10/force_left_aligned.csv')]))
    visited=[]
    def download(name,**kw):
        visited.append(name)
        if 'P10' in name:
            assert not original.exists()
            raise aq.StoragePause('test reached next participant')
        return job.raw/name
    monkeypatch.setattr(job,'download',download)
    with pytest.raises(aq.StoragePause,match='next participant'):sources.process_feel(job)
    assert visited==['Kitchen/P09/force_left_aligned.csv','Kitchen/P10/force_left_aligned.csv']
    assert replacement.exists() and not original.exists()
