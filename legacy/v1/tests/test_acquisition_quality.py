import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest
from PIL import Image

from tactilelatent.acquisition_quality import (
    UnavailableTactile, parse_contact_csv, timestamp_ambiguity, supervised_history_ends)
from tactilelatent.acquisition_sources import nearest_indices, target_artifact
from tactilelatent.acquisition_data import clip_indices
from tactilelatent.acquisition_pilots import hrdex_native
from tactilelatent.acquisition_archives import tacquad


def test_duplicate_capture_clocks_remain_native_and_ambiguous_frames_are_masked(tmp_path):
    times=np.arange(50,dtype=float)/30
    times[21]=times[20]
    sensor=np.arange(200,dtype=float)/100
    values=np.arange(200)[:,None]
    camera,conflicts,audit=timestamp_ambiguity(times,sensor,values)
    assert audit['duplicate_camera_pairs']==1 and not camera[20:22].any()
    assert camera.sum()==48 and len(conflicts)==0
    with pytest.raises(ValueError):nearest_indices(times,sensor)
    ix,observed,error,_=nearest_indices(times,sensor,allow_equal_queries=True)
    observed &= camera
    class Job:
        root=tmp_path
        def check_space(self,*a,**k):pass
    target=target_artifact(Job(),'duplicate',values[ix],times,['force'],valid=observed[:,None],
                           rgb_frame_valid=camera,allow_equal_timestamps=True)
    with np.load(target['path']) as f:
        np.testing.assert_array_equal(f['timestamps'],times)
        assert not f['valid'][20:22].any()
        np.testing.assert_array_equal(f['values'],values[ix])
    row=dict(frames=len(times),targets=target,views={'0':dict(frame_offset=0,frames=len(times))})
    # A stride-two clip can contain just one of the two tied frames; still reject it.
    with pytest.raises(ValueError,match='ambiguous'):clip_indices(row,24)
    assert len(clip_indices(row,49)[0])==8


def test_repeated_identical_sensor_packets_are_distinct_from_conflicting_packets():
    camera=np.arange(5,dtype=float)
    sensor=np.array([0.,1.,1.,2.,2.,3.,4.])
    values=np.array([0,4,4,6,9,10,11])[:,None]
    mask,conflicts,audit=timestamp_ambiguity(camera,sensor,values)
    assert mask.all() and audit['duplicate_sensor_pairs']==2
    np.testing.assert_array_equal(conflicts,[2.])
    backward=camera.copy();backward[3]=0
    with pytest.raises(ValueError,match='Backward'):timestamp_ambiguity(backward,sensor,values)


def test_partial_capture_keeps_only_supported_complete_histories():
    times=np.arange(80,dtype=float)/30
    observed=np.ones(80,bool);observed[:20]=False
    ends=supervised_history_ends(times,observed)
    assert ends[0]==34 and ends[-1]==79
    mapped=np.arange(80);mapped[55:]=-1
    ends=supervised_history_ends(times,observed,mapped)
    assert ends[0]==34 and ends[-1]==54
    for end in ends:
        ix=np.arange(end-14,end+1,2)
        assert observed[ix].all() and (mapped[ix]>=0).all()
    observed[:]=False
    assert not len(supervised_history_ends(times,observed,mapped))


def test_empty_tacquad_interval_requires_evidence_of_absent_stream():
    payload=b'plastic_bottle,17,124,23,149,0,-1,original description\n'
    records,issues=parse_contact_csv(payload,'contact_indoor.csv',
                                    absent_streams={('plastic_bottle','duragel')})
    assert not issues and records[0]['intervals']['duragel'] is None
    assert records[0]['intervals']['gelsight_mini']==[17,124]
    assert records[0]['raw_intervals']['duragel']==[0,-1]
    assert not records[0]['interval_valid']['duragel']
    _,issues=parse_contact_csv(payload,'contact_indoor.csv')
    assert len(issues)==1 and issues[0]['sensor']=='duragel'


def test_reversed_or_noninteger_contact_annotation_is_reported_without_reordering():
    records,issues=parse_contact_csv(b'object,50,10,0,30,0,40,text\n'
                                    b'bad,0.5,10,0,30,0,40,text\n','contact_fine.csv')
    assert len(records)==1 and len(issues)==2
    assert records[0]['intervals']['gelsight_mini'] is None
    assert records[0]['raw_intervals']['gelsight_mini']==[50,10]
    assert records[0]['intervals']['digit']==[0,30]


def test_annotation_cannot_create_observations_for_a_missing_archive_folder():
    absent={('missing_object',sensor) for sensor in ('gelsight_mini','digit','duragel')}
    records,issues=parse_contact_csv(b'missing_object,10,20,10,20,10,20,text\n',
                                    'contact_indoor.csv',absent_streams=absent)
    assert not issues and not any(records[0]['interval_valid'].values())
    assert all(value is None for value in records[0]['intervals'].values())
    assert records[0]['raw_intervals']['digit']==[10,20]


def test_empty_hrdex_tactile_never_silently_substitutes_force_feedback(tmp_path):
    root=tmp_path/'raw/hand';root.mkdir(parents=True)
    np.save(root/'tactile.npy',np.empty(0))
    np.save(root/'force.npy',np.array([[65535,1,2,3,4,5]]))
    calls=[]
    class Job:
        def download(self,name):calls.append(name);return tmp_path/name
    with pytest.raises(UnavailableTactile):hrdex_native(Job(),dict(raw_path='raw',embodiment='inspire_dftp'))
    assert calls==['raw/hand/tactile.npy','raw/hand/force.npy']


def test_tacquad_indexes_remaining_sensors_when_one_contact_stream_is_absent(tmp_path):
    image=io.BytesIO();Image.fromarray(np.array([[0,65535]],np.uint16)).save(image,format='PNG')
    payload=io.BytesIO()
    with zipfile.ZipFile(payload,'w') as z:
        z.writestr('contact_indoor.csv','bottle,0,0,0,0,0,-1,unchanged description\n')
        z.writestr('data_indoor/bottle/gelsight/0.png',image.getvalue())
        z.writestr('data_indoor/bottle/digit/0.png',image.getvalue())
    original=payload.getvalue();parts=[]
    boundaries=np.linspace(0,len(original),8).astype(int)
    for i,(lo,hi) in enumerate(zip(boundaries[:-1],boundaries[1:]),1):
        p=tmp_path/f'tacquad.zip.{i:03}';p.write_bytes(original[lo:hi]);parts.append(p)
    class Job:
        root=tmp_path/'out'
        spec=dict(revision='pinned')
        def inventory(self):return dict(files=[dict(path=p.name) for p in parts])
        def download(self,name):return tmp_path/name
        def report(self,**kw):pass
    Job.root.mkdir();tacquad(Job())
    manifest=json.loads((Job.root/'manifest.json').read_text())
    assert manifest['excluded_from_joint_training']
    quality=json.loads((Job.root/'quality.json').read_text())
    assert quality['contacts']==1 and quality['unavailable_sensor_intervals']==1
    assert not quality['contact_annotation_issues']
    assert (Job.root/'contacts/contact_indoor.csv').read_bytes()==b'bottle,0,0,0,0,0,-1,unchanged description\n'
    record=json.loads((Job.root/'contact_intervals.jsonl').read_text())
    assert record['intervals']['duragel'] is None
    assert record['interval_valid']['digit'] and record['interval_valid']['gelsight_mini']
