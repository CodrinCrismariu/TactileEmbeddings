import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from tactilelatent.vjepa_cache import (CACHE_VERSION, partition_plan, open_partition, commit_prefix,
    finalize_partition, VJEPAFeatureCache, CachedAlignedClips)
from tactilelatent.aligned_frames import write_json


def fixture_plan():
    rows=[dict(source='a',split='train',episode_id='one',views={'0':'ego','2':'wrist'}),
          dict(source='a',split='val',episode_id='two',views={'0':'ego'}),
          dict(source='ftp_motiontrans',split='train',episode_id='heldout',views={'0':'ego'})]
    dataset=SimpleNamespace(episodes=rows,windows=[(0,0),(0,15),(1,0),(2,0)])
    plans=partition_plan(dataset)
    return plans,next(p for p in plans if p['source']=='a' and p['split']=='train')


def test_partition_separates_sources_splits_and_camera_offsets():
    plans,p=fixture_plan()
    assert {(v['source'],v['split']) for v in plans}=={('a','train'),('a','val'),('ftp_motiontrans','train')}
    assert p['clips'].tolist()==[[0,0,0],[0,15,2]]
    assert p['camera_clips']==4 and p['episodes'][0]['camera_slots']==[0,2]


def test_interrupted_write_resume_and_exact_float32_roundtrip(tmp_path):
    plans,p=fixture_plan();folder=tmp_path/'a'/'train'
    array,start,finished=open_partition(folder,p);assert start==0 and not finished
    expected=np.arange(4*4*768,dtype=np.float32).reshape(4,4,768)/8192
    array[:2]=expected[:2];commit_prefix(folder,array,p,1)
    array[2:]=-999;array.flush();del array  # interrupted, uncommitted second clip
    array,start,finished=open_partition(folder,p)
    assert start==1 and not finished
    np.testing.assert_array_equal(array[:2],expected[:2])
    array[2:]=expected[2:];commit_prefix(folder,array,p,2);del array
    complete=finalize_partition(folder,p)
    assert complete['complete'] and complete['clips']==2
    write_json(tmp_path/'index.json',dict(version=CACHE_VERSION,signature={'test':1}))
    with pytest.raises(ValueError,match='not complete'):VJEPAFeatureCache(tmp_path)
    write_json(tmp_path/'complete.json',{'complete':True})
    reader=VJEPAFeatureCache(tmp_path,{'test':1})
    for start,offset in [(0,0),(15,2)]:
        features,mask=reader.read('a','train','one',np.arange(8)*2+start)
        assert features.dtype==torch.float32 and features.shape==(4,3,768)
        assert mask[0].tolist()==[True,False,True] and not features[:,1].any()
        np.testing.assert_array_equal(features[:,[0,2]].permute(1,0,2),expected[offset:offset+2])
    with pytest.raises(ValueError,match='Stale'):VJEPAFeatureCache(tmp_path,{'test':2})
    with pytest.raises(ValueError,match='exact ordered'):reader.read('a','train','one',np.arange(8)[::-1])
    with pytest.raises(KeyError,match='Full clip'):reader.read('a','train','one',np.arange(8)*2+1)
    with pytest.raises(KeyError,match='Episode absent'):reader.read('a','train','heldout',np.arange(8)*2)
    array,start,finished=open_partition(folder,p);assert start==2 and finished
    changed={**p,'clips':p['clips'].copy()};changed['clips'][0,1]=1
    with pytest.raises(ValueError,match='Corrupt'):open_partition(folder,changed)


def test_incomplete_and_nonfinite_data_cannot_be_published(tmp_path):
    _,p=fixture_plan();folder=tmp_path/'a'/'train'
    array,_,_=open_partition(folder,p);array[:]=np.nan;commit_prefix(folder,array,p,2);del array
    with pytest.raises(ValueError,match='Nonfinite'):finalize_partition(folder,p)
    assert not (folder/'complete.json').exists()


def test_cached_dataset_never_reads_rgb_and_preserves_targets(monkeypatch):
    import tactilelatent.vjepa_cache as module
    row=dict(source='ftp_motiontrans',split='val',episode_id='motion')
    aligned=SimpleNamespace(frames=8,stride=2,rgb_size=384,episodes=[row],windows=[(0,15)],indices_by_source={'ftp_motiontrans':[0]},caches={},include_targets=True)
    class Cache:
        def read(self,source,split,episode,indices):
            assert (source,split,episode)==('ftp_motiontrans','val','motion')
            assert indices.tolist()==list(range(15,30,2))
            return torch.ones(4,3,768),torch.ones(8,3,dtype=torch.bool)
    values=np.arange(48,dtype=np.float32).reshape(8,6,1,1)
    monkeypatch.setattr(module,'targets',lambda row,indices:values.copy())
    monkeypatch.setattr(module,'sensor_indices',lambda row,indices:indices)
    def fail(*args,**kwargs):raise AssertionError('RGB decoding is forbidden for cached training')
    monkeypatch.setattr(module,'scene_rgb',fail)
    item=CachedAlignedClips(aligned,Cache())[0]
    assert 'rgb' not in item and torch.equal(item['target'],torch.from_numpy(values))
    assert item['video_features'].shape==(4,3,768)
