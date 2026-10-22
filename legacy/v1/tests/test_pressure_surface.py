import json
from pathlib import Path
import threading
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tactilelatent import pressure_surface as ps


def test_missing_models_do_not_advertise_an_unrunnable_surface(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path);monkeypatch.delenv('EGOPRESSURE_MANO_PATH',raising=False)
    result=ps.surface_assets()
    assert not any(r['available'] for r in result['sides'].values())
    assert result['native_texture_shape']==[224,224] and not result['needs_mano_pickle']


def test_uv_seam_indices_and_hand_sides_are_preserved(tmp_path,monkeypatch):
    root=tmp_path/'user_models';root.mkdir()
    (root/'MANO_UV_left.obj').write_text('vt 0 0\nvt 1 0\nvt 1 1\nvt 0 .5\nvt 0 1\n'
        'f 1/1 2/2 3/3\nf 1/4 3/3 4/5\n')
    monkeypatch.chdir(tmp_path);monkeypatch.setenv('EGOPRESSURE_MANO_PATH',str(root))
    result=ps.surface_assets();mesh=result['sides']['left']['mesh']
    assert result['sides']['left']['available'] and not result['sides']['right']['available']
    assert mesh['faces']==[[0,1,2],[0,2,3]]
    assert mesh['uv_faces']==[[0,1,2],[3,2,4]]
    assert mesh['uv'][0]==[0.,0.] and mesh['uv'][2]==[1.,1.]
    assert mesh['image_origin']=='top' and mesh['uv_origin']=='bottom'
    assert mesh['pixel_mapping']=='column=u*(W-1); row=(1-v)*(H-1)'


def test_non_uv_or_out_of_range_mesh_stays_unavailable(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path);monkeypatch.delenv('EGOPRESSURE_MANO_PATH',raising=False)
    root=tmp_path/'mano_models';root.mkdir()
    (root/'MANO_UV_left.obj').write_text('v 0 0 0\nf 1 2 3\n')
    (root/'MANO_UV_right.obj').write_text('vt 2 0\nvt 0 0\nvt 0 1\nf 1/1 2/2 3/3\n')
    result=ps.surface_assets()
    assert not any(r['available'] for r in result['sides'].values())
    assert all(r.get('error') for r in result['sides'].values())


def test_surface_pose_joins_native_frame_identifier_and_rejects_wrong_hand(tmp_path):
    path=tmp_path/'annotations.parquet'
    vertices=np.arange(778*3,dtype=float).tolist();joints=np.arange(63,dtype=float).tolist()
    pq.write_table(pa.Table.from_pylist([
        dict(frame=19,has_annotation=True,hand_side='left',vertices=vertices,joint_position=joints),
        dict(frame=21,has_annotation=False,hand_side='left',vertices=None,joint_position=None)]),path)
    row=dict(source='egopressure',source_frame_ids=[19,20,21],hand_side='left',annotation_path=str(path))
    viewer=SimpleNamespace(lock=threading.RLock(),_episode=lambda episode,frame:row)
    result=ps.surface_frame(viewer,0,0)
    assert result['source_frame_id']==19 and result['frame']==0 and result['available']
    np.testing.assert_array_equal(np.array(result['vertices']).reshape(-1),vertices)
    assert not ps.surface_frame(viewer,0,1)['available']
    assert not ps.surface_frame(viewer,0,2)['available']
    row['hand_side']='right'
    with pytest.raises(ValueError,match='hand side'):ps.surface_frame(viewer,0,0)
