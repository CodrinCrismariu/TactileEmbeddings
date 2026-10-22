import copy
from dataclasses import replace

import numpy as np
import pytest
import torch

from tactilelatent.marker_targets import MarkerConfig, MarkerTracker, detect_markers, MARKER_SOURCES
from tactilelatent.marker_data import marker_collate
from tactilelatent.marker_model import MarkerSpatialVJEPA, marker_loss, spatial_marker_loss, model_from_state, MODEL_TYPE


def image(dx=0,dy=0,missing=None):
    y,x=np.mgrid[:160,:200]
    out=np.full((160,200),180,np.float32)
    for i,(cy,cx) in enumerate(( (yy,xx) for yy in (30,60,90,120) for xx in (30,60,90,120,150))):
        if i==missing:continue
        out-=140*np.exp(-((x-cx-dx)**2+(y-cy-dy)**2)/6)
    return np.repeat(np.clip(out,0,255).astype(np.uint8)[...,None],3,-1)


def test_signed_subpixel_displacements_and_missing_recovery():
    tracker=MarkerTracker(image())
    result=tracker.update(image(-2.25,1.5))
    assert result['valid'].all()
    np.testing.assert_allclose(result['displacement_px'],np.tile([-2.25,1.5],(20,1)),atol=.12)
    hidden=tracker.update(image(-2.25,1.5,missing=3))
    assert hidden['valid'].sum()==19
    assert np.isnan(hidden['displacement_px'][~hidden['valid']]).all()
    assert tracker.update(image())['valid'].all()
    np.testing.assert_allclose(tracker.update(image())['displacement_px'],0,atol=1e-5)


def test_large_motion_not_clipped_to_gate_or_subtracted_as_jitter():
    tracker=MarkerTracker(image())
    moved=tracker.update(image(3,2))
    np.testing.assert_allclose(moved['displacement_px'],np.tile([3,2],(20,1)),atol=.12)
    rejected=tracker.update(image(13,0))
    assert not rejected['valid'].any()


def item(n):
    return dict(source='ftp_unit',marker_target=torch.ones(8,n,2),marker_valid=torch.ones(8,n,dtype=torch.bool),
                marker_reference=torch.rand(n,2),marker_surface=torch.zeros(n,dtype=torch.long),marker_spacing=torch.full((n,),30.))


def test_variable_point_padding_and_nan_masked_gradient():
    batch=marker_collate([item(2),item(3)])
    assert batch['marker_target'].shape==(2,8,3,2)
    assert not batch['marker_valid'][0,:,2].any()
    batch['marker_target'][0,:,2]=float('nan')
    pred=torch.zeros_like(batch['marker_target'],requires_grad=True)
    loss=marker_loss(pred,batch['marker_target'],batch['marker_valid'],batch['marker_surface'])
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(pred.grad).all() and (pred.grad[0,:,2]==0).all()
    no_support=marker_loss(pred.detach(),torch.full_like(pred,float('nan')),torch.zeros_like(batch['marker_valid']),batch['marker_surface'])
    assert no_support==0


def test_surfaces_have_equal_loss_weight_despite_marker_counts():
    pred=torch.zeros(1,1,4,2)
    target=torch.tensor([[[[1.,1.],[1.,1.],[1.,1.],[3.,3.]]]])
    surfaces=torch.tensor([[0,0,0,1]])
    loss=marker_loss(pred,target,torch.ones(1,1,4,dtype=torch.bool),surfaces)
    assert loss.item()==pytest.approx((.95+2.95)/2)


def test_spatial_marker_model_gradients_roles_and_roundtrip():
    torch.manual_seed(45)
    cfg=dict(sensor_code_dims={'egotouch':3},numeric_shapes={'ftp_rdp':[1,1,1]},
        statistics={'egotouch':{'mean':[0.]*3,'std':[1.]*3}},width=16,heads=4,layers=1,
        feedforward=32,dropout=0.,grid_size=2,input_dim=12,marker_hidden=16,
        marker_scales={s:.05 for s in MARKER_SOURCES})
    model=MarkerSpatialVJEPA(**cfg).train()
    features=torch.randn(2,4,3,4,12);mask=torch.ones(2,8,3,dtype=torch.bool)
    batch=marker_collate([item(2),item(3)]);batch['source']=['freetacman']*2
    out=model(features,mask,'freetacman',marker_reference=batch['marker_reference'],marker_surface=batch['marker_surface'])
    assert out['mu'].shape==(2,8,32) and out['marker_displacement'].shape==(2,8,3,2)
    assert not set(MARKER_SOURCES)&set(model.adapters)
    loss,_,kl=spatial_marker_loss(model,out,batch);loss.backward()
    assert kl==0 and torch.isfinite(model.visual.frame_queries.grad).all()
    assert model.visual.frame_queries.grad.abs().sum()>0
    roles=torch.tensor([[0,0,2]]).expand(2,-1)
    torch.testing.assert_close(out['mu'],model.visual.encode(features,mask,roles)[0])
    state=dict(model_type=MODEL_TYPE,model_config=cfg,model=copy.deepcopy(model.state_dict()))
    restored=model_from_state(state).eval()
    torch.testing.assert_close(restored.visual.encode(features,mask)[0],model.visual.encode(features,mask)[0])
    assert 'sensor_code' in restored(features,mask,'egotouch')
    assert restored(features,mask,'ftp_rdp')['pred'].shape==(2,8,1,1,1)
    with pytest.raises(ValueError,match='reference geometry'):restored(features,mask,'ftp_unit')


def test_marker_dataset_never_reads_rgb_tactile_targets(monkeypatch):
    from tactilelatent.marker_data import MarkerTargetClips
    import tactilelatent.aligned_frames as frames
    class Dataset:
        episodes=[dict(source='ftp_vla_touch',episode_id='test',frames=15,sensor_frame_order=list(range(14,-1,-1)))]
        windows=[(0,0)];indices_by_source={'ftp_vla_touch':[0]};span=15;stride=2;rgb_size=384
        def __getitem__(self,index):raise AssertionError('Underlying tactile RGB loader called')
    class Cache:
        def read(self,row,indices):
            assert np.array_equal(indices,np.arange(0,15,2))
            return item(3)
    monkeypatch.setattr(frames,'scene_rgb',lambda row,indices,size:(torch.zeros(8,3,3,2,2),torch.ones(8,3,dtype=torch.bool)))
    result=MarkerTargetClips(Dataset(),Cache())[0]
    assert 'target' not in result and 'code_target' not in result
    assert result['sensor_frame_indices'].tolist()==list(range(14,-1,-2))
