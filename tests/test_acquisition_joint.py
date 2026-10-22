"""Chronological histories and native direct heads for the integrated sources."""
import numpy as np
import pytest
import torch

from tactilelatent.acquisition_joint import history_mapping,AllSourceVJEPA


def row(offset=0):
    return dict(frames=20,targets=dict(path='unused',sha256='unused'),views={'0':dict(
        role='ego',frame_offset=offset,frames=20,path='unused')})


def test_sharded_rgb_offset_is_not_compared_with_episode_length():
    native=dict(timestamps=np.arange(20)/30)
    indices,views=history_mapping(row(1000),18,native)
    np.testing.assert_array_equal(indices,np.arange(4,19,2))
    np.testing.assert_array_equal(views['0'],1000+indices)
    with pytest.raises(ValueError,match='boundary'):history_mapping(row(1000),12,native)


def test_explicit_rgb_maps_reject_duplicates_and_missing_history():
    r=row();r['views']['0'].update(frame_indices_key='rgb_frame_indices',frame_indices_artifact='unused')
    native=dict(timestamps=np.arange(20)/30,rgb_frame_indices=np.arange(20),rgb_frame_valid=np.ones(20,bool))
    indices,views=history_mapping(r,18,native);np.testing.assert_array_equal(views['0'],indices)
    native['rgb_frame_indices'][6]=4
    with pytest.raises(ValueError,match='Nonchronological|nonchronological'):history_mapping(r,18,native)
    native['rgb_frame_indices']=np.arange(20);native['rgb_frame_valid'][8]=False
    with pytest.raises(ValueError,match='ambiguous'):history_mapping(r,18,native)


def test_direct_heads_preserve_native_zero_and_exceed_sigmoid_range():
    torch.manual_seed(45)
    cfg=dict(sensor_code_dims={'ftp_unit':2},numeric_shapes={'feel_lab':[5],'hrdexdb_allegro_v5':[4]},
        statistics={'ftp_unit':dict(mean=[0.,0.],std=[1.,1.])},input_dim=8,latent_dim=32,
        width=16,heads=2,layers=1,feedforward=32,dropout=0.,grid_size=2,prediction_mode='current_frame',
        direct_numeric={'feel_lab':dict(scale=[.2]*5),'hrdexdb_allegro_v5':dict(scale=[100.]*4)})
    model=AllSourceVJEPA(**cfg)
    y=torch.tensor([[0.,1.,2.,1000.]])
    torch.testing.assert_close(model.decode_numeric('hrdexdb_allegro_v5',model.normalize_numeric('hrdexdb_allegro_v5',y)),y)
    assert isinstance(model.numeric_heads['feel_lab'],torch.nn.Linear)
    features=torch.randn(2,4,3,4,8);mask=torch.ones(2,8,3,dtype=torch.bool)
    output=model(features,mask,'feel_lab')
    assert output['pred'].shape==(2,1,5) and output['mu'].shape==(2,1,32)
    loss=(output['pred']-2).square().mean();loss.backward()
    assert model.numeric_heads['feel_lab'].weight.grad.abs().sum()>0
    # Actual checkpoint model configuration round-trips without an AE for Lab.
    restored=AllSourceVJEPA(**cfg);restored.load_state_dict(model.state_dict(),strict=True)
    model.eval();restored.eval()
    torch.testing.assert_close(model(features,mask,'feel_lab')['pred'],restored(features,mask,'feel_lab')['pred'])
