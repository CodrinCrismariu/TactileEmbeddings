import numpy as np
import pytest
import torch

from tactilelatent.pressure_teacher import (PressureAutoencoder,balanced_frame_mean,choose_training_indices,
    reconstruction_loss,metric_sums,summarize_metrics,pressure_logits)
from tactilelatent.sensor_autoencoder import SensorAutoencoder


def test_active_pressure_gradients_survive_saturated_sigmoid():
    logits=torch.full((1,1,8,8),-150.,requires_grad=True)
    target=torch.zeros_like(logits);target[0,0,3,4]=.4
    valid=torch.ones_like(target,dtype=torch.bool)
    old=(logits.sigmoid()-target).abs().mean()
    assert torch.autograd.grad(old,logits,retain_graph=True)[0][0,0,3,4]==0
    new=reconstruction_loss(logits,target,valid)
    gradient=torch.autograd.grad(new,logits)[0]
    assert torch.isfinite(new) and gradient[0,0,3,4]<-.1


def test_background_pixel_count_does_not_dilute_active_region_loss():
    small=torch.tensor([[[[.4,0.]]]])
    large=torch.zeros(1,1,1,10000);large[...,0]=.4
    a=balanced_frame_mean(small,small,torch.ones_like(small,dtype=torch.bool))
    b=balanced_frame_mean(large,large,torch.ones_like(large,dtype=torch.bool))
    torch.testing.assert_close(a,b)


def test_zero_and_missing_frames_remain_distinct():
    target=torch.zeros(2,1,8,8);target[1]=float('nan')
    logits=torch.full_like(target,-4.,requires_grad=True)
    loss=reconstruction_loss(logits,target,torch.isfinite(target));loss.backward()
    assert torch.isfinite(loss) and logits.grad[0].abs().sum()>0 and logits.grad[1].abs().sum()==0
    zero=summarize_metrics(metric_sums(torch.zeros_like(target),target,torch.isfinite(target)))
    assert zero['all_count']==64 and zero['background_mae']==0


def test_perfect_reconstruction_beats_collapsed_zero_on_selection_metric():
    y=torch.zeros(2,1,16,16);y[0,0,3:6,2:5]=.4;valid=torch.ones_like(y,dtype=torch.bool)
    perfect=summarize_metrics(metric_sums(y,y,valid))
    zero=summarize_metrics(metric_sums(torch.zeros_like(y),y,valid))
    assert perfect['balanced_mae']==0 and zero['contact_to_zero_ratio']==1
    assert zero['balanced_mae']>0 and zero['predicted_contact_pressure_ratio']==0


def test_frame_sampling_keeps_real_zero_frames_and_explicit_ids():
    rng=np.random.default_rng(45);valid=np.arange(100);positive=np.arange(20)
    ix=choose_training_indices(valid,positive,16,rng)
    assert len(ix)==len(set(ix))==16 and np.isin(ix,positive).sum()==12
    ix=choose_training_indices(valid,[],16,rng);assert len(ix)==16
    ix=choose_training_indices([],[],16,rng);assert len(ix)==0


def test_logits_training_preserves_single_sigmoid_output_architecture():
    model=SensorAutoencoder([1,16,16],channels=[4,8],latent_dim=64)
    x=torch.rand(2,1,16,16);valid=torch.ones_like(x,dtype=torch.bool)
    torch.testing.assert_close(pressure_logits(model,x,valid).sigmoid(),model(x,valid))
    assert model.encode(x,valid).shape==(2,64) and model(x,valid).shape==x.shape


def test_conditioned_teacher_checkpoint_preserves_raw_target_mask_and_single_output(tmp_path):
    from tactilelatent.expansion_models import load_teacher
    model=PressureAutoencoder(target_shape=[1,16,16],channels=[4,8],latent_dim=64,
        input_mean=.001,input_scale=.02)
    target=torch.zeros(2,1,16,16);target[0,0,4,5]=.5;target[1,0,2,3]=float('nan')
    before=target.clone();valid=torch.isfinite(target);output=model(target,valid)
    assert torch.isfinite(output).all() and output.shape==target.shape
    torch.testing.assert_close(target,before,equal_nan=True)
    path=tmp_path/'teacher.pt';torch.save(dict(model_type='expansion_pressure_autoencoder_v2',
        model_config=model.config,model=model.state_dict()),path)
    restored,_=load_teacher(path)
    torch.testing.assert_close(restored(target,valid),output)
    assert all(not p.requires_grad for p in restored.parameters())
    from tactilelatent.sensor_autoencoder import make_autoencoder
    generic=make_autoencoder(model.config);generic.load_state_dict(model.state_dict())
    torch.testing.assert_close(generic(target,valid),output)


def test_conditioned_codes_use_separate_cache_with_matching_teacher_and_frame_ids(tmp_path,monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from tactilelatent import data_expansion as data
    from tactilelatent import expansion_training as training
    from tactilelatent.aligned_frames import write_json,file_hash,json_hash
    root=tmp_path/'output';root.mkdir();monkeypatch.setattr(data,'ROOT',root);monkeypatch.setattr(data,'RESERVE',0)
    monkeypatch.setattr(training,'ROOT',root)
    norm=root/'pressure_normalization.json';write_json(norm,dict(scale_counts=10.))
    rows=[]
    for split in ['train','val']:
        target=tmp_path/(split+'.parquet')
        pq.write_table(pa.Table.from_pylist([dict(frame=i+1,pressure_map=np.ones(224*224,np.float32).tolist(),
            pressure_map_range=[0.,float(i+1)]) for i in range(2)]),target)
        rows.append(dict(source='egopressure',episode_id=split,split=split,frames=2,
            source_frame_ids=[1,2],target_path=str(target),pressure_normalization=str(norm)))
    model=PressureAutoencoder(target_shape=[1,224,224],channels=[4,8],latent_dim=64,input_mean=.001,input_scale=.02)
    checkpoint=root/'new.pt';torch.save(dict(model_type='expansion_pressure_autoencoder_v2',
        model_config=model.config,model=model.state_dict(),config=dict(
        target_snapshot_sha256=json_hash({r['episode_id']:file_hash(r['target_path']) for r in rows}),
        pressure_normalization_sha256=file_hash(norm))),checkpoint)
    destination=root/'codes/repair_v2'
    result=training.build_codes('egopressure',rows,checkpoint,device='cpu',destination=destination)
    assert result['complete'] and result['teacher_sha256']==file_hash(checkpoint)
    assert (destination/'verification.json').exists() and not (root/'codes/egopressure').exists()
    for row in result['episodes']:
        with np.load(row['path']) as values:
            assert values['codes'].shape==(2,64) and values['valid'].all()
            np.testing.assert_array_equal(values['frame_indices'],[0,1])
