import copy

import numpy as np
import pytest
import torch

from tactilelatent.aligned_frames import file_hash
from tactilelatent.bimanual_mirror import load_policy
from tactilelatent.mirror_balanced_training import FramePool, reflect_batch, paired_loss, quality, train_source


def test_pair_loss_keeps_both_gradients_positive_and_emphasizes_worse_side():
    a=torch.tensor(1.,requires_grad=True);b=torch.tensor(3.,requires_grad=True)
    paired_loss(a,b).backward()
    assert a.grad==.25 and b.grad==.75
    with pytest.raises(ValueError):paired_loss(a,b,.5)


def marker_item(points,frames):
    y=torch.arange(frames*points*2).float().reshape(frames,points,2)/20
    y[0,0]=float('nan');valid=torch.isfinite(y).all(-1)
    return dict(marker_target=y,marker_valid=valid,marker_reference=torch.linspace(.1,.9,2*points).reshape(points,2),
                marker_surface=torch.arange(points)%4,marker_spacing=torch.linspace(10.,20.,points))


def test_cross_episode_pool_mirrors_geometry_vectors_masks_and_padding(monkeypatch):
    from tactilelatent import mirror_preparation as prep
    rows=[dict(source='baihu',split='train',episode_id=str(i),platform=p,frames=3)
          for i,p in enumerate(['D-WHEEL','PIKA'])]
    monkeypatch.setattr(prep.NativeTargets,'read',lambda self,r,indices:marker_item(3+int(r['episode_id']),len(indices)))
    pool=FramePool(rows,'train',6)
    assert len(pool)==6 and set(pool.groups)=={'D-WHEEL','PIKA'}
    batch=pool.batch([0,3,1,4])
    spec=load_policy('configs/bimanual_mirror_policy_v2.json')['sources']['baihu']
    mirrored=reflect_batch(batch,spec);restored=reflect_batch(mirrored,spec)
    for key in batch:torch.testing.assert_close(restored[key],batch[key],equal_nan=True)
    actual=torch.arange(4)[None]<batch['marker_count'][:,None]
    assert torch.equal(mirrored['marker_surface'][actual],3-batch['marker_surface'][actual])
    torch.testing.assert_close(mirrored['marker_reference'][...,0][actual],1-batch['marker_reference'][...,0][actual])
    assert not mirrored['marker_reference'][~actual].any()
    torch.testing.assert_close(mirrored['marker_target'][...,0],-batch['marker_target'][...,0],equal_nan=True)
    assert torch.equal(mirrored['marker_valid'],batch['marker_valid'])
    assert torch.equal(mirrored['frame_indices'],batch['frame_indices'])


def metrics(a,b,active_a=None,active_b=None):
    def one(x,active):
        return dict(mae=x,all_zero_baseline_mae=1.,active_mae=x*2 if active is None else active,
                    active_zero_baseline_mae=2.)
    return dict(original=one(a,active_a),mirrored=one(b,active_b))


def test_parity_requires_good_absolute_quality_and_active_motion_accuracy():
    reference=metrics(.1,.3);initial=metrics(.105,.2)
    assert quality(metrics(.105,.11),reference,initial)['parity_achieved']
    assert not quality(metrics(.19,.195),reference,initial)['accepted']
    candidate=quality(metrics(.105,.11,.21,.3),reference,initial)
    assert candidate['accepted'] and not candidate['parity_achieved']
    candidate=metrics(.105,.11)
    candidate['original']['all_zero_baseline_mae']=.01
    assert not quality(candidate,reference,initial)['accepted']


def test_parity_cannot_hide_one_bad_baihu_platform():
    reference=metrics(.1,.3);initial=metrics(.105,.2);candidate=metrics(.105,.11)
    for value,platform in [(reference,metrics(.1,.3)),(initial,metrics(.105,.2)),(candidate,metrics(.105,.18))]:
        for variant in value:value[variant]['by_platform']={'PIKA':platform[variant]}
    result=quality(candidate,reference,initial)
    assert result['accepted'] and not result['parity_achieved']
    assert result['ratios']['PIKA/mae']>1.1


def test_training_checkpoint_resume_matches_uninterrupted_and_completed_run_is_noop(tmp_path,monkeypatch):
    from tactilelatent import mirror_preparation as prep
    from tactilelatent.sensor_autoencoder import SensorAutoencoder
    import tactilelatent.mirror_balanced_training as training
    torch.set_num_threads(1);torch.manual_seed(45)
    model=SensorAutoencoder([2,3,3],latent_dim=2,channels=[2])
    path=tmp_path/'parent.pt';torch.save(dict(model_config=model.config,model=model.state_dict()),path)
    info=dict(checkpoint=str(path),sha256=file_hash(path),code_dim=2)
    parent=dict(info,original_checkpoint=str(path))
    rows=[dict(source='egotouch',episode_id=s,split=s,frames=4) for s in ['train','val']]
    monkeypatch.setattr(prep,'target_snapshot',lambda *args:'native-targets')
    def read(self,row,indices):
        target=torch.linspace(.1,.9,72).reshape(4,2,3,3)[indices].clone()
        return dict(target=target,target_mask=torch.ones_like(target,dtype=torch.bool))
    monkeypatch.setattr(prep.NativeTargets,'read',read)
    policy=load_policy('configs/bimanual_mirror_policy_v2.json')
    config=dict(epochs=2,val_frames=4,train_frames=4,batch_size=2,learning_rate=1e-4,
        weight_decay=1e-4,min_learning_rate=1e-6,seed=45,resample_every=5,worse_weight=.25,
        parity_gap=.1,min_delta=.0001,min_epochs=2,patience=10)
    full=tmp_path/'full';full.mkdir();interrupted=tmp_path/'interrupted';interrupted.mkdir()
    train_source('egotouch',rows,info,parent,policy,full,config,torch.device('cpu'))
    original_checkpoint=training.atomic_checkpoint
    def interrupt(path,state):
        original_checkpoint(path,state)
        if path.name=='last.pt' and state['continuation_epoch']==1:raise InterruptedError('simulated shutdown')
    monkeypatch.setattr(training,'atomic_checkpoint',interrupt)
    with pytest.raises(InterruptedError):
        train_source('egotouch',rows,info,parent,policy,interrupted,config,torch.device('cpu'))
    monkeypatch.setattr(training,'atomic_checkpoint',original_checkpoint)
    train_source('egotouch',rows,info,parent,policy,interrupted,config,torch.device('cpu'))
    checkpoints=[torch.load(p/'teachers/egotouch/last.pt',weights_only=False) for p in [full,interrupted]]
    for key,value in checkpoints[0]['model'].items():
        torch.testing.assert_close(value,checkpoints[1]['model'][key],rtol=0,atol=0)
    assert checkpoints[0]['continuation_epoch']==checkpoints[1]['continuation_epoch']==2
    before=file_hash(interrupted/'teachers/egotouch/last.pt')
    result=train_source('egotouch',rows,info,parent,policy,interrupted,config,torch.device('cpu'))
    assert result['training_complete'] and before==file_hash(interrupted/'teachers/egotouch/last.pt')
