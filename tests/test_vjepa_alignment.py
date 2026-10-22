import copy
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from tactilelatent.vjepa_model import (PairLatentEncoder, matched_models, model_from_state,
    MODEL_TYPE, clip_view_mask, video_features, VJEPABackbone)
from tactilelatent.train_alignment import alignment_loss, sam_replay
from tactilelatent.train_vjepa import make_training_state, train_microbatch, commit_pair
from tactilelatent.sam import rng_snapshot, restore_rng


def setup_pair():
    cfg = dict(sensor_code_dims={'egotouch': 3}, numeric_shapes={'ftp_rdp': [1, 1, 1]},
        statistics={'egotouch': {'mean': [0.]*3, 'std': [1.]*3}},
        input_dim=6, latent_dim=32, adapter_hidden_dims=[8,8])
    return cfg, matched_models(cfg)


def test_frame_pair_projection_order_and_camera_mask():
    visual = PairLatentEncoder(1, 1, 'deterministic')
    with torch.no_grad():
        visual.prior_mu.weight.copy_(torch.tensor([[1.], [10.]])); visual.prior_mu.bias.zero_()
    x = torch.arange(1.,5.).reshape(1,4,1,1).expand(-1,-1,3,-1).clone()
    x[:,:,2] = 999.
    mask = torch.tensor([True,True,False]).reshape(1,1,3).expand(1,8,3)
    mu, lv = visual.encode(x,mask)
    torch.testing.assert_close(mu.flatten(),torch.tensor([1.,10.,2.,20.,3.,30.,4.,40.]))
    assert torch.isneginf(lv).all()
    with pytest.raises(ValueError, match='available camera'):
        visual.encode(x,torch.zeros_like(mask))
    broken=mask.clone();broken[:,2,0]=False
    with pytest.raises(ValueError,match='constant'):
        clip_view_mask(broken)


def test_matched_weights_determinism_and_stochastic_gradient():
    cfg, pair=setup_pair(); d,s=pair.values()
    for name, value in d.state_dict().items():
        assert torch.equal(value,s.state_dict()[name])
    x=torch.randn(2,4,3,6);mask=torch.ones(2,8,3,dtype=torch.bool)
    before=rng_snapshot()['cpu']
    a=d(x,mask,'egotouch');b=d(x,mask,'egotouch',sample_latent=True)
    assert torch.equal(before,rng_snapshot()['cpu'])
    assert torch.equal(a['sensor_code'],b['sensor_code'])
    assert not any('logvar' in name for name in d.state_dict())
    s.train();a=s(x,mask,'egotouch');b=s(x,mask,'egotouch')
    assert not torch.equal(a['sensor_code'],b['sensor_code'])
    batch={'code_target':torch.randn(2,8,3),'code_valid':torch.ones(2,8,dtype=torch.bool)}
    loss,_,_=alignment_loss(a,batch,.001);loss.backward()
    assert s.visual.prior_mu.weight.grad.abs().sum()>0
    assert s.visual.prior_logvar.weight.grad.abs().sum()>0
    with pytest.raises(ValueError,match='KL penalty'):
        alignment_loss(d(x,mask,'egotouch'),batch,.001)
    restored=model_from_state(dict(model_type=MODEL_TYPE,model_config={**cfg,'latent_mode':'deterministic'},model=d.state_dict()))
    assert torch.equal(restored(x,mask,'egotouch')['sensor_code'],d(x,mask,'egotouch')['sensor_code'])


def test_features_only_encode_available_ordered_videos_without_gradients():
    class Fake(nn.Module):
        dim=1
        def forward(self,x):
            self.seen=x.detach().clone()
            return x[:,0,:,0,0].reshape(len(x),4,2).mean(-1).unsqueeze(-1)
    b=Fake();rgb=torch.zeros(1,8,3,3,2,2)
    rgb[0,:,0]=torch.arange(8.).reshape(8,1,1,1)
    rgb[0,:,2]=100+torch.arange(8.).reshape(8,1,1,1)
    rgb.requires_grad_()
    mask=torch.tensor([True,False,True]).reshape(1,1,3).expand(1,8,3)
    encoded=video_features(b,dict(rgb=rgb,view_mask=mask),'cpu')
    assert encoded.shape==(1,4,3,1) and not encoded.requires_grad
    torch.testing.assert_close(encoded[0,:,0,0],torch.tensor([.5,2.5,4.5,6.5]))
    torch.testing.assert_close(encoded[0,:,2,0],torch.tensor([100.5,102.5,104.5,106.5]))
    assert not encoded[0,:,1].any() and len(b.seen)==2


def test_backbone_train_keeps_encoder_eval():
    b=VJEPABackbone.__new__(VJEPABackbone);nn.Module.__init__(b)
    b.encoder=nn.Sequential(nn.Linear(2,2),nn.Dropout()).requires_grad_(False)
    b.train();assert not b.encoder.training and not any(p.requires_grad for p in b.parameters())


def test_rgb_size_does_not_change_targets_or_indices(monkeypatch):
    import tactilelatent.alignment_data as module
    import numpy as np
    dataset=module.AlignedClipDataset.__new__(module.AlignedClipDataset)
    dataset.windows=[(0,5)];dataset.episodes=[{'source':'ftp_rdp','episode_id':'ep'}]
    dataset.span=15;dataset.stride=2;dataset.rgb_size=384;dataset.caches={};dataset.include_targets=True
    seen=[]
    def scene(row,indices,size):
        seen.append((indices.tolist(),size))
        return torch.zeros(8,3,3,1,1),torch.ones(8,3,dtype=torch.bool)
    monkeypatch.setattr(module,'scene_rgb',scene)
    monkeypatch.setattr(module,'sensor_indices',lambda row,indices:indices+100)
    native=np.arange(8*2*21*21,dtype=np.float32).reshape(8,2,21,21)
    monkeypatch.setattr(module,'targets',lambda row,indices:native.copy())
    item=dataset[0]
    assert seen==[(list(range(5,20,2)),384)]
    assert item['sensor_frame_indices'].tolist()==list(range(105,120,2))
    assert torch.equal(item['target'],torch.from_numpy(native))


def test_paired_optimizer_resume_matches_uninterrupted(tmp_path):
    torch.set_num_threads(2)
    model_cfg,pair=setup_pair()
    cfg=dict(output=str(tmp_path),epochs=30,seed=45,batch_size=2,accumulation=2,
        learning_rate=5e-5,min_learning_rate=1e-6,weight_decay=.05,warmup_epochs=2,
        sam_rho=.05,beta_max=.001,save_viewer_epochs=True)
    opt,sched,sam=make_training_state(pair,cfg,4)
    x=torch.randn(2,4,3,6);mask=torch.ones(2,8,3,dtype=torch.bool)
    batch=dict(code_target=torch.randn(2,8,3),code_valid=torch.ones(2,8,dtype=torch.bool))
    def update(models,opts,schedulers,sams):
        replay={n:[] for n in models}
        for _ in range(2):
            train_microbatch(models,opts,sams,replay,x,mask,batch,'egotouch',.001,2,torch.device('cpu'))
        for n,m in models.items():
            before=rng_snapshot()['cpu']
            sam_replay(m,opts[n],sams[n],replay[n],torch.device('cpu'))
            assert torch.equal(before,rng_snapshot()['cpu'])
            schedulers[n].step()
    update(pair,opt,sched,sam)
    validation={n:{'egotouch':{'supervision_loss':1.}} for n in pair}
    generator=torch.Generator().manual_seed(45)
    commit_pair(tmp_path,1,pair,opt,sched,sam,model_cfg,cfg,{}, {},{n:1. for n in pair},validation,0.,generator)
    pair_state=torch.load(tmp_path/'pair_state.pt',weights_only=False)
    update(pair,opt,sched,sam)
    resumed=matched_models(model_cfg);o2,sc2,sa2=make_training_state(resumed,cfg,4)
    for n,m in resumed.items():
        saved=torch.load(tmp_path/n/'epoch_001.pt',weights_only=False)
        m.load_state_dict(saved['model']);o2[n].load_state_dict(saved['optimizer'])
        sc2[n].load_state_dict(saved['scheduler']);sa2[n].load_state_dict(saved['sam'])
    restore_rng(pair_state['rng']);update(resumed,o2,sc2,sa2)
    for n,m in pair.items():
        for key,value in m.state_dict().items():
            torch.testing.assert_close(value,resumed[n].state_dict()[key],atol=0,rtol=0)
        assert sc2[n].state_dict()==sched[n].state_dict() and sa2[n].steps==sam[n].steps
