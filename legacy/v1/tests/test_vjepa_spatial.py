import copy
import pytest
import torch

from tactilelatent.vjepa_spatial import SpatialFrameEncoder,JointSpatialVJEPA,MODEL_TYPE,model_from_state
from tactilelatent.train_alignment import alignment_loss,sam_replay
from tactilelatent.sam import SAMStep,rng_snapshot


def config():
    return dict(sensor_code_dims={'egotouch':3},numeric_shapes={'ftp_rdp':[1,1,1]},
        statistics={'egotouch':{'mean':[0.]*3,'std':[1.]*3}},width=16,heads=4,layers=2,
        feedforward=32,dropout=.1,grid_size=2,adapter_hidden_dims=(8,8),input_dim=12)


def inputs():
    return torch.randn(2,4,3,4,12),torch.ones(2,8,3,dtype=torch.bool)


def test_ordered_frame_latents_and_spatial_information_not_averaged():
    torch.manual_seed(45);m=JointSpatialVJEPA(**config()).eval();x,mask=inputs()
    a=m(x,mask,'egotouch');b=m(x,mask,'egotouch')
    assert a['mu'].shape==(2,8,32) and torch.equal(a['mu'],b['mu'])
    assert not torch.equal(a['mu'][:,0],a['mu'][:,1])
    assert not any('logvar' in n for n,_ in m.named_parameters())
    # Exchange image regions, preserving the exact arithmetic spatial mean.
    swapped=x.flip(3)
    torch.testing.assert_close(x.mean(3),swapped.mean(3))
    assert not torch.allclose(m(swapped,mask,'egotouch')['mu'],a['mu'])


def test_missing_cameras_are_masked_and_freetacman_is_scene_role():
    m=JointSpatialVJEPA(**config()).eval();x,mask=inputs();mask[:,:,1:]=False
    a=m.visual.encode(x,mask)[0];x[:,:,1:]=float('nan')
    torch.testing.assert_close(m.visual.encode(x,mask)[0],a,rtol=0,atol=0)
    mask[:]=False
    with pytest.raises(ValueError,match='available'):m.visual.encode(x,mask)
    m=JointSpatialVJEPA(**{**config(),'sensor_code_dims':{'freetacman':3},'statistics':{'freetacman':{'mean':[0.]*3,'std':[1.]*3}}}).eval()
    x,mask=inputs();roles=torch.tensor([[0,0,2]]).expand(2,-1)
    torch.testing.assert_close(m(x,mask,'freetacman')['mu'],m.visual.encode(x,mask,roles)[0])


def test_full_training_sam_and_resume_equivalence():
    torch.manual_seed(45);m=JointSpatialVJEPA(**config()).train();x,mask=inputs()
    batch=dict(code_target=torch.randn(2,8,3),code_valid=torch.ones(2,8,dtype=torch.bool))
    opt=torch.optim.AdamW(m.parameters(),lr=.001);sam=SAMStep(opt,.05)
    def step(model,optimizer,sam):
        before=rng_snapshot();loss,_,kl=alignment_loss(model(x,mask,'egotouch'),batch,0.)
        assert kl==0;loss.backward()
        for name,p in model.named_parameters():
            if name.startswith('visual.') and p.grad is not None:assert torch.isfinite(p.grad).all()
        replay=[dict(features=x,mask=mask,source='egotouch',batch=batch,beta=0.,rng=before)]
        sam_replay(model,optimizer,sam,replay,torch.device('cpu'))
    before=m.visual.frame_queries.detach().clone();step(m,opt,sam)
    assert not torch.equal(before,m.visual.frame_queries)
    state=copy.deepcopy(dict(model_type=MODEL_TYPE,model_config=config(),model=m.state_dict()))
    restored=model_from_state(state).train();opt2=torch.optim.AdamW(restored.parameters(),lr=.001)
    opt2.load_state_dict(copy.deepcopy(opt.state_dict()));sam2=SAMStep(opt2,.05);sam2.load_state_dict(sam.state_dict())
    rng=torch.get_rng_state();step(m,opt,sam);torch.set_rng_state(rng);step(restored,opt2,sam2)
    for name,p in m.state_dict().items():torch.testing.assert_close(p,restored.state_dict()[name],rtol=0,atol=0)


def test_deterministic_rejects_gaussian_and_kl():
    with pytest.raises(ValueError,match='deterministic'):JointSpatialVJEPA(**config(),latent_mode='gaussian')
    m=JointSpatialVJEPA(**config());x,mask=inputs();out=m(x,mask,'egotouch')
    with pytest.raises(ValueError,match='KL'):alignment_loss(out,dict(code_target=torch.zeros(2,8,3),code_valid=torch.ones(2,8,dtype=torch.bool)),.001)
