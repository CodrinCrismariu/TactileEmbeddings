import copy
import pytest
import torch

from tactilelatent.vjepa_cached_training import encoded_inputs, check_resume_contract, LEGACY_TRAINER_SHA256
from tactilelatent.vjepa_model import matched_models
from tactilelatent.train_vjepa import make_training_state, train_microbatch
from tactilelatent.train_alignment import sam_replay


def test_cached_inputs_are_detached_and_forbid_live_fallback():
    values=torch.randn(2,4,3,768,requires_grad=True)
    batch={'video_features':values,'view_mask':torch.ones(2,8,3,dtype=torch.bool)}
    output=encoded_inputs(None,batch,torch.device('cpu'))
    assert torch.equal(output,values) and not output.requires_grad
    with pytest.raises(ValueError,match='instantiate'):encoded_inputs(object(),batch,'cpu')
    with pytest.raises(ValueError,match='fallback'):encoded_inputs(None,{'view_mask':batch['view_mask']},'cpu')
    with pytest.raises(ValueError,match='float32'):encoded_inputs(None,{**batch,'video_features':values.half()},'cpu')
    invalid=values.detach().clone();invalid[0,0,0,0]=float('nan')
    with pytest.raises(ValueError,match='float32'):encoded_inputs(None,{**batch,'video_features':invalid},'cpu')


def migration_example():
    cfg=dict(epochs=30,latent_dim=32,learning_rate=5e-5,feature_cache='cache')
    provenance=dict(data={'same':True},teachers={'unchanged':True},backbone={'id':'same'},
        implementation={'train_vjepa.py':'new','vjepa_model.py':'model','vjepa_cached_training.py':'helper'},
        feature_cache={'verified':True})
    saved=dict(config={k:v for k,v in cfg.items() if k!='feature_cache'},provenance=copy.deepcopy(provenance))
    saved['provenance'].pop('feature_cache')
    saved['provenance']['implementation'].pop('vjepa_cached_training.py')
    saved['provenance']['implementation']['train_vjepa.py']=LEGACY_TRAINER_SHA256
    return saved,cfg,provenance


def test_resume_adoption_only_permits_the_verified_input_change():
    saved,cfg,p=migration_example()
    with pytest.raises(ValueError,match='differs'):check_resume_contract(saved,cfg,p)
    assert check_resume_contract(saved,cfg,p,True)
    assert not check_resume_contract({'config':cfg,'provenance':p},cfg,p)
    for key,value in [('epochs',31),('latent_dim',16),('learning_rate',1e-3)]:
        with pytest.raises(ValueError,match='cannot change'):
            check_resume_contract(saved,{**cfg,key:value},p,True)
    for key in ['data','teachers','backbone']:
        with pytest.raises(ValueError,match='cannot change'):
            check_resume_contract(saved,cfg,{**p,key:{'changed':True}},True)
    altered=copy.deepcopy(p);altered['implementation']['vjepa_model.py']='changed'
    with pytest.raises(ValueError,match='cannot change'):check_resume_contract(saved,cfg,altered,True)


def test_both_cached_models_update_without_rgb_and_deterministic_kl_is_zero():
    torch.set_num_threads(2)
    mc=dict(sensor_code_dims={'egotouch':3},numeric_shapes={},statistics={'egotouch':{'mean':[0.]*3,'std':[1.]*3}},adapter_hidden_dims=[8,8])
    models=matched_models(mc)
    cfg=dict(learning_rate=5e-5,min_learning_rate=1e-6,epochs=30,warmup_epochs=2,accumulation=1,weight_decay=.05,sam_rho=.05)
    opts,schedulers,sams=make_training_state(models,cfg,2)
    initial={n:m.visual.prior_mu.weight.detach().clone() for n,m in models.items()}
    batch=dict(video_features=torch.randn(2,4,3,768),view_mask=torch.ones(2,8,3,dtype=torch.bool),
               code_target=torch.randn(2,8,3),code_valid=torch.ones(2,8,dtype=torch.bool))
    encoded=encoded_inputs(None,batch,'cpu')
    replay={n:[] for n in models}
    result=train_microbatch(models,opts,sams,replay,encoded,batch['view_mask'],batch,'egotouch',.001,1,torch.device('cpu'))
    assert result['deterministic'][1][2]==0.
    assert replay['deterministic'][0]['beta']==0. and replay['stochastic'][0]['beta']==.001
    for name,model in models.items():
        sam_replay(model,opts[name],sams[name],replay[name],torch.device('cpu'))
        assert not torch.equal(model.visual.prior_mu.weight,initial[name])
        assert sams[name].steps==1
