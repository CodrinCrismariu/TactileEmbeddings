import copy

import pytest
import torch

from tactilelatent.train import make_scheduler
from tactilelatent.train_alignment import alignment_scheduler,extension_config
from tactilelatent.train_vjepa_markers import check_epoch_extension,CODE_MODEL_TYPE


def state_and_request():
    cfg=dict(epochs=20,accumulation=2,warmup_epochs=2,learning_rate=.001,min_learning_rate=.00001,seed=45)
    parameter=torch.nn.Parameter(torch.ones(1));opt=torch.optim.AdamW([parameter],lr=cfg['learning_rate'])
    scheduler=make_scheduler(opt,cfg,10)
    for _ in range(45):opt.step();scheduler.step()
    provenance=dict(data={'original':True},implementation={'train_vjepa_markers.py':'old','marker_code_cache.py':'same'})
    state=dict(config=cfg,epoch=8,scheduler=scheduler.state_dict(),optimizer=opt.state_dict(),
               provenance=provenance,model_config={'latent_dim':32},model_type=CODE_MODEL_TYPE)
    extended=extension_config(cfg,state,30)
    request=dict(original_config=cfg,target_epochs=30,after_epoch=9,lr_extension=extended['lr_extension'],
                 previous_trainer_sha256='old',updated_trainer_sha256='new')
    current=copy.deepcopy(provenance);current['implementation']['train_vjepa_markers.py']='new'
    return state,extended,current,request


def test_epoch_extension_retains_optimizer_lr_and_decays_smoothly_to_new_end():
    state,cfg,provenance,request=state_and_request()
    check_epoch_extension(state,cfg,state['model_config'],provenance,request)
    opt=torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))],lr=cfg['learning_rate'])
    scheduler=alignment_scheduler(opt,cfg,10)
    opt.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
    anchor=opt.param_groups[0]['lr'];assert anchor==cfg['lr_extension']['anchor_lr']
    rates=[]
    for _ in range(150-45):opt.step();scheduler.step();rates.append(opt.param_groups[0]['lr'])
    assert rates[0]<=anchor and rates[0]>anchor*.99
    assert all(a>=b for a,b in zip(rates,rates[1:]))
    assert rates[-1]==pytest.approx(cfg['min_learning_rate'])
    # Regular later resume preserves the extended schedule exactly.
    continuation=copy.deepcopy(scheduler.state_dict());saved_opt=copy.deepcopy(opt.state_dict())
    clone_opt=torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))],lr=cfg['learning_rate'])
    clone_scheduler=alignment_scheduler(clone_opt,cfg,10)
    clone_opt.load_state_dict(saved_opt);clone_scheduler.load_state_dict(continuation)
    assert clone_opt.param_groups[0]['lr']==opt.param_groups[0]['lr']


@pytest.mark.parametrize('change',['data','cache_implementation','model','configuration','trainer','anchor'])
def test_epoch_extension_rejects_changes_other_than_authorized_duration(change):
    state,cfg,provenance,request=state_and_request();mc=copy.deepcopy(state['model_config'])
    if change=='data':provenance['data']={'original':False}
    elif change=='cache_implementation':provenance['implementation']['marker_code_cache.py']='changed'
    elif change=='model':mc['latent_dim']=16
    elif change=='configuration':cfg['seed']=46
    elif change=='trainer':provenance['implementation']['train_vjepa_markers.py']='unrecognized'
    elif change=='anchor':request['lr_extension']=dict(anchor_update=44,anchor_lr=0.)
    with pytest.raises(ValueError,match='Epoch extension cannot change'):
        check_epoch_extension(state,cfg,mc,provenance,request)
