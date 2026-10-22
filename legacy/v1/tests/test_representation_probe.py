import pytest
import torch

from tactilelatent.representation_probe import pooled_dino,check_pair,feature_table,paired_difference


def cache(episodes=('train',),dimension=32):
    count=len(episodes)
    return dict(values=torch.randn(count,2,dimension),target=torch.rand(count,2,6,1,1),
        valid=torch.ones(count,2,6,1,1,dtype=torch.bool),episode_id=list(episodes))


def test_pool_ignores_absent_cameras_and_preserves_time():
    values=torch.zeros(1,2,3,4,2)
    values[0,0,0]=2;values[0,0,1]=6;values[0,0,2]=1000
    values[0,1,0]=10;values[0,1,1]=1000;values[0,1,2]=1000
    mask=torch.tensor([[[True,True,False],[True,False,False]]])
    actual=pooled_dino(values,mask)
    torch.testing.assert_close(actual,torch.tensor([[[4.,4.],[10.,10.]]]))


def test_pool_rejects_missing_all_cameras():
    with pytest.raises(ValueError):
        pooled_dino(torch.zeros(1,2,3,4,2),torch.zeros(1,2,3,dtype=torch.bool))


def test_heldout_episode_leakage_rejected():
    with pytest.raises(ValueError,match='leakage'):
        check_pair(cache(),cache())


def test_missing_targets_and_nonfinite_features_rejected():
    train,validation=cache(),cache(('val',))
    validation['valid'][0,0,0]=False
    with pytest.raises(ValueError):
        check_pair(train,validation)
    validation=cache(('val',));validation['values'][0,0,0]=float('nan')
    with pytest.raises(ValueError):
        check_pair(train,validation)


def test_dimensions_must_match_and_target_clip_order_is_retained():
    with pytest.raises(ValueError,match='dimensions'):
        check_pair(cache(),cache(('val',),768))
    item=cache(('first','second'))
    table=feature_table(item,[1,0],torch.device('cpu'))
    assert table['mu'].shape==(4,32) and table['y'].shape==(4,6)
    assert torch.isneginf(table['logvar']).all()
    torch.testing.assert_close(table['mu'],item['values'][[1,0]].flatten(0,1))
    torch.testing.assert_close(table['y'],item['target'][[1,0]].flatten(0,1).flatten(1))


def test_paired_bootstrap_uses_equal_episode_observations():
    a=dict(validation=dict(episode_errors=[dict(episode_id='one',absolute_sum=10.,count=100),dict(episode_id='two',absolute_sum=20.,count=200)]))
    b=dict(validation=dict(episode_errors=[dict(episode_id='two',absolute_sum=18.,count=200),dict(episode_id='one',absolute_sum=9.,count=100)]))
    result=paired_difference(a,b)
    assert result['candidate_minus_reference_mae']==pytest.approx(-.01)
    assert result['paired_episode_bootstrap_ci95']==pytest.approx([-.01,-.01])
    b['validation']['episode_errors'][0]['count']=199
    with pytest.raises(ValueError):
        paired_difference(a,b)


def test_complete_head_fit_preserves_cached_features(tmp_path):
    from tactilelatent.representation_probe import fit_probe
    torch.set_num_threads(2)
    train=cache(tuple(f'train-{i}' for i in range(30)))
    validation=cache(('val-one','val-two'))
    original=train['values'].clone()
    train['values'].requires_grad_()
    result=fit_probe(train,validation,tmp_path,epochs=2,patience=1,device=torch.device('cpu'))
    assert result['head']==[32,512,512,512,6]
    assert result['fit_clips']+result['tune_clips']==30
    assert result['selected_head_epochs'] in (1,2)
    assert result['validation']['mae']>0 and len(result['validation']['per_channel_mae'])==6
    assert (tmp_path/'head.pt').exists()
    assert train['values'].grad is None
    torch.testing.assert_close(train['values'].detach(),original,rtol=0,atol=0)


def test_temporal_baseline_contains_all_frames_and_correct_difference():
    from tactilelatent.representation_probe import temporal_summary
    x=torch.tensor([[[1.,2.],[3.,4.],[8.,9.]]])
    summary=temporal_summary(x)
    torch.testing.assert_close(summary[...,:2],x)
    torch.testing.assert_close(summary[...,2:4],x.mean(1,keepdim=True).expand_as(x))
    torch.testing.assert_close(summary[...,4:],torch.tensor([[[0.,0.],[2.,2.],[5.,5.]]]))


def test_fit_only_normalizer_does_not_use_tune_features():
    from tactilelatent.representation_probe import fit_statistics
    item=cache(('fit','tune'),dimension=2)
    item['values'][0]=torch.tensor([[1.,2.],[3.,4.]])
    item['values'][1]=1000000.
    mean,std=fit_statistics(item,[0])
    torch.testing.assert_close(mean,torch.tensor([2.,3.]))
    torch.testing.assert_close(std,torch.tensor([1.,1.]))


def test_sweep_selects_lr_from_tune_and_reports_head_variation(tmp_path):
    from tactilelatent.representation_probe import fit_probe_sweep
    from tactilelatent.frozen_probe import episode_partition
    torch.set_num_threads(2)
    train=cache(tuple(f'train-{i}' for i in range(30)),dimension=4)
    validation=cache(('val-one','val-two'),dimension=4)
    result=fit_probe_sweep(train,validation,tmp_path,seeds=(45,46),learning_rates=(.0003,.001),epochs=1,patience=1,device=torch.device('cpu'))
    assert len(result['tuning_trials'])==4 and len(result['seed_results'])==2
    expected=min(result['learning_rates'],key=lambda rate:sum(r['selection']['best_tune_loss'] for r in result['tuning_trials'] if r['learning_rate']==rate)/2)
    assert result['selected_learning_rate']==expected
    assert result['validation']['mae_head_seed_std']>=0
    normalizer=torch.load(tmp_path/'normalization.pt',weights_only=False)
    assert all(episode_partition(e,45)=='fit' for e in normalizer['fit_episode_ids'])
    assert not any('validation' in r for r in result['tuning_trials'])
