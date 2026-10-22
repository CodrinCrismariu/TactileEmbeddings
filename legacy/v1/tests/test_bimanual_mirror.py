import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tactilelatent.bimanual_mirror import (load_policy,require_spec,validate_spec,
    transform_targets,mirror_rgb,apply_batch,sample_flags,verify_evidence)
from tactilelatent.mirror_preparation import quality,fit_pair_statistics,sampled_batches
from tactilelatent.mirror_joint import PairCodeCache,MirroredCodeClips,augment


def grid():return dict(status='verified',kind='grid',hand_permutation=[1,0],flip_width=True)
def scalar():return dict(status='verified',kind='scalar',hand_permutation=[1,0])
def marker(surfaces=2):
    return dict(status='verified',kind='markers',surface_permutation=list(reversed(range(surfaces))),
        surface_transforms=[[[-1,0,1],[0,1,0]] for _ in range(surfaces)])
def policy(spec,source='egotouch'):return dict(sources={source:spec})


def sample(source='egotouch',batch=2):
    rgb=torch.arange(batch*8*3*3*3*4).float().reshape(batch,8,3,3,3,4)
    y=torch.rand(batch,1,2,3,4)
    return dict(source=[source]*batch,episode_id=['a','b'][:batch],
        rgb=rgb,view_mask=torch.ones(batch,8,3,dtype=torch.bool),
        target=y,target_mask=torch.ones_like(y,dtype=torch.bool),
        frame_indices=torch.full((batch,1),14),history_frame_indices=torch.arange(0,15,2)[None].expand(batch,-1),
        sensor_frame_indices=torch.full((batch,1),14),
        code_target=torch.zeros(batch,1,5),code_valid=torch.ones(batch,1,dtype=torch.bool),
        mirror_code_target=torch.ones(batch,1,5)*7,mirror_code_valid=torch.ones(batch,1,dtype=torch.bool))


def test_grid_mirror_preserves_anatomical_support_and_missing_zeros():
    y=torch.arange(2*21*21).reshape(1,2,21,21).float();y[0,0,3,5]=float('nan');y[0,1,2,8]=0
    item=dict(target=y,target_mask=torch.isfinite(y),frame_indices=torch.tensor([12]))
    m=transform_targets(item,grid());assert m['target'][0,0,2,12]==0
    assert torch.isnan(m['target'][0,1,3,15]) and not m['target_mask'][0,1,3,15]
    twice=transform_targets(m,grid())
    torch.testing.assert_close(twice['target'],y,equal_nan=True,rtol=0,atol=0)
    assert torch.equal(twice['target_mask'],item['target_mask'])
    assert torch.equal(m['frame_indices'],item['frame_indices'])
    with pytest.raises(ValueError,match='never mirror latent'):
        transform_targets(dict(item,code_target=torch.ones(1,8)),grid())


def test_real_source_policy_proves_egotouch_masks_and_blocks_unknown_axes():
    p=load_policy('configs/bimanual_mirror_policy_v1.json');report=verify_evidence(p)
    assert report['egotouch']['verified'] and report['ftp_rdp_bimanual']['verified']
    for s in ('opengraph','ftp_unit_bimanual','baihu'):
        with pytest.raises(ValueError,match='blocked'):require_spec(p,s)
    for s in ('egopressure','ftp_motiontrans','freetacman','ftp_vitamin'):
        with pytest.raises(ValueError,match='bilateral'):require_spec(p,s)


def test_rgb_camera_roles_and_missing_masks_swap_through_whole_history():
    b=sample();b['view_mask'][0,:,1]=False;b['view_mask'][1,:,0]=False
    x,m=mirror_rgb(b['rgb'],b['view_mask'])
    torch.testing.assert_close(x[:,:,1],b['rgb'][:,:,2].flip(-1),rtol=0,atol=0)
    assert not m[0,:,2].any() and not m[1,:,0].any()
    xx,mm=mirror_rgb(x,m)
    torch.testing.assert_close(xx,b['rgb'],rtol=0,atol=0);assert torch.equal(mm,b['view_mask'])


@pytest.mark.parametrize('surfaces',[2,4])
def test_markers_reflect_vectors_reference_and_surface_identity_together(surfaces):
    n=surfaces*3;reference=torch.linspace(.1,.9,n*2).reshape(n,2)
    item=dict(marker_target=torch.arange(n*4).float().reshape(2,n,2)/10,
        marker_valid=torch.ones(2,n,dtype=torch.bool),marker_reference=reference,
        marker_surface=torch.arange(n)%surfaces,marker_spacing=torch.arange(n).float()+1)
    item['marker_valid'][0,0]=False;item['marker_target'][0,0]=float('nan')
    out=transform_targets(item,marker(surfaces))
    torch.testing.assert_close(out['marker_target'][...,0],-item['marker_target'][...,0],equal_nan=True)
    torch.testing.assert_close(out['marker_reference'][:,0],1-reference[:,0])
    assert torch.equal(out['marker_surface'],surfaces-1-item['marker_surface'])
    twice=transform_targets(out,marker(surfaces))
    for k in item:torch.testing.assert_close(twice[k],item[k],equal_nan=True)


def test_marker_batch_padding_never_becomes_reflected_geometry():
    b=sample('ftp_unit_bimanual');b.pop('target');b.pop('target_mask')
    b.update(marker_target=torch.randn(2,1,5,2),marker_valid=torch.ones(2,1,5,dtype=torch.bool),
        marker_reference=torch.rand(2,5,2),marker_surface=torch.zeros(2,5,dtype=torch.long),
        marker_spacing=torch.ones(2,5),mirror_marker_count=torch.tensor([3,5]))
    b['marker_reference'][0,3:]=0;b['marker_target'][0,:,3:]=0;b['marker_valid'][0,:,3:]=False
    m=apply_batch(b,policy(marker(),'ftp_unit_bimanual'),['ftp_unit_bimanual'],flags=[True,True])
    assert torch.equal(m['marker_reference'][0,3:],torch.zeros(2,2))
    assert not m['marker_valid'][0,:,3:].any()
    assert torch.equal(m['marker_surface'][0,:3],torch.ones(3,dtype=torch.long))
    assert torch.equal(b['marker_surface'],torch.zeros(2,5,dtype=torch.long))


def test_per_clip_choice_uses_matching_encoded_codes_and_leaves_ids_unchanged():
    b=sample();before=copy.deepcopy(b)
    m=apply_batch(b,policy(grid()),['egotouch'],flags=[True,False])
    assert torch.equal(m['code_target'][0],before['mirror_code_target'][0])
    assert torch.equal(m['code_target'][1],before['code_target'][1])
    assert torch.equal(m['target'][0],before['target'][0].flip(1,3))
    for k in ('frame_indices','sensor_frame_indices','history_frame_indices'):assert torch.equal(m[k],b[k])
    assert 'mirror_code_target' not in m and 'mirror_code_target' in b
    assert torch.equal(b['rgb'],before['rgb'])
    del b['mirror_code_target']
    with pytest.raises(ValueError,match='freshly encoded'):apply_batch(b,policy(grid()),['egotouch'],flags=[True,False])


def test_scalar_force_swap_preserves_signed_encoding_and_measurement_zeros():
    x=torch.tensor([[[[.2]],[[.8]]],[[[.5]],[[.5]]]])
    y=transform_targets(dict(target=x,target_mask=torch.ones_like(x,dtype=torch.bool)),scalar())['target']
    assert torch.equal(y[:,0],x[:,1]) and torch.equal(y[:,1],x[:,0])
    assert torch.equal(y[1],x[1])


def test_taxel_map_keeps_finger_identity_and_round_trips():
    p=np.arange(80).reshape(5,4,4)[:,:,::-1].reshape(-1).tolist()
    spec=dict(kind='taxels',hand_permutation=[1,0],channel_permutation=[p,p])
    item=dict(target=torch.arange(160).float().reshape(1,2,1,80))
    m=transform_targets(item,spec)
    assert m['target'][0,0,0,0]==83 and m['target'][0,0,0,16]==99
    assert torch.equal(transform_targets(m,spec)['target'],item['target'])


def test_invalid_geometry_and_non_involutive_channel_maps_fail():
    with pytest.raises(ValueError):validate_spec(dict(kind='scalar',hand_permutation=[1,2,0]))
    s=marker();s['surface_transforms'][1][0][2]=.2
    with pytest.raises(ValueError,match='self-inverting'):validate_spec(s)
    with pytest.raises(ValueError,match='explicit'):validate_spec(dict(kind='taxels',hand_permutation=[1,0],channel_permutation=[list(range(80))]))


def test_augmentation_draws_reproduce_without_global_rng_and_disabled_is_exact_noop():
    b=sample();f=sample_flags(b,.5,45,3,12)
    torch.manual_seed(998);torch.rand(500)
    assert torch.equal(sample_flags(b,.5,45,3,12),f)
    assert sample_flags(b,0,45,3,12).sum()==0 and sample_flags(b,1,45,3,12).sum()==2
    assert augment(b,{},3,12) is b
    assert apply_batch(b,policy(grid()),[],flags=[True,True]) is b


def test_quality_gate_rejects_reconstruction_regressions_and_zero_collapse():
    def metrics(mae,zero=1.):return dict(mae=mae,all_mae=mae,all_zero_baseline_mae=zero,contact_count=4,contact_to_zero_ratio=.2)
    baseline=dict(original=metrics(.1),mirrored=metrics(.3))
    assert quality(dict(original=metrics(.105),mirrored=metrics(.2)),baseline)['accepted']
    assert not quality(dict(original=metrics(.12),mirrored=metrics(.1)),baseline)['accepted']
    assert not quality(dict(original=metrics(.1),mirrored=metrics(.29)),baseline)['accepted']
    assert not quality(dict(original=metrics(0.,0.),mirrored=metrics(0.,0.)),baseline)['accepted']


def test_balanced_statistics_use_both_variants_and_train_only_contract():
    x=np.array([[1.,3.],[3.,7.]])
    s=fit_pair_statistics(np.stack([x.sum(0),(x*x).sum(0)]),2)
    np.testing.assert_allclose(s['mean'],[2.,5.]);np.testing.assert_allclose(s['std'],[1.,2.])
    assert s['variant_weights']==[.5,.5] and s['fit_split']=='train'
    with pytest.raises(ValueError):fit_pair_statistics(np.zeros((2,2)),0)


def test_teacher_sampling_is_episode_balanced_validation_fixed_and_no_cross_split():
    rows=[dict(episode_id=str(i),frames=30,split='train' if i<4 else 'val') for i in range(6)]
    train=list(sampled_batches(rows,'train',12,1));assert len({r['episode_id'] for r,_ in train})==4
    val=list(sampled_batches(rows,'val',12,1));other=list(sampled_batches(rows,'val',12,9))
    assert [r['episode_id'] for r,_ in val]==[r['episode_id'] for r,_ in other]
    for (r,a),(_,b) in zip(val,other):assert r['split']=='val';np.testing.assert_array_equal(a,b)


def test_real_encoder_pair_cache_preserves_chronology_train_statistics_and_resume(tmp_path,monkeypatch):
    from tactilelatent.aligned_frames import file_hash,json_hash
    from tactilelatent.sensor_autoencoder import SensorAutoencoder
    from tactilelatent.mirror_preparation import cache_codes,NativeTargets
    import tactilelatent.data_expansion as expansion
    monkeypatch.setattr(expansion,'reserve',lambda additional=0:None)
    torch.manual_seed(45)
    model=SensorAutoencoder([2,21,21],latent_dim=5,channels=[2])
    checkpoint=tmp_path/'teacher.pt'
    torch.save(dict(model_config=model.config,model=model.state_dict()),checkpoint)
    teacher=dict(checkpoint=str(checkpoint),sha256=file_hash(checkpoint),code_dim=5)
    rows=[]
    for split in ('train','val'):
        p=tmp_path/f'{split}.npz';y=np.zeros((3,21,21),np.float32)
        y[:,3,5]=np.arange(1,4)/5.;z=y.copy()/2;z[:,7,8]=.7 if split=='val' else .1
        y[:,0,0]=np.nan
        np.savez(p,left_pressure_grid=y,right_pressure_grid=z)
        rows.append(dict(source='egotouch',episode_id=split,split=split,frames=3,target_path=str(p),
            target_kind='pressure_grid',target_shape=[2,21,21],target_scale=1.))
    pol=policy(grid());root=tmp_path/'prepared'
    index=cache_codes('egotouch',rows,teacher,pol,root,torch.device('cpu'),chunk_frames=2)
    assert index['complete'] and (root/'codes/egotouch/verification.json').exists()
    with np.load(index['episodes'][0]['path']) as f:
        train=f['codes'].copy();np.testing.assert_array_equal(f['frame_indices'],[0,1,2])
        assert train.shape==(2,3,5) and not np.allclose(train[0],train[1])
    np.testing.assert_allclose(index['statistics']['mean'],train.reshape(-1,5).mean(0),atol=1e-7)
    same=cache_codes('egotouch',rows,teacher,pol,root,torch.device('cpu'),chunk_frames=2)
    assert same==index
    cache=PairCodeCache(root/'codes/egotouch/index.json',teacher['sha256'],json_hash(pol))
    codes,valid=cache.read(rows[0],[2,0]);assert codes.shape==(2,2,5) and valid.all()
    with pytest.raises(ValueError,match='contract'):cache.read(dict(rows[0],frames=4),[0])
    with pytest.raises(ValueError,match='outside'):cache.read(rows[0],[3])
    with open(index['episodes'][0]['path'],'ab') as f:f.write(b'corrupt')
    with pytest.raises(ValueError,match='Corrupt'):cache_codes('egotouch',rows,teacher,pol,root,torch.device('cpu'))


def test_sampler_windows_and_counts_remain_unchanged_when_attaching_pair_codes():
    from types import SimpleNamespace
    class Dataset:
        episodes=[dict(source='egotouch',episode_id='a')];windows=[(0,0)];indices_by_source={'egotouch':[0]}
        def __len__(self):return 1
        def __getitem__(self,i):return dict(source='egotouch',frame_indices=torch.tensor([14]))
    class Codes:
        def read(self,row,indices):return torch.arange(10).reshape(2,1,5).float(),torch.ones(2,1,dtype=torch.bool)
    base=Dataset();wrapped=MirroredCodeClips(base,{'egotouch':Codes()},['egotouch'])
    assert wrapped.windows is base.windows and wrapped.indices_by_source is base.indices_by_source
    item=wrapped[0];assert torch.equal(item['code_target'],torch.arange(5).reshape(1,5).float())
    assert torch.equal(item['mirror_code_target'],torch.arange(5,10).reshape(1,5).float())
    assert len(wrapped)==len(base)


def test_sam_replay_keeps_selected_mirrored_codes_and_checkpoint_predictions():
    from tactilelatent.marker_code_model import MarkerCodeVJEPA,model_from_state,MODEL_TYPE
    from tactilelatent.sam import SAMStep,rng_snapshot
    from tactilelatent.train_alignment import alignment_loss
    from tactilelatent.train_vjepa_markers import replay_update
    torch.manual_seed(45)
    config=dict(sensor_code_dims={'egotouch':5,'ftp_unit':4},numeric_shapes={},
        statistics={s:dict(mean=[0.]*n,std=[1.]*n) for s,n in [('egotouch',5),('ftp_unit',4)]},
        input_dim=12,latent_dim=32,width=16,heads=4,layers=2,feedforward=32,grid_size=2,
        dropout=.1,prediction_mode='current_frame')
    model=MarkerCodeVJEPA(**config).train();batch=sample()
    batch=apply_batch(batch,policy(grid()),['egotouch'],flags=[True,False])
    features=torch.randn(2,4,3,4,12);before=rng_snapshot();out=model(features,batch['view_mask'],'egotouch')
    loss,_,_=alignment_loss(out,batch,0.);loss.backward()
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4);sam=SAMStep(optimizer,.05)
    selected=batch['code_target'].clone();rgb=batch['rgb'].clone()
    replay_update(model,optimizer,sam,[dict(features=features,batch=batch,rng=before)],torch.device('cpu'))
    assert sam.steps==1 and torch.equal(selected,batch['code_target']) and torch.equal(rgb,batch['rgb'])
    model.eval();restored=model_from_state(dict(model_type=MODEL_TYPE,model_config=config,model=model.state_dict())).eval()
    torch.testing.assert_close(model(features,batch['view_mask'],'egotouch')['mu'],
        restored(features,batch['view_mask'],'egotouch')['mu'],rtol=0,atol=0)


def test_paired_validation_measures_prediction_equivariance_without_touching_primary_data(monkeypatch):
    from tactilelatent.mirror_joint import paired_validate
    import tactilelatent.vjepa_spatial as spatial
    b=sample('ftp_rdp_bimanual')
    b.pop('code_target');b.pop('code_valid');b.pop('mirror_code_target');b.pop('mirror_code_valid')
    b['rgb']=b['rgb']/b['rgb'].max()
    b['target']=(b['rgb'].mean((1,3,4,5))[:,1:3]*.1+.2).reshape(2,1,2,1,1)
    b['target_mask']=torch.ones_like(b['target'],dtype=torch.bool)
    original=copy.deepcopy(b)
    monkeypatch.setattr(spatial,'spatial_video_features',lambda backbone,batch,device:batch['rgb'].mean((1,3,4,5)))
    class Model:
        def eval(self):return self
        def __call__(self,features,mask,source,**kwargs):
            return dict(pred=(features[:,1:3]*.1+.2).reshape(2,1,2,1,1))
    class Teachers:
        def supervision(self,batch,model):return dict(target=batch['target'],target_mask=batch['target_mask'])
    cfg=dict(seed=45,mirror_augmentation=dict(enabled=True,probability=.5,sources=['ftp_rdp_bimanual'],
        policy='configs/bimanual_mirror_policy_v1.json'))
    result=paired_validate(Model(),None,Teachers(),[b],torch.device('cpu'),cfg)['ftp_rdp_bimanual']
    assert result['original_mae']==0 and result['mirrored_mae']==0 and result['equivariance_mae']==0
    assert torch.equal(b['rgb'],original['rgb']) and torch.equal(b['target'],original['target'])


def test_readiness_gates_reject_unknown_geometry_missing_sources_and_heldout_data(tmp_path):
    from tactilelatent.aligned_frames import json_hash
    from tactilelatent.mirror_joint import read_preparation
    policy_path='configs/bimanual_mirror_policy_v1.json';pol=load_policy(policy_path)
    ready=dict(policy_sha256=json_hash(pol),sources={'ftp_rdp_bimanual':dict(ready=True,numeric=True)})
    (tmp_path/'ready.json').write_text(json.dumps(ready))
    (tmp_path/'policy_audit.json').write_text(json.dumps(dict(policy_sha256=json_hash(pol),sources=verify_evidence(pol))))
    cfg=dict(seed=45,heldout_source='ftp_motiontrans',prediction_mode='current_frame',mirror_augmentation=dict(
        enabled=True,probability=.5,sources=['ftp_rdp_bimanual'],policy=policy_path,preparation=str(tmp_path)))
    assert read_preparation(cfg,{'ftp_rdp_bimanual'}) is not None
    with pytest.raises(ValueError,match='absent'):read_preparation(cfg,{'egotouch'})
    wrong=copy.deepcopy(cfg);wrong['mirror_augmentation']['sources']=['opengraph']
    with pytest.raises(ValueError,match='within-finger'):read_preparation(wrong)
    wrong['mirror_augmentation']['sources']=['ftp_motiontrans']
    with pytest.raises(ValueError,match='bilateral'):read_preparation(wrong)
    wrong=copy.deepcopy(cfg);wrong['heldout_source']='egotouch'
    with pytest.raises(ValueError,match='held-out'):read_preparation(wrong)
    ready['policy_sha256']='changed';(tmp_path/'ready.json').write_text(json.dumps(ready))
    with pytest.raises(ValueError,match='policy changed'):read_preparation(cfg)


def test_unprepared_full_configuration_never_silently_drops_requested_sources():
    from tactilelatent.mirror_joint import read_preparation
    cfg=json.loads(Path('configs/joint_vjepa21_spatial_current_32d_mirror.json').read_text())
    with pytest.raises(ValueError,match='opengraph') as error:read_preparation(cfg)
    assert 'ftp_unit_bimanual' in str(error.value)
    expanded=json.loads(Path('configs/joint_vjepa21_spatial_expanded_32d_mirror.json').read_text())
    assert expanded['excluded_sources']==['freetacman'] and expanded['heldout_source']=='ftp_motiontrans'
    assert expanded['latent_dim']==32 and expanded['prediction_mode']=='current_frame'
    assert not any('egopressure'==s for s in expanded['mirror_augmentation']['sources'])


def test_verification_replays_encoder_chunks_including_short_remainder():
    from tactilelatent.mirror_cache_verification import verification_chunks
    chunks=verification_chunks(260)
    assert [len(c) for c in chunks]==[128,128,4]
    assert [int(c[0]) for c in chunks]==[0,128,256]
    assert [len(c) for c in verification_chunks(97)]==[97]
    with pytest.raises(ValueError):verification_chunks(0)


def test_batch_sensitive_encoder_is_verified_without_relaxing_tolerance(tmp_path,monkeypatch):
    import tactilelatent.mirror_preparation as preparation
    from tactilelatent.mirror_cache_verification import verify_codes
    from tactilelatent.aligned_frames import file_hash
    class Model:
        def encode(self,target,valid):
            # Stand-in for a batch-shape-dependent GPU convolution kernel.
            return target[:,0,0,:2]+len(target)*.0001
    class Reader:
        def read(self,row,indices):
            y=torch.zeros(len(indices),2,1,2);y[:,0,0,0]=torch.from_numpy(indices).float()/300
            return dict(target=y,target_mask=torch.ones_like(y,dtype=torch.bool))
    monkeypatch.setattr(preparation,'NativeTargets',Reader)
    monkeypatch.setattr(preparation,'load_model',lambda info,device:(Model(),{}))
    rows=[dict(source='egotouch',episode_id=s,split=s,frames=260) for s in ('train','val')]
    records=[]
    for row in rows:
        codes=np.zeros((2,260,2),np.float32)
        for start in range(0,260,128):
            ix=np.arange(start,min(start+128,260));original=Reader().read(row,ix)
            for variant,value in enumerate([original,transform_targets(original,dict(kind='grid',hand_permutation=[1,0],flip_width=True))]):
                codes[variant,ix]=Model().encode(value['target'],value['target_mask']).numpy()
        path=tmp_path/(row['split']+'.npz')
        np.savez(path,codes=codes,valid=np.ones((2,260),bool),frame_indices=np.arange(260))
        records.append(dict(episode_id=row['episode_id'],path=str(path),sha256=file_hash(path)))
    index=dict(signature=dict(source='egotouch',implementation={'mirror_preparation.py':file_hash(preparation.__file__)}),
               teacher={},episodes=records)
    (tmp_path/'index.json').write_text(json.dumps(index))
    p=policy(dict(status='verified',kind='grid',hand_permutation=[1,0],flip_width=True))
    result=verify_codes(rows,index,p,torch.device('cpu'))
    assert result['passed'] and result['max_absolute_error']==0
    assert result['tolerance']==dict(rtol=2e-5,atol=2e-5)
    assert len(result['episodes'])==12
    with np.load(records[0]['path']) as f:
        isolated=Reader().read(rows[0],np.array([0,130,259]))
        assert not np.allclose(f['codes'][0,[0,130,259]],Model().encode(isolated['target'],isolated['target_mask']).numpy(),rtol=2e-5,atol=2e-5)
