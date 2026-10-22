import torch

from tactilelatent.aligned_frames import file_hash
from tactilelatent.marker_autoencoder import MarkerAutoencoder
from tactilelatent.marker_code_model import MarkerCodeVJEPA,MixedMarkerTeachers,model_from_state,MODEL_TYPE
from tactilelatent.marker_targets import MARKER_SOURCES
from tactilelatent.train_alignment import alignment_loss
from tactilelatent.train_vjepa_markers import replay_update
from tactilelatent.sam import SAMStep,rng_snapshot


def test_frozen_marker_codes_train_visual_adapters_without_decoder_gradients(tmp_path):
    torch.set_num_threads(2);torch.manual_seed(45)
    ae=MarkerAutoencoder(latent_dim=4,width=32,queries=4,displacement_scale=.05)
    path=tmp_path/'teacher.pt';torch.save(dict(model_type='marker_autoencoder_v1',model_config=ae.config,model=ae.state_dict()),path)
    info=dict(checkpoint=str(path),sha256=file_hash(path),representation='marker_displacement')
    teachers=MixedMarkerTeachers(dict(sources={s:info for s in MARKER_SOURCES}),torch.device('cpu'))
    cfg=dict(sensor_code_dims={s:4 for s in MARKER_SOURCES},numeric_shapes={'ftp_rdp':[1,1,1]},
        statistics={s:dict(mean=[0.]*4,std=[1.]*4) for s in MARKER_SOURCES},
        input_dim=12,width=16,heads=4,layers=1,feedforward=32,grid_size=2,dropout=.1)
    model=MarkerCodeVJEPA(**cfg).train();features=torch.randn(2,4,3,4,12)
    batch=dict(source=['ftp_unit']*2,view_mask=torch.ones(2,8,3,dtype=torch.bool),
        marker_target=torch.randn(2,8,9,2)*.05,marker_valid=torch.ones(2,8,9,dtype=torch.bool),
        marker_reference=torch.rand(2,9,2),marker_surface=torch.zeros(2,9,dtype=torch.long))
    batch['marker_valid'][0,0]=False;batch['marker_target'][0,0]=float('nan')
    batch.update(teachers.supervision(batch,model))
    assert batch['code_target'].shape==(2,8,4) and not batch['code_target'].requires_grad
    assert not batch['code_valid'][0,0]
    before=rng_snapshot();out=model(features,batch['view_mask'],'ftp_unit');loss,_,_=alignment_loss(out,batch,0.)
    loss.backward();assert model.visual.frame_queries.grad.abs().sum()>0
    assert all(not p.requires_grad and p.grad is None for m in teachers.models.values() for p in m.parameters())
    opt=torch.optim.AdamW(model.parameters(),lr=.001);sam=SAMStep(opt,.05)
    replay_update(model,opt,sam,[dict(features=features,batch=batch,rng=before)],torch.device('cpu'))
    assert sam.steps==1
    reconstructed=teachers.decode_marker('ftp_unit',model.raw_code('ftp_unit',out['sensor_code']),batch)
    assert reconstructed.shape==(2,8,9,2) and torch.isfinite(reconstructed).all()
    state=dict(model_type=MODEL_TYPE,model_config=cfg,model=model.state_dict())
    restored=model_from_state(state).eval();model.eval()
    torch.testing.assert_close(restored(features,batch['view_mask'],'ftp_unit')['sensor_code'],model(features,batch['view_mask'],'ftp_unit')['sensor_code'])


def test_code_training_does_not_require_old_optical_ae_files(tmp_path,monkeypatch):
    import json
    import tactilelatent.train_vjepa_markers as module
    native=['egotouch','opentouch','opengraph','egotactile','ftp_vitamin'];teacher=tmp_path/'native.pt';teacher.write_bytes(b'native teacher')
    sources={s:dict(checkpoint=str(teacher),sha256=file_hash(teacher),code_dim=2) for s in native}
    sources.update({s:dict(checkpoint='/nonexistent/old_optical_ae.pt',sha256='unused',code_dim=256) for s in MARKER_SOURCES})
    registry=tmp_path/'teachers.json';registry.write_text(json.dumps(dict(schema_version=1,sources=sources)))
    normalization=tmp_path/'norm.json';normalization.write_text(json.dumps(dict(data={'test':True},
        teacher_hashes={s:v['sha256'] for s,v in sources.items()},statistics={s:dict(mean=[0.,0.],std=[1.,1.]) for s in native})))
    audit=tmp_path/'audit.json';audit.write_text(json.dumps(dict(signature={'test':True},passed=True,VLA_complete_pixels=True)))
    monkeypatch.setattr(module,'manifest_signature',lambda _: {'test':True})
    seen=[]
    monkeypatch.setattr(module,'AlignedClipDataset',lambda manifest,split,**kwargs:seen.append(kwargs) or object())
    cfg=dict(latent_dim=32,heldout_source='ftp_motiontrans',stochastic=False,beta_max=0,manifest='test',
        audit=str(audit),teachers=str(registry),normalization=str(normalization),adapter_hidden_dims=[128,128],attention={})
    _,_,pinned,mc,_=module.prepare_code_inputs(cfg)
    assert set(pinned['sources'])==set(native) and set(mc['sensor_code_dims'])==set(native)
    assert len(seen[0]['sources'])==11 and 'ftp_motiontrans' not in seen[0]['sources']
