import json

import numpy as np
import torch

from tactilelatent.aligned_frames import file_hash,write_json
from tactilelatent.baihu_autoencoder import BaihuMarkerAutoencoder
from tactilelatent.expansion_models import ExpansionMarkerAutoencoder,load_teacher
from tactilelatent.baihu_ae_data import BaihuAEFrames,evaluate_baihu


def test_normalized_projection_prevents_all_hidden_gelu_units_dying(tmp_path):
    torch.set_num_threads(1);torch.manual_seed(45)
    old=ExpansionMarkerAutoencoder(width=32,queries=2)
    torch.manual_seed(45);new=BaihuMarkerAutoencoder(width=32,queries=2)
    torch.testing.assert_close(old.point_encoder[0].weight,new.point_encoder[0].weight)
    torch.testing.assert_close(old.to_code[1].weight,new.to_code[1].weight)
    torch.testing.assert_close(old.to_code[3].weight,new.to_code[4].weight)
    # Same very negative hidden inputs as the failed production checkpoint.
    x=torch.linspace(-70.,-7.,32).repeat(4,1).requires_grad_(True)
    old_hidden=old.to_code[2](x)
    new_hidden=new.to_code[3](new.to_code[2](x))
    assert (old_hidden==0).all() and torch.isfinite(new_hidden).all()
    assert (new_hidden!=0).any()
    new_hidden.square().mean().backward()
    assert x.grad.norm()>1e-6
    y=torch.randn(3,19,2)*.03;valid=torch.ones(3,19,dtype=torch.bool)
    valid[1,3]=False;y[1,3]=float('nan')
    ref=torch.rand_like(y);surface=torch.arange(19)[None].expand(3,-1)%4
    pred=new(y,valid,ref,surface);loss=new.loss(pred,y,valid,surface);loss.backward()
    assert new.to_code[1].weight.grad.abs().sum()>0
    assert new.point_encoder[0].weight.grad.abs().sum()>0
    path=tmp_path/'v2.pt';torch.save(dict(model_type='expansion_marker_autoencoder_v2',
        model_config=new.config,model=new.state_dict()),path)
    restored,_=load_teacher(path)
    torch.testing.assert_close(restored(y,valid,ref,surface),pred)
    legacy=tmp_path/'v1.pt';torch.save(dict(model_type='expansion_marker_autoencoder_v1',
        model_config=old.config,model=old.state_dict()),legacy)
    restored,_=load_teacher(legacy)
    torch.testing.assert_close(restored(y,valid,ref,surface),old(y,valid,ref,surface))


def test_frame_pool_masks_padding_preserves_native_frames_and_validation(tmp_path,monkeypatch):
    import tactilelatent.baihu_ae_data as module
    rows=[];datasets={}
    for split in ('train','val'):
        for episode,k in enumerate((8,9)):
            path=tmp_path/f'{split}_{episode}.npz';path.write_bytes(f'{split}_{episode}'.encode())
            row=dict(source='baihu',episode_id=f'{split}_{episode}',split=split,frames=9,marker_path=str(path))
            rows.append(row)
            write_json(path.with_name('complete.json'),dict(markers=k))
            # Distinct per-frame values expose accidental temporal reassignment.
            d=dict(displacement_px=np.arange(9,dtype=np.float32)[:,None,None]*np.ones((9,k,2),np.float32),
                spacing_px=np.ones(k,np.float32)*2,valid=np.ones((9,k),bool),
                reference_uv=np.zeros((k,2),np.float32),surface_id=np.arange(k)%4)
            d['valid'][2,0]=False;d['displacement_px'][2,0]=np.nan
            datasets[row['episode_id']]=(d,dict(tracks_sha256=file_hash(path)))
    monkeypatch.setattr(module,'marker_arrays',lambda row:datasets[row['episode_id']])
    train=BaihuAEFrames(rows,'train',10)
    repeat=BaihuAEFrames(rows,'train',10)
    assert train.signature==repeat.signature
    b=train.batch(np.arange(len(train)))
    for i,frame in enumerate(b['frame']):
        assert torch.allclose(b['target'][i,1],torch.ones(2)*int(frame)/2)
    assert not train.data['valid'][train.data['episode']==0,-1].any()
    assert torch.equal(train.data['frame']%2,torch.zeros_like(train.data['frame']))
    val=BaihuAEFrames(rows,'val',6)
    assert [r['indices'] for r in val.records]==[[0,4,8],[0,4,8]]
    # A constant decoder is explicitly caught by the input-dependence metric.
    class ZeroModel(torch.nn.Module):
        displacement_scale=torch.tensor(1.)
        def forward(self,y,valid,reference,surface):return torch.zeros_like(y)
        def encode(self,y,valid,reference,surface):return torch.zeros(len(y),32)
        def loss(self,p,y,valid,surface):return torch.nan_to_num(y).sum()*0
    metrics=evaluate_baihu(ZeroModel(),val)
    assert metrics['zero_input_output_change_px']==0 and metrics['effective_rank']==0
    assert metrics['mae']==metrics['zero_baseline_mae']
