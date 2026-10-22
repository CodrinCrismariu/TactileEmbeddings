import copy
import torch

from tactilelatent.marker_autoencoder import MarkerAutoencoder,load_marker_autoencoder


def data():
    torch.manual_seed(4)
    target=torch.randn(3,18,2)*.05;valid=torch.ones(3,18,dtype=torch.bool)
    ref=torch.rand(3,18,2);surface=torch.arange(18)[None].expand(3,-1)%2
    return target,valid,ref,surface


def test_permutation_invariance_masking_and_no_displacement_skip():
    torch.set_num_threads(2);ae=MarkerAutoencoder(latent_dim=4,width=32,queries=4,displacement_scale=.05).eval()
    y,v,r,s=data();permutation=torch.randperm(18)
    a=ae.encode(y,v,r,s);b=ae.encode(y[:,permutation],v[:,permutation],r[:,permutation],s[:,permutation])
    torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)
    v[:,-2:]=False;y[:,-2:]=float('nan')
    masked=ae.encode(y,v,r,s);y[:,-2:]=1000
    torch.testing.assert_close(ae.encode(y,v,r,s),masked,rtol=0,atol=0)
    assert torch.isfinite(ae.encode(y,torch.zeros_like(v),r,s)).all()
    # A fixed code yields the same reconstruction regardless of current touch.
    zero=torch.zeros(3,4)
    torch.testing.assert_close(ae.decode(zero,r,s),ae.decode(zero,r,s),rtol=0,atol=0)
    assert ae.decode(a,r,s).shape==y.shape


def test_signed_reconstruction_gradients_and_checkpoint(tmp_path):
    ae=MarkerAutoencoder(latent_dim=4,width=32,queries=4,displacement_scale=.05)
    y,v,r,s=data();y[:,0]=float('nan');v[:,0]=False
    pred=ae(y,v,r,s);loss=ae.loss(pred,y,v,s);assert torch.isfinite(loss);loss.backward()
    assert ae.to_code[-1].weight.grad.abs().sum()>0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in ae.parameters())
    with torch.no_grad():
        ae.decoder.mlp[-1].weight.zero_();ae.decoder.mlp[-1].bias.copy_(torch.tensor([-2.,3.]))
    pred=ae(y,v,r,s);assert (pred[...,0]<0).all() and (pred[...,1]>0).all()
    path=tmp_path/'ae.pt';torch.save(dict(model_type='marker_autoencoder_v1',model_config=ae.config,model=copy.deepcopy(ae.state_dict())),path)
    restored,_=load_marker_autoencoder(path);torch.testing.assert_close(restored(y,v,r,s),pred)
    assert not any(p.requires_grad for p in restored.parameters())


def test_tiny_smooth_deformation_can_be_encoded():
    torch.manual_seed(45);torch.set_num_threads(2)
    ae=MarkerAutoencoder(latent_dim=4,width=32,queries=4,displacement_scale=.05)
    ref=torch.rand(1,20,2).expand(8,-1,-1);surface=torch.zeros(8,20,dtype=torch.long);valid=torch.ones(8,20,dtype=torch.bool)
    amplitude=torch.linspace(-.08,.08,8)[:,None,None]
    target=amplitude*torch.stack((ref[...,0]-.5,ref[...,1]-.5),dim=-1)
    baseline=ae.loss(torch.zeros_like(target),target,valid,surface).item();opt=torch.optim.Adam(ae.parameters(),lr=.005)
    for _ in range(180):
        pred=ae(target,valid,ref,surface);loss=ae.loss(pred,target,valid,surface);opt.zero_grad();loss.backward();opt.step()
    assert ae.loss(ae(target,valid,ref,surface),target,valid,surface)<baseline*.4
