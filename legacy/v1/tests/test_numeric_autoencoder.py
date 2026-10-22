import numpy as np
import pytest
import torch

from tactilelatent.numeric_autoencoder import NumericAutoencoder,numeric_loss,load_numeric_teacher,MODEL_TYPE


def make_model():
    return NumericAutoencoder(4,2,[12,8],[.1,10.,1.,2.],[False,False,True,False])


def test_numeric_zero_signed_roundtrip_and_missing_masks():
    model=make_model()
    raw=torch.tensor([[0.,-12.,6.,-1.],[.01,40.,0.,359.]])
    torch.testing.assert_close(model.from_normalized(model.normalize(raw)),raw)
    valid=torch.ones_like(raw,dtype=torch.bool);valid[0,0]=False
    nan=raw.clone();nan[0,0]=float('nan')
    torch.testing.assert_close(model.encode(nan,valid),model.encode(raw,valid))
    assert torch.isfinite(model(nan,valid)).all()
    # Signed wrench/direction outputs remain signed; force-only output is nonnegative.
    result=model.from_normalized(torch.tensor([[-1.,-1.,-1.,-1.]]))
    assert (result[0,[0,1,3]]<0).all() and result[0,2]==0


def test_invalid_values_never_supervise_or_change_gradients():
    prediction=torch.tensor([[.1,.2,.3,.4]],requires_grad=True)
    target=torch.tensor([[0.,float('nan'),.5,0.]])
    valid=torch.tensor([[True,False,True,True]])
    loss=numeric_loss(prediction,target,valid,torch.tensor([True]*4));loss.backward()
    assert prediction.grad[0,1]==0 and torch.isfinite(loss)
    with pytest.raises(ValueError,match='observed'):numeric_loss(prediction,target,torch.zeros_like(valid),torch.ones(4,dtype=torch.bool))


def test_small_teacher_learns_and_checkpoint_decode_matches(tmp_path):
    torch.manual_seed(45);model=make_model();opt=torch.optim.AdamW(model.parameters(),lr=.01)
    t=torch.linspace(-1,1,128);x=torch.stack([t*.1,t*10,t.abs(),t*2],dim=1)
    valid=torch.ones_like(x,dtype=torch.bool)
    with torch.no_grad():before=(model(x,valid)-x).abs().mean()
    for _ in range(200):
        y=model.decode_normalized(model.encode(x,valid));loss=numeric_loss(y,model.normalize(x),valid,model.nonnegative)
        opt.zero_grad();loss.backward();opt.step()
    assert (model(x,valid)-x).abs().mean()<before*.2
    path=tmp_path/'best.pt';torch.save(dict(model_type=MODEL_TYPE,model_config=model.config,model=model.state_dict()),path)
    loaded,_=load_numeric_teacher(path)
    torch.testing.assert_close(loaded(x,valid),model(x,valid))
