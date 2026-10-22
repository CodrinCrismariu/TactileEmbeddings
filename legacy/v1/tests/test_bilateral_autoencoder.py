import torch

from tactilelatent.bilateral_autoencoder import BilateralVectorAutoencoder,BilateralMarkerAutoencoder,load_bilateral_teacher
from tactilelatent.bimanual_mirror import load_policy,transform_targets
from tactilelatent.mirror_mapping_review import candidate_spec


def test_taxel_hand_sharing_preserves_full_code_and_mirror_equivariance_with_missing_channels():
    torch.manual_seed(45)
    model=BilateralVectorAutoencoder(hidden_dims=[32,16]).eval()
    y=torch.rand(3,2,1,80);valid=torch.rand_like(y)>.2;y[~valid]=float('nan')
    spec=candidate_spec('opengraph','horizontal_all_fingers')
    mirrored=transform_targets(dict(target=y,target_mask=valid),spec)
    original=model(y,valid);reflection=model(mirrored['target'],mirrored['target_mask'])
    expected=transform_targets(dict(target=original),spec)['target']
    torch.testing.assert_close(reflection,expected,rtol=1e-5,atol=1e-6)
    assert model.encode(y,valid).shape==(3,16)
    code=model.encode(y,valid).reshape(3,2,8)
    torch.testing.assert_close(model.encode(mirrored['target'],mirrored['target_mask']).reshape(3,2,8),code.flip(1))


def test_marker_hand_sharing_handles_unequal_counts_missing_points_and_padding():
    torch.manual_seed(45)
    model=BilateralMarkerAutoencoder(width=16,heads=2,queries=2).eval()
    y=torch.randn(2,1,9,2)*.02;valid=torch.rand(2,1,9)>.2
    y[~valid]=float('nan');reference=torch.rand(2,9,2)
    surface=torch.tensor([[0]*3+[1]*5+[0],[0]*4+[1]*5]);valid[0,0,-1]=False
    reference[0,-1]=0
    spec=load_policy('configs/bimanual_mirror_policy_v2.json')['sources']['ftp_unit_bimanual']
    original=model(y[:,0],valid[:,0],reference,surface)
    for i in range(2):
        batch=dict(marker_target=y[i],marker_valid=valid[i],marker_reference=reference[i],marker_surface=surface[i])
        mirrored=transform_targets(batch,spec)
        actual=model(mirrored['marker_target'],mirrored['marker_valid'],mirrored['marker_reference'][None],mirrored['marker_surface'][None])[0]
        expected=transform_targets(dict(batch,marker_target=original[i:i+1]),spec)['marker_target'][0]
        torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-6)
    assert model.encode(y[:,0],valid[:,0],reference,surface).shape==(2,16)
    code=torch.randn(2,16,requires_grad=True)
    model.decode(code,reference,surface).square().mean().backward()
    assert code.grad.abs().sum()>0


def test_checkpoint_load_preserves_vector_and_marker_decoder_outputs(tmp_path):
    for model in [BilateralVectorAutoencoder(hidden_dims=[16]),BilateralMarkerAutoencoder(width=16,heads=2,queries=2)]:
        path=tmp_path/(model.model_type+'.pt')
        torch.save(dict(model_type=model.model_type,model_config=model.config,model=model.state_dict()),path)
        restored,_=load_bilateral_teacher(path)
        code=torch.randn(2,16)
        args=() if isinstance(model,BilateralVectorAutoencoder) else (torch.rand(2,7,2),torch.randint(2,(2,7)))
        torch.testing.assert_close(model.decode(code,*args),restored.decode(code,*args),rtol=0,atol=0)
        assert all(not p.requires_grad for p in restored.parameters())
