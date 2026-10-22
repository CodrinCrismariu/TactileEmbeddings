import json

import numpy as np
import pytest
import torch

from tactilelatent.bimanual_mirror import load_policy, require_spec, transform_targets
from tactilelatent.mirror_mapping_review import candidates, candidate_spec, marker_payload


@pytest.mark.parametrize('source,count',[('opengraph',9),('ftp_unit_bimanual',4),('baihu',32)])
def test_review_hypotheses_are_explicit_and_never_promote_training(source,count):
    entries=candidates(source)
    assert len(entries)==count and len({x['id'] for x in entries})==count
    policy=load_policy('configs/bimanual_mirror_policy_v1.json')
    with pytest.raises(ValueError,match='blocked'):
        require_spec(policy,source)
    with pytest.raises(ValueError,match='Unknown'):
        candidate_spec(source,'not-a-mapping')


def test_opengraph_preserves_named_fingers_and_mask_locations():
    y=torch.arange(160.).reshape(2,1,80)
    valid=torch.ones_like(y,dtype=torch.bool);valid[0,0,79]=False
    for entry in candidates('opengraph'):
        spec=entry['spec'];b=dict(target=y,target_mask=valid)
        result=transform_targets(b,spec)
        restored=transform_targets(result,spec)
        torch.testing.assert_close(restored['target'],y)
        assert torch.equal(restored['target_mask'],valid)
        for finger in range(5):
            values=result['target'][0,0,finger*16:(finger+1)*16]
            assert set(values.tolist())==set(y[1,0,finger*16:(finger+1)*16].tolist())


def test_requested_horizontal_mirror_reflects_each_finger_without_reversing_finger_order():
    spec=candidate_spec('opengraph','horizontal_all_fingers')
    y=torch.arange(160.).reshape(2,1,80);valid=torch.ones_like(y,dtype=torch.bool)
    valid[1,0,2]=False
    result=transform_targets(dict(target=y,target_mask=valid),spec)
    expected=y.reshape(2,5,4,4).flip(0).flip(-1).reshape_as(y)
    expected_mask=valid.reshape(2,5,4,4).flip(0).flip(-1).reshape_as(valid)
    torch.testing.assert_close(result['target'],expected)
    assert torch.equal(result['target_mask'],expected_mask)
    assert candidates('opengraph')[0]['id']=='horizontal_all_fingers'


@pytest.mark.parametrize('source,surfaces,shape',[('ftp_unit_bimanual',2,[240,320]),('baihu',4,[240,240])])
def test_marker_candidates_roundtrip_native_signed_vectors_and_unequal_counts(source,surfaces,shape):
    surface=torch.tensor([i for i in range(surfaces) for _ in range(i+2)])
    n=len(surface);torch.manual_seed(45)
    b=dict(marker_surface=surface,marker_reference=torch.rand(n,2),
           marker_target=torch.randn(1,n,2)*.2,marker_valid=torch.ones(1,n,dtype=torch.bool),
           marker_spacing=torch.linspace(12.,24.,n))
    b['marker_valid'][0,1]=False;b['marker_target'][0,1]=float('nan')
    for entry in candidates(source):
        transformed=transform_targets(b,entry['spec'])
        restored=transform_targets(transformed,entry['spec'])
        for key in b:
            torch.testing.assert_close(restored[key],b[key],equal_nan=True)
        # A unit-length displacement is changed only by the linear reflection,
        # never by the reference point's translation or the UV pixel scaling.
        torch.testing.assert_close(transformed['marker_target'].norm(dim=-1),
                                   b['marker_target'].norm(dim=-1),equal_nan=True)
        payload=marker_payload(dict(source=source,target_shape=[surfaces*3,*shape]),
            transformed,[str(i) for i in range(surfaces)],0,
            prediction=torch.nan_to_num(transformed['marker_target'][0]).numpy())
        assert len(payload['optical_panels'])==surfaces
        assert payload['marker_displacement']['measured_px'][1] is None
        expected=transformed['marker_reference'].numpy()*np.array(shape[::-1])-transformed['marker_reference'].numpy()
        np.testing.assert_allclose(payload['marker_displacement']['reference_px'],expected,atol=2e-5)
        json.dumps(payload,allow_nan=False)
