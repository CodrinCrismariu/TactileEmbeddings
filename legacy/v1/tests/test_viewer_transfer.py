import copy
import pytest
import torch

from tactilelatent.frozen_probe import head
from tactilelatent.viewer_transfer import load_transfer_head


def saved():
    return dict(head=head(32,6,hidden_dims=[8,8,8]).state_dict(),config={'hidden_dims':[8,8,8]},
        mean=torch.linspace(-2,2,32),std=torch.linspace(.01,2,32),feature_normalization='fit_zscore')


def test_exported_probe_retains_fit_normalization_inside_autocast():
    s=saved();x=torch.randn(5,32);model=load_transfer_head(s,32).eval()
    raw=head(32,6,hidden_dims=[8,8,8]).eval();raw.load_state_dict(s['head'])
    expected=raw((x-s['mean'])/s['std'])
    with torch.autocast('cpu',torch.bfloat16):actual=model(x)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    assert not torch.allclose(actual,raw(x))


def test_legacy_probe_unchanged_and_bad_stats_rejected():
    s=saved();s.pop('feature_normalization');model=load_transfer_head(s,32)
    assert set(model.state_dict())==set(s['head'])
    for field in ('mean','std'):
        bad=saved();bad[field][0]=float('nan')
        with pytest.raises(ValueError,match='normalization'):load_transfer_head(bad,32)
    bad=saved();bad['std'][0]=0.
    with pytest.raises(ValueError,match='normalization'):load_transfer_head(bad,32)
