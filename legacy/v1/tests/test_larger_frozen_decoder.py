import pytest
import torch
from tactilelatent.frozen_probe import head


def test_larger_head_shapes_and_preserved_default():
    small = head(32, 6)
    large = head(32, 6, hidden_dims=(1024, 1024, 512))
    assert sum(p.numel() for p in small.parameters()) == 19974
    assert sum(p.numel() for p in large.parameters()) == 1611270
    x = torch.randn(4, 32)
    assert small(x).shape == large(x).shape == (4, 6)
    assert list(small.state_dict()) == ['0.weight', '0.bias', '3.weight', '3.bias']
    y = large(x)
    assert ((y >= 0) & (y <= 1)).all()
    y.sum().backward()
    assert all(p.grad is not None for p in large.parameters())


def test_invalid_hidden_width():
    with pytest.raises(ValueError):
        head(32, 6, hidden_dims=(0,))
