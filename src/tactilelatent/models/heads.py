"""Construct generic sensor heads from registry-supplied dimensions.

Inputs are shared-latent dimensions, target dimensions, and training-only code
statistics. Outputs are PyTorch modules used by ``JointModel``; these factories
do not load datasets or choose source-specific physical conventions.
"""

import torch
from torch import nn


def code_adapter(latent_dim, code_dim, hidden_dims):
    """Map each [..., latent_dim] vector to an unconstrained teacher code."""
    layers = []
    previous = latent_dim
    for width in hidden_dims:
        layers.extend((nn.Linear(previous, width), nn.GELU()))
        previous = width
    layers.append(nn.Linear(previous, code_dim))
    return nn.Sequential(*layers)


def code_normalization(statistics, dimension):
    """Store verified [dimension] training mean/std as checkpointed buffers."""
    mean = torch.as_tensor(statistics["mean"], dtype=torch.float32)
    std = torch.as_tensor(statistics["std"], dtype=torch.float32)
    if (
        mean.shape != (dimension,)
        or std.shape != (dimension,)
        or not torch.isfinite(mean).all()
        or not torch.isfinite(std).all()
        or (std < 1e-6).any()
    ):
        raise ValueError("Invalid training-only sensor-code normalization")
    norm = nn.Module()
    norm.register_buffer("mean", mean)
    norm.register_buffer("std", std)
    return norm


def bounded_numeric_head(latent_dim, shape):
    """Produce flattened [0, 1] targets; the caller restores native dimensions."""
    cells = int(torch.tensor(shape).prod())
    return nn.Sequential(nn.Linear(latent_dim, cells), nn.Sigmoid())
