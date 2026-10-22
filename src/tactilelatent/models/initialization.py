"""Explicit random-number compatibility for the extracted baseline model.

The original constructor created an unused pair projection before replacing its
visual encoder. It also initialized bounded heads before replacing two with
linear heads. Reproducing these draws preserves the real seed-45 epoch-0 model,
parameter ordering, and RNG state without importing historical implementations.
"""

from torch import nn


def consume_discarded_pair_projection(input_dim, latent_dim):
    """Consume precisely the old deterministic pair encoder's random draws."""
    nn.Linear(input_dim, 2 * latent_dim)
