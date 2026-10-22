"""Six-source extension of the SAME image-only Gaussian-latent VAE method.

All sources share DINO, camera fusion, temporal/spatial attention, and the
32D Gaussian bottleneck. Each sensor layout has its own small decoder. This
contains no tactile-informed encoder, hand-pose input, or CVAE branch.
"""

import torch
from torch import nn

from .model import EgoTouchLatentModel, load_vae_weights
from .prepare_joint import TARGETS


class JointTactileVAE(nn.Module):
    def __init__(self, input_dim=768, width=768, heads=8, temporal_layers=3,
                 cross_view_layers=2, latent_dim=32, dropout=.1, stochastic=True,
                 targets=None):
        super().__init__()
        # The original six-source experiment keeps its default registry.
        # Larger experiments can pass only their training sources, avoiding
        # an untrained decoder for a dataset held out for evaluation.
        self.targets = dict(TARGETS if targets is None else targets)
        # Keep the already-trained EgoTouch encoder and pressure decoder.
        self.visual = EgoTouchLatentModel(
            input_dim=input_dim, width=width, heads=heads,
            temporal_layers=temporal_layers, cross_view_layers=cross_view_layers,
            latent_dim=latent_dim, dropout=dropout, stochastic=stochastic,
        )
        self.stochastic = stochastic

        # The other sensors do not share a label geometry: a taxel vector,
        # pressure map, and RGB optical-tactile image need distinct decoders.
        self.decoders = nn.ModuleDict()
        for source, (_, shape, _) in self.targets.items():
            if source == "egotouch":
                continue
            cells = shape[0] * shape[1] * shape[2]
            self.decoders[source] = nn.Sequential(
                nn.Linear(latent_dim, 512), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(512, cells), nn.Sigmoid(),
            )

    def initialize_from_egotouch(self, weights):
        """Warm-start shared image reasoning and EgoTouch decoder weights."""
        return load_vae_weights(self.visual, weights)

    def forward(self, features, view_mask, source, sample_latent=None):
        if source not in self.targets:
            raise ValueError(f"Unknown joint source: {source}")
        mu, logvar = self.visual.encode(features, view_mask)
        if sample_latent is None:
            sample_latent = self.stochastic and self.training
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp() if sample_latent else mu
        if source == "egotouch":
            pred = self.visual.decode_latent(z)
        else:
            shape = self.targets[source][1]
            pred = self.decoders[source](z).reshape(*z.shape[:-1], *shape)
        return {"pred": pred, "mu": mu, "logvar": logvar}
