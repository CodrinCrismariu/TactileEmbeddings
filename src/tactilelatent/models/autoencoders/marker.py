"""Variable-size marker-set autoencoders and signed displacement decoders.

Each marker supplies displacement, validity, reference coordinates, and surface
identity. Attention uses a null support point for completely missing frames.
Decoding uses only sensor codes and static reference geometry; point ordering
and geometry are retained rather than rasterized or downsampled.
"""

import torch
from torch import nn
from torch.nn import functional as F


class MarkerDisplacementHead(nn.Module):
    """Predict signed (dx,dy) in marker-spacing units at reference coordinates.

    Coordinates and surface identity are static geometry, never the measured
    current displacement. The variable-size point representation avoids
    assigning a universal marker index across episodes/sensors.
    """

    def __init__(self, latent_dim=32, hidden=128, surfaces=2):
        super().__init__()
        self.surface = nn.Embedding(surfaces, 8)
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim + 2 + 8, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
        )
        nn.init.normal_(self.mlp[-1].weight, std=0.001)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, z, reference, surface):
        """Map [B, T, D] codes and [B, N, 2] geometry to [B, T, N, 2]."""
        if (
            z.ndim != 3
            or reference.ndim != 3
            or reference.shape[-1] != 2
            or (surface.shape != reference.shape[:2])
        ):
            raise ValueError("Expected frame latents [B,T,D] and marker geometry [B,N,2]")
        geometry = torch.cat((reference, self.surface(surface)), dim=-1)
        value = torch.cat(
            (
                z[:, :, None, :].expand(-1, -1, reference.shape[1], -1),
                geometry[:, None, :, :].expand(-1, z.shape[1], -1, -1),
            ),
            dim=-1,
        )
        return self.mlp(value)


def marker_loss(prediction, target, valid, surface, scale=1.0):
    """Robust point loss, equal weighting of supported frames/surfaces.

    Lost markers and NaNs do not contribute, including to gradients. Whole
    missing frames/surfaces are excluded rather than treated as stationary.
    """
    if (
        prediction.shape != target.shape
        or valid.shape != target.shape[:-1]
        or surface.shape != (target.shape[0], target.shape[2])
    ):
        raise ValueError("Mismatched marker predictions, masks or surface IDs")
    valid = valid.bool() & torch.isfinite(target).all(-1)
    safe = torch.where(valid[..., None], target, torch.zeros_like(target)).detach().float()
    scale = torch.as_tensor(scale, device=prediction.device, dtype=torch.float32).clamp_min(1e-06)
    error = F.smooth_l1_loss(
        prediction.float() / scale, safe / scale, reduction="none", beta=0.1
    ).mean(-1)
    losses = []
    for sid in surface.unique():
        mask = valid & (surface[:, None, :] == sid)
        count = mask.sum(-1)
        supported = count > 0
        values = torch.where(mask, error, 0.0).sum(-1) / count.clamp_min(1)
        if supported.any():
            losses.append(values[supported].mean())
    return torch.stack(losses).mean() if losses else prediction.sum() * 0.0


class MarkerAutoencoder(nn.Module):
    """Compress variable-size signed marker displacement into fixed codes.

    Inputs are [frames, points, 2] displacement/reference positions and
    [frames, points] validity/surface IDs. True means a tracked observation;
    padded or lost markers never become zero-displacement supervision.
    Displacement uses marker-spacing units and reference coordinates use the
    normalized sensor plane. A null token keeps all-missing attention finite.
    """

    def __init__(
        self,
        latent_dim=32,
        width=128,
        queries=8,
        heads=4,
        displacement_scale=1.0,
        architecture="marker_set",
    ):
        super().__init__()
        if (
            architecture != "marker_set"
            or min(latent_dim, width, queries, heads) < 1
            or width % heads
            or (displacement_scale <= 0)
        ):
            raise ValueError("Invalid marker autoencoder configuration")
        self.config = dict(
            architecture=architecture,
            latent_dim=latent_dim,
            width=width,
            queries=queries,
            heads=heads,
            displacement_scale=float(displacement_scale),
        )
        self.register_buffer("displacement_scale", torch.tensor(float(displacement_scale)))
        self.surface = nn.Embedding(2, 8)
        self.point_encoder = nn.Sequential(
            nn.Linear(13, width), nn.GELU(), nn.Linear(width, width), nn.GELU(), nn.LayerNorm(width)
        )
        self.queries = nn.Parameter(torch.empty(queries, width))
        nn.init.normal_(self.queries, std=0.02)
        self.null = nn.Parameter(torch.zeros(1, 1, width))
        self.attention = nn.MultiheadAttention(width, heads, dropout=0.0, batch_first=True)
        self.to_code = nn.Sequential(
            nn.Flatten(), nn.Linear(queries * width, width), nn.GELU(), nn.Linear(width, latent_dim)
        )
        self.decoder = MarkerDisplacementHead(latent_dim, width)

    def encode(self, target, valid, reference, surface):
        """Read valid marker features into [frames, latent_dim] sensor codes."""
        if (
            target.ndim != 3
            or target.shape[-1] != 2
            or valid.shape != target.shape[:-1]
            or (reference.shape != target.shape)
            or (surface.shape != valid.shape)
        ):
            raise ValueError(
                "Expected [frames,points,2] vectors/geometry and matching masks/surfaces"
            )
        valid = valid.bool() & torch.isfinite(target).all(-1)
        clean = torch.where(valid[..., None], target, 0.0).float() / self.displacement_scale
        features = torch.cat(
            (clean, reference.float(), valid[..., None].float(), self.surface(surface)), dim=-1
        )
        memory = self.point_encoder(features)
        memory = torch.cat((memory, self.null.expand(len(memory), -1, -1)), dim=1)
        padding = torch.cat(
            (~valid, torch.zeros(len(valid), 1, dtype=torch.bool, device=valid.device)), dim=1
        )
        query = self.queries[None].expand(len(memory), -1, -1)
        pooled, _ = self.attention(
            query, memory, memory, key_padding_mask=padding, need_weights=False
        )
        return self.to_code(pooled)

    def decode(self, code, reference, surface):
        """Reconstruct signed vectors using code and static geometry alone."""
        return self.decoder(code[:, None], reference, surface)[:, 0] * self.displacement_scale

    def forward(self, target, valid, reference, surface):
        return self.decode(self.encode(target, valid, reference, surface), reference, surface)

    def loss(self, prediction, target, valid, surface):
        return marker_loss(
            prediction[:, None], target[:, None], valid[:, None], surface, self.displacement_scale
        )


class ExpansionMarkerAutoencoder(MarkerAutoencoder):
    """Four-surface checkpoint format with an explicit surface embedding."""

    def __init__(self, surfaces=4, **kwargs):
        super().__init__(**kwargs)
        if surfaces != 4:
            raise ValueError("Baihu requires exactly four ordered surfaces")
        self.surface = nn.Embedding(surfaces, 8)
        self.decoder = MarkerDisplacementHead(
            self.config["latent_dim"], self.config["width"], surfaces
        )
        self.config["surfaces"] = surfaces


class BaihuMarkerAutoencoder(ExpansionMarkerAutoencoder):
    """Four-surface teacher format with pre-GELU code-layer normalization.

    The extra LayerNorm is part of the recorded model format; it does not
    change marker geometry, displacement units, or surface ownership.
    """

    def __init__(self, code_normalization="pre_gelu_layer_norm", **kwargs):
        if code_normalization != "pre_gelu_layer_norm":
            raise ValueError("Unsupported Baihu sensor-code normalization")
        super().__init__(**kwargs)
        flatten, hidden, activation, output = self.to_code.children()
        self.to_code = nn.Sequential(
            flatten, hidden, nn.LayerNorm(self.config["width"]), activation, output
        )
        self.config["code_normalization"] = code_normalization
