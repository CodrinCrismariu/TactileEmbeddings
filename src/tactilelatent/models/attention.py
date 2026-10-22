"""Read full video patch tokens with one current-frame attention query.

Edit this file to change how spatial, temporal, and camera information becomes
the shared representation. Dataset identities never enter this encoder.
Inputs are patch tokens and camera masks; outputs are current-frame context
vectors or deterministic shared latents. ``JointModel.encode`` is the public
model entry point for this computation.
"""

import math

import torch
from torch import nn


def clip_view_mask(mask):
    """Reduce [B, 8, cameras] availability to [B, cameras].

    True means available. Camera availability must stay constant throughout
    each clip, and every clip must contain at least one available camera.
    """
    if mask.ndim != 3 or mask.shape[1] != 8:
        raise ValueError("Expected camera availability [batch,8,cameras]")
    if not torch.equal(mask, mask[:, :1].expand_as(mask)):
        raise ValueError("Camera availability must be constant within a clip")
    available = mask[:, 0].bool()
    if not available.any(-1).all():
        raise ValueError("Every clip needs an available camera")
    return available


def positional_encoding(length, width):
    positions = torch.arange(length, dtype=torch.float32)[:, None]
    scale = torch.exp(torch.arange(0, width, 2, dtype=torch.float32) * (-math.log(10000.0) / width))
    encoding = torch.zeros(length, width)
    encoding[:, 0::2] = torch.sin(positions * scale)
    encoding[:, 1::2] = torch.cos(positions * scale)
    return encoding


class SpatialFrameEncoder(nn.Module):
    """[B,4,3,576,768] patch tokens -> [B,1,32] deterministic latents.

    All four temporal groups and all available views remain attention memory.
    One query reads that memory for the current tactile reading. Positional
    encodings distinguish patch locations, time groups, and camera roles.
    Attribute names intentionally preserve existing checkpoint parameter keys.
    """

    def __init__(
        self,
        input_dim=768,
        latent_dim=32,
        width=256,
        heads=4,
        layers=2,
        feedforward=1024,
        dropout=0.1,
        grid_size=24,
        prediction_mode="current_frame",
    ):
        super().__init__()
        if width % 4 or width % heads or layers < 1 or grid_size < 1:
            raise ValueError("Invalid attention dimensions")
        if prediction_mode not in ("all_frames", "current_frame"):
            raise ValueError("Unknown frame prediction mode")
        self.latent_dim = latent_dim
        self.grid_size = grid_size
        self.prediction_mode = prediction_mode

        # Keep initialization order unchanged for the baseline seed.
        self.project = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, width))
        self.camera_roles = nn.Embedding(3, width)
        queries = 1 if prediction_mode == "current_frame" else 8
        self.frame_queries = nn.Parameter(torch.empty(queries, width))
        self.pair_slot = nn.Embedding(2, width)
        nn.init.trunc_normal_(self.frame_queries, std=0.02)

        row = positional_encoding(grid_size, width // 2)[:, None, :].expand(-1, grid_size, -1)
        col = positional_encoding(grid_size, width // 2)[None, :, :].expand(grid_size, -1, -1)
        self.register_buffer(
            "spatial_position", torch.cat((row, col), -1).reshape(grid_size**2, width)
        )
        self.register_buffer("group_position", positional_encoding(4, width))
        self.register_buffer("frame_position", positional_encoding(8, width))

        layer = nn.TransformerDecoderLayer(
            width,
            heads,
            feedforward,
            dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, layers)
        self.bottleneck = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, 128),
            nn.GELU(),
            nn.Linear(128, latent_dim),
        )

    def encode_context(self, features, view_mask, camera_roles=None):
        """Read patch memory with learned frame queries.

        ``features`` is [B, 4, 3, grid_size**2, input_dim]. ``view_mask`` is
        [B, 8, 3] availability, and optional [B, 3] role indices identify the
        camera slots. The result is [B, Q, width], with Q=1 for the current
        frame or Q=8 for the supported all-frame checkpoint format.

        Decoder blocks update query tokens through self-attention, attention
        over visual memory, and an MLP. The visual memory is not itself updated
        by these blocks. The bottleneck consumes the final block's queries.
        """
        available = clip_view_mask(view_mask)
        batch_size = features.shape[0]
        patches = self.grid_size**2
        if features.ndim != 5 or features.shape[1:4] != (4, 3, patches):
            raise ValueError("Expected full [batch,4,3,patches,width] spatial tokens")
        if camera_roles is None:
            camera_roles = torch.arange(3, device=features.device)[None].expand(batch_size, -1)
        if camera_roles.shape != (batch_size, 3):
            raise ValueError("Camera roles must have shape [batch,3]")

        # Mask before projection: NaNs in absent cameras cannot enter attention.
        clean = features.masked_fill(~available[:, None, :, None, None], 0.0)
        memory = self.project(clean)
        memory = (
            memory
            + self.spatial_position[None, None, None]
            + self.group_position[None, :, None, None]
            + self.camera_roles(camera_roles)[:, None, :, None]
        )
        memory = memory.reshape(batch_size, 4 * 3 * patches, -1)
        padding = (~available[:, None, :, None]).expand(batch_size, 4, 3, patches)
        padding = padding.reshape(batch_size, -1)

        slots = (
            torch.tensor([7], device=features.device)
            if self.prediction_mode == "current_frame"
            else torch.arange(8, device=features.device)
        )
        query = self.frame_queries + self.frame_position[slots] + self.pair_slot(slots % 2)
        query = query[None].expand(batch_size, -1, -1)
        return self.decoder(query, memory, memory_key_padding_mask=padding)

    def encode(self, features, view_mask, camera_roles=None):
        """Return [B, Q, latent_dim] means and deterministic log-variance.

        ``logvar`` is identically negative infinity for compatibility with
        historical prediction APIs; the current model learns no variance.
        """
        latent = self.bottleneck(self.encode_context(features, view_mask, camera_roles))
        return latent, torch.full_like(latent, -torch.inf)
