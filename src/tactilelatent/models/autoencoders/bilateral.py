"""Shared-hand teachers with explicit anatomical reflection conventions.

A full code concatenates the two hand codes. Mirroring measurements before
encoding is required; arbitrary sensor-code coordinate permutations are not
used as augmentation. Native marker counts and per-finger taxels are retained.
"""

import torch
from torch import nn

from .marker import MarkerAutoencoder
from .pressure import SensorAutoencoder


class BilateralVectorAutoencoder(nn.Module):
    """Encode two native 80-taxel hands with a shared canonical-hand teacher.

    Targets and validity have shape [B, 2, 1, 80]. The right hand is reflected
    within each 4x4 finger grid before encoding. The full [B, latent_dim] code
    concatenates left/right half-codes; decoding restores the native layout.
    """

    model_type = "bilateral_vector_autoencoder_v1"

    def __init__(
        self,
        target_shape=(2, 1, 80),
        latent_dim=16,
        hidden_dims=(128, 128),
        architecture="mlp",
        output_mode="sigmoid",
        channels=(32, 64),
        bilateral=True,
    ):
        super().__init__()
        if (
            list(target_shape) != [2, 1, 80]
            or latent_dim % 2
            or (not bilateral)
            or (architecture != "mlp")
        ):
            raise ValueError("Expected two native 80-taxel hands and an even full code dimension")
        self.config = dict(
            target_shape=list(target_shape),
            latent_dim=latent_dim,
            hidden_dims=list(hidden_dims),
            architecture="mlp",
            output_mode=output_mode,
            channels=list(channels),
            bilateral=True,
        )
        self.hand_dim = latent_dim // 2
        self.shared = SensorAutoencoder(
            [1, 1, 80],
            architecture="mlp",
            latent_dim=self.hand_dim,
            hidden_dims=hidden_dims,
            output_mode=output_mode,
        )
        self.register_buffer("reflection", torch.arange(80).reshape(5, 4, 4).flip(-1).flatten())

    @property
    def encoder(self):
        return self.shared.encoder

    @property
    def decoder(self):
        return self.shared.decoder

    def encode(self, target, valid=None):
        if target.ndim != 4 or tuple(target.shape[1:]) != (2, 1, 80):
            raise ValueError("Expected native two-hand taxels")
        valid = torch.isfinite(target) if valid is None else valid.bool() & torch.isfinite(target)
        canonical = torch.cat([target[:, :1], target[:, 1:].index_select(-1, self.reflection)], 0)
        mask = torch.cat([valid[:, :1], valid[:, 1:].index_select(-1, self.reflection)], 0)
        code = self.shared.encode(canonical, mask)
        return torch.stack(code.chunk(2), 1).flatten(1)

    def decode(self, code):
        if code.ndim != 2 or code.shape[1] != 2 * self.hand_dim:
            raise ValueError("Wrong bilateral code shape")
        hands = torch.cat(code.reshape(len(code), 2, self.hand_dim).unbind(1), 0)
        left, right = self.shared.decode(hands).chunk(2)
        return torch.cat([left, right.index_select(-1, self.reflection)], 1)

    def forward(self, target, valid=None):
        return self.decode(self.encode(target, valid))


class BilateralMarkerAutoencoder(nn.Module):
    """Share a marker teacher across two reflected hand conventions.

    Targets/reference geometry are [B, N, 2], validity and surface IDs [B, N].
    Surface 0 owns the left half-code; surface 1 owns the right half-code.
    Reflection changes reference u to 1-u and signed dx to -dx, leaving v/dy
    unchanged. Geometry, validity, and hand ownership remain explicit.
    """

    model_type = "bilateral_marker_autoencoder_v1"

    def __init__(
        self,
        latent_dim=16,
        width=128,
        queries=8,
        heads=4,
        displacement_scale=1.0,
        architecture="marker_set",
        bilateral=True,
    ):
        super().__init__()
        if latent_dim % 2 or not bilateral or architecture != "marker_set":
            raise ValueError("Even bilateral code dimension required")
        self.config = dict(
            latent_dim=latent_dim,
            width=width,
            queries=queries,
            heads=heads,
            displacement_scale=displacement_scale,
            architecture=architecture,
            bilateral=True,
        )
        self.hand_dim = latent_dim // 2
        self.shared = MarkerAutoencoder(
            latent_dim=self.hand_dim,
            width=width,
            queries=queries,
            heads=heads,
            displacement_scale=displacement_scale,
        )

    @property
    def decoder(self):
        return self.shared.decoder

    @property
    def displacement_scale(self):
        return self.shared.displacement_scale

    @staticmethod
    def reflect_reference(reference):
        """Reflect normalized reference positions across the sensor's u axis."""
        return torch.stack([1 - reference[..., 0], reference[..., 1]], -1)

    @staticmethod
    def reflect_vector(value):
        """Reflect signed displacement vectors by reversing dx only."""
        return torch.stack([-value[..., 0], value[..., 1]], -1)

    def encode(self, target, valid, reference, surface):
        if (
            target.shape != reference.shape
            or surface.shape != valid.shape
            or target.shape[:-1] != valid.shape
        ):
            raise ValueError("Marker geometry/mask shapes differ")
        if ((surface < 0) | (surface > 1)).any():
            raise ValueError("Two ordered hand surfaces required")
        canonical = torch.cat([target, self.reflect_vector(target)], 0)
        ref = torch.cat([reference, self.reflect_reference(reference)], 0)
        mask = torch.cat([valid & (surface == 0), valid & (surface == 1)], 0)
        code = self.shared.encode(canonical, mask, ref, torch.zeros_like(mask, dtype=torch.long))
        return torch.stack(code.chunk(2), 1).flatten(1)

    def decode(self, code, reference, surface):
        if code.ndim != 2 or code.shape[1] != 2 * self.hand_dim:
            raise ValueError("Wrong bilateral code shape")
        hands = torch.cat(code.reshape(len(code), 2, self.hand_dim).unbind(1), 0)
        ref = torch.cat([reference, self.reflect_reference(reference)], 0)
        left, right = self.shared.decode(
            hands, ref, torch.zeros(ref.shape[:-1], device=ref.device, dtype=torch.long)
        ).chunk(2)
        return torch.where((surface == 0)[..., None], left, self.reflect_vector(right))

    def forward(self, target, valid, reference, surface):
        return self.decode(self.encode(target, valid, reference, surface), reference, surface)

    def loss(self, prediction, target, valid, surface):
        return self.shared.loss(prediction, target, valid, surface)
