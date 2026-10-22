"""Native-shape sensor autoencoders; code size is independent of visual latent."""
import math

import torch
from torch import nn

from .tactile_autoencoder import TactileAutoencoder


class SensorAutoencoder(nn.Module):
    def __init__(self, target_shape, architecture="conv", latent_dim=64,
                 channels=(32, 64), hidden_dims=(256, 128), output_mode="sigmoid"):
        super().__init__()
        self.shape = tuple(target_shape)
        if len(self.shape) != 3 or min(self.shape) < 1 or latent_dim < 1:
            raise ValueError("Positive [C,H,W] and sensor code dimension are required")
        if architecture not in ("conv", "mlp") or output_mode != "sigmoid":
            raise ValueError("Choose conv/mlp and sigmoid output")
        widths = channels if architecture == "conv" else hidden_dims
        if (architecture == "conv" and not widths) or any(width < 1 for width in widths):
            raise ValueError("Positive widths are required; conv needs at least one stage")
        if architecture == "conv" and min(self.shape[1:]) <= 1:
            raise ValueError("Use MLP for sensor vectors and force/torque channels")
        self.config = {"target_shape": list(self.shape), "architecture": architecture,
                       "latent_dim": latent_dim, "channels": list(channels),
                       "hidden_dims": list(hidden_dims), "output_mode": output_mode}
        cells = math.prod(self.shape)
        if architecture == "mlp":
            encoder, previous = [nn.Flatten()], 2 * cells
            for width in hidden_dims:
                encoder.extend([nn.Linear(previous, width), nn.GELU()]); previous = width
            encoder.append(nn.Linear(previous, latent_dim))
            decoder, previous = [], latent_dim
            for width in reversed(hidden_dims):
                decoder.extend([nn.Linear(previous, width), nn.GELU()]); previous = width
            decoder.extend([nn.Linear(previous, cells), nn.Sigmoid(), nn.Unflatten(1, self.shape)])
        else:
            encoder, previous, sizes = [], 2 * self.shape[0], [self.shape[1:]]
            for width in channels:
                encoder.extend([nn.Conv2d(previous, width, 3, stride=2, padding=1), nn.GELU()])
                sizes.append(tuple((n + 1) // 2 for n in sizes[-1])); previous = width
            hidden_cells = channels[-1] * math.prod(sizes[-1])
            encoder.extend([nn.Flatten(), nn.Linear(hidden_cells, latent_dim)])
            decoder = [nn.Linear(latent_dim, hidden_cells), nn.GELU(),
                       nn.Unflatten(1, (channels[-1], *sizes[-1]))]
            for i in reversed(range(len(channels))):
                output = channels[i - 1] if i else self.shape[0]
                decoder.extend([nn.Upsample(size=sizes[i], mode="nearest"),
                                nn.Conv2d(channels[i], output, 3, padding=1),
                                nn.GELU() if i else nn.Sigmoid()])
        self.encoder, self.decoder = nn.Sequential(*encoder), nn.Sequential(*decoder)

    def encode(self, target, valid=None):
        if target.ndim != 4 or tuple(target.shape[1:]) != self.shape:
            raise ValueError(f"Expected [batch,{self.shape}]")
        valid = torch.isfinite(target) if valid is None else valid.bool()
        if valid.shape != target.shape:
            raise ValueError("Target and validity mask shapes differ")
        return self.encoder(torch.cat((torch.where(valid, target, 0.), valid.to(target.dtype)), dim=1))

    def decode(self, code):
        return self.decoder(code)

    def forward(self, target, valid=None):
        return self.decode(self.encode(target, valid))


def make_autoencoder(config):
    """Load both native-shape models and all historical EgoTouch checkpoints."""
    if 'input_scale' in config:
        from .pressure_teacher import PressureAutoencoder
        return PressureAutoencoder(**config)
    return SensorAutoencoder(**config) if "target_shape" in config else TactileAutoencoder(**config)
