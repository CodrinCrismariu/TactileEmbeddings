"""Small deterministic EgoTouch autoencoders: pressure -> code -> pressure.

The encoder also sees the validity mask, so missing cells differ from real zeros.
There are no skip connections: every reconstructed cell passes through the code.
The decoder consumes ONLY the code and can later be frozen for RGB training.
"""

import torch
from torch import nn


PRESETS = {
    "small": {"channels": [16, 32], "hidden_dims": [128]},
    "medium": {"channels": [32, 64], "hidden_dims": [512]},
    "large": {"channels": [32, 64, 128], "hidden_dims": [1024, 512]},
}


class ClippedPressureOutput(nn.Module):
    """Single unconstrained output: raw for training, clipped for inference.

    Training against raw values preserves corrective gradients outside [0,1].
    Clamping during inference enforces sensor bounds and permits exact zeros.
    """
    def forward(self, raw):
        return raw if self.training else raw.clamp(0., 1.)


class HurdlePressureOutput(nn.Module):
    """Separate a nonzero decision from positive pressure magnitude.

    Training forward uses a differentiable probability-weighted value. Evaluation
    forward uses a hard decision and can emit EXACT zero. These probabilities
    are learned classifiers, not a claim of calibrated predictive uncertainty.
    """
    def __init__(self, probability_threshold=.5):
        super().__init__()
        self.probability_threshold = probability_threshold

    def components(self, logits):
        nonzero_logits, magnitude_logits = logits.chunk(2, dim=1)
        probability = nonzero_logits.sigmoid()
        magnitude = magnitude_logits.sigmoid()
        return {"nonzero_logits": nonzero_logits, "nonzero_probability": probability,
                "magnitude": magnitude, "soft": probability * magnitude,
                "hard": torch.where(probability >= self.probability_threshold,
                                    magnitude, torch.zeros_like(magnitude))}

    def forward(self, logits):
        parts = self.components(logits)
        return parts["soft"] if self.training else parts["hard"]


class TactileAutoencoder(nn.Module):
    def __init__(self, architecture="conv", latent_dim=32, channels=(32, 64),
                 hidden_dims=(512,), output_mode="sigmoid", zero_probability_threshold=.5):
        super().__init__()
        if architecture not in ("conv", "mlp") or latent_dim < 1:
            raise ValueError("Choose conv or mlp and a positive latent dimension")
        widths = channels if architecture == "conv" else hidden_dims
        if not widths or any(w < 1 for w in widths):
            raise ValueError("Network widths must be positive and nonempty")
        if architecture == "conv" and len(channels) > 5:
            raise ValueError("At most five downsampling layers fit a 21x21 grid")
        if output_mode not in ("sigmoid", "clamp", "hurdle") or not 0 < zero_probability_threshold < 1:
            raise ValueError("Choose sigmoid/clamp/hurdle output and a probability threshold in (0,1)")
        self.output_mode = output_mode
        self.config = {"architecture": architecture, "latent_dim": latent_dim,
                       "channels": list(channels), "hidden_dims": list(hidden_dims),
                       "output_mode": output_mode, "zero_probability_threshold": zero_probability_threshold}
        output_channels = 4 if output_mode == "hurdle" else 2

        # Four input channels: left/right pressure, then left/right valid masks.
        if architecture == "mlp":
            encoder = [nn.Flatten()]
            previous = 4 * 21 * 21
            for width in hidden_dims:
                encoder += [nn.Linear(previous, width), nn.GELU()]
                previous = width
            encoder += [nn.Linear(previous, latent_dim)]
            decoder = []
            previous = latent_dim
            for width in reversed(hidden_dims):
                decoder += [nn.Linear(previous, width), nn.GELU()]
                previous = width
            if output_mode == "sigmoid":
                # Preserve historical state-dict keys and behavior.
                decoder += [nn.Linear(previous, 2 * 21 * 21), nn.Sigmoid(),
                            nn.Unflatten(1, (2, 21, 21))]
            else:
                decoder += [nn.Linear(previous, output_channels * 21 * 21),
                            nn.Unflatten(1, (output_channels, 21, 21)),
                            HurdlePressureOutput(zero_probability_threshold) if output_mode == "hurdle"
                            else ClippedPressureOutput()]
        else:
            encoder, sizes = [], [21]
            previous = 4
            for width in channels:
                encoder += [nn.Conv2d(previous, width, 3, stride=2, padding=1), nn.GELU()]
                sizes.append((sizes[-1] + 1) // 2)
                previous = width
            cells = channels[-1] * sizes[-1] ** 2
            encoder += [nn.Flatten(), nn.Linear(cells, latent_dim)]
            decoder = [nn.Linear(latent_dim, cells), nn.GELU(),
                       nn.Unflatten(1, (channels[-1], sizes[-1], sizes[-1]))]
            # Explicit sizes invert 21 -> 11 -> 6 (etc.) without odd-size cropping.
            for i in reversed(range(len(channels))):
                out_channels = channels[i - 1] if i else output_channels
                decoder += [nn.Upsample(size=(sizes[i], sizes[i]), mode="nearest"),
                            nn.Conv2d(channels[i], out_channels, 3, padding=1),
                            nn.GELU() if i else (nn.Sigmoid() if output_mode == "sigmoid" else
                                                ClippedPressureOutput() if output_mode == "clamp" else
                                                HurdlePressureOutput(zero_probability_threshold))]
        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)

    def encode(self, pressure, valid=None):
        if pressure.ndim != 4 or pressure.shape[1:] != (2, 21, 21):
            raise ValueError("Expected pressure [batch, 2, 21, 21]")
        valid = torch.isfinite(pressure) if valid is None else valid.bool()
        if valid.shape != pressure.shape:
            raise ValueError("Pressure and mask shapes differ")
        safe = torch.where(valid, pressure, 0.)
        return self.encoder(torch.cat((safe, valid.to(safe.dtype)), dim=1))

    def decode_components(self, code):
        if self.output_mode == "clamp":
            raw = self.decoder[:-1](code)
            return {"soft": raw, "hard": raw.clamp(0., 1.)}
        if self.output_mode != "hurdle":
            prediction = self.decoder(code)
            return {"soft": prediction, "hard": prediction}
        # Keep the output module inside decoder so standalone decoder exports
        # retain their evaluation behavior, while exposing logits for supervision.
        logits = self.decoder[:-1](code)
        return self.decoder[-1].components(logits)

    def decode(self, code, hard=None):
        if hard is None:
            return self.decoder(code)
        return self.decode_components(code)["hard" if hard else "soft"]

    def forward(self, pressure, valid=None):
        return self.decode(self.encode(pressure, valid))
