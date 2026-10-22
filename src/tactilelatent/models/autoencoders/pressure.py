"""Native dense sensor autoencoders with explicit validity-mask inputs.

SensorAutoencoder handles arbitrary [C,H,W] targets; TactileAutoencoder retains
EgoTouch's two 21x21 hand grids. PressureAutoencoder adds fixed input conditioning
for sparse UV pressure. Output shapes and checkpoint module names stay exact.
The decoder always consumes the code alone, without a target skip connection.
"""

import math

import torch
from torch import nn


class ClippedPressureOutput(nn.Module):
    """Single unconstrained output: raw for training, clipped for inference.

    Training against raw values preserves corrective gradients outside [0,1].
    Clamping during inference enforces sensor bounds and permits exact zeros.
    """

    def forward(self, raw):
        return raw if self.training else raw.clamp(0.0, 1.0)


class HurdlePressureOutput(nn.Module):
    """Separate a nonzero decision from positive pressure magnitude.

    Training forward uses a differentiable probability-weighted value. Evaluation
    forward uses a hard decision and can emit EXACT zero. These probabilities
    are learned classifiers, not a claim of calibrated predictive uncertainty.
    """

    def __init__(self, probability_threshold=0.5):
        super().__init__()
        self.probability_threshold = probability_threshold

    def components(self, logits):
        nonzero_logits, magnitude_logits = logits.chunk(2, dim=1)
        probability = nonzero_logits.sigmoid()
        magnitude = magnitude_logits.sigmoid()
        return {
            "nonzero_logits": nonzero_logits,
            "nonzero_probability": probability,
            "magnitude": magnitude,
            "soft": probability * magnitude,
            "hard": torch.where(
                probability >= self.probability_threshold, magnitude, torch.zeros_like(magnitude)
            ),
        }

    def forward(self, logits):
        parts = self.components(logits)
        return parts["soft"] if self.training else parts["hard"]


class TactileAutoencoder(nn.Module):
    """Reconstruct native [B, 2, 21, 21] pressure from a compact sensor code.

    Encoding concatenates cleaned pressure and its validity mask; valid zeros
    remain observed measurements. Decoding receives only [B, latent_dim] codes.
    Output-mode alternatives are retained for historical checkpoint loading;
    selected teacher metadata determines the actual reconstruction behavior.
    """

    def __init__(
        self,
        architecture="conv",
        latent_dim=32,
        channels=(32, 64),
        hidden_dims=(512,),
        output_mode="sigmoid",
        zero_probability_threshold=0.5,
    ):
        super().__init__()
        if architecture not in ("conv", "mlp") or latent_dim < 1:
            raise ValueError("Choose conv or mlp and a positive latent dimension")
        widths = channels if architecture == "conv" else hidden_dims
        if not widths or any((w < 1 for w in widths)):
            raise ValueError("Network widths must be positive and nonempty")
        if architecture == "conv" and len(channels) > 5:
            raise ValueError("At most five downsampling layers fit a 21x21 grid")
        if (
            output_mode not in ("sigmoid", "clamp", "hurdle")
            or not 0 < zero_probability_threshold < 1
        ):
            raise ValueError(
                "Choose sigmoid/clamp/hurdle output and a probability threshold in (0,1)"
            )
        self.output_mode = output_mode
        self.config = {
            "architecture": architecture,
            "latent_dim": latent_dim,
            "channels": list(channels),
            "hidden_dims": list(hidden_dims),
            "output_mode": output_mode,
            "zero_probability_threshold": zero_probability_threshold,
        }
        output_channels = 4 if output_mode == "hurdle" else 2
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
                decoder += [
                    nn.Linear(previous, 2 * 21 * 21),
                    nn.Sigmoid(),
                    nn.Unflatten(1, (2, 21, 21)),
                ]
            else:
                decoder += [
                    nn.Linear(previous, output_channels * 21 * 21),
                    nn.Unflatten(1, (output_channels, 21, 21)),
                    HurdlePressureOutput(zero_probability_threshold)
                    if output_mode == "hurdle"
                    else ClippedPressureOutput(),
                ]
        else:
            encoder, sizes = ([], [21])
            previous = 4
            for width in channels:
                encoder += [nn.Conv2d(previous, width, 3, stride=2, padding=1), nn.GELU()]
                sizes.append((sizes[-1] + 1) // 2)
                previous = width
            cells = channels[-1] * sizes[-1] ** 2
            encoder += [nn.Flatten(), nn.Linear(cells, latent_dim)]
            decoder = [
                nn.Linear(latent_dim, cells),
                nn.GELU(),
                nn.Unflatten(1, (channels[-1], sizes[-1], sizes[-1])),
            ]
            for i in reversed(range(len(channels))):
                out_channels = channels[i - 1] if i else output_channels
                decoder += [
                    nn.Upsample(size=(sizes[i], sizes[i]), mode="nearest"),
                    nn.Conv2d(channels[i], out_channels, 3, padding=1),
                    nn.GELU()
                    if i
                    else nn.Sigmoid()
                    if output_mode == "sigmoid"
                    else ClippedPressureOutput()
                    if output_mode == "clamp"
                    else HurdlePressureOutput(zero_probability_threshold),
                ]
        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)

    def encode(self, pressure, valid=None):
        """Return [B, latent_dim] codes; True mask cells are observed pressure."""
        if pressure.ndim != 4 or pressure.shape[1:] != (2, 21, 21):
            raise ValueError("Expected pressure [batch, 2, 21, 21]")
        valid = torch.isfinite(pressure) if valid is None else valid.bool()
        if valid.shape != pressure.shape:
            raise ValueError("Pressure and mask shapes differ")
        safe = torch.where(valid, pressure, 0.0)
        return self.encoder(torch.cat((safe, valid.to(safe.dtype)), dim=1))

    def decode_components(self, code):
        if self.output_mode == "clamp":
            raw = self.decoder[:-1](code)
            return {"soft": raw, "hard": raw.clamp(0.0, 1.0)}
        if self.output_mode != "hurdle":
            prediction = self.decoder(code)
            return {"soft": prediction, "hard": prediction}
        logits = self.decoder[:-1](code)
        return self.decoder[-1].components(logits)

    def decode(self, code, hard=None):
        if hard is None:
            return self.decoder(code)
        return self.decode_components(code)["hard" if hard else "soft"]

    def forward(self, pressure, valid=None):
        return self.decode(self.encode(pressure, valid))


class SensorAutoencoder(nn.Module):
    """Reconstruct fixed native [B, C, H, W] targets using an MLP or CNN.

    The mask enters the encoder alongside measurements, preserving the
    distinction between missing cells and valid zeros. The decoder consumes
    only [B, latent_dim] sensor codes and emits the exact configured target
    shape in [0, 1]. Physical units are restored outside the model by codecs.
    """

    def __init__(
        self,
        target_shape,
        architecture="conv",
        latent_dim=64,
        channels=(32, 64),
        hidden_dims=(256, 128),
        output_mode="sigmoid",
    ):
        super().__init__()
        self.shape = tuple(target_shape)
        if len(self.shape) != 3 or min(self.shape) < 1 or latent_dim < 1:
            raise ValueError("Positive [C,H,W] and sensor code dimension are required")
        if architecture not in ("conv", "mlp") or output_mode != "sigmoid":
            raise ValueError("Choose conv/mlp and sigmoid output")
        widths = channels if architecture == "conv" else hidden_dims
        if architecture == "conv" and (not widths) or any((width < 1 for width in widths)):
            raise ValueError("Positive widths are required; conv needs at least one stage")
        if architecture == "conv" and min(self.shape[1:]) <= 1:
            raise ValueError("Use MLP for sensor vectors and force/torque channels")
        self.config = {
            "target_shape": list(self.shape),
            "architecture": architecture,
            "latent_dim": latent_dim,
            "channels": list(channels),
            "hidden_dims": list(hidden_dims),
            "output_mode": output_mode,
        }
        cells = math.prod(self.shape)
        if architecture == "mlp":
            encoder, previous = ([nn.Flatten()], 2 * cells)
            for width in hidden_dims:
                encoder.extend([nn.Linear(previous, width), nn.GELU()])
                previous = width
            encoder.append(nn.Linear(previous, latent_dim))
            decoder, previous = ([], latent_dim)
            for width in reversed(hidden_dims):
                decoder.extend([nn.Linear(previous, width), nn.GELU()])
                previous = width
            decoder.extend([nn.Linear(previous, cells), nn.Sigmoid(), nn.Unflatten(1, self.shape)])
        else:
            encoder, previous, sizes = ([], 2 * self.shape[0], [self.shape[1:]])
            for width in channels:
                encoder.extend([nn.Conv2d(previous, width, 3, stride=2, padding=1), nn.GELU()])
                sizes.append(tuple(((n + 1) // 2 for n in sizes[-1])))
                previous = width
            hidden_cells = channels[-1] * math.prod(sizes[-1])
            encoder.extend([nn.Flatten(), nn.Linear(hidden_cells, latent_dim)])
            decoder = [
                nn.Linear(latent_dim, hidden_cells),
                nn.GELU(),
                nn.Unflatten(1, (channels[-1], *sizes[-1])),
            ]
            for i in reversed(range(len(channels))):
                output = channels[i - 1] if i else self.shape[0]
                decoder.extend(
                    [
                        nn.Upsample(size=sizes[i], mode="nearest"),
                        nn.Conv2d(channels[i], output, 3, padding=1),
                        nn.GELU() if i else nn.Sigmoid(),
                    ]
                )
        self.encoder, self.decoder = (nn.Sequential(*encoder), nn.Sequential(*decoder))

    def encode(self, target, valid=None):
        """Encode target/mask tensors without changing the native grid layout."""
        if target.ndim != 4 or tuple(target.shape[1:]) != self.shape:
            raise ValueError(f"Expected [batch,{self.shape}]")
        valid = torch.isfinite(target) if valid is None else valid.bool()
        if valid.shape != target.shape:
            raise ValueError("Target and validity mask shapes differ")
        return self.encoder(
            torch.cat((torch.where(valid, target, 0.0), valid.to(target.dtype)), dim=1)
        )

    def decode(self, code):
        return self.decoder(code)

    def forward(self, target, valid=None):
        return self.decode(self.encode(target, valid))


class PressureAutoencoder(SensorAutoencoder):
    """Same 64D convolutional AE with fixed train-only input conditioning.

    Decoder targets and outputs retain the existing bounded pressure scale.
    Conditioning makes sparse pressure visible beside the constant valid mask.
    ``input_mean`` and ``input_scale`` are fixed training statistics, not values
    estimated separately for each episode or input frame.
    """

    def __init__(self, input_mean=0.0, input_scale=1.0, **kwargs):
        super().__init__(**kwargs)
        if not math.isfinite(input_mean) or not math.isfinite(input_scale) or input_scale <= 0:
            raise ValueError("Input statistics must be finite with positive scale")
        self.input_mean = float(input_mean)
        self.input_scale = float(input_scale)
        self.config.update(input_mean=self.input_mean, input_scale=self.input_scale)

    def encode(self, target, valid=None):
        return super().encode((target - self.input_mean) / self.input_scale, valid)
