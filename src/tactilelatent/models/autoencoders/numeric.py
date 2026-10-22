"""Small native-vector teachers with fixed, zero-preserving asinh scaling.

Signed wrench channels remain signed; channels declared nonnegative are
constrained on native decoding. Missing coordinates are distinct from zeros.
The published normalization and state-dict keys are preserved.
"""

import torch
from torch import nn


class NumericAutoencoder(nn.Module):
    """Encode [frames, channels] readings while preserving channel constraints.

    ``scale`` is a fixed positive value per native channel; ``nonnegative``
    declares which decoded channels may be clipped at zero. Encoding includes
    a validity indicator for each channel, and [frames, latent_dim] codes
    reconstruct values in the original channel units through inverse asinh.
    """

    def __init__(
        self,
        input_dim,
        latent_dim,
        hidden_dims,
        scale,
        nonnegative,
        architecture="mlp",
        output_mode="native numeric",
    ):
        super().__init__()
        if input_dim < 1 or latent_dim < 1 or latent_dim > input_dim or (not hidden_dims):
            raise ValueError("Positive native/code dimensions and hidden widths required")
        scale = torch.as_tensor(scale, dtype=torch.float32)
        nonnegative = torch.as_tensor(nonnegative, dtype=torch.bool)
        if (
            scale.shape != (input_dim,)
            or nonnegative.shape != scale.shape
            or (not torch.isfinite(scale).all())
            or (scale <= 0).any()
        ):
            raise ValueError("Invalid fixed native scaling or channel constraints")
        self.register_buffer("scale", scale)
        self.register_buffer("nonnegative", nonnegative)
        self.input_dim, self.latent_dim = (input_dim, latent_dim)
        self.config = dict(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=list(hidden_dims),
            scale=scale.tolist(),
            nonnegative=nonnegative.tolist(),
            architecture=architecture,
            output_mode=output_mode,
        )
        layers = []
        previous = input_dim * 2
        for width in hidden_dims:
            if width < 1:
                raise ValueError("Hidden widths must be positive")
            layers.extend([nn.Linear(previous, width), nn.GELU()])
            previous = width
        layers.append(nn.Linear(previous, latent_dim))
        self.encoder = nn.Sequential(*layers)
        layers = []
        previous = latent_dim
        for width in reversed(hidden_dims):
            layers.extend([nn.Linear(previous, width), nn.GELU()])
            previous = width
        layers.append(nn.Linear(previous, input_dim))
        self.decoder = nn.Sequential(*layers)

    def normalize(self, values):
        """Apply fixed channel scaling and asinh; native zero stays zero."""
        return torch.asinh(values / self.scale)

    def encode(self, values, valid=None):
        if values.ndim != 2 or values.shape[1] != self.input_dim:
            raise ValueError("Expected [frames,native channels]")
        valid = torch.isfinite(values) if valid is None else valid.bool() & torch.isfinite(values)
        if valid.shape != values.shape:
            raise ValueError("Validity mask shape differs")
        values = torch.where(valid, values, 0.0)
        return self.encoder(torch.cat([self.normalize(values), valid.to(values.dtype)], dim=-1))

    def decode_normalized(self, code):
        """Return transformed channel values before native-unit constraints."""
        return self.decoder(code)

    def from_normalized(self, values):
        """Restore native units and only the declared nonnegative constraints."""
        native = torch.sinh(values.clamp(-15, 15)) * self.scale
        return torch.where(self.nonnegative, native.clamp_min(0), native)

    def decode(self, code):
        return self.from_normalized(self.decode_normalized(code))

    def forward(self, values, valid=None):
        return self.decode(self.encode(values, valid))


def numeric_loss(prediction, target, valid, nonnegative, nonzero_weight=3.0):
    """Loss in fixed transformed coordinates; train even negative raw outputs.

    Physical nonnegativity is enforced on native decoding. Computing this loss
    before that projection avoids dead gradients at zero. No second contact head.
    """
    valid = valid.bool() & torch.isfinite(target)
    target = torch.where(valid, target, 0.0)
    weights = valid * (1 + (nonzero_weight - 1) * (nonnegative & (target != 0)))
    error = prediction - target
    loss = (
        torch.nn.functional.smooth_l1_loss(prediction, target, reduction="none", beta=0.1)
        + 0.25 * error.abs()
    )
    counts = weights.sum(0)
    active = counts > 0
    if not active.any():
        raise ValueError("No observed numeric targets")
    return ((loss * weights).sum(0) / counts.clamp_min(1))[active].mean()
