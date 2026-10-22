"""Gaussian or deterministic visual bottleneck -> sensor adapters.

Frozen tactile encoders supply cached supervision. Frozen decoders are an
explicit inference/evaluation utility, never part of the training forward.
"""
import torch
from torch import nn

from .aligned_frames import file_hash
from .model import EgoTouchLatentModel
from .sensor_autoencoder import make_autoencoder


MODEL_TYPE = "joint_sensor_alignment_v1"


class DeterministicVisualEncoder(EgoTouchLatentModel):
    """One learned latent projection; no learned variance or latent sampling.

    The historical encode API returns (latent, logvar). Negative infinity
    denotes exactly zero variance for compatible inference/probe tools; it is
    not a learned Gaussian or a predictive-uncertainty estimate.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.prior_logvar
        self.stochastic = False

    def encode(self, features, view_mask):
        latent = self.prior_mu(self.encode_context(features, view_mask))
        return latent, torch.full_like(latent, -torch.inf)


class JointSensorAlignment(nn.Module):
    def __init__(self, sensor_code_dims, numeric_shapes, statistics, input_dim=768,
                 width=768, heads=8, temporal_layers=3, cross_view_layers=2,
                 latent_dim=32, dropout=.1, stochastic=True, adapter_hidden_dims=(128, 128),
                 latent_mode="gaussian"):
        super().__init__()
        if latent_dim < 1 or any(n < 1 for n in adapter_hidden_dims):
            raise ValueError("Positive shared/adapter dimensions required")
        if set(sensor_code_dims) & set(numeric_shapes):
            raise ValueError("Sources cannot have both code and numeric heads")
        if latent_mode not in ("gaussian", "deterministic") or (latent_mode == "deterministic" and stochastic):
            raise ValueError("Deterministic mode requires stochastic=False")
        self.latent_mode = latent_mode
        self.model_type = MODEL_TYPE
        self.sensor_code_dims = dict(sensor_code_dims)
        self.numeric_shapes = dict(numeric_shapes)
        self.stochastic = stochastic
        encoder_class = DeterministicVisualEncoder if latent_mode == "deterministic" else EgoTouchLatentModel
        self.visual = encoder_class(input_dim, width, heads, temporal_layers,
                                    cross_view_layers, latent_dim, dropout, stochastic)
        self.visual.decoder = nn.Identity()  # Remove the historical pressure decoder.
        self.adapters, self.numeric_heads = nn.ModuleDict(), nn.ModuleDict()
        self.normalization = nn.ModuleDict()
        for source, dimension in sensor_code_dims.items():
            layers, previous = [], latent_dim
            for hidden in adapter_hidden_dims:
                layers.extend([nn.Linear(previous, hidden), nn.GELU()])
                previous = hidden
            layers.append(nn.Linear(previous, dimension))
            self.adapters[source] = nn.Sequential(*layers)
            norm = nn.Module()
            mean = torch.as_tensor(statistics[source]["mean"], dtype=torch.float32)
            std = torch.as_tensor(statistics[source]["std"], dtype=torch.float32)
            if mean.shape != (dimension,) or std.shape != (dimension,) or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std < 1e-6).any():
                raise ValueError("Invalid train-only code normalization")
            norm.register_buffer("mean", mean)
            norm.register_buffer("std", std)
            self.normalization[source] = norm
        for source, shape in numeric_shapes.items():
            cells = int(torch.tensor(shape).prod())
            self.numeric_heads[source] = nn.Sequential(nn.Linear(latent_dim, cells), nn.Sigmoid())

    def forward(self, features, view_mask, source, sample_latent=None):
        if source not in self.adapters and source not in self.numeric_heads:
            raise ValueError(f"Untrained source: {source}")
        mu, logvar = self.visual.encode(features, view_mask)
        sample = self.stochastic and self.training if sample_latent is None else sample_latent
        if self.latent_mode == "deterministic":
            sample = False
        z = mu + torch.randn_like(mu) * (.5 * logvar).exp() if sample else mu
        output = {"mu": mu, "logvar": logvar}
        if self.latent_mode == "deterministic":
            output["latent_mode"] = "deterministic"
        if source in self.adapters:
            output["sensor_code"] = self.adapters[source](z)  # Standardized code.
        else:
            output["pred"] = self.numeric_heads[source](z).reshape(*z.shape[:-1], *self.numeric_shapes[source])
        return output

    def raw_code(self, source, standardized):
        norm = self.normalization[source]
        return standardized.float() * norm.std + norm.mean


class FrozenSensorDecoders:
    """Load one pinned decoder at a time; excluded from all model optimizers."""
    def __init__(self, registry, device):
        self.registry, self.device = registry, device
        self.source = self.decoder = None

    def decode(self, source, raw_code):
        if self.source != source:
            self.decoder = None
            info = self.registry["sources"][source]
            if file_hash(info["checkpoint"]) != info["sha256"]:
                raise ValueError("Decoder teacher provenance mismatch")
            saved = torch.load(info["checkpoint"], map_location="cpu", weights_only=False)
            ae = make_autoencoder(saved["model_config"])
            ae.load_state_dict(saved["model"])
            self.decoder = ae.decoder.to(self.device).eval().requires_grad_(False)
            self.source = source
        leading = raw_code.shape[:-1]
        return self.decoder(raw_code.flatten(0, -2)).reshape(*leading, *self.registry["sources"][source]["target_shape"])


def model_from_state(state):
    if state.get("model_type") != MODEL_TYPE:
        raise ValueError("Not a sensor-code alignment checkpoint")
    model = JointSensorAlignment(**state["model_config"])
    model.load_state_dict(state["model"], strict=True)
    return model
