"""Editable shared encoder and generic source-routed prediction heads.

The encoder never receives a dataset ID. Head names only select the output
representation after encoding; adding a source requires no branch in this file.
Inputs are frozen patch tokens, camera availability, and an output-head name.
Outputs contain shared latent means and either standardized sensor codes or
direct numeric predictions. Use ``model_from_state`` for strict saved-model
loading and ``JointModel.encode`` for frozen-representation probes.
"""

import torch
from torch import nn

from .attention import SpatialFrameEncoder
from .heads import bounded_numeric_head, code_adapter, code_normalization
from .initialization import consume_discarded_pair_projection

MODEL_TYPE = "joint_vjepa21_spatial_marker_codes_v1"


class JointModel(nn.Module):
    """Frozen video tokens -> attention -> shared latent -> a registered head.

    ``sensor_code_dims``, ``numeric_shapes``, and ``direct_numeric`` are generic
    head specifications compiled from the selected sources. Their insertion
    order is preserved for baseline initialization and checkpoint compatibility.
    """

    def __init__(
        self,
        sensor_code_dims,
        numeric_shapes,
        statistics,
        input_dim=768,
        latent_dim=32,
        adapter_hidden_dims=(128, 128),
        latent_mode="deterministic",
        width=256,
        heads=4,
        layers=2,
        feedforward=1024,
        dropout=0.1,
        grid_size=24,
        prediction_mode="current_frame",
        direct_numeric=None,
        initialization="baseline_v1",
    ):
        super().__init__()
        if latent_mode != "deterministic":
            raise ValueError("The current spatial model has a deterministic latent")
        if initialization != "baseline_v1":
            raise ValueError("Unknown initialization policy")
        if latent_dim < 1 or any(n < 1 for n in adapter_hidden_dims):
            raise ValueError("Positive shared-latent and adapter dimensions required")
        if set(sensor_code_dims) & set(numeric_shapes):
            raise ValueError("A source cannot have both code and numeric heads")

        self.latent_mode = latent_mode
        self.model_type = MODEL_TYPE
        self.stochastic = False
        self.sensor_code_dims = dict(sensor_code_dims)
        self.numeric_shapes = dict(numeric_shapes)
        self.prediction_mode = prediction_mode
        self.direct_numeric = direct_numeric or {}
        if not set(self.direct_numeric) <= set(self.numeric_shapes):
            raise ValueError("Direct numeric specifications need corresponding head shapes")

        # Reserve the original visual module's position in parameter iteration.
        self.visual = nn.Identity()
        consume_discarded_pair_projection(input_dim, latent_dim)
        self.adapters = nn.ModuleDict()
        self.numeric_heads = nn.ModuleDict()
        self.normalization = nn.ModuleDict()
        for source, dimension in self.sensor_code_dims.items():
            self.adapters[source] = code_adapter(latent_dim, dimension, adapter_hidden_dims)
            self.normalization[source] = code_normalization(statistics[source], dimension)
        for source, shape in self.numeric_shapes.items():
            self.numeric_heads[source] = bounded_numeric_head(latent_dim, shape)

        self.visual = SpatialFrameEncoder(
            input_dim,
            latent_dim,
            width,
            heads,
            layers,
            feedforward,
            dropout,
            grid_size,
            prediction_mode,
        )
        self.direct_scales = nn.ModuleDict()
        for source, info in self.direct_numeric.items():
            dimension = len(info["scale"])
            self.numeric_heads[source] = nn.Linear(latent_dim, dimension)
            norm = nn.Module()
            norm.register_buffer("scale", torch.tensor(info["scale"], dtype=torch.float32))
            self.direct_scales[source] = norm

    def encode(self, features, view_mask, camera_roles=None):
        """Return [B, Q, latent_dim] means without accessing a source name.

        Inputs follow ``SpatialFrameEncoder.encode_context``. The second
        returned tensor is negative-infinite log-variance, a deterministic
        compatibility value rather than learned predictive uncertainty.
        """
        return self.visual.encode(features, view_mask, camera_roles)

    def forward(
        self, features, view_mask, source, sample_latent=None, camera_roles=None, **unused_geometry
    ):
        """Encode a clip and route its latent through one registered head.

        ``source`` selects a head only after shared encoding. Code predictions
        use training-standardized teacher coordinates; numeric predictions use
        their source codec's bounded or fixed transformed coordinates. Native
        reconstruction and physical units require the corresponding decoder.
        """
        # sample_latent is accepted for checkpoint/probe API compatibility;
        # this deterministic model does not sample or learn a variance.
        mu, logvar = self.encode(features, view_mask, camera_roles)
        output = dict(mu=mu, logvar=logvar, latent_mode="deterministic")
        if source in self.adapters:
            output["sensor_code"] = self.adapters[source](mu)
        elif source in self.numeric_heads:
            output["pred"] = self.numeric_heads[source](mu).reshape(
                *mu.shape[:-1],
                *self.numeric_shapes[source],
            )
        else:
            raise ValueError(f"Untrained source: {source}")
        return output

    def raw_code(self, source, standardized):
        """Undo training-only z-scoring before a frozen sensor decoder."""
        norm = self.normalization[source]
        return standardized.float() * norm.std + norm.mean

    def normalize_numeric(self, source, value):
        """Map direct native readings into fixed, zero-preserving asinh space."""
        return torch.asinh(value / self.direct_scales[source].scale)

    def decode_numeric(self, source, value):
        """Restore direct numeric units, enforcing their nonnegative support."""
        native = torch.sinh(value.float().clamp(-15, 15)) * self.direct_scales[source].scale
        return native.clamp_min(0)


def model_from_state(state):
    """Build a current-format model and load every recorded state key strictly.

    This constructs modules and restores weights; it does not load V-JEPA,
    select a device, resume an optimizer, or invoke historical model classes.
    """
    if state.get("model_type") != MODEL_TYPE:
        raise ValueError("Expected a current spatial sensor-code checkpoint")
    model = JointModel(**state["model_config"])
    model.load_state_dict(state["model"], strict=True)
    return model
