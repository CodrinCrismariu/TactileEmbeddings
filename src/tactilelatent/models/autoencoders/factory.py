"""Construct and strictly load native-target autoencoder checkpoint formats.

Inputs are saved model configuration, format identifiers, and checkpoint paths.
Outputs are actual maintained modules plus their original checkpoint metadata.
Use ``load_teacher(path, expected_sha256=digest)`` for frozen reconstruction;
the loader never dispatches on a dataset name or repairs incompatible weights.
"""

import hashlib
from pathlib import Path

import torch

from .bilateral import BilateralMarkerAutoencoder, BilateralVectorAutoencoder
from .marker import BaihuMarkerAutoencoder, ExpansionMarkerAutoencoder, MarkerAutoencoder
from .numeric import NumericAutoencoder
from .pressure import PressureAutoencoder, SensorAutoencoder, TactileAutoencoder

TEACHER_TYPES = {
    "marker_autoencoder_v1": MarkerAutoencoder,
    "bilateral_vector_autoencoder_v1": BilateralVectorAutoencoder,
    "bilateral_marker_autoencoder_v1": BilateralMarkerAutoencoder,
    "expansion_marker_autoencoder_v1": ExpansionMarkerAutoencoder,
    "expansion_marker_autoencoder_v2": BaihuMarkerAutoencoder,
    "expansion_pressure_autoencoder_v1": SensorAutoencoder,
    "expansion_pressure_autoencoder_v2": PressureAutoencoder,
    "native-numeric-ae-v1": NumericAutoencoder,
}


def checkpoint_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def make_autoencoder(config, model_type=None):
    """Construct an actual implementation using its recorded checkpoint format."""
    if model_type is not None:
        if model_type not in TEACHER_TYPES:
            raise ValueError(f"Unknown sensor-teacher checkpoint type: {model_type}")
        return TEACHER_TYPES[model_type](**config)
    # The first dense AEs predate a model_type field; their saved config is
    # unambiguous. These branches adapt a serialization format, not a source ID.
    if config.get("bilateral"):
        cls = BilateralVectorAutoencoder if "target_shape" in config else BilateralMarkerAutoencoder
    elif "input_scale" in config:
        cls = PressureAutoencoder
    elif "input_dim" in config:
        cls = NumericAutoencoder
    elif config.get("architecture") == "marker_set":
        cls = (
            BaihuMarkerAutoencoder
            if "code_normalization" in config
            else (ExpansionMarkerAutoencoder if config.get("surfaces") == 4 else MarkerAutoencoder)
        )
    else:
        cls = SensorAutoencoder if "target_shape" in config else TactileAutoencoder
    return cls(**config)


def load_teacher(path, device="cpu", expected_sha256=None):
    """Return ``(frozen_model, saved_state)`` after strict format/hash checks.

    All parameters have gradients disabled and the model is in evaluation mode.
    Loading trusts the selected checkpoint's serialized Python data; callers
    must select artifacts through verified source metadata.
    """
    if expected_sha256 is not None and checkpoint_hash(path) != expected_sha256:
        raise ValueError("Frozen sensor-teacher checkpoint hash mismatch")
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = make_autoencoder(state["model_config"], state.get("model_type"))
    model.load_state_dict(state["model"], strict=True)
    model = model.to(device).eval().requires_grad_(False)
    return model, state
