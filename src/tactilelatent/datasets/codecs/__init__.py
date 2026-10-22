"""Convert declared sensor coordinates separately from neural reconstruction.

Inputs are native values and ``SourceSpec`` normalization metadata. Outputs retain
source dimensions; only documented transforms alter units or supervision scale.
"""

import torch


class SensorCodec:
    """Apply a source's declared numeric transform and documented unit inverse."""

    def __init__(self, spec):
        self.spec = spec

    def encode_supervision(self, native):
        """Map native numeric values to the configured training coordinates."""
        transform = self.spec.numeric_transform
        if transform.get("method") == "asinh(native/train RMS)":
            scale = native.new_tensor(transform["scale"])
            return torch.asinh(native / scale)
        return native

    def decode_supervision(self, normalized):
        """Decode numeric predictions using the declared bounded inverse."""
        transform = self.spec.numeric_transform
        if transform.get("method") == "asinh(native/train RMS)":
            scale = normalized.new_tensor(transform["scale"])
            return (torch.sinh(normalized.float().clamp(-15, 15)) * scale).clamp_min(0)
        return normalized

    def physical_values(self, values, geometry=None):
        """Return marker pixels where defined; otherwise preserve released units.

        Marker values are signed spacing-normalized vectors. Multiplication by
        per-marker pixel spacing restores pixels, with geometry broadcast over
        time. Released pressure counts are never relabeled as SI force.
        """
        if self.spec.target_kind == "marker_displacement":
            spacing = geometry["spacing"]
            while spacing.ndim < values.ndim - 1:
                spacing = spacing.unsqueeze(-2)
            return values * spacing.unsqueeze(-1)
        return values
