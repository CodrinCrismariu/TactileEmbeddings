"""Decode raw sensor codes with the source's pinned, frozen teacher.

Inputs are raw (not z-scored) teacher codes and optional static marker geometry.
Outputs preserve the teacher's native target shape and source codec scale.
For example, ``SensorDecoder(spec).decode(code, geometry)`` reconstructs targets
without providing measured displacement as a skip connection.
"""

import torch

from .autoencoders import load_teacher


class SensorDecoder:
    """Native reconstruction adapter; normalization remains with the source codec.

    Raw codes may have leading [batch,query] dimensions. Marker geometry is
    static calibration information. Current readings and validity never enter
    the frozen decoder as skip connections.
    """

    def __init__(self, spec, device="cpu"):
        if not spec.teacher.get("sha256"):
            raise ValueError("Sensor decoding requires a pinned teacher hash")
        self.spec = spec
        self.model, self.state = load_teacher(
            spec.teacher["checkpoint"],
            device,
            spec.teacher["sha256"],
        )
        self.device = torch.device(device)

    @torch.inference_mode()
    def decode(self, raw_code, geometry=None):
        """Decode [..., code_dim], preserving every leading code dimension.

        Marker geometry supplies ``reference`` [..., N, 2] and ``surface``
        [..., N], or a shared static [N, 2]/[N] layout. Reference coordinates
        are normalized sensor coordinates; signed output displacement remains
        in the marker teacher's spacing units. Missingness is masked by the
        caller rather than represented as a decoded zero.
        """
        raw_code = raw_code.to(self.device).float()
        leading = tuple(raw_code.shape[:-1])
        code = raw_code.reshape(-1, raw_code.shape[-1])
        if self.spec.target_kind != "marker_displacement":
            result = self.model.decode(code)
        else:
            geometry = geometry or {}
            if "reference" not in geometry or "surface" not in geometry:
                raise ValueError("Marker decoding requires reference coordinates and surface IDs")
            reference = geometry["reference"].to(self.device)
            surface = geometry["surface"].to(self.device)
            if reference.ndim == 2:
                reference = reference.expand(*leading, *reference.shape)
                surface = surface.expand(*leading, surface.shape[-1])
            elif len(leading) == 2 and reference.ndim == 3:
                reference = reference[:, None].expand(*leading, *reference.shape[-2:])
                surface = surface[:, None].expand(*leading, surface.shape[-1])
            if tuple(reference.shape[:-2]) != leading or tuple(surface.shape[:-1]) != leading:
                raise ValueError("Static marker geometry does not match code batch dimensions")
            points = reference.shape[-2]
            result = self.model.decode(
                code, reference.reshape(-1, points, 2), surface.reshape(-1, points)
            )
        return result.reshape(*leading, *result.shape[1:])
