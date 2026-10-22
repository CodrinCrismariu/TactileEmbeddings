"""Versioned native sensor-coordinate reflection conventions.

These transforms preserve anatomical/surface identity, signed vectors and masks.
They never infer correspondence or transform arbitrary latent coordinates.
Inputs are native sample tensors and a verified policy; outputs retain shape and
measurement support while transforming hand ownership and local coordinates.
"""

import json
from pathlib import Path

import numpy as np
import torch


def load_policy(path):
    """Read a versioned policy and validate mappings marked as verified."""
    policy = json.loads(Path(path).read_text())
    if policy.get("schema_version") != 1 or not isinstance(policy.get("sources"), dict):
        raise ValueError("Expected a versioned source-coordinate mirror policy")
    for source, spec in policy["sources"].items():
        if spec.get("status") == "verified":
            validate_spec(spec)
            if not spec.get("evidence"):
                raise ValueError(f"Missing mirror-coordinate evidence: {source}")
    return policy


def require_spec(policy, source):
    """Return a verified source mapping, rejecting unresolved geometry."""
    if source not in policy["sources"]:
        raise ValueError(f"Mirroring is restricted to bilateral sources: {source}")
    spec = policy["sources"][source]
    if spec.get("status") != "verified":
        raise ValueError(
            f"{source}: mirror mapping blocked: {spec.get('reason', 'unverified geometry')}"
        )
    validate_spec(spec)
    return spec


def validate_spec(spec):
    """Require self-inverting hand, taxel or marker-coordinate correspondences.

    Marker affine maps support verified local x/y reflections; an unverified
    axis swap or guessed correspondence is rejected.
    """
    kind = spec["kind"]
    if kind not in ("grid", "taxels", "scalar", "markers"):
        raise ValueError("Unknown tactile mirror representation")
    perm = np.asarray(
        spec["surface_permutation"] if kind == "markers" else spec["hand_permutation"]
    )
    if (
        perm.ndim != 1
        or sorted(perm.tolist()) != list(range(len(perm)))
        or not np.array_equal(perm[perm], np.arange(len(perm)))
    ):
        raise ValueError("Mirror identities must form an involutive permutation")
    if kind == "markers":
        transforms = np.asarray(spec["surface_transforms"], np.float64)
        if transforms.shape != (len(perm), 2, 3) or not np.isfinite(transforms).all():
            raise ValueError("Every marker surface requires a finite coordinate transform")
        for sid, dst in enumerate(perm):
            a, b = transforms[sid, :, :2], transforms[sid, :, 2]
            reverse, offset = transforms[dst, :, :2], transforms[dst, :, 2]
            # Current native marker contract supports x/y reflections. An axis
            # swap needs native image aspect ratios and must not be guessed.
            if not np.allclose(a, np.diag(np.diag(a))) or not np.allclose(a.T @ a, np.eye(2)):
                raise ValueError("Marker mirrors support verified local-axis reflections only")
            if not np.allclose(reverse @ a, np.eye(2)) or not np.allclose(reverse @ b + offset, 0):
                raise ValueError("Marker coordinate transform is not self-inverting")
    if kind == "taxels":
        channels = np.asarray(spec["channel_permutation"], np.int64)
        if channels.ndim != 2 or len(channels) != len(perm):
            raise ValueError("Taxel mirroring requires an explicit per-hand anatomical map")
        for hand, dst in enumerate(perm):
            p = channels[hand]
            if sorted(p.tolist()) != list(range(len(p))) or not np.array_equal(
                channels[dst][p], np.arange(len(p))
            ):
                raise ValueError(
                    "Taxel correspondence must preserve all channels and invert exactly"
                )


def transform_targets(item, spec):
    """Transform one unpadded tactile sample (time dimension is optional).

    Marker point identity/order is retained; surface ownership and coordinates
    transform together. NaNs and measured zero remain distinct.
    """
    validate_spec(spec)
    if "code_target" in item:
        raise ValueError(
            "Transform native targets before encoding; never mirror latent coordinates"
        )
    out = dict(item)
    kind = spec["kind"]
    if kind != "markers":
        order = torch.tensor(
            spec["hand_permutation"], dtype=torch.long, device=item["target"].device
        )
        for key in ("target", "target_mask"):
            if key not in item:
                continue
            value = item[key]
            if value.shape[-3] != len(order):
                raise ValueError("Unexpected bilateral target layout")
            value = value.index_select(-3, order)
            if kind == "grid" and spec.get("flip_width"):
                value = value.flip(-1)
            elif kind == "taxels":
                channels = torch.as_tensor(spec["channel_permutation"], device=value.device)
                if value.shape[-2] != 1 or value.shape[-1] != channels.shape[1]:
                    raise ValueError("Taxel geometry differs from the verified channel map")
                value = torch.stack(
                    [
                        value[..., hand, :, :].index_select(-1, channels[hand])
                        for hand in range(len(order))
                    ],
                    dim=-3,
                )
            out[key] = value
        return out
    surface = item["marker_surface"].long()
    if (
        surface.ndim != 1
        or (surface < 0).any()
        or (surface >= len(spec["surface_permutation"])).any()
    ):
        raise ValueError("Unexpected marker surface identities")
    transforms = torch.as_tensor(
        spec["surface_transforms"], device=surface.device, dtype=item["marker_reference"].dtype
    )[surface]
    matrix, offset = transforms[..., :2], transforms[..., 2]
    out["marker_reference"] = torch.einsum("nij,nj->ni", matrix, item["marker_reference"]) + offset
    out["marker_target"] = torch.einsum("nij,...nj->...ni", matrix, item["marker_target"])
    out["marker_surface"] = torch.as_tensor(spec["surface_permutation"], device=surface.device)[
        surface
    ]
    return out


def mirror_rgb(rgb, view_mask):
    """Reflect RGB width and exchange left/right wrist slots and availability.

    Camera slots are scene, left wrist, right wrist; leading time/batch dimensions
    are preserved. All frames in a clip receive the same geometric operation.
    """
    if rgb.shape[-4] != 3 or view_mask.shape[-1] != 3:
        raise ValueError("Expected scene/left-wrist/right-wrist camera slots")
    order = torch.tensor([0, 2, 1], device=rgb.device)
    return rgb.index_select(-4, order).flip(-1), view_mask.index_select(
        -1, order.to(view_mask.device)
    )
