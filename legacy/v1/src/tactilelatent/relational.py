"""Masked temporal geometry of frozen tactile codes; no cross-source matching."""
import json
import math
from pathlib import Path

import torch

HELDOUT = "ftp_motiontrans"


def read_floors(path, data_signature, cache_signatures):
    artifact = json.loads(Path(path).read_text())
    if (artifact.get("schema_version") != 1 or artifact.get("data") != data_signature
            or artifact.get("cache_signatures") != cache_signatures
            or artifact.get("split") != "train" or HELDOUT in artifact.get("sources", {})):
        raise ValueError("Relational floors have stale or held-out provenance")
    if set(artifact["sources"]) != set(cache_signatures):
        raise ValueError("A floor is required for every AE-backed training source")
    floors = {s: float(row["distance_floor"]) for s, row in artifact["sources"].items()}
    if any(not math.isfinite(v) or v <= 0 for v in floors.values()):
        raise ValueError("Finite positive code-space floors required")
    return floors


def trajectory_distances(value, valid, floor):
    """Return normalized distances and frame positions, or None for no signal."""
    if not math.isfinite(floor) or floor <= 0:
        raise ValueError("Positive finite distance floor required")
    valid = valid.bool() & torch.isfinite(value).all(-1)
    positions = valid.nonzero().flatten()
    if len(positions) < 2:
        return None
    selected = value.detach().float()[positions]
    distance = torch.cdist(selected, selected)
    off_diagonal = ~torch.eye(len(positions), device=value.device, dtype=torch.bool)
    magnitude = distance[off_diagonal].square().mean().sqrt()
    if magnitude <= floor:
        return None
    return distance / magnitude.clamp_min(floor), positions, float(magnitude)


def relational_loss(mu, code, valid, floor, epsilon=1e-6):
    """Equal weight per informative clip; teachers never receive gradients.

    Near-static clips retain the ordinary reconstruction loss but do not enter
    this normalized geometry objective. Diagonals and invalid frames are absent.
    """
    if mu.ndim != 3 or code.ndim != 3 or mu.shape[:2] != code.shape[:2] or valid.shape != code.shape[:2]:
        raise ValueError("Expected aligned [batch,time,dimension] trajectories")
    if epsilon <= 0 or not torch.isfinite(mu).all():
        raise ValueError("Invalid latent values or numerical floor")
    losses, magnitudes = [], []
    for predicted, teacher, mask in zip(mu.float(), code, valid):
        descriptor = trajectory_distances(teacher, mask, floor)
        if descriptor is None:
            continue
        target, positions, magnitude = descriptor
        distance = torch.cdist(predicted[positions], predicted[positions])
        pairs = ~torch.eye(len(positions), device=mu.device, dtype=torch.bool)
        scale = distance[pairs].square().mean().clamp_min(epsilon ** 2).sqrt()
        losses.append((distance[pairs] / scale - target[pairs]).square().mean())
        magnitudes.append(magnitude)
    loss = torch.stack(losses).mean() if losses else mu.sum() * 0
    return loss, {"relational_loss": float(loss.detach()), "relational_clips": len(losses),
                  "relational_skipped": len(mu) - len(losses),
                  "tactile_distance_rms": sum(magnitudes) / max(len(magnitudes), 1)}


def extra_loss(output, batch, source, cfg, floors, epoch, payload=None):
    """The same pure loss function is used for the first pass and SAM replay."""
    if source == HELDOUT:
        raise ValueError("MotionTrans cannot enter representation objectives")
    loss, metrics = output["mu"].sum() * 0, {}
    relational = cfg.get("relational", {})
    if relational.get("weight", 0) > 0 and source in floors:
        value, metrics = relational_loss(output["mu"], batch["code_target"],
                                          batch["code_valid"], floors[source])
        weight = relational["weight"] * min(1., (epoch + 1) / relational.get("ramp_epochs", 2))
        loss = loss + weight * value
        metrics["relational_weight"] = weight
    elif relational.get("weight", 0) > 0:
        metrics["relational_numeric_omitted"] = len(output["mu"])
    if payload is not None:
        from .trajectory_matching import trajectory_loss
        value, stats = trajectory_loss(output["mu"], payload)
        weight = cfg["trajectory"]["weight"] * min(1., (epoch + 1) / cfg["trajectory"].get("ramp_epochs", 2))
        loss = loss + weight * value
        metrics.update(stats, trajectory_weight=weight)
    return loss, metrics
