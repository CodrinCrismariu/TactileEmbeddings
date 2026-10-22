"""Apply verified source-coordinate mirroring to uniform batches.

Inputs are native batches, registry policies and deterministic scheduling keys.
Outputs carry reflected RGB, transformed targets and corresponding frozen codes;
clip counts, chronology, validity and source weights are unchanged.
"""

import hashlib
import json

import torch

from ..datasets.sources.mirroring import mirror_rgb, transform_targets


def sample_flags(batch, probability, seed, epoch, step):
    """Return one reproducible mirror decision per clip without global RNG draws.

    Episode/frame identity and seed/epoch/step determine each decision, allowing
    checkpoint resume and SAM replay to reuse the identical augmentation.
    """
    if not 0 <= probability <= 1:
        raise ValueError("Mirror probability must be in [0,1]")
    # No worker-local RNG: epoch-boundary resume and SAM replay cannot redraw a
    # different augmentation. IDs also distinguish individual clips in a batch.
    values = []
    for i, episode in enumerate(batch["episode_id"]):
        frames = batch["frame_indices"][i].tolist()
        key = json.dumps([seed, epoch, step, i, episode, frames], separators=(",", ":")).encode()
        draw = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / 2**64
        values.append(draw < probability)
    return torch.tensor(values, dtype=torch.bool)


def augment_samples(batch, registry, seed, epoch, step, probability=None, flags=None):
    """Return a consistently mirrored batch before visual encoding/dropout.

    Exchange wrist slots and availability, apply verified native sensor geometry,
    and retrieve separately encoded mirrored teacher codes. ``flags`` can replay
    exact decisions. Missing measurements stay missing, and invalid padded marker
    points never acquire support through a coordinate transform.
    """
    from dataclasses import replace

    source = batch.source
    spec = registry[source].spec
    if not spec.mirror:
        if flags is not None and torch.as_tensor(flags).any():
            raise ValueError("Source has no verified mirror mapping")
        return batch
    probability = registry.mirror_probability if probability is None else probability
    if not 0 <= probability <= 1:
        raise ValueError("Mirror probability must be in [0,1]")
    n = len(batch.episode_ids)
    scheduling = dict(episode_id=batch.episode_ids, frame_indices=batch.frame_indices)
    flags = (
        sample_flags(scheduling, probability, seed, epoch, step)
        if flags is None
        else torch.as_tensor(flags, dtype=torch.bool).cpu()
    )
    if flags.shape != (n,):
        raise ValueError("One mirror decision is required per clip")
    if not flags.any():
        return batch
    rgb = batch.visual.rgb.clone()
    cameras = batch.visual.view_mask.clone()
    mapped = (
        batch.visual.rgb_frame_indices.clone()
        if batch.visual.rgb_frame_indices is not None
        else None
    )
    values = batch.target.values.clone()
    valid = batch.target.valid.clone()
    geometry = {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in batch.target.geometry.items()
    }
    supervised = batch.supervision.values.clone()
    supported = batch.supervision.valid.clone()
    for index in flags.nonzero().flatten().tolist():
        rgb[index], cameras[index] = mirror_rgb(
            batch.visual.rgb[index], batch.visual.view_mask[index]
        )
        if mapped is not None:
            mapped[index] = mapped[index].index_select(
                -1, torch.tensor([0, 2, 1], device=mapped.device)
            )
        native = {"target": batch.target.values[index], "target_mask": batch.target.valid[index]}
        if batch.target.kind == "marker_displacement":
            count = int(batch.target.geometry["count"][index])
            native = dict(
                marker_target=batch.target.values[index, ..., :count, :],
                marker_valid=batch.target.valid[index, ..., :count],
                marker_reference=batch.target.geometry["reference"][index, :count],
                marker_surface=batch.target.geometry["surface"][index, :count],
                marker_spacing=batch.target.geometry["spacing"][index, :count],
            )
        changed = transform_targets(native, spec.mirror)
        if batch.target.kind == "marker_displacement":
            values[index, ..., :count, :] = changed["marker_target"]
            valid[index, ..., :count] = changed["marker_valid"]
            geometry["reference"][index, :count] = changed["marker_reference"]
            geometry["surface"][index, :count] = changed["marker_surface"]
        else:
            values[index] = changed["target"]
            valid[index] = changed["target_mask"]
        if spec.supervision_kind == "code":
            row = batch.provenance[index]["native_row"]
            next_orientation = not batch.provenance[index].get("mirror_applied", False)
            code, code_valid = registry[source].cache.read(
                row, batch.frame_indices[index].cpu().numpy(), mirrored=next_orientation
            )
            supervised[index] = code.to(supervised.device)
            supported[index] = code_valid.to(supported.device)
        else:
            supervised[index] = registry[source].codec.encode_supervision(values[index])
            supported[index] = valid[index]
    provenance = [
        dict(row, mirror_applied=bool(row.get("mirror_applied", False)) != bool(flag))
        for row, flag in zip(batch.provenance, flags.tolist())
    ]
    states = [row["mirror_applied"] for row in provenance]
    orientation = "mirrored" if all(states) else "original" if not any(states) else "mixed"
    return replace(
        batch,
        visual=replace(batch.visual, rgb=rgb, view_mask=cameras, rgb_frame_indices=mapped),
        target=replace(batch.target, values=values, valid=valid, geometry=geometry),
        supervision=replace(
            batch.supervision, values=supervised, valid=supported, orientation=orientation
        ),
        provenance=provenance,
    )
