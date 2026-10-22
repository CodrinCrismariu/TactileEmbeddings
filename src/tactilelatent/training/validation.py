"""Evaluate native reconstruction through source-declared frozen decoders.

Inputs are uniform validation batches and model predictions. Outputs are
per-source supervision losses, masked native reconstruction metrics, and
optional paired mirror diagnostics. ``validate`` leaves ordinary validation
unaugmented unless mirroring is explicitly requested; the engine uses
``paired_validate`` separately for equivariance reporting.
"""

from collections import defaultdict
from pathlib import Path

import torch

from .objectives import forward_batch, supervision_loss


class FrozenDecoders:
    """Lazily load pinned teachers and restore native prediction dimensions.

    Teacher codes are first unstandardized with the joint model's training
    statistics. Marker decoding receives only static reference positions and
    surface IDs; measured displacement is never a decoder skip connection.
    """

    def __init__(self, registry, project_root, device):
        self.registry, self.root, self.device, self.loaded = (
            registry,
            Path(project_root),
            device,
            {},
        )

    def teacher(self, source):
        if source not in self.loaded:
            from ..models.autoencoders import load_teacher

            info = self.registry.specs[source].teacher
            path = Path(info["checkpoint"])
            if not path.is_absolute():
                path = self.root / path
            # Constructors initialize on CPU before loading frozen weights.
            # Lazy validation loading must not advance the training RNG; the
            # historical trainer constructed these teachers before task setup.
            with torch.random.fork_rng(devices=[]):
                self.loaded[source] = load_teacher(
                    path, self.device, expected_sha256=info["sha256"]
                )[0]
        return self.loaded[source]

    @torch.no_grad()
    def decode(self, model, output, batch):
        """Return [B, Q, *native_shape] predictions in the target codec's scale."""
        if batch.supervision.kind == "direct":
            return (
                model.decode_numeric(batch.source, output["pred"])
                if batch.source in model.direct_numeric
                else output["pred"].float()
            )
        code = model.raw_code(batch.source, output["sensor_code"])
        return self.decode_raw(batch.source, code, batch.target)

    @torch.no_grad()
    def decode_raw(self, source, code, target):
        teacher = self.teacher(source)
        leading = code.shape[:-1]
        with torch.autocast(self.device.type, enabled=False):
            if "reference" in target.geometry:
                reference, surface = target.geometry["reference"], target.geometry["surface"]
                b, q, d = code.shape
                n = reference.shape[-2]
                ref = reference[:, None].expand(-1, q, -1, -1).reshape(b * q, n, 2)
                sid = surface[:, None].expand(-1, q, -1).reshape(b * q, n)
                prediction = teacher.decode(code.float().reshape(-1, d), ref, sid)
            else:
                prediction = teacher.decode(code.float().reshape(-1, code.shape[-1]))
        return prediction.reshape(*leading, *prediction.shape[1:])


def move_batch(batch, device):
    """Keep RGB on host for SAM replay, moving supervision and masks only."""
    from dataclasses import replace

    from ..contracts import move_tensors

    visual = replace(
        batch.visual,
        view_mask=batch.visual.view_mask.to(device),
        camera_roles=(
            batch.visual.camera_roles.to(device) if batch.visual.camera_roles is not None else None
        ),
    )
    return replace(
        batch,
        visual=visual,
        target=move_tensors(batch.target, device),
        supervision=move_tensors(batch.supervision, device),
    )


@torch.no_grad()
def validate(model, backbone, loader, registry, project_root, device, *, mirror=False, seed=45):
    """Measure observed native targets without fitting or updating parameters.

    Validity masks distinguish missing readings from measured zeros. Metric
    reductions and units come from each ``SourceSpec``: bounded pressure/image
    error, native numeric error, or signed marker error in spacing/pixel units.
    Frozen-teacher reconstruction is reported as the code-supervision oracle.
    """
    from ..models.backbone import spatial_video_features

    model.eval()
    decoders = FrozenDecoders(registry, project_root, device)
    totals = defaultdict(lambda: defaultdict(float))
    channels = {}
    for step, original in enumerate(loader):
        if mirror:
            if not registry.specs[original.source].mirror:
                continue
            original = registry.augment(
                original, seed=seed, epoch=0, step=step, flags=[True] * len(original.episode_ids)
            )
        batch = move_batch(original, device)
        with torch.autocast(device.type, torch.bfloat16, enabled=device.type == "cuda"):
            features = spatial_video_features(backbone, original, device)
            output = forward_batch(model, features, batch)
            loss = supervision_loss(output, batch)
        prediction = decoders.decode(model, output, batch).float()
        target = batch.target.values.float()
        if prediction.shape != target.shape or not torch.isfinite(prediction).all():
            raise ValueError("Native reconstruction shape or finite-value contract violated")
        valid = batch.target.expanded_mask().bool() & torch.isfinite(target)
        safe = torch.where(valid, target, 0.0)
        error = torch.where(valid, (prediction - safe).abs(), 0.0)
        count = valid.sum().item()
        row = totals[batch.source]
        row["batches"] += 1
        row["loss"] += loss.item()
        row["absolute"] += error.sum().item()
        row["count"] += count
        row["batch_mae"] += error.sum().item() / max(count, 1)
        row["batch_square"] += ((prediction - safe).square() * valid).sum().item() / max(count, 1)
        if "spacing" in batch.target.geometry:
            spacing = batch.target.geometry["spacing"][:, None, :, None]
            pixel_error = error * spacing
            row["batch_pixel_mae"] += pixel_error.sum().item() / max(count, 1)
            row["batch_endpoint"] += torch.linalg.vector_norm(
                pixel_error, dim=-1
            ).sum().item() / max(count / 2, 1)
            row["batch_zero_pixel"] += (safe.abs() * spacing * valid).sum().item() / max(count, 1)
        if batch.supervision.kind == "code":
            truth = model.raw_code(batch.source, batch.supervision.values)
            oracle = decoders.decode_raw(batch.source, truth, batch.target).float()
            absolute = torch.where(valid, (oracle - safe).abs(), 0.0).sum().item()
            row["oracle_absolute"] += absolute
            row["oracle_batch_mae"] += absolute / max(count, 1)
        if batch.target.kind == "native_numeric":
            axes = (0, 1, *range(3, target.ndim))
            sums = error.sum(axes).cpu()
            counts = valid.sum(axes).cpu()
            if batch.source not in channels:
                channels[batch.source] = [sums.double(), counts.double()]
            else:
                channels[batch.source][0] += sums
                channels[batch.source][1] += counts
    result = {}
    for source, row in totals.items():
        spec = registry.specs[source]
        n = row["batches"]
        by_cells = spec.metadata.get("native_metric_reduction") == "cells"
        mae = row["absolute"] / max(row["count"], 1) if by_cells else row["batch_mae"] / n
        item = dict(
            supervision_loss=row["loss"] / n, target=spec.target_kind, valid_cells=int(row["count"])
        )
        if "batch_pixel_mae" in row:
            item.update(
                marker_mae_spacing=mae,
                marker_mae_px=row["batch_pixel_mae"] / n,
                endpoint_error_px=row["batch_endpoint"] / n,
                zero_displacement_baseline_mae_px=row["batch_zero_pixel"] / n,
            )
        elif spec.target_kind == "native_numeric":
            item["native_mae"] = mae
        else:
            item["normalized_mae"] = mae
        if "oracle_absolute" in row:
            item["frozen_ae_mae"] = (
                row["oracle_absolute"] / max(row["count"], 1)
                if by_cells
                else row["oracle_batch_mae"] / n
            )
        if spec.target_kind == "optical_rgb":
            import math

            item["psnr"] = -10 * math.log10(max(row["batch_square"] / n, 1e-15))
        if source in channels:
            sums, counts = channels[source]
            item["per_channel_mae"] = (sums / counts.clamp_min(1)).tolist()
            for name, indices in spec.metric_groups.items():
                ix = torch.as_tensor(indices, dtype=torch.long)
                item[name + "_mae"] = float(sums[ix].sum() / counts[ix].sum().clamp_min(1))
        result[source] = item
    return result


@torch.no_grad()
def paired_validate(model, backbone, loader, registry, project_root, device, *, seed=45):
    """Compare original/mirrored reconstruction and prediction equivariance.

    Source adapters transform measurements and predicted native outputs using
    the same verified convention. Latents and arbitrary code coordinates are
    never flipped. Only eligible sources contribute; these diagnostics never
    select the best training checkpoint.
    """
    from dataclasses import replace

    from ..models.backbone import spatial_video_features

    model.eval()
    decoders = FrozenDecoders(registry, project_root, device)
    totals = defaultdict(lambda: defaultdict(float))
    for step, original in enumerate(loader):
        if not registry.specs[original.source].mirror:
            continue
        flags = [True] * len(original.episode_ids)
        mirrored = registry.augment(original, seed=seed, epoch=0, step=step, flags=flags)
        predictions = []
        for item in (original, mirrored):
            batch = move_batch(item, device)
            with torch.autocast(device.type, torch.bfloat16, enabled=device.type == "cuda"):
                features = spatial_video_features(backbone, item, device)
                output = forward_batch(model, features, batch)
            predictions.append(decoders.decode(model, output, batch).float())
        # The source adapter transforms the prediction in exactly the native
        # coordinate convention of its measurement. Do not guess latent flips.
        predicted_sample = replace(
            original, target=replace(original.target, values=predictions[0].cpu())
        )
        expected = registry.augment(
            predicted_sample, seed=seed, epoch=0, step=step, flags=flags
        ).target.values.to(device)
        row = totals[original.source]
        for name, prediction, item in zip(
            ("original", "mirrored"), predictions, (original, mirrored)
        ):
            target, mask = item.target.values.to(device), item.target.expanded_mask().to(device)
            row[name] += torch.where(mask, (prediction - target).abs(), 0.0).sum().item()
            row[name + "_cells"] += mask.sum().item()
        mask = mirrored.target.expanded_mask().to(device)
        row["equivariance"] += (
            torch.where(mask, (predictions[1] - expected).abs(), 0.0).sum().item()
        )
        row["equivariance_cells"] += mask.sum().item()
    return {
        source: {
            name: row[name] / max(1, row[name + "_cells"])
            for name in ("original", "mirrored", "equivariance")
        }
        for source, row in totals.items()
    }
