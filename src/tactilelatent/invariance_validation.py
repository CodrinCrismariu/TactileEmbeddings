"""Native reconstruction and relational validation in one visual forward pass."""
from collections import defaultdict
import math

import torch

from .alignment_model import FrozenSensorDecoders
from .joint_all_config import ALL_TARGETS
from .relational import HELDOUT, relational_loss
from .train import features
from .train_alignment import alignment_loss


@torch.no_grad()
def validate(model, backbone, loader, registry, device, floors):
    model.eval()
    decoders = FrozenSensorDecoders(registry, device)
    totals = defaultdict(lambda: defaultdict(float))
    for original in loader:
        source = original["source"][0]
        if source == HELDOUT or any(s != source for s in original["source"]):
            raise ValueError("Invalid representation-validation source")
        batch = {k: v.to(device) if torch.is_tensor(v) and k != "rgb" else v for k, v in original.items()}
        with torch.autocast(device.type, torch.bfloat16, enabled=device.type == "cuda"):
            output = model(features(backbone, original, device), batch["view_mask"], source, sample_latent=False)
        row = totals[source]
        target, mask = batch["target"].float(), batch["target_mask"].bool()
        if "sensor_code" in output:
            valid = batch["code_valid"].bool()
            difference = output["sensor_code"].float() - batch["code_target"]
            row["task"] += difference.square().mean(-1)[valid].sum().item()
            row["task_count"] += valid.sum().item()
            if source in floors:
                loss, stats = relational_loss(output["mu"], batch["code_target"], valid, floors[source])
                row["relational_sum"] += loss.item() * stats["relational_clips"]
                row["relational_clips"] += stats["relational_clips"]
                row["relational_skipped"] += stats["relational_skipped"]
            values = []
            for kind in (output["sensor_code"], batch["code_target"], torch.zeros_like(batch["code_target"])):
                raw = model.raw_code(source, kind).flatten(0, 1)
                values.append(torch.cat([decoders.decode(source, chunk) for chunk in raw.split(4)]).reshape_as(target))
            predicted, oracle, baseline = values
            row["oracle"] += ((oracle - target).abs() * mask).sum().item()
            row["baseline"] += ((baseline - target).abs() * mask).sum().item()
        else:
            predicted = output["pred"].float()
            _, loss, _ = alignment_loss(output, batch)
            row["task"] += loss.item() * mask.sum().item()
            row["task_count"] += mask.sum().item()
        difference = predicted - target
        if not torch.isfinite(predicted).all():
            raise FloatingPointError(f"Nonfinite native reconstruction: {source}")
        row["absolute"] += (difference.abs() * mask).sum().item()
        row["squared"] += (difference.square() * mask).sum().item()
        row["cells"] += mask.sum().item()
        adjacent = mask[:, 1:] & mask[:, :-1]
        delta = (predicted[:, 1:] - predicted[:, :-1]) - (target[:, 1:] - target[:, :-1])
        row["temporal"] += (delta.abs() * adjacent).sum().item()
        row["temporal_cells"] += adjacent.sum().item()
    result = {}
    for source, row in totals.items():
        count = max(row["cells"], 1)
        result[source] = {"supervision_loss": row["task"] / max(row["task_count"], 1),
                          "normalized_mae": row["absolute"] / count,
                          "temporal_change_mae": row["temporal"] / max(row["temporal_cells"], 1),
                          "valid_cells": int(row["cells"])}
        if source in model.adapters:
            result[source].update(code_mse=result[source]["supervision_loss"],
                frozen_ae_mae=row["oracle"] / count, mean_code_baseline_mae=row["baseline"] / count)
        if source in floors:
            result[source].update(relational_loss=row["relational_sum"] / row["relational_clips"] if row["relational_clips"] else None,
                relational_clips=int(row["relational_clips"]), relational_skipped=int(row["relational_skipped"]))
        if ALL_TARGETS[source][0] == "optical_rgb":
            result[source]["psnr"] = -10 * math.log10(max(row["squared"] / count, 1e-15))
    return result
