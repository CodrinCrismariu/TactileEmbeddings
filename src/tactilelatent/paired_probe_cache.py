"""Reuse one frozen DINO forward for several frozen encoders on identical clips."""
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from .alignment_data import evaluation_dataset
from .train import features


@torch.no_grad()
def extract_split(backbone, models, manifest, source, split, indices, destinations,
                  checkpoint_hashes, manifest_hash, device, batch_size=8, workers=4, progress=None):
    """Write the existing frozen_probe cache format; preserve all targets/indices."""
    signatures = {name: {"checkpoint_sha256": checkpoint_hashes[name], "manifest_sha256": manifest_hash,
        "source": source, "split": split, "indices": indices, "with_targets": True} for name in models}
    pending = []
    for name in models:
        path = Path(destinations[name])
        if path.exists():
            saved = torch.load(path, map_location="cpu", weights_only=False)
            if saved.get("signature") != signatures[name]:
                raise ValueError(f"Stale paired feature cache: {path}")
        else:
            pending.append(name)
    if not pending:
        return
    dataset = evaluation_dataset(manifest, split=split, sources=[source], include_targets=True)
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, num_workers=workers,
                        pin_memory=device.type == "cuda", shuffle=False)
    moments = {name: {"mu": [], "logvar": []} for name in pending}
    targets, valid, episodes = [], [], []
    backbone.eval()
    for model in models.values():
        model.eval()
    for step, batch in enumerate(loader):
        if any(s != source for s in batch["source"]):
            raise ValueError("Mixed-source probe batch")
        with torch.autocast(device.type, torch.bfloat16, enabled=device.type == "cuda"):
            encoded = features(backbone, batch, device)
            mask = batch["view_mask"].to(device)
            for name in pending:
                mu, logvar = models[name].visual.encode(encoded, mask)
                moments[name]["mu"].append(mu.float().cpu())
                moments[name]["logvar"].append(logvar.float().cpu())
        targets.append(batch["target"].cpu()); valid.append(batch["target_mask"].cpu())
        episodes.extend(batch["episode_id"])
        if progress and step % 100 == 0:
            progress(split, step, len(loader))
    shared = {"target": torch.cat(targets), "valid": torch.cat(valid),
              "episode_id": episodes, "indices": indices}
    for name in pending:
        result = {"signature": signatures[name], **shared,
                  **{key: torch.cat(value) for key, value in moments[name].items()}}
        path = Path(destinations[name]); path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp"); torch.save(result, temporary); temporary.replace(path)
