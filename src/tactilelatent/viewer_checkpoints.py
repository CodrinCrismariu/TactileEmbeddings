"""Small immutable inference snapshots for selecting every completed epoch."""
from pathlib import Path

import torch


def save_viewer_epoch(checkpoint, state):
    if not state["config"].get("save_viewer_epochs", True):
        return
    folder = Path(checkpoint).parent / "viewer_epochs"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"epoch_{state['epoch'] + 1:03d}.pt"
    if path.exists():
        return
    keys = ("model", "model_type", "model_config", "config", "provenance", "teachers",
            "epoch", "backbone", "backbone_revision", "validation")
    inference = {key: state[key] for key in keys if key in state}
    inference["inference_only"] = True
    from .sam import check_checkpoint_budget
    check_checkpoint_budget(inference)
    temporary = path.with_suffix(".tmp")
    torch.save(inference, temporary)
    temporary.replace(path)
