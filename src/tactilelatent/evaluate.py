"""Evaluate one EgoTouch checkpoint on the official test splits."""
import argparse
import json
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from .data import RawEgoTouch
from .losses import metric_counts, summarize
from .model import DinoBackbone, EgoTouchLatentModel, MODEL_ID, REVISION, load_vae_weights
from .train import features


@torch.no_grad()
def evaluate_split(model, backbone, loader, device, view_mode, threshold, split):
    model.eval()
    counts = torch.zeros(7, dtype=torch.float64, device=device)
    started = time.perf_counter()
    for step, batch in enumerate(loader):
        mask = batch["view_mask"].to(device)
        if view_mode == "ego":
            mask[:, :, 1:] = False
        with torch.autocast("cuda", torch.bfloat16, enabled=device.type == "cuda"):
            output = model(features(backbone, batch, device), mask)
        counts += metric_counts(output["pressure"].float(), batch["target"].to(device),
                                batch["target_mask"].to(device), threshold)
        if step % 100 == 0:
            print(json.dumps({"event": "progress", "split": split, "views": view_mode,
                              "batch": step, "batches": len(loader),
                              "elapsed_seconds": time.perf_counter() - started}), flush=True)
    return summarize(counts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--splits", nargs="+", default=["test_seen", "test_unseen"])
    parser.add_argument("--views", nargs="+", choices=["ego", "all"], default=["ego", "all"])
    args = parser.parse_args()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = state["config"]
    if cfg.get("latent_objective") != "vae": raise ValueError("This evaluator supports image-only VAE checkpoints")
    if state["backbone"] != MODEL_ID or state["backbone_revision"] != REVISION:
        raise ValueError("Backbone mismatch")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = DinoBackbone().to(device)
    keys = ("width", "heads", "temporal_layers", "cross_view_layers", "latent_dim",
            "dropout", "stochastic")
    model = EgoTouchLatentModel(input_dim=backbone.dim,
        **{key: cfg[key] for key in keys}).to(device)
    load_vae_weights(model, state["model"])
    results = {"checkpoint": args.checkpoint, "checkpoint_epoch": state["epoch"], "splits": {}}
    common = dict(root=cfg["root"], frames=cfg["frames"], stride=cfg["stride"], size=cfg["size"])
    for split in args.splits:
        dataset = RawEgoTouch(split=split, **common)
        loader = DataLoader(dataset, batch_size=cfg["batch_size"], num_workers=cfg["workers"], pin_memory=True)
        results["splits"][split] = {"windows": len(dataset), "missing_episodes": len(dataset.missing_episodes)}
        for view_mode in args.views:
            metrics = evaluate_split(model, backbone, loader, device, view_mode,
                                     cfg["evaluation_threshold"], split)
            results["splits"][split][view_mode] = metrics
            print(json.dumps({"event": "result", "split": split, "views": view_mode,
                              "metrics": metrics}), flush=True)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(results, indent=2))
    temporary.replace(destination)
    print(json.dumps({"event": "complete", "output": str(destination)}), flush=True)


if __name__ == "__main__":
    main()
