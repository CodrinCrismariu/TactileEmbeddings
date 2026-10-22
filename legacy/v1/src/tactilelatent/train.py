"""Train the image-only EgoTouch Gaussian-latent pressure predictor.

Read this file from main() downward. One epoch repeatedly:
  images -> frozen DINO patch features -> trainable latent model -> pressure;
  compare pressure to the measured grids -> backpropagate -> optimizer step.

Validation uses the Gaussian MEAN, not a random sample. No tactile pressure
or hand pose is ever given to the visual model as an input. See
ARCHITECTURE_GUIDE.md for every model tensor shape and TRAINING_GUIDE.md for
an explanation of the training settings.
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .data import RawEgoTouch
from .losses import metric_counts, objective, standard_normal_kl, summarize
from .model import (
    MODEL_ID,
    REVISION,
    DinoBackbone,
    EgoTouchLatentModel,
    load_vae_weights,
)


MODEL_CONFIG_KEYS = (
    "width", "heads", "temporal_layers", "cross_view_layers",
    "latent_dim", "dropout", "stochastic",
)


def features(backbone, batch, device):
    """Encode RGB frames into [B,T,3,196,768] frozen DINO patch features.

    The DataLoader gives RGB as [B,T,3 cameras,3 colors,H,W]. DINO accepts
    ordinary images [images,3,H,W], so we flatten batch/time/camera first.
    Image chunks of eight keep backbone peak memory manageable. Missing
    cameras get zero features and are later ignored by the model's view mask.

    Note: training's random wrist dropout happens *after* the data loader's
    availability mask. DINO may still encode a dropped wrist, but the model
    masks its contribution; this preserves the previous training behavior.
    """
    rgb = batch["rgb"].to(device)
    batch_size, times, views, colors, height, width = rgb.shape
    available = batch["view_mask"].to(device).reshape(-1)
    images = rgb.reshape(-1, colors, height, width)[available]
    encoded = torch.cat([backbone(chunk) for chunk in images.split(8)])

    # Restore the original camera slots after encoding only available images.
    result = encoded.new_zeros(batch_size * times * views, *encoded.shape[1:])
    result[available] = encoded
    return result.reshape(batch_size, times, views, *encoded.shape[1:])


def select(dataset, limit):
    """Optionally take evenly spaced windows for a smaller development run."""
    if limit is None or len(dataset) <= limit:
        return dataset
    indices = np.linspace(0, len(dataset) - 1, limit, dtype=int).tolist()
    return Subset(dataset, indices)


@torch.no_grad()
def evaluate(model, backbone, loader, device, view_mode, threshold):
    """Evaluate mean-latent pressure predictions against a zero baseline.

    In ego mode only the chest camera is available. In all mode the three
    cameras can contribute. Both modes use the SAME trained model weights.
    """
    model.eval()
    counts = torch.zeros(7, dtype=torch.float64, device=device)
    zero_counts = counts.clone()

    for batch in loader:
        view_mask = batch["view_mask"].to(device)
        if view_mode == "ego":
            view_mask[:, :, 1:] = False

        with torch.autocast("cuda", torch.bfloat16, enabled=device.type == "cuda"):
            output = model(features(backbone, batch, device), view_mask)

        target = batch["target"].to(device)
        valid = batch["target_mask"].to(device)
        counts += metric_counts(output["pressure"].float(), target, valid, threshold)
        zero_counts += metric_counts(torch.zeros_like(target), target, valid, threshold)

    return {"model": summarize(counts), "zero": summarize(zero_counts)}


def kl_weight(cfg, epoch_progress):
    """Increase the KL regularizer gradually so reconstruction learns first."""
    maximum = cfg.get("beta_max", cfg.get("beta", 0.0))
    warmup_epochs = cfg.get("beta_warmup_epochs", 0.0)
    if warmup_epochs <= 0:
        return maximum
    return maximum * min(1.0, epoch_progress / warmup_epochs)


def make_scheduler(optimizer, cfg, batches_per_epoch):
    """Linear learning-rate warmup, then cosine decay to the configured floor.

    The scheduler advances once per OPTIMIZER UPDATE, not once per microbatch.
    There are ceil(microbatches / accumulation) updates in each epoch.
    """
    updates_per_epoch = math.ceil(batches_per_epoch / cfg["accumulation"])
    total_updates = updates_per_epoch * cfg["epochs"]
    warmup_updates = updates_per_epoch * cfg["warmup_epochs"]

    def schedule(update):
        if update < warmup_updates:
            return (update + 1) / max(1, warmup_updates)
        progress = min(1.0, (update - warmup_updates) / max(1, total_updates - warmup_updates))
        floor = cfg["min_learning_rate"] / cfg["learning_rate"]
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def train_one_epoch(model, backbone, loader, optimizer, scheduler, cfg, device, epoch):
    """Run all clips once; return average loss per microbatch.

    A microbatch is two clips in the active config. Its loss is divided by 144
    before backward(), so gradients from 144 microbatches accumulate into one
    optimizer update (nominal effective batch size 288 clips). The final
    accumulation group may be shorter at the end of an epoch.
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0

    for step, batch in enumerate(loader):
        # The chest view stays on. Each wrist is independently hidden with
        # probability 0.3, consistently across this clip's eight times.
        view_mask = batch["view_mask"].to(device)
        keep_wrist = torch.rand(len(view_mask), 1, 2, device=device) >= cfg["wrist_dropout"]
        view_mask[:, :, 1:] &= keep_wrist

        # Autocast uses bfloat16 for suitable GPU operations, reducing memory
        # use while retaining fp32 model weights and optimizer state.
        with torch.autocast("cuda", torch.bfloat16, enabled=device.type == "cuda"):
            target = batch["target"].to(device)
            valid = batch["target_mask"].to(device)
            output = model(features(backbone, batch, device), view_mask)

            progress = epoch + step / max(1, len(loader))
            beta = kl_weight(cfg, progress)
            loss = objective(
                output, target, valid, beta=beta,
                threshold=cfg["pressure_threshold"],
                contact_weight=cfg["contact_weight"],
                tv_weight=cfg["tv_weight"],
                normalize_by_valid=True,
            )

            # These are diagnostics only. The objective recomputes its own KL.
            mu, logvar = output["mu"], output["logvar"]
            latent_std = (0.5 * logvar).exp().mean()
            kl = standard_normal_kl(mu, logvar)

        (loss / cfg["accumulation"]).backward()
        total_loss += loss.item()

        update_due = (step + 1) % cfg["accumulation"] == 0 or step + 1 == len(loader)
        if update_due:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        if step % cfg["log_every"] == 0:
            print(json.dumps({
                "epoch": epoch, "batch": step, "loss": loss.item(),
                "kl": kl.item(), "latent_std": latent_std.item(),
                "beta": beta, "lr": scheduler.get_last_lr()[0],
            }), flush=True)

    return total_loss / len(loader)


def save_checkpoint(path, model, optimizer, scheduler, epoch, cfg, generator, device):
    """Save enough state to resume an interrupted run at the next epoch."""
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "config": cfg,
        "backbone": MODEL_ID,
        "backbone_revision": REVISION,
        "torch_rng": torch.get_rng_state(),
        "loader_rng": generator.get_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
    }
    from .sam import check_checkpoint_budget
    check_checkpoint_budget(state)
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/egotouch_vae_image_only_latent32.json")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    if cfg.get("latent_objective") != "vae":
        raise ValueError("Only the image-only VAE objective is supported")

    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Index windows without decoding every frame now. The Dataset lazily reads
    # MP4 frames and pressure arrays when the DataLoader asks for a clip.
    data_options = dict(
        root=cfg["root"], frames=cfg["frames"], stride=cfg["stride"],
        size=cfg["size"], episode_limit=cfg.get("episode_limit"),
    )
    train_set = select(RawEgoTouch(split="train", **data_options), cfg.get("train_windows"))
    val_set = select(RawEgoTouch(split="val", **data_options), cfg.get("validation_windows"))
    print(json.dumps({
        "event": "indexed", "train_windows": len(train_set),
        "validation_windows": len(val_set),
    }), flush=True)

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_set, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["workers"], generator=generator, pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=cfg["batch_size"],
        num_workers=cfg["workers"], pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = DinoBackbone().to(device)
    model_options = {key: cfg[key] for key in MODEL_CONFIG_KEYS}
    model = EgoTouchLatentModel(input_dim=backbone.dim, **model_options).to(device)

    output_dir = Path(cfg["output"])
    checkpoint = output_dir / "last.pt"

    # A weight initialization is NOT a resume: it starts a new optimizer and
    # a fresh 20-epoch schedule. The old checkpoint itself is never changed.
    initial = cfg.get("init_checkpoint")
    if initial and not checkpoint.exists():
        source = torch.load(initial, map_location="cpu", weights_only=False)
        if source["backbone_revision"] != REVISION:
            raise ValueError("Initialization backbone mismatch")
        if source["config"].get("latent_objective") != "vae":
            raise ValueError("Initialization requires a VAE checkpoint")
        ignored = load_vae_weights(model, source["model"])
        print(json.dumps({
            "event": "initialized", "checkpoint": initial,
            "checkpoint_epoch": source["epoch"],
            "ignored_inactive_tensors": ignored,
        }), flush=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"], betas=(0.9, 0.999),
    )
    scheduler = make_scheduler(optimizer, cfg, len(train_loader))

    output_dir.mkdir(parents=True, exist_ok=True)
    start_epoch = 0
    if checkpoint.exists() and not args.resume:
        raise FileExistsError(f"{checkpoint} exists; pass --resume or choose a new output")
    if args.resume:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if state["config"] != cfg or state["backbone_revision"] != REVISION:
            raise ValueError("Checkpoint/config mismatch")
        load_vae_weights(model, state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = state["epoch"] + 1
        torch.set_rng_state(state["torch_rng"])
        generator.set_state(state["loader_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])

    (output_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    started = time.perf_counter()
    for epoch in range(start_epoch, cfg["epochs"]):
        average_loss = train_one_epoch(
            model, backbone, train_loader, optimizer, scheduler, cfg, device, epoch,
        )
        validation = {
            mode: evaluate(model, backbone, val_loader, device, mode, cfg["evaluation_threshold"])
            for mode in cfg["evaluation_views"]
        }
        record = {
            "epoch": epoch, "train_loss": average_loss,
            "validation": validation, "elapsed_seconds": time.perf_counter() - started,
        }
        print(json.dumps(record), flush=True)
        with (output_dir / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        save_checkpoint(checkpoint, model, optimizer, scheduler, epoch, cfg, generator, device)


if __name__ == "__main__":
    main()
