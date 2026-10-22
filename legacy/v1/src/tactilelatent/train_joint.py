"""Optional six-dataset image-only VAE training entry point.

This is a separate experiment from the established EgoTouch-only run. The
shared visual encoder predicts the same 32D Gaussian latent from images; a
source-specific decoder predicts each dataset's native tactile target. The
code does not claim dataset invariance merely because an encoder is shared.
"""

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .joint_data import JointClipDataset, SourceBatchSampler
from .joint_all_config import ALL_TARGETS
from .joint_model import JointTactileVAE
from .losses import objective, standard_normal_kl
from .model import DinoBackbone, MODEL_ID, REVISION
from .prepare_joint import TARGETS
from .train import MODEL_CONFIG_KEYS, features, kl_weight, make_scheduler
from .sam import SAMStep, rng_snapshot, restore_rng, check_storage_budget


def sam_replay_step(model, optimizer, sam, replay, accumulation, device):
    """Replay one accumulated batch with identical stochastic noise/masks.

    Features and targets are detached; only shared-model/decoder weights get
    gradients. Preserve the RNG position after the first pass, so replay does
    not change randomness for the next batch. Partial final batches retain
    the baseline's accumulation scaling.
    """
    after_first_pass = rng_snapshot()
    norm = sam.perturb()
    total = 0.
    try:
        for batch in replay:
            restore_rng(batch["rng"])
            with torch.autocast("cuda", torch.bfloat16, enabled=device.type == "cuda"):
                prediction = model(batch["features"], batch["mask"], batch["source"])
                loss = joint_loss(prediction, batch["target"], batch["valid"],
                                  batch["source"], batch["beta"])
            (loss / accumulation).backward()
            total += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    except BaseException:
        optimizer.zero_grad(set_to_none=True)
        raise
    finally:
        sam.restore()
        restore_rng(after_first_pass)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    sam.steps += 1
    return {"sam_first_gradient_norm": norm, "sam_perturbed_loss": total / len(replay)}


def joint_loss(output, target, valid, source, beta):
    """Use pressure-specific loss only where the label is a pressure grid."""
    if source == "egotouch":
        return objective({"pressure": output["pred"], "mu": output["mu"],
                          "logvar": output["logvar"]}, target, valid,
                         beta=beta, threshold=.1, contact_weight=3.,
                         tv_weight=.01, normalize_by_valid=True)

    # Taxel vectors and optical tactile RGB do not share EgoTouch's geometry
    # or contact threshold. Their normalized native values use ordinary
    # masked reconstruction; each source has its own decoder.
    safe = torch.where(valid, target, 0.)
    difference = output["pred"] - safe
    reconstruction = ((difference.square() + .5 * difference.abs()) * valid).sum()
    reconstruction = reconstruction / valid.sum().clamp_min(1)
    return reconstruction + beta * standard_normal_kl(output["mu"], output["logvar"])


@torch.no_grad()
def validate(model, backbone, loader, device):
    model.eval()
    errors = defaultdict(float)
    cells = defaultdict(int)
    for batch in loader:
        source = batch["source"][0]
        if any(name != source for name in batch["source"]):
            raise ValueError("Joint batch must contain one source")
        mask = batch["view_mask"].to(device)
        with torch.autocast("cuda", torch.bfloat16, enabled=device.type == "cuda"):
            output = model(features(backbone, batch, device), mask, source)
        target = batch["target"].to(device)
        valid = batch["target_mask"].to(device)
        safe = torch.where(valid, target, 0.)
        errors[source] += ((output["pred"].float() - safe).abs() * valid).sum().item()
        cells[source] += valid.sum().item()
    return {source: {"normalized_mae": errors[source] / max(cells[source], 1),
                     "valid_cells": cells[source]} for source in sorted(cells)}


def resolve_train_sources(cfg, target_registry, manifest_sources):
    """Select either an explicit source list or the established holdout setup."""
    if manifest_sources != set(target_registry):
        raise ValueError("Manifest source set differs from the selected target registry")
    heldout = cfg.get("heldout_source")
    explicit = cfg.get("train_sources")
    if explicit is not None:
        if heldout or not isinstance(explicit, list) or not explicit or len(explicit) != len(set(explicit)):
            raise ValueError("Explicit train_sources must be unique, nonempty, and have no heldout_source")
        if not set(explicit) <= manifest_sources:
            raise ValueError("Explicit training source is absent from the manifest")
        return sorted(explicit)
    if cfg.get("dataset_family") == "joint_all" and not heldout:
        raise ValueError("The twelve-source experiment requires a holdout or explicit train_sources")
    if heldout and heldout not in target_registry:
        raise ValueError("Held-out source must be in the manifest and target registry")
    if heldout == "egotouch" and cfg.get("init_checkpoint"):
        raise ValueError("An EgoTouch-pretrained checkpoint would leak the held-out EgoTouch source")
    return sorted(manifest_sources - ({heldout} if heldout else set()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/joint_six_vae32.json")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                        help="One epoch, two clips per source, separate smoke output")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    if args.smoke:
        cfg = {**cfg, "epochs": 1, "train_windows_per_source_per_epoch": 2,
               "validation_windows_per_source": 2, "workers": 0, "log_every": 1,
               "checkpoint_every": 0, "output": cfg["output"] + "_smoke"}
    optimizer_method = cfg.get("optimizer_method", "adamw")
    if optimizer_method not in ("adamw", "sam_adamw"):
        raise ValueError("Unknown optimizer_method")
    use_sam = optimizer_method == "sam_adamw"
    if use_sam and (cfg.get("init_checkpoint") or cfg.get("init_joint_checkpoint")):
        raise ValueError("This SAM comparison must start from fresh learned weights")
    if cfg.get("latent_objective") != "vae":
        raise ValueError("Joint training supports only the image-only VAE method")
    target_registry = ALL_TARGETS if cfg.get("dataset_family") == "joint_all" else TARGETS
    manifest_sources = {row["source"] for row in json.loads(Path(cfg["manifest"]).read_text())["episodes"]}
    heldout = cfg.get("heldout_source")
    train_sources = resolve_train_sources(cfg, target_registry, manifest_sources)
    train_targets = {name: target_registry[name] for name in train_sources}

    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    train_set = JointClipDataset(cfg["manifest"], split="train", sources=train_sources)
    val_set = JointClipDataset(cfg["manifest"], split="val", sources=train_sources)
    if set(train_set.indices_by_source) != set(train_sources) or set(val_set.indices_by_source) != set(train_sources):
        raise ValueError("Every training source needs both train and validation clips")
    train_batches = SourceBatchSampler(
        train_set, cfg["batch_size"], seed=cfg["seed"],
        max_windows_per_source=cfg["train_windows_per_source_per_epoch"],
        sampling_mode=cfg.get("sampling_mode", "random"),
    )
    val_batches = SourceBatchSampler(
        val_set, cfg["batch_size"], seed=cfg["seed"],
        max_windows_per_source=cfg["validation_windows_per_source"],
    )
    train_loader = DataLoader(train_set, batch_sampler=train_batches,
                              num_workers=cfg["workers"], pin_memory=True)
    val_loader = DataLoader(val_set, batch_sampler=val_batches,
                            num_workers=cfg["workers"], pin_memory=True)
    print(json.dumps({"event": "indexed", "train_windows": len(train_set),
                      "validation_windows": len(val_set),
                      "training_sources": train_sources, "heldout_source": heldout,
                      "train_batches_per_epoch": len(train_loader)}), flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = DinoBackbone().to(device)
    model = JointTactileVAE(input_dim=backbone.dim,
                            targets=train_targets,
                            **{key: cfg[key] for key in MODEL_CONFIG_KEYS}).to(device)
    output_dir = Path(cfg["output"])
    checkpoint = output_dir / "last.pt"
    initial = cfg.get("init_checkpoint")
    initial_joint = cfg.get("init_joint_checkpoint")
    if initial and initial_joint:
        raise ValueError("Choose either EgoTouch or joint initialization, not both")
    if initial and not checkpoint.exists():
        source = torch.load(initial, map_location="cpu", weights_only=False)
        if source["backbone_revision"] != REVISION or source["config"].get("latent_objective") != "vae":
            raise ValueError("Initialization requires a matching EgoTouch image-only VAE checkpoint")
        ignored = model.initialize_from_egotouch(source["model"])
        print(json.dumps({"event": "initialized", "checkpoint": initial,
                          "ignored_inactive_tensors": ignored}), flush=True)
    if initial_joint and not checkpoint.exists():
        source = torch.load(initial_joint, map_location="cpu", weights_only=False)
        previous = source["config"]
        if source["backbone_revision"] != REVISION or any(
            previous.get(key) != cfg.get(key)
            for key in ("dataset_family", "manifest", "heldout_source", "latent_objective",
                        *MODEL_CONFIG_KEYS)
        ):
            raise ValueError("Joint initialization must match the source set and model architecture")
        model.load_state_dict(source["model"], strict=True)
        print(json.dumps({"event": "initialized_joint", "checkpoint": initial_joint,
                          "source_completed_epoch": source["epoch"],
                          "optimizer_and_schedule": "new"}), flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"],
                                  weight_decay=cfg["weight_decay"])
    scheduler = make_scheduler(optimizer, cfg, len(train_loader))
    sam = SAMStep(optimizer, cfg.get("sam_rho", .05)) if use_sam else None
    output_dir.mkdir(parents=True, exist_ok=True)
    start_epoch = 0
    if checkpoint.exists() and not args.resume:
        raise FileExistsError(f"{checkpoint} exists; pass --resume or choose a new output")
    if args.resume:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if state["config"] != cfg or state["backbone_revision"] != REVISION:
            raise ValueError("Joint checkpoint/config mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        if sam:
            sam.load_state_dict(state["sam"])
        start_epoch = state["epoch"] + 1
        torch.set_rng_state(state["torch_rng"])
        if "python_rng" in state:
            random.setstate(state["python_rng"])
        if "numpy_rng" in state:
            np.random.set_state(state["numpy_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])

    (output_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    started = time.perf_counter()
    for epoch in range(start_epoch, cfg["epochs"]):
        train_batches.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        source_loss = defaultdict(float)
        source_count = defaultdict(int)
        replay = []
        sam_diagnostics = defaultdict(float)
        sam_updates = 0
        if sam:
            latent_sum = torch.zeros(cfg["latent_dim"], device=device)
            latent_cross = torch.zeros(cfg["latent_dim"], cfg["latent_dim"], device=device)
            latent_var_sum = torch.zeros((), device=device)
            latent_count = 0
        for step, batch in enumerate(train_loader):
            source = batch["source"][0]
            if any(name != source for name in batch["source"]):
                raise ValueError("Joint batch must contain one source")
            mask = batch["view_mask"].to(device)
            # Drop wrist views only when a chest/main view remains. A
            # FreeTacMan clip has just one wrist/scene view and must keep it.
            if mask[:, :, 0].all():
                keep_wrist = torch.rand(len(mask), 1, 2, device=device) >= cfg["wrist_dropout"]
                mask[:, :, 1:] &= keep_wrist
            target = batch["target"].to(device)
            valid = batch["target_mask"].to(device)
            beta = kl_weight(cfg, epoch + step / max(len(train_loader), 1))
            # DINO is frozen; compute once, then reuse exactly the same patches
            # in SAM's second pass rather than decoding/extracting RGB twice.
            with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, enabled=device.type == "cuda"):
                encoded = features(backbone, batch, device)
            stochastic_state = rng_snapshot() if use_sam else None
            with torch.autocast("cuda", torch.bfloat16, enabled=device.type == "cuda"):
                output = model(encoded, mask, source)
                loss = joint_loss(output, target, valid, source, beta)
            (loss / cfg["accumulation"]).backward()
            if use_sam:
                replay.append({"features": encoded.detach(), "mask": mask, "source": source,
                               "target": target, "valid": valid, "beta": beta,
                               "rng": stochastic_state})
                means = output["mu"].detach().float().reshape(-1, cfg["latent_dim"])
                latent_sum += means.sum(0)
                latent_cross += means.T @ means
                latent_var_sum += output["logvar"].detach().float().exp().sum()
                latent_count += len(means)
            source_loss[source] += loss.item()
            source_count[source] += 1

            if (step + 1) % cfg["accumulation"] == 0 or step + 1 == len(train_loader):
                if sam:
                    diagnostics = sam_replay_step(model, optimizer, sam, replay,
                                                  cfg["accumulation"], device)
                    for key, value in diagnostics.items():
                        sam_diagnostics[key] += value
                    sam_updates += 1
                    replay.clear()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if step % cfg["log_every"] == 0:
                print(json.dumps({"epoch": epoch, "batch": step, "source": source,
                                  "loss": loss.item(), "beta": beta,
                                  "lr": scheduler.get_last_lr()[0]}), flush=True)

        validation = validate(model, backbone, val_loader, device)
        result = {"epoch": epoch, "train_loss_by_source": {
            name: source_loss[name] / source_count[name] for name in sorted(source_count)},
            "validation": validation, "elapsed_seconds": time.perf_counter() - started}
        if sam:
            latent_average = latent_sum / latent_count
            covariance = latent_cross / latent_count - latent_average[:, None] * latent_average[None, :]
            spectrum = torch.linalg.eigvalsh(covariance.cpu().double()).clamp_min(0)
            proportions = spectrum / spectrum.sum().clamp_min(1e-12)
            effective_rank = (-(proportions * proportions.clamp_min(1e-12).log()).sum()).exp().item()
            result.update({"optimizer_method": optimizer_method, "sam_rho": sam.rho,
                           "sam_updates": sam_updates,
                           "latent": {"effective_rank_mu": effective_rank,
                                      "mean_coordinate_variance_mu": covariance.diag().mean().item(),
                                      "mean_predicted_variance": (latent_var_sum / (latent_count * cfg["latent_dim"])).item()},
                           **{key: value / max(sam_updates, 1)
                              for key, value in sam_diagnostics.items()}})
        print(json.dumps(result), flush=True)
        with (output_dir / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(result) + "\n")
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(), "epoch": epoch,
                 "config": cfg, "backbone": MODEL_ID, "backbone_revision": REVISION,
                 "torch_rng": torch.get_rng_state(),
                 "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                 "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}
        if sam:
            state["sam"] = sam.state_dict()
        temporary = checkpoint.with_suffix(".tmp")
        if sam:
            # AdamW checkpoint contains weights and two moment tensors. Check
            # conservatively including atomic-write overhead; never delete data.
            estimated_bytes = sum(p.numel() * p.element_size() for p in model.parameters()) * 3 + 10_000_000
            check_storage_budget(Path.cwd(), additional=estimated_bytes)
        torch.save(state, temporary)
        temporary.replace(checkpoint)
        from .viewer_checkpoints import save_viewer_epoch
        save_viewer_epoch(checkpoint, state)
        if cfg.get("checkpoint_every") and (epoch + 1) % cfg["checkpoint_every"] == 0:
            # Hard links retain a completed checkpoint without duplicating bytes.
            periodic = output_dir / f"epoch_{epoch + 1:03d}.pt"
            if not periodic.exists():
                periodic.hardlink_to(checkpoint)


if __name__ == "__main__":
    main()
