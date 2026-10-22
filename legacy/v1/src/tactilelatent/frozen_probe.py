"""Frozen joint-encoder features and common head training for evaluations."""

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from .joint_all_config import ALL_TARGETS
from .joint_data import JointClipDataset
from .joint_model import JointTactileVAE
from .model import DinoBackbone, MODEL_ID, REVISION
from .train import MODEL_CONFIG_KEYS, features
from .alignment_data import evaluation_dataset


CHECKPOINT = "processing/runs/joint_all_holdout_vla_8k_50ep/last.pt"
MANIFEST = "processing/joint_all/manifest.json"
SOURCES = tuple(sorted(set(ALL_TARGETS) - {"ftp_vla_touch"}))


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_encoder(checkpoint, device, heldout_source="ftp_vla_touch"):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = state["config"]
    if state["backbone"] != MODEL_ID or state["backbone_revision"] != REVISION:
        raise ValueError("Checkpoint DINO version differs from this code")
    deterministic = (state.get("model_type") == "joint_sensor_alignment_v1"
                     and cfg.get("latent_objective") == "deterministic"
                     and state["model_config"].get("latent_mode") == "deterministic"
                     and not cfg.get("stochastic"))
    if (heldout_source not in ALL_TARGETS or (cfg.get("latent_objective") != "vae" and not deterministic)
            or cfg.get("heldout_source") != heldout_source or cfg.get("train_sources") is not None):
        raise ValueError(f"Expected the joint VAE with {heldout_source} held out")
    targets = {name: target for name, target in ALL_TARGETS.items() if name != heldout_source}
    backbone = DinoBackbone().to(device).eval()
    if state.get("model_type") == "joint_sensor_alignment_v1":
        from .alignment_model import model_from_state
        from .aligned_frames import manifest_signature
        if manifest_signature(cfg["manifest"]) != state["provenance"]["data"]:
            raise ValueError("Alignment evaluation data provenance differs")
        model = model_from_state(state).to(device).eval()
        model._alignment_manifest = cfg["manifest"]
        model._alignment_data_signature = state["provenance"]["data"]
    else:
        model = JointTactileVAE(input_dim=backbone.dim, targets=targets,
                                **{key: cfg[key] for key in MODEL_CONFIG_KEYS}).to(device).eval()
        model.load_state_dict(state["model"], strict=True)
    backbone.requires_grad_(False)
    model.requires_grad_(False)
    return backbone, model, state


def episode_partition(episode_id, seed=45, tune_fraction=.1):
    """Stable episode-level train/tune split made only within manifest train."""
    digest = hashlib.sha256(f"probe:{seed}:{episode_id}".encode()).digest()
    return "tune" if int.from_bytes(digest[:8], "big") / 2**64 < tune_fraction else "fit"


def select_indices(dataset, partition=None, cap=None, seed=45):
    indices = [i for i, (episode, _) in enumerate(dataset.windows)
               if partition is None or episode_partition(dataset.episodes[episode]["episode_id"], seed) == partition]
    if cap is not None and len(indices) > cap:
        rng = np.random.default_rng(seed)
        indices = sorted(rng.choice(indices, cap, replace=False).tolist())
    if not indices:
        raise ValueError(f"No clips for partition={partition}")
    return indices


@torch.no_grad()
def cached_latents(backbone, model, manifest, source, split, indices, path,
                   checkpoint_hash, manifest_hash, device, with_targets=False,
                   batch_size=2, workers=4):
    """Cache mu/logvar, optionally tactile targets; never cache input RGB."""
    path = Path(path)
    if getattr(model, "model_type", None) == "joint_sensor_alignment_v1":
        from .aligned_frames import manifest_signature
        if manifest_signature(manifest) != model._alignment_data_signature:
            raise ValueError("Feature extraction requires the checkpoint's canonical manifest")
    signature = {"checkpoint_sha256": checkpoint_hash, "manifest_sha256": manifest_hash,
                 "source": source, "split": split, "indices": indices,
                 "with_targets": with_targets}
    if path.exists():
        cached = torch.load(path, map_location="cpu", weights_only=False)
        if cached.get("signature") != signature:
            raise ValueError(f"Stale or incompatible feature cache: {path}")
        return cached
    dataset = evaluation_dataset(manifest, split=split, sources=[source], include_targets=with_targets)
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=device.type == "cuda")
    mu, logvar, target, valid, episodes = [], [], [], [], []
    backbone.eval(); model.eval()
    for step, batch in enumerate(loader):
        if any(name != source for name in batch["source"]):
            raise ValueError("Mixed source batch")
        with torch.autocast("cuda", torch.bfloat16, enabled=device.type == "cuda"):
            m, v = model.visual.encode(features(backbone, batch, device),
                                        batch["view_mask"].to(device))
        mu.append(m.float().cpu()); logvar.append(v.float().cpu())
        episodes.extend(batch["episode_id"])
        if with_targets:
            target.append(batch["target"].cpu())
            valid.append(batch["target_mask"].cpu())
        if step % 100 == 0:
            print(json.dumps({"event": "extract", "source": source, "split": split,
                              "batch": step, "batches": len(loader)}), flush=True)
    cached = {"signature": signature, "mu": torch.cat(mu), "logvar": torch.cat(logvar),
              "episode_id": episodes, "indices": indices}
    if with_targets:
        cached["target"] = torch.cat(target)
        cached["valid"] = torch.cat(valid)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(cached, temporary)
    temporary.replace(path)
    return cached


def head(latent_dim, output_dim, dropout=.1, classification=False, hidden_dims=(512,)):
    """Build a probe MLP; the original single-512-layer head stays the default."""
    if not hidden_dims or any(width < 1 for width in hidden_dims):
        raise ValueError("Head needs positive hidden dimensions")
    layers = []
    previous = latent_dim
    for width in hidden_dims:
        layers.extend([nn.Linear(previous, width), nn.GELU(), nn.Dropout(dropout)])
        previous = width
    layers.append(nn.Linear(previous, output_dim))
    if not classification:
        layers.append(nn.Sigmoid())
    return nn.Sequential(*layers)


def latent_draw(mu, logvar, generator=None):
    noise = torch.randn(mu.shape, device=mu.device, dtype=mu.dtype, generator=generator)
    return mu + noise * (.5 * logvar).exp()


def fit_head(network, fit, tune, loss_function, *, epochs=100, patience=10,
             batch_size=256, lr=.001, weight_decay=.0001, seed=45,
             fixed_epochs=None, device=None):
    """Fit a head on mean and sampled z; tune loss never updates weights."""
    if device is None:
        device = next(network.parameters()).device
    optimizer = torch.optim.AdamW(network.parameters(), lr=lr, weight_decay=weight_decay)
    rng = torch.Generator(device=device).manual_seed(seed)
    order_rng = torch.Generator().manual_seed(seed)
    best_loss, best_epoch, best_state, stale = float("inf"), 0, None, 0
    count = fit["mu"].shape[0]
    history = []
    for epoch in range(fixed_epochs or epochs):
        network.train()
        perm = torch.randperm(count, generator=order_rng)
        total = 0.; seen = 0
        for ids in perm.split(batch_size):
            ids = ids.to(device)
            mu, logvar, y = (fit[key][ids] for key in ("mu", "logvar", "y"))
            z = latent_draw(mu, logvar, rng)
            loss = .5 * (loss_function(network(mu), y) + loss_function(network(z), y))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(ids); seen += len(ids)
        record = {"epoch": epoch, "train_loss": total / seen}
        if tune is not None:
            network.eval()
            with torch.no_grad():
                losses = []
                # A fixed seed keeps the tuning comparison stable across epochs.
                tune_rng = torch.Generator(device=device).manual_seed(seed + 991)
                for ids in torch.arange(tune["mu"].shape[0], device=device).split(batch_size):
                    mu, logvar, y = (tune[key][ids] for key in ("mu", "logvar", "y"))
                    z = latent_draw(mu, logvar, tune_rng)
                    value = .5 * (loss_function(network(mu), y) + loss_function(network(z), y))
                    losses.append((value.item(), len(ids)))
                tune_loss = sum(value * n for value, n in losses) / sum(n for _, n in losses)
            record["tune_loss"] = tune_loss
            if tune_loss < best_loss - 1e-5:
                best_loss, best_epoch, stale = tune_loss, epoch + 1, 0
                best_state = {key: value.detach().cpu().clone()
                              for key, value in network.state_dict().items()}
            else:
                stale += 1
        history.append(record)
        print(json.dumps({"event": "head_epoch", **record}), flush=True)
        if fixed_epochs is None and tune is not None and stale >= patience:
            break
    if tune is not None:
        network.load_state_dict(best_state)
    return {"selected_epochs": fixed_epochs or best_epoch, "history": history,
            "best_tune_loss": None if tune is None else best_loss}
