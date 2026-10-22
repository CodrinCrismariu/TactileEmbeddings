"""Fresh joint VAE training with a balanced, alternating dataset adversary.

Each group of tactile minibatches is preceded by several classifier-only
updates on recent detached latents. During the subsequent model update both
classifiers are frozen; the stronger one sends a reversed gradient to the
visual encoder. A newly initialized challenger joins the persistent head each
epoch, while independent episode-disjoint probes remain evaluation-only.
"""

import argparse
import json
import random
import shutil
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .alternating_adversary import (
    BalancedLatentBank, choose_challenger, fit_classifiers, reset_challenger,
    reversed_loss,
)
from .frozen_probe import head, sha256
from .joint_all_config import ALL_TARGETS
from .joint_data import JointClipDataset, SourceBatchSampler
from .joint_model import JointTactileVAE
from .model import DinoBackbone, MODEL_ID, REVISION
from .train import MODEL_CONFIG_KEYS, features, kl_weight, make_scheduler
from .train_joint import joint_loss, validate
from .train_joint_adversarial import (
    calibrate, escalate, fixed_diagnostic, latent_statistics, reversal_strength,
    source_weights,
)


def checkpoint_state(model, persistent, challenger, model_optimizer,
                     classifier_optimizer, scheduler, bank, epoch, cfg, reversal,
                     diagnostics, record, device):
    """Save every state needed to resume at the next epoch boundary."""
    return {
        "model": model.state_dict(), "adversary": persistent.state_dict(),
        "challenger": challenger.state_dict(),
        "optimizer": model_optimizer.state_dict(),
        "classifier_optimizer": classifier_optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "bank": bank.state_dict(),
        "epoch": epoch, "config": cfg, "backbone": MODEL_ID,
        "backbone_revision": REVISION, "reversal": reversal,
        "diagnostics": diagnostics, "last_record": record,
        "torch_rng": torch.get_rng_state(),
        "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
    }


def save_checkpoint(path, state):
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)
    from .viewer_checkpoints import save_viewer_epoch
    save_viewer_epoch(path, state)


def restore_checkpoint(state, model, persistent, challenger, model_optimizer,
                       classifier_optimizer, scheduler, bank, cfg, device):
    if state["config"] != cfg or state["backbone_revision"] != REVISION:
        raise ValueError("Resume config/backbone mismatch")
    model.load_state_dict(state["model"], strict=True)
    persistent.load_state_dict(state["adversary"], strict=True)
    challenger.load_state_dict(state["challenger"], strict=True)
    model_optimizer.load_state_dict(state["optimizer"])
    classifier_optimizer.load_state_dict(state["classifier_optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    bank.load_state_dict(state["bank"])
    torch.set_rng_state(state["torch_rng"])
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    if device.type == "cuda":
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    persistent.requires_grad_(False).eval()
    challenger.requires_grad_(False).eval()
    return state["epoch"] + 1, state["reversal"], state["diagnostics"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/joint_all_alternating_fresh_8k_50ep.json")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    if cfg.get("init_checkpoint") is not None or cfg.get("init_joint_checkpoint") is not None:
        raise ValueError("Alternating run must initialize fresh")
    if cfg["heldout_source"] != "ftp_vla_touch" or cfg["latent_objective"] != "vae":
        raise ValueError("Expected the 11-source VAE with ftp_vla_touch held out")
    sources = tuple(sorted(set(ALL_TARGETS) - {cfg["heldout_source"]}))
    manifest_sources = {row["source"] for row in
                        json.loads(Path(cfg["manifest"]).read_text())["episodes"]}
    if manifest_sources != set(ALL_TARGETS):
        raise ValueError("Manifest must contain the expected twelve sources")
    manifest_hash = sha256(cfg["manifest"])
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    train_set = JointClipDataset(cfg["manifest"], split="train", sources=sources)
    val_set = JointClipDataset(cfg["manifest"], split="val", sources=sources)
    sampler = SourceBatchSampler(train_set, cfg["batch_size"], seed=cfg["seed"],
        max_windows_per_source=cfg["train_windows_per_source_per_epoch"],
        sampling_mode=cfg["sampling_mode"])
    val_sampler = SourceBatchSampler(val_set, cfg["batch_size"], seed=cfg["seed"],
        max_windows_per_source=cfg["validation_windows_per_source"])
    train_loader = DataLoader(train_set, batch_sampler=sampler, num_workers=cfg["workers"],
                              pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_set, batch_sampler=val_sampler, num_workers=cfg["workers"],
                            pin_memory=torch.cuda.is_available())
    print(json.dumps({"event": "indexed", "sources": sources, "heldout": cfg["heldout_source"],
                      "train_batches_per_epoch": len(train_loader)}), flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = DinoBackbone().to(device).eval().requires_grad_(False)
    model = JointTactileVAE(input_dim=backbone.dim,
        targets={source: ALL_TARGETS[source] for source in sources},
        **{key: cfg[key] for key in MODEL_CONFIG_KEYS}).to(device)
    persistent = head(cfg["latent_dim"], len(sources), classification=True).to(device)
    challenger = head(cfg["latent_dim"], len(sources), classification=True).to(device)
    classifiers = (persistent, challenger)
    persistent.requires_grad_(False).eval()
    challenger.requires_grad_(False).eval()
    model_optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"],
                                         weight_decay=cfg["weight_decay"])
    classifier_optimizer = torch.optim.AdamW(
        list(persistent.parameters()) + list(challenger.parameters()),
        lr=cfg["classifier_learning_rate"], weight_decay=cfg["weight_decay"])
    scheduler = make_scheduler(model_optimizer, cfg, len(train_loader))
    bank = BalancedLatentBank(sources, cfg["bank_capacity"], cfg["bank_minimum"])
    weights = source_weights(train_set, cfg["train_windows_per_source_per_epoch"], sources)
    output_dir = Path(cfg["output"])
    checkpoint = output_dir / "last.pt"
    if checkpoint.exists() and not args.resume:
        raise FileExistsError(f"Run exists: {checkpoint}; use --resume")
    if args.resume and not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    output_dir.mkdir(parents=True, exist_ok=True)
    start_epoch, diagnostics = 0, {}
    reversal = {"start": 0., "target": 0., "ramp_start": 1, "ramp_epochs": 5}
    if args.resume:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        start_epoch, reversal, diagnostics = restore_checkpoint(
            state, model, persistent, challenger, model_optimizer,
            classifier_optimizer, scheduler, bank, cfg, device)
        metrics_path = output_dir / "metrics.jsonl"
        existing = [json.loads(line) for line in metrics_path.read_text().splitlines()] if metrics_path.exists() else []
        if existing and existing[-1]["epoch"] > state["epoch"]:
            raise ValueError("Metrics are ahead of the resumable checkpoint")
        if not existing or existing[-1]["epoch"] < state["epoch"]:
            with metrics_path.open("a") as stream:
                stream.write(json.dumps(state["last_record"]) + "\n")
    (output_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    started = time.perf_counter()
    for epoch in range(start_epoch, cfg["epochs"]):
        sampler.set_epoch(epoch)
        if epoch > 0:
            reset_challenger(challenger, classifier_optimizer)
            fit_classifiers((challenger,), classifier_optimizer, bank,
                steps=cfg["challenger_pretrain_steps"],
                clips_per_source=cfg["classifier_clips_per_source"], device=device)
        model.train()
        model_optimizer.zero_grad(set_to_none=True)
        task_sums, class_sums = defaultdict(float), defaultdict(float)
        accuracy_sums = defaultdict(float)
        head_accuracy_sums = [defaultdict(float), defaultdict(float)]
        source_counts = defaultdict(int)
        latent_sum = torch.zeros(cfg["latent_dim"], dtype=torch.float64, device=device)
        latent_outer = torch.zeros(cfg["latent_dim"], cfg["latent_dim"], dtype=torch.float64, device=device)
        latent_count, classifier_steps = 0, 0
        strength = reversal_strength(epoch, reversal)
        for step, batch in enumerate(train_loader):
            if step % cfg["accumulation"] == 0:
                classifier_steps += fit_classifiers(
                    classifiers, classifier_optimizer, bank,
                    steps=cfg["classifier_steps_per_update"],
                    clips_per_source=cfg["classifier_clips_per_source"], device=device)
            source = batch["source"][0]
            if any(name != source for name in batch["source"]):
                raise ValueError("Mixed-source batch")
            mask = batch["view_mask"].to(device)
            if mask[:, :, 0].all():
                keep_wrist = torch.rand(len(mask), 1, 2, device=device) >= cfg["wrist_dropout"]
                mask[:, :, 1:] &= keep_wrist
            target, valid = batch["target"].to(device), batch["target_mask"].to(device)
            beta = kl_weight(cfg, epoch + step / max(len(train_loader), 1))
            with torch.autocast("cuda", torch.bfloat16, enabled=device.type == "cuda"):
                output = model(features(backbone, batch, device), mask, source)
                task = joint_loss(output, target, valid, source, beta)
                mu, logvar = output["mu"], output["logvar"]
                sampled = mu + torch.randn_like(mu) * (.5 * logvar).exp()
                selected, class_losses, class_accuracies = choose_challenger(
                    classifiers, mu, sampled, sources.index(source))
                if strength > 0 and bank.ready():
                    adversarial = reversed_loss(classifiers[selected], mu, sampled,
                                                sources.index(source), strength)
                    loss = task + weights[source] * adversarial
                else:
                    loss = task
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite objective at epoch {epoch}, batch {step}")
            (loss / cfg["accumulation"]).backward()
            bank.add(source, mu, sampled)
            task_sums[source] += task.item()
            class_sums[source] += class_losses[selected]
            accuracy_sums[source] += class_accuracies[selected]
            for index in range(2):
                head_accuracy_sums[index][source] += class_accuracies[index]
            source_counts[source] += 1
            flat = mu.detach().double().reshape(-1, cfg["latent_dim"])
            latent_sum += flat.sum(0)
            latent_outer += flat.T @ flat
            latent_count += len(flat)
            if (step + 1) % cfg["accumulation"] == 0 or step + 1 == len(train_loader):
                nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                model_optimizer.step(); scheduler.step(); model_optimizer.zero_grad(set_to_none=True)
            if step % cfg["log_every"] == 0:
                print(json.dumps({"epoch": epoch, "batch": step, "source": source,
                    "task_loss": task.item(), "class_loss": class_losses[selected],
                    "class_accuracy": class_accuracies[selected],
                    "persistent_accuracy": class_accuracies[0],
                    "challenger_accuracy": class_accuracies[1],
                    "selected_head": selected, "bank_min_count": min(bank.counts().values()),
                    "classifier_steps": classifier_steps, "grl_lambda": strength,
                    "beta": beta, "lr": scheduler.get_last_lr()[0]}), flush=True)
        validation = validate(model, backbone, val_loader, device)
        if epoch == 0 and cfg["epochs"] > 1:
            cpu_rng = torch.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state_all() if device.type == "cuda" else []
            _, ratios = calibrate(backbone, model, persistent, train_set, sources,
                                  weights, cfg, device)
            base = max(cfg["grl_initial_min"], min(cfg["grl_max"],
                cfg["grl_initial_gradient_ratio"] * float(np.median(list(ratios.values())))))
            reversal.update(start=0., target=base, ramp_start=1, ramp_epochs=5)
            diagnostics["calibration"] = {"lambda_base": base,
                "task_to_class_gradient_ratio": ratios}
            torch.set_rng_state(cpu_rng)
            if device.type == "cuda":
                torch.cuda.set_rng_state_all(cuda_rng)
        diagnostic = None
        if epoch + 1 in cfg["diagnostic_epochs"]:
            diagnostic = fixed_diagnostic(backbone, model, sources, cfg, epoch, output_dir,
                                          manifest_hash, device)
            macro = max(diagnostic["deterministic"]["macro_accuracy"],
                        diagnostic["monte_carlo"]["macro_accuracy"])
            diagnostics[str(epoch + 1)] = {"metrics": diagnostic, "controller_macro": macro}
            reversal = escalate(reversal, epoch, macro, cfg["diagnostic_threshold"],
                                cfg["grl_max"])
        record = {"epoch": epoch,
            "train_loss_by_source": {s: task_sums[s] / source_counts[s] for s in sources},
            "classifier_loss_by_source": {s: class_sums[s] / source_counts[s] for s in sources},
            "classifier_accuracy_by_source": {s: accuracy_sums[s] / source_counts[s] for s in sources},
            "persistent_accuracy_by_source": {s: head_accuracy_sums[0][s] / source_counts[s]
                                               for s in sources},
            "challenger_accuracy_by_source": {s: head_accuracy_sums[1][s] / source_counts[s]
                                              for s in sources},
            "classifier_steps": classifier_steps, "bank_counts": bank.counts(),
            "validation": validation, "grl_lambda": strength,
            "next_grl_target": reversal["target"],
            "latent": latent_statistics(latent_sum, latent_outer, latent_count),
            "diagnostic_macro_accuracy": None if diagnostic is None else macro,
            "elapsed_seconds": time.perf_counter() - started}
        print(json.dumps({"event": "epoch_complete", **record}), flush=True)
        state = checkpoint_state(model, persistent, challenger, model_optimizer,
            classifier_optimizer, scheduler, bank, epoch, cfg, reversal, diagnostics,
            record, device)
        save_checkpoint(checkpoint, state)
        with (output_dir / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        if epoch + 1 in cfg["snapshot_epochs"]:
            shutil.copy2(checkpoint, output_dir / f"epoch_{epoch + 1:02d}.pt")
    print(json.dumps({"event": "complete", "epochs": cfg["epochs"],
                      "checkpoint": str(checkpoint)}), flush=True)


if __name__ == "__main__":
    main()
