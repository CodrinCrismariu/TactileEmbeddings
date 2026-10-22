"""Fresh 11-source joint VAE training with a gradient-reversed source probe."""

import argparse
import copy
import json
import math
import random
import shutil
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset

from .evaluate_dataset_classifier import class_table, evaluate as evaluate_identity
from .frozen_probe import cached_latents, fit_head, head, select_indices, sha256
from .joint_all_config import ALL_TARGETS
from .joint_data import JointClipDataset, SourceBatchSampler
from .joint_model import JointTactileVAE
from .model import DinoBackbone, MODEL_ID, REVISION
from .train import MODEL_CONFIG_KEYS, features, kl_weight, make_scheduler
from .train_joint import joint_loss, validate


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, strength):
        ctx.strength = float(strength)
        return tensor.view_as(tensor)

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.strength * gradient, None


def reverse(tensor, strength):
    return GradientReverse.apply(tensor, strength)


def classification_loss(classifier, mu, logvar, source_index, strength, weight,
                        *, generator=None):
    """One mean and one sampled latent per frame; only encoder sees reversal."""
    noise = torch.randn(mu.shape, device=mu.device, dtype=mu.dtype, generator=generator)
    sampled = mu + noise * (.5 * logvar).exp()
    label = torch.full(mu.shape[:-1], source_index, dtype=torch.long, device=mu.device)
    mean_logits = classifier(reverse(mu, strength))
    sample_logits = classifier(reverse(sampled, strength))
    raw = .5 * (F.cross_entropy(mean_logits.flatten(0, 1), label.flatten()) +
                F.cross_entropy(sample_logits.flatten(0, 1), label.flatten()))
    accuracy = .5 * ((mean_logits.argmax(-1) == label).float().mean() +
                     (sample_logits.argmax(-1) == label).float().mean())
    return raw * weight, raw.detach(), accuracy.detach()


def source_weights(train_set, cap, sources):
    quotas = {source: min(len(train_set.indices_by_source[source]), cap) for source in sources}
    total = sum(quotas.values())
    return {source: total / (len(sources) * count) for source, count in quotas.items()}


def reversal_strength(epoch, state):
    if epoch == 0 or state["target"] == 0:
        return 0.
    progress = min(1., max(0., (epoch - state["ramp_start"] + 1) / state["ramp_epochs"]))
    return state["start"] + (state["target"] - state["start"]) * progress


def escalate(state, epoch, diagnostic_macro, threshold=.14, maximum=2.):
    previous = reversal_strength(epoch, state)
    updated = copy.deepcopy(state)
    if diagnostic_macro > threshold and previous < maximum:
        updated.update(start=previous, target=min(maximum, state["target"] * 2),
                       ramp_start=epoch + 1, ramp_epochs=2)
    return updated


def calibrate(backbone, model, classifier, dataset, sources, weights, cfg, device):
    """Aim for adversarial latent gradients at 20% of task gradients."""
    model.eval(); classifier.eval()
    ratios = {}
    rng = torch.Generator(device=device).manual_seed(cfg["seed"] + 901)
    for source_index, source in enumerate(sources):
        ids = dataset.indices_by_source[source][:2]
        batch = next(iter(DataLoader(Subset(dataset, ids), batch_size=2, num_workers=0)))
        mask = batch["view_mask"].to(device)
        target, valid = batch["target"].to(device), batch["target_mask"].to(device)
        with torch.autocast("cuda", torch.bfloat16, enabled=device.type == "cuda"):
            output = model(features(backbone, batch, device), mask, source, sample_latent=True)
            task = joint_loss(output, target, valid, source, cfg["beta_max"] / 10)
        latent = (output["mu"], output["logvar"])
        task_grad = torch.autograd.grad(task, latent, retain_graph=True, allow_unused=True)
        # GRL(1) flips the sign but preserves the norm used for calibration.
        with torch.autocast("cuda", torch.bfloat16, enabled=device.type == "cuda"):
            direct, _, _ = classification_loss(
                classifier, output["mu"], output["logvar"], source_index, 1.,
                weights[source], generator=rng)
        adversarial_grad = torch.autograd.grad(direct, latent, allow_unused=True)
        task_norm = math.sqrt(sum(gradient.float().square().sum().item()
                                  for gradient in task_grad if gradient is not None))
        class_norm = math.sqrt(sum(gradient.float().square().sum().item()
                                   for gradient in adversarial_grad if gradient is not None))
        if not math.isfinite(task_norm) or not math.isfinite(class_norm) or class_norm <= 0:
            raise FloatingPointError(f"Bad calibration gradient for {source}")
        ratios[source] = task_norm / class_norm
    median = float(np.median(list(ratios.values())))
    return max(.01, min(.5, .2 * median)), ratios


def fixed_diagnostic(backbone, model, sources, cfg, epoch, output_dir,
                     manifest_hash, device):
    """Fresh post-hoc probe; uses only train-split episodes, never final val."""
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if device.type == "cuda" else []
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    try:
        fitting, scoring = [], {}
        for source_index, source in enumerate(sources):
            dataset = JointClipDataset(cfg["manifest"], split="train", sources=[source])
            fit_ids = select_indices(dataset, "fit", cap=cfg["diagnostic_fit_clips"],
                                     seed=cfg["seed"] + source_index)
            score_ids = select_indices(dataset, "tune", cap=cfg["diagnostic_score_clips"],
                                       seed=cfg["seed"] + source_index)
            signature = f"{cfg['seed']}:epoch-{epoch}:source-{source}"
            folder = output_dir / "diagnostics" / f"epoch_{epoch + 1:02d}"
            fitting.append(cached_latents(backbone, model, cfg["manifest"], source,
                "train", fit_ids, folder / f"{source}_fit.pt", signature,
                manifest_hash, device, workers=cfg["workers"]))
            scoring[source] = cached_latents(backbone, model, cfg["manifest"], source,
                "train", score_ids, folder / f"{source}_score.pt", signature,
                manifest_hash, device, workers=cfg["workers"])
        probe = head(cfg["latent_dim"], len(sources), classification=True).to(device)
        fit_head(probe, class_table(fitting, device), None, F.cross_entropy,
                 fixed_epochs=cfg["diagnostic_probe_epochs"], seed=cfg["seed"] + epoch,
                 device=device)
        return evaluate_identity(probe, scoring, device, cfg["diagnostic_samples"],
                                 cfg["seed"] + epoch + 2026)
    finally:
        torch.set_rng_state(cpu_rng)
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(cuda_rng)
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)


def latent_statistics(total, outer, count):
    if count < 2:
        return {"pooled_variance": 0., "effective_rank": 0.}
    mean = total / count
    covariance = outer / count - torch.outer(mean, mean)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
    variance = eigenvalues.sum()
    probabilities = eigenvalues / variance.clamp_min(1e-12)
    rank = torch.exp(-(probabilities * probabilities.clamp_min(1e-12).log()).sum())
    return {"pooled_variance": variance.item(), "effective_rank": rank.item()}


def save_checkpoint(path, model, classifier, optimizer, scheduler, epoch, cfg,
                    reversal, diagnostics, record, device):
    state = {"model": model.state_dict(), "adversary": classifier.state_dict(),
             "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
             "epoch": epoch, "config": cfg, "backbone": MODEL_ID,
             "backbone_revision": REVISION, "reversal": reversal,
             "diagnostics": diagnostics, "last_record": record,
             "torch_rng": torch.get_rng_state(),
             "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
             "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)
    from .viewer_checkpoints import save_viewer_epoch
    save_viewer_epoch(path, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/joint_all_grl_fresh_8k_50ep.json")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    if cfg.get("init_checkpoint") is not None or cfg.get("init_joint_checkpoint") is not None:
        raise ValueError("This adversarial experiment must initialize fresh")
    if cfg["heldout_source"] != "ftp_vla_touch" or cfg["latent_objective"] != "vae":
        raise ValueError("Expected the 11-source VAE with ftp_vla_touch held out")
    sources = tuple(sorted(set(ALL_TARGETS) - {cfg["heldout_source"]}))
    manifest_sources = {row["source"] for row in json.loads(Path(cfg["manifest"]).read_text())["episodes"]}
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
    classifier = head(cfg["latent_dim"], len(sources), classification=True).to(device)
    optimizer = torch.optim.AdamW([
        {"params": model.parameters(), "lr": cfg["learning_rate"]},
        {"params": classifier.parameters(), "lr": cfg["classifier_learning_rate"]}],
        weight_decay=cfg["weight_decay"])
    scheduler = make_scheduler(optimizer, cfg, len(train_loader))
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
        if state["config"] != cfg or state["backbone_revision"] != REVISION:
            raise ValueError("Resume config/backbone mismatch")
        model.load_state_dict(state["model"], strict=True)
        classifier.load_state_dict(state["adversary"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = state["epoch"] + 1
        reversal, diagnostics = state["reversal"], state["diagnostics"]
        torch.set_rng_state(state["torch_rng"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
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
        model.train(); classifier.train()
        optimizer.zero_grad(set_to_none=True)
        task_sums, class_sums, accuracy_sums = defaultdict(float), defaultdict(float), defaultdict(float)
        source_counts = defaultdict(int)
        latent_sum = torch.zeros(cfg["latent_dim"], dtype=torch.float64, device=device)
        latent_outer = torch.zeros(cfg["latent_dim"], cfg["latent_dim"], dtype=torch.float64, device=device)
        latent_count = 0
        strength = reversal_strength(epoch, reversal)
        for step, batch in enumerate(train_loader):
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
                class_objective, class_raw, class_accuracy = classification_loss(
                    classifier, output["mu"], output["logvar"], sources.index(source),
                    strength, weights[source])
                loss = task + class_objective
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite objective at epoch {epoch}, batch {step}")
            (loss / cfg["accumulation"]).backward()
            task_sums[source] += task.item(); class_sums[source] += class_raw.item()
            accuracy_sums[source] += class_accuracy.item(); source_counts[source] += 1
            flat = output["mu"].detach().double().reshape(-1, cfg["latent_dim"])
            latent_sum += flat.sum(0)
            latent_outer += flat.T @ flat
            latent_count += len(flat)
            if (step + 1) % cfg["accumulation"] == 0 or step + 1 == len(train_loader):
                nn.utils.clip_grad_norm_(list(model.parameters()) + list(classifier.parameters()), 1.,
                                         error_if_nonfinite=True)
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
            if step % cfg["log_every"] == 0:
                print(json.dumps({"epoch": epoch, "batch": step, "source": source,
                    "task_loss": task.item(), "class_loss": class_raw.item(),
                    "class_accuracy": class_accuracy.item(), "grl_lambda": strength,
                    "beta": beta, "lr": scheduler.get_last_lr()[0]}), flush=True)
        validation = validate(model, backbone, val_loader, device)
        if epoch == 0 and cfg["epochs"] > 1:
            cpu_rng = torch.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state_all() if device.type == "cuda" else []
            base, ratios = calibrate(backbone, model, classifier, train_set, sources,
                                     weights, cfg, device)
            reversal.update(start=0., target=base, ramp_start=1, ramp_epochs=5)
            diagnostics["calibration"] = {"lambda_base": base, "task_to_class_gradient_ratio": ratios}
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
            reversal = escalate(reversal, epoch, macro,
                                cfg["diagnostic_threshold"], cfg["grl_max"])
        record = {"epoch": epoch, "train_loss_by_source": {
            source: task_sums[source] / source_counts[source] for source in sources},
            "classifier_loss_by_source": {
            source: class_sums[source] / source_counts[source] for source in sources},
            "classifier_accuracy_by_source": {
            source: accuracy_sums[source] / source_counts[source] for source in sources},
            "validation": validation, "grl_lambda": strength,
            "next_grl_target": reversal["target"],
            "latent": latent_statistics(latent_sum, latent_outer, latent_count),
            "diagnostic_macro_accuracy": None if diagnostic is None else macro,
            "elapsed_seconds": time.perf_counter() - started}
        print(json.dumps({"event": "epoch_complete", **record}), flush=True)
        save_checkpoint(checkpoint, model, classifier, optimizer, scheduler, epoch,
                        cfg, reversal, diagnostics, record, device)
        with (output_dir / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        if epoch + 1 in cfg["snapshot_epochs"]:
            shutil.copy2(checkpoint, output_dir / f"epoch_{epoch + 1:02d}.pt")
    print(json.dumps({"event": "complete", "epochs": cfg["epochs"],
                      "checkpoint": str(checkpoint)}), flush=True)


if __name__ == "__main__":
    main()
