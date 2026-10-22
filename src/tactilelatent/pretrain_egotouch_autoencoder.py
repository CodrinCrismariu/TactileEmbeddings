"""Tactile-only pretraining. Run with --help; see EGOTOUCH_AUTOENCODER.md.

Read pressure NPZs from the existing manifest. Never decode RGB or load DINO.
Use official episode splits, sample individual frames, and select by val MAE.
"""

import argparse
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .tactile_autoencoder import PRESETS, TactileAutoencoder


class PressureFrames(Dataset):
    """Index frames without loading video; cache a few decompressed episodes.

    Batch sampling below groups frames by episode to avoid decompressing a
    different NPZ for every frame. Episode order and frame order change each epoch.
    """
    def __init__(self, manifest, split, stride=2, cache_episodes=2):
        if stride < 1:
            raise ValueError("Frame stride must be positive")
        rows = json.loads(Path(manifest).read_text())["episodes"]
        ego = [r for r in rows if r["source"] == "egotouch"]
        ids, paths = set(), set()
        for row in ego:
            path = str(Path(row["target_path"]).resolve())
            if row["episode_id"] in ids or path in paths:
                raise ValueError("Duplicate EgoTouch episode or target path across manifest splits")
            ids.add(row["episode_id"]); paths.add(path)
        self.episodes = [r for r in ego if r["split"] == split]
        self.stride, self.cache_episodes = stride, cache_episodes
        lengths = [(int(r["frames"]) + stride - 1) // stride for r in self.episodes]
        if not lengths or min(lengths) < 1:
            raise ValueError(f"No valid EgoTouch episodes in {split}")
        self.offsets = np.concatenate(([0], np.cumsum(lengths)))
        self.cache = OrderedDict()

    def __len__(self):
        return int(self.offsets[-1])

    def __getitem__(self, index):
        episode = int(np.searchsorted(self.offsets, index, side="right") - 1)
        row = self.episodes[episode]
        if episode not in self.cache:
            with np.load(row["target_path"]) as file:
                array = np.stack([file[f"{side}_pressure_grid"] for side in ("left", "right")], axis=1)
            if array.shape != (row["frames"], 2, 21, 21):
                raise ValueError(f"Pressure shape/manifest mismatch: {row['target_path']}")
            self.cache[episode] = array.astype(np.float32, copy=False)
            while len(self.cache) > self.cache_episodes:
                self.cache.popitem(last=False)
        self.cache.move_to_end(episode)
        frame = int(index - self.offsets[episode]) * self.stride
        target = torch.from_numpy(self.cache[episode][frame].copy())
        valid = torch.isfinite(target)
        if not valid.any() or ((target[valid] < 0) | (target[valid] > 1)).any():
            raise ValueError(f"Invalid normalized pressure: {row['target_path']} frame {frame}")
        return {"target": torch.where(valid, target, 0.), "valid": valid}


def frame_batches(dataset, batch_size, cap, seed, shuffle):
    """Uniform frame subset, grouped into episode-local batches for efficient I/O.

    cap=0 uses every indexed frame. The same seed gives the same frames across
    architectures; validation always uses a fixed seed and fixed subset.
    """
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(len(dataset), cap, replace=False)) if 0 < cap < len(dataset) else None
    episodes = rng.permutation(len(dataset.episodes)) if shuffle else range(len(dataset.episodes))
    for episode in episodes:
        lo, hi = dataset.offsets[episode:episode + 2]
        indices = (np.arange(lo, hi) if selected is None else
                   selected[np.searchsorted(selected, lo):np.searchsorted(selected, hi)].copy())
        if shuffle:
            rng.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            yield indices[start:start + batch_size].tolist()


def reconstruction_loss(prediction, target, valid, contact_weight=3., threshold=.1, mae_weight=.5):
    """Existing pressure regression convention: weighted (MSE + 0.5 MAE).

    No KL or spatial smoothing: this is a deterministic reconstruction baseline.
    Divide by valid-cell count, matching the project's active pressure objective.
    """
    safe = torch.where(valid, target, 0.)
    error = prediction - safe
    weights = torch.where(safe > threshold, contact_weight, 1.) * valid
    return ((error.square() + mae_weight * error.abs()) * weights).sum() / valid.sum().clamp_min(1)


def autoencoder_loss(parts, target, valid, cfg):
    """For a hurdle decoder, supervise zero/nonzero and positive magnitude separately.

    A strictly positive measured value is nonzero, even if below the .05 contact
    metric threshold. Invalid taxels never become negative classifier examples.
    No hard gate is used in the training loss, avoiding dead zero-output gradients.
    """
    options = (cfg["contact_weight"], cfg["loss_threshold"], cfg.get("mae_weight", .5))
    if "nonzero_logits" not in parts:
        return reconstruction_loss(parts["soft"], target, valid, *options)
    nonzero = (torch.where(valid, target, 0.) > 0) & valid
    bce = F.binary_cross_entropy_with_logits(parts["nonzero_logits"], nonzero.float(), reduction="none")
    classification = (bce * valid).sum() / valid.sum().clamp_min(1)
    magnitude = reconstruction_loss(parts["magnitude"], target, nonzero, *options)
    return cfg.get("nonzero_loss_weight", 1.) * classification + magnitude


def selection_score(metrics, name):
    """Fixed validation criterion; independent of each trial's training loss.

    balanced_mae weights global, contact and high-pressure MAE equally. It is
    intentionally NOT a population-weighted average. Missing regions (e.g. a
    tiny smoke subset without high pressures) fall back to overall MAE.
    """
    if name == "mae":
        return metrics["mae"]
    if name != "balanced_mae":
        raise ValueError(f"Unknown selection metric: {name}")
    values = [metrics["mae"], metrics["contact_mae"], metrics["high_pressure_mae"]]
    return sum(metrics["mae"] if value is None else value for value in values) / 3


# Fill these when reading older autoencoder checkpoints: defaults reproduce
# their original constant-LR, MAE-selected training behavior.
EXTENDED_DEFAULTS = {"mae_weight": .5, "selection_metric": "mae", "scheduler": "constant",
                     "lr_patience": 8, "lr_factor": .5, "min_learning_rate": 1e-6,
                     "output_mode": "sigmoid", "zero_probability_threshold": .5, "nonzero_loss_weight": 1.}


def check_resume_config(previous, current):
    previous = {**EXTENDED_DEFAULTS, **previous}
    previous["model"] = {"output_mode": "sigmoid", "zero_probability_threshold": .5, **previous["model"]}
    if current["epochs"] < previous["epochs"]:
        raise ValueError("Resume may extend, but cannot reduce the configured epoch budget")
    if {k: v for k, v in previous.items() if k != "epochs"} != {
            k: v for k, v in current.items() if k != "epochs"}:
        raise ValueError("Resume configuration or manifest differs from checkpoint")


def run_epoch(model, loader, device, cfg, optimizer=None):
    model.train(optimizer is not None)
    # Global cell counts avoid bias from short batches and unequal valid masks.
    totals = torch.zeros(18, dtype=torch.float64, device=device)
    frames = 0
    example = None
    with torch.set_grad_enabled(optimizer is not None):
        for batch in loader:
            target, valid = batch["target"].to(device), batch["valid"].to(device)
            parts = model.decode_components(model.encode(target, valid))
            pred = parts["hard"]  # Consistent exact-zero decision for train/val metrics.
            loss = autoencoder_loss(parts, target, valid, cfg)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite autoencoder loss")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
            with torch.no_grad():
                error = (pred - target).abs()
                contact = (target > cfg["evaluation_threshold"]) & valid
                predicted_contact = (pred > cfg["evaluation_threshold"]) & valid
                high = (target >= .5) & valid
                true_zero = (target == 0) & valid
                predicted_zero = (pred == 0) & valid
                true_nonzero = (target > 0) & valid
                values = [loss.detach() * valid.sum(), valid.sum(),
                          error[valid].sum(), error[valid].square().sum(),
                          error[contact].sum(), contact.sum(), error[high].sum(), high.sum(),
                          (contact & predicted_contact).sum(),
                          (predicted_contact & ~contact).sum(),
                          (contact & ~predicted_contact).sum(), target[valid].abs().sum(),
                          true_zero.sum(), (true_zero & predicted_zero).sum(),
                          true_nonzero.sum(), (true_nonzero & predicted_zero).sum(),
                          predicted_zero.sum(), (contact & predicted_zero).sum()]
                totals += torch.stack(values).double()
                frames += len(target)
                if example is None:
                    example = (target[:4].cpu(), pred[:4].detach().cpu(), valid[:4].cpu())
    (loss_sum, cells, absolute, squared, contact_error, n_contact, high_error, n_high,
     tp, fp, fn, zero_error, n_zero, correct_zero, n_nonzero, missed_nonzero, predicted_zeros,
     missed_contact_by_zero) = totals.tolist()
    if cells == 0:
        raise ValueError("No valid pressure cells evaluated")
    metrics = {"frames": frames, "valid_cells": int(cells), "loss": loss_sum / cells,
               "mae": absolute / cells, "mse": squared / cells,
               "contact_mae": contact_error / n_contact if n_contact else None,
               "contact_cells": int(n_contact), "high_pressure_mae": high_error / n_high if n_high else None,
               "high_pressure_cells": int(n_high),
               "contact_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
               "contact_iou": tp / (tp + fp + fn) if tp + fp + fn else None,
               "zero_prediction_mae": zero_error / cells,
               "true_zero_cells": int(n_zero), "true_nonzero_cells": int(n_nonzero),
               "exact_zero_recall": correct_zero / n_zero if n_zero else None,
               "false_zero_rate_on_nonzero": missed_nonzero / n_nonzero if n_nonzero else None,
               "zero_output_miss_rate_on_contact": missed_contact_by_zero / n_contact if n_contact else None,
               "predicted_zero_fraction": predicted_zeros / cells}
    return metrics, example


def save_examples(example, path):
    """Rows: frames. Columns: true L/R, reconstructed L/R, absolute error L/R.

    All panels have the SAME grayscale scale [0,1]; missing cells are magenta.
    """
    target, pred, valid = example
    rows = []
    for y, p, mask in zip(target, pred, valid):
        panels = []
        for array in (y, p, (p - y).abs()):
            for hand in range(2):
                gray = (array[hand].clamp(0, 1).numpy() * 255).astype(np.uint8)
                rgb = np.repeat(gray[..., None], 3, axis=-1)
                rgb[~mask[hand].numpy()] = [255, 0, 255]
                panels.append(rgb)
        rows.append(np.concatenate(panels, axis=1))
    grid = Image.fromarray(np.concatenate(rows, axis=0))
    grid.resize((grid.width * 6, grid.height * 6), Image.Resampling.NEAREST).save(path)


def atomic_save(payload, path):
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--manifest", default="processing/joint_all/manifest.json")
    parser.add_argument("--output", required=True, help="New run directory; never reuse across ablations")
    parser.add_argument("--architecture", choices=("conv", "mlp"), default="conv")
    parser.add_argument("--size", choices=tuple(PRESETS), default="medium")
    parser.add_argument("--sensor-code-dim", "--latent-dim", dest="latent_dim", type=int, default=32,
                        help="Tactile autoencoder code dimension; independent of the shared visual latent")
    parser.add_argument("--channels", type=int, nargs="+", help="Override convolution widths, e.g. 16 32 64")
    parser.add_argument("--hidden-dims", type=int, nargs="+", help="Override MLP hidden widths, e.g. 512 256")
    parser.add_argument("--output-mode", choices=("sigmoid", "clamp", "hurdle"), default="sigmoid",
                        help="Sigmoid is the current default; clamp and hurdle retain earlier experiments")
    parser.add_argument("--zero-probability-threshold", type=float, default=.5,
                        help="Legacy hurdle mode only: minimum nonzero probability")
    parser.add_argument("--nonzero-loss-weight", type=float, default=1., help="Legacy hurdle mode only: BCE weight")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=.0001)
    parser.add_argument("--weight-decay", type=float, default=.0001)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--max-train-frames", type=int, default=100000, help="Fresh uniform subset each epoch; 0 = all")
    parser.add_argument("--max-val-frames", type=int, default=20000, help="Fixed subset for model selection; 0 = all")
    parser.add_argument("--contact-weight", type=float, default=3., help="1 removes contact weighting")
    parser.add_argument("--mae-weight", type=float, default=.5, help="Absolute-error weight alongside squared error")
    parser.add_argument("--selection-metric", choices=("mae", "balanced_mae"), default="mae",
                        help="Checkpoint/scheduler criterion; balanced_mae averages global/contact/high-pressure MAE")
    parser.add_argument("--scheduler", choices=("constant", "plateau"), default="constant")
    parser.add_argument("--lr-patience", type=int, default=8)
    parser.add_argument("--lr-factor", type=float, default=.5)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--loss-threshold", type=float, default=.1)
    parser.add_argument("--evaluation-threshold", type=float, default=.05)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume", action="store_true",
                        help="Resume last completed epoch; only --epochs may be increased")
    parser.add_argument("--smoke", action="store_true", help="One epoch, 16 train/val frames, CPU, separate _smoke output")
    args = parser.parse_args(argv)
    for name in ("epochs", "batch_size", "frame_stride", "latent_dim"):
        if getattr(args, name) < 1:
            parser.error(f"{name} must be positive")
    if min(args.workers, args.max_train_frames, args.max_val_frames, args.weight_decay) < 0:
        parser.error("Workers, caps and weight decay must be nonnegative")
    if args.learning_rate <= 0 or args.contact_weight <= 0:
        parser.error("Learning rate and contact weight must be positive")
    if args.nonzero_loss_weight <= 0 or not 0 < args.zero_probability_threshold < 1:
        parser.error("Nonzero loss weight must be positive; probability threshold must be in (0,1)")
    if args.mae_weight < 0 or args.lr_patience < 0 or not 0 < args.lr_factor < 1:
        parser.error("MAE weight and patience must be nonnegative; LR factor must be in (0,1)")
    if not 0 < args.min_learning_rate <= args.learning_rate:
        parser.error("Minimum learning rate must be positive and no greater than initial LR")
    if not 0 <= args.loss_threshold <= 1 or not 0 <= args.evaluation_threshold <= 1:
        parser.error("Pressure thresholds must be in [0,1]")
    if args.channels is not None and args.architecture != "conv":
        parser.error("--channels applies only to conv")
    if args.hidden_dims is not None and args.architecture != "mlp":
        parser.error("--hidden-dims applies only to mlp")
    if args.smoke:
        args.output += "_smoke"
        args.epochs, args.max_train_frames, args.max_val_frames = 1, 16, 16
        args.batch_size, args.workers, args.device = 8, 0, "cpu"
    return args


def main(argv=None):
    args = parse_args(argv)
    cfg = {k: v for k, v in vars(args).items() if k != "resume"}
    cfg["model"] = {"architecture": args.architecture, "latent_dim": args.latent_dim,
                    "channels": args.channels or PRESETS[args.size]["channels"],
                    "hidden_dims": args.hidden_dims or PRESETS[args.size]["hidden_dims"],
                    "output_mode": args.output_mode, "zero_probability_threshold": args.zero_probability_threshold}
    cfg["manifest_sha256"] = hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest()
    output = Path(args.output)
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError(f"Run directory is not empty: {output}; choose a new output or --resume")
    if args.resume and not (output / "last.pt").exists():
        raise FileNotFoundError(f"No checkpoint to resume: {output / 'last.pt'}")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    if device.type == "cpu":
        torch.set_num_threads(min(torch.get_num_threads(), 4))
    train = PressureFrames(args.manifest, "train", args.frame_stride)
    val = PressureFrames(args.manifest, "val", args.frame_stride)
    model = TactileAutoencoder(**cfg["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                 factor=args.lr_factor, patience=args.lr_patience, min_lr=args.min_learning_rate)
                 if args.scheduler == "plateau" else None)
    best, best_mae, start = float("inf"), float("inf"), 0
    if args.resume:
        checkpoint = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
        check_resume_config(checkpoint["config"], cfg)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
        torch.set_rng_state(checkpoint["torch_rng"])
        random.setstate(checkpoint["python_rng"]); np.random.set_state(checkpoint["numpy_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        start = checkpoint["epoch"] + 1
        best = checkpoint.get("best_score", checkpoint["best_val_mae"])
        best_mae = checkpoint["best_val_mae"]
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(cfg, indent=2))
    summary = {"model": cfg["model"], "encoder_parameters": sum(p.numel() for p in model.encoder.parameters()),
               "decoder_parameters": sum(p.numel() for p in model.decoder.parameters()),
               "train_frames_available": len(train), "val_frames_available": len(val),
               "train_episodes": len(train.episodes), "val_episodes": len(val.episodes), "device": str(device)}
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"event": "initialized", **summary}), flush=True)
    for epoch in range(start, args.epochs):
        began = time.perf_counter()
        learning_rate = optimizer.param_groups[0]["lr"]
        train_loader = DataLoader(train, batch_sampler=list(frame_batches(train, args.batch_size,
            args.max_train_frames, args.seed + epoch, True)), num_workers=args.workers)
        val_loader = DataLoader(val, batch_sampler=list(frame_batches(val, args.batch_size,
            args.max_val_frames, args.seed + 100000, False)), num_workers=args.workers)
        train_metrics, _ = run_epoch(model, train_loader, device, cfg, optimizer)
        val_metrics, example = run_epoch(model, val_loader, device, cfg)
        score = selection_score(val_metrics, args.selection_metric)
        improved = score < best
        best = min(best, score)
        best_mae = min(best_mae, val_metrics["mae"])
        if scheduler is not None:
            scheduler.step(score)
        record = {"epoch": epoch + 1, "train": train_metrics, "val": val_metrics,
                  "best_val_mae": best_mae, "selection_metric": args.selection_metric,
                  "selection_score": score, "best_score": best,
                  "learning_rate": learning_rate, "next_learning_rate": optimizer.param_groups[0]["lr"],
                  "seconds": time.perf_counter() - began}
        checkpoint = {"epoch": epoch, "config": cfg, "model": model.state_dict(),
                      "optimizer": optimizer.state_dict(), "best_val_mae": best_mae,
                      "best_score": best, "scheduler": scheduler.state_dict() if scheduler else None,
                      "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
                      "numpy_rng": np.random.get_state(),
                      "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}
        if improved:
            # Inference artifact contains both standalone modules, without optimizer state.
            atomic_save({"model_config": cfg["model"], "model": model.state_dict(),
                         "encoder": model.encoder.state_dict(), "decoder": model.decoder.state_dict(),
                         "epoch": epoch, "validation": val_metrics, "config": cfg}, output / "best.pt")
            (output / "best_metrics.json").write_text(json.dumps(record, indent=2))
            save_examples(example, output / "reconstructions.png")
        atomic_save(checkpoint, output / "last.pt")
        with (output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
