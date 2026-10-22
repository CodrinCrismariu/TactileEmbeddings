"""Fit a native sensor decoder while keeping the completed joint encoder frozen.

Legacy defaults retain the historical VLA_touch evaluation. Pass --source
ftp_motiontrans for the current sensor-disjoint experiment.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from .frozen_probe import (CHECKPOINT, MANIFEST, cached_latents, episode_partition,
                           fit_head, head, latent_draw, load_encoder, seed_all,
                           select_indices, sha256)
from .alignment_data import evaluation_dataset as JointClipDataset
from .joint_all_config import ALL_TARGETS


LEVELS = (.5, .8, .9, .95)


def training_fraction(cache, fraction, seed):
    """Nested episode subsets; keep fit and tune represented at every budget."""
    if not 0 < fraction <= 1:
        raise ValueError("Training fraction must be in (0,1]")
    if fraction == 1:
        return cache
    chosen = set()
    episodes = sorted(set(cache["episode_id"]))
    for partition in ("fit", "tune"):
        group = [e for e in episodes if episode_partition(e, seed) == partition]
        group.sort(key=lambda e: hashlib.sha256(f"fraction:{seed}:{e}".encode()).digest())
        chosen.update(group[:max(1, math.ceil(fraction * len(group)))])
    ids = [i for i, e in enumerate(cache["episode_id"]) if e in chosen]
    n = len(cache["episode_id"])
    return {key: value[ids] if torch.is_tensor(value) and value.ndim and len(value) == n else
            [value[i] for i in ids] if isinstance(value, list) and len(value) == n else value
            for key, value in cache.items()}


def episode_diagnostics(point, target, episodes, seed, bootstrap=5000):
    pred, truth = point.detach().cpu().numpy(), target.detach().cpu().numpy()
    centered_pred, centered_true, rows = [], [], []
    for episode in sorted(set(episodes)):
        ids = np.array([i for i, e in enumerate(episodes) if e == episode])
        p, y = pred[ids].reshape(-1, pred.shape[-1]), truth[ids].reshape(-1, truth.shape[-1])
        rows.append({"episode_id": episode, "absolute_sum": float(np.abs(p - y).sum()),
                     "count": int(y.size), "per_channel_mae": np.abs(p - y).mean(0).tolist()})
        centered_pred.append(p - p.mean(0)); centered_true.append(y - y.mean(0))
    p, y = np.concatenate(centered_pred), np.concatenate(centered_true)
    denominator = np.sqrt(np.square(p).sum(0) * np.square(y).sum(0))
    correlation = np.divide((p * y).sum(0), denominator, out=np.zeros_like(denominator), where=denominator > 0)
    totals, counts = np.array([r["absolute_sum"] for r in rows]), np.array([r["count"] for r in rows])
    rng = np.random.default_rng(seed)
    samples = rng.integers(len(rows), size=(bootstrap, len(rows)))
    estimates = totals[samples].sum(1) / counts[samples].sum(1)
    return {"episode_errors": rows, "episode_bootstrap_mae_ci95": np.quantile(estimates, [.025, .975]).tolist(),
            "within_episode_channel_correlations": correlation.tolist(),
            "bootstrap_unit": "episode", "bootstrap_replicates": bootstrap}


def regression_loss(pred, target):
    difference = pred - target
    return (difference.square() + .5 * difference.abs()).mean()


def frame_table(cache, clip_ids, device):
    return {"mu": cache["mu"][clip_ids].flatten(0, 1).to(device),
            "logvar": cache["logvar"][clip_ids].flatten(0, 1).to(device),
            "y": cache["target"][clip_ids].flatten(0, 1).flatten(1).to(device)}


def quantile_sorted(ordered, probability):
    position = probability * (ordered.shape[0] - 1)
    lo = math.floor(position); hi = math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def new_group():
    return {"n": 0, "crps": 0., "std": 0., "error": 0.,
            "sx": 0., "sy": 0., "sxx": 0., "syy": 0., "sxy": 0.,
            "interval": {str(level): {"covered": 0, "below": 0, "above": 0,
                                      "width": 0.} for level in LEVELS}}


def update_group(group, selected, y, pred_mean, pred_std, crps, intervals):
    n = int(selected.sum().item())
    if not n:
        return
    group["n"] += n
    x = pred_std[selected].double()
    error = (pred_mean[selected] - y[selected]).abs().double()
    group["crps"] += crps[selected].double().sum().item()
    group["std"] += x.sum().item()
    group["error"] += error.sum().item()
    group["sx"] += x.sum().item(); group["sy"] += error.sum().item()
    group["sxx"] += x.square().sum().item(); group["syy"] += error.square().sum().item()
    group["sxy"] += (x * error).sum().item()
    for level, (lo, hi) in intervals.items():
        values = group["interval"][str(level)]
        values["covered"] += int(((y >= lo) & (y <= hi) & selected).sum().item())
        values["below"] += int(((y < lo) & selected).sum().item())
        values["above"] += int(((y > hi) & selected).sum().item())
        values["width"] += (hi - lo)[selected].double().sum().item()


def finish_group(group):
    n = max(group["n"], 1)
    covariance = group["sxy"] - group["sx"] * group["sy"] / n
    vx = group["sxx"] - group["sx"] ** 2 / n
    vy = group["syy"] - group["sy"] ** 2 / n
    correlation = covariance / math.sqrt(vx * vy) if vx > 0 and vy > 0 else None
    result = {"valid_pixels": group["n"], "crps": group["crps"] / n,
              "mean_predictive_std": group["std"] / n,
              "mean_absolute_error": group["error"] / n,
              "std_error_pearson": correlation, "intervals": {}}
    for level in LEVELS:
        values = group["interval"][str(level)]
        coverage = values["covered"] / n
        result["intervals"][str(level)] = {
            "empirical_coverage": coverage, "coverage_error": coverage - level,
            "mean_width": values["width"] / n,
            "below_fraction": values["below"] / n,
            "above_fraction": values["above"] / n,
        }
    return result


def rgb_tile(array):
    if array.ndim == 3 and array.shape[0] == 3:
        array = array.transpose(1, 2, 0)
    return Image.fromarray((np.clip(array, 0, 1) * 255).astype(np.uint8)).resize(
        (160, 160), Image.Resampling.NEAREST)


def save_examples(examples, output):
    output.mkdir(parents=True, exist_ok=True)
    chosen = {"first": examples[0],
              "worst_error": max(examples, key=lambda x: x["clip_mae"]),
              "most_uncertain": max(examples, key=lambda x: x["std_mean"])}
    for name, row in chosen.items():
        if row["target"].shape[-2:] == (1, 1):
            # Torque is a channel vector, not an RGB sensor image. A labelled
            # table shows predictions, spread and misses without inventing a
            # spatial geometry or force units for these six readings.
            canvas = Image.new("RGB", (850, 45 + 25 * row["target"].shape[0]), "white")
            draw = ImageDraw.Draw(canvas)
            draw.text((8, 8), "Channel       target       z=mu       MC mean       MC std       90% miss", fill="black")
            for channel in range(row["target"].shape[0]):
                values = [row[key].reshape(-1)[channel] for key in ("target", "point", "mean", "std", "miss")]
                label = f"finger_{channel + 1}" if channel < 4 else f"thumb_{channel - 3}"
                draw.text((8, 40 + 25 * channel), label + "      " + "      ".join(f"{float(v):.5f}" for v in values), fill="black")
            canvas.save(output / f"{name}.png")
            continue
        canvas = Image.new("RGB", (160 * 5, 185), "white")
        draw = ImageDraw.Draw(canvas)
        items = [("target", row["target"]), ("z=mu", row["point"]),
                 ("MC mean", row["mean"]), ("MC std x4", row["std"] * 4),
                 ("90% miss", row["miss"].astype(np.float32))]
        for column, (label, value) in enumerate(items):
            canvas.paste(rgb_tile(value), (column * 160, 25))
            draw.text((column * 160 + 4, 5), label, fill="black")
        canvas.save(output / f"{name}.png")


@torch.no_grad()
def evaluate(network, train, validation, device, samples, seed, output,
             target_shape=(3, 32, 32)):
    network.eval()
    mu = validation["mu"].to(device)
    logvar = validation["logvar"].to(device)
    target = validation["target"].to(device).flatten(2)
    valid = validation["valid"].to(device).flatten(2)
    if not valid.all():
        raise ValueError("This evaluation requires fully valid target readings")
    if tuple(validation["target"].shape[2:]) != tuple(target_shape):
        raise ValueError("Target layout does not match evaluation target_shape")
    baseline = train["target"].to(device).mean((0, 1)).flatten()
    point = torch.cat([network(chunk) for chunk in mu.flatten(0, 1).split(512)]).reshape_as(target)
    absolute = (point - target).abs()
    high_threshold = torch.quantile(absolute[valid], .75).item()
    count = valid.sum().item()
    groups = {"all": new_group(), "high_error_quartile": new_group(),
              "other_pixels": new_group()}
    total = {"point_abs": absolute[valid].double().sum().item(),
             "point_sq": ((point - target).square()[valid]).double().sum().item(),
             "baseline_abs": ((target - baseline).abs()[valid]).double().sum().item(),
             "ensemble_abs": 0., "ensemble_sq": 0., "sample_abs": 0.,
             "best_clip_abs": 0.}
    pit = torch.zeros(10, dtype=torch.long, device=device)
    rng = torch.Generator(device=device).manual_seed(seed)
    examples = []
    # Keep historical aggregate keys for evaluator compatibility. For torque,
    # valid_pixels means scalar channel observations, not physical pixels.
    channel_groups = [new_group() for _ in range(target_shape[0])] if tuple(target_shape[1:]) == (1, 1) else []
    for start in range(0, len(mu), 2):
        end = min(start + 2, len(mu))
        z = latent_draw(mu[start:end].unsqueeze(0).expand(samples, -1, -1, -1),
                        logvar[start:end].unsqueeze(0).expand(samples, -1, -1, -1), rng)
        draws = torch.cat([network(chunk) for chunk in z.reshape(-1, z.shape[-1]).split(512)])
        draws = draws.reshape(samples, end - start, mu.shape[1], -1)
        y = target[start:end]
        mask = valid[start:end]
        mean = draws.mean(0); std = draws.std(0, unbiased=False)
        ordered = draws.sort(dim=0).values
        weights = (2 * torch.arange(1, samples + 1, device=device) - samples - 1).view(-1, 1, 1, 1)
        crps = (draws - y).abs().mean(0) - (weights * ordered).sum(0) / (samples * samples)
        intervals = {level: (quantile_sorted(ordered, (1 - level) / 2),
                             quantile_sorted(ordered, (1 + level) / 2)) for level in LEVELS}
        selected = {"all": mask,
                    "high_error_quartile": mask & (absolute[start:end] >= high_threshold),
                    "other_pixels": mask & (absolute[start:end] < high_threshold)}
        for name, region in selected.items():
            update_group(groups[name], region, y, mean, std, crps, intervals)
        for channel, group in enumerate(channel_groups):
            channel_mask = torch.zeros_like(mask)
            channel_mask[..., channel] = mask[..., channel]
            update_group(group, channel_mask, y, mean, std, crps, intervals)
        total["ensemble_abs"] += (mean - y).abs()[mask].double().sum().item()
        total["ensemble_sq"] += (mean - y).square()[mask].double().sum().item()
        total["sample_abs"] += (draws - y).abs().double().sum().item() / samples
        clip_errors = (draws - y).abs().mean(dim=(2, 3))  # [draw, clip]
        total["best_clip_abs"] += clip_errors.min(0).values.double().sum().item()
        bin_ids = ((draws <= y).float().mean(0) * 10).long().clamp(max=9)
        pit += torch.bincount(bin_ids[mask], minlength=10)
        for local in range(end - start):
            frame = mu.shape[1] // 2
            lo, hi = intervals[.9]
            miss = ((y[local, frame] < lo[local, frame]) |
                    (y[local, frame] > hi[local, frame])).reshape(*target_shape)
            examples.append({"target": y[local, frame].reshape(*target_shape).cpu().numpy(),
                             "point": point[start + local, frame].reshape(*target_shape).cpu().numpy(),
                             "mean": mean[local, frame].reshape(*target_shape).cpu().numpy(),
                             "std": std[local, frame].reshape(*target_shape).cpu().numpy(),
                             "miss": miss.cpu().numpy(),
                             "clip_mae": absolute[start + local].mean().item(),
                             "std_mean": std[local].mean().item()})
    save_examples(examples, output / "examples")
    result = {"validation_clips": len(mu), "sample_count": samples,
            "deterministic_mae": total["point_abs"] / count,
            "deterministic_mse": total["point_sq"] / count,
            "ensemble_mean_mae": total["ensemble_abs"] / count,
            "ensemble_mean_mse": total["ensemble_sq"] / count,
            "mean_image_baseline_mae": total["baseline_abs"] / count,
            "expected_sample_mae": total["sample_abs"] / count,
            "best_of_samples_clip_mae": total["best_clip_abs"] / len(mu),
            "high_error_threshold": high_threshold,
            "regions": {name: finish_group(group) for name, group in groups.items()},
            "pit_histogram_10_bins": (pit / pit.sum()).cpu().tolist()}
    if channel_groups:
        result["mean_target_baseline_mae"] = result["mean_image_baseline_mae"]
        result["per_channel"] = {}
        for channel, group in enumerate(channel_groups):
            label = f"finger_{channel + 1}" if channel < 4 else f"thumb_{channel - 3}"
            values = finish_group(group)
            values["deterministic_mae"] = absolute[..., channel].mean().item()
            values["mean_target_baseline_mae"] = (target[..., channel] - baseline[channel]).abs().mean().item()
            result["per_channel"][label] = values
        result["metric_space"] = "normalized torque: 0.5 + 0.5*tanh(raw/1000); not verified SI units"
        if "episode_id" in validation:
            result.update(episode_diagnostics(point, target, validation["episode_id"], seed))
            torch.save({"point": point.cpu(), "target": target.cpu(), "episode_id": validation["episode_id"]},
                       output / "point_predictions.pt")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--source", default="ftp_vla_touch", choices=("ftp_vla_touch", "ftp_motiontrans"))
    parser.add_argument("--manifest", default=MANIFEST)
    parser.add_argument("--output", default="processing/evaluations/vla_touch_frozen_decoder")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[512],
                        help="Decoder hidden widths; original baseline is 512")
    parser.add_argument("--cache-dir", help="Reuse compatible frozen feature caches without copying")
    parser.add_argument("--train-fraction", type=float, default=1.,
                        help="Nested episode-level label budget; feature caches remain reusable")
    parser.add_argument("--allow-intermediate", action="store_true",
                        help="Explicitly permit a completed epoch before the final epoch")
    parser.add_argument("--overwrite", action="store_true",
                        help="Explicitly replace a completed evaluation in --output")
    args = parser.parse_args()
    if args.samples < 2:
        parser.error("--samples must be at least 2")
    if any(width < 1 for width in args.hidden_dims):
        parser.error("--hidden-dims must be positive")
    if not 0 < args.train_fraction <= 1:
        parser.error("--train-fraction must be in (0,1]")
    seed_all(args.seed)
    output = Path(args.output)
    if (output / "results.json").exists() and not args.overwrite:
        raise FileExistsError(f"Completed evaluation exists: {output / 'results.json'}")
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_hash, manifest_hash = sha256(args.checkpoint), sha256(args.manifest)
    backbone, encoder, state = load_encoder(args.checkpoint, device, args.source)
    if state["epoch"] != state["config"]["epochs"] - 1 and not args.allow_intermediate:
        raise ValueError("Holdout evaluation requires the final completed checkpoint")
    latent_dim = state["config"]["latent_dim"]
    target_shape = ALL_TARGETS[args.source][1]
    output_dim = math.prod(target_shape)
    data = {}
    for split in ("train", "val"):
        dataset = JointClipDataset(args.manifest, split=split, sources=[args.source])
        indices = select_indices(dataset)
        data[split] = cached_latents(backbone, encoder, args.manifest, args.source,
            split, indices, Path(args.cache_dir or output) / f"{split}_features.pt", checkpoint_hash, manifest_hash,
            device, with_targets=True, workers=args.workers)
    del backbone, encoder
    train = training_fraction(data["train"], args.train_fraction, args.seed)
    if set(train["episode_id"]) & set(data["val"]["episode_id"]):
        raise ValueError("Training and validation episodes overlap")
    fit_ids = [i for i, episode in enumerate(train["episode_id"])
               if episode_partition(episode, args.seed) == "fit"]
    tune_ids = [i for i, episode in enumerate(train["episode_id"])
                if episode_partition(episode, args.seed) == "tune"]
    if not fit_ids or not tune_ids:
        raise ValueError("Episode-level fit/tune split is empty")
    fit, tune = frame_table(train, fit_ids, device), frame_table(train, tune_ids, device)
    decoder = head(latent_dim, output_dim, hidden_dims=args.hidden_dims).to(device)
    selection = fit_head(decoder, fit, tune, regression_loss, epochs=args.epochs,
                         patience=args.patience, seed=args.seed, device=device)
    seed_all(args.seed)
    decoder = head(latent_dim, output_dim, hidden_dims=args.hidden_dims).to(device)
    refit = fit_head(decoder, frame_table(train, range(len(train["mu"])), device), None,
                     regression_loss, fixed_epochs=selection["selected_epochs"],
                     seed=args.seed, device=device)
    metrics = evaluate(decoder, train, data["val"], device, args.samples,
                       args.seed + 2026, output, target_shape)
    deterministic = state["model_config"].get("latent_mode") == "deterministic" if "model_config" in state else False
    result = {"checkpoint": args.checkpoint, "checkpoint_sha256": checkpoint_hash,
              "checkpoint_epoch": state["epoch"], "manifest_sha256": manifest_hash,
              "checkpoint_is_final": state["epoch"] == state["config"]["epochs"] - 1,
              "source": args.source, "frozen_encoder": True, "latent": "deterministic" if deterministic else "mu_logvar",
              "head": "-".join(map(str, [latent_dim, *args.hidden_dims, output_dim])), "seed": args.seed,
              "head_parameters": sum(p.numel() for p in decoder.parameters()),
              "target_shape": target_shape,
              "train_fraction": args.train_fraction,
              "calibration_note": "Empirical calibration of predictions from this decoder; latent samples do not measure decoder-weight uncertainty or guarantee all plausible outcomes.",
              "fit_clips": len(fit_ids), "tune_clips": len(tune_ids),
              "train_clips": len(train["mu"]), "selected_epochs": selection["selected_epochs"],
              "selection": selection, "refit": refit, "validation": metrics}
    if deterministic:
        result["calibration_note"] = "Deterministic latent and evaluation head: latent draws are identical. Zero-width MC intervals are not calibrated predictive uncertainty."
    if sha256(args.checkpoint) != checkpoint_hash:
        raise ValueError("Checkpoint changed during evaluation; use a stable snapshot")
    torch.save({"head": decoder.cpu().state_dict(), "config": vars(args),
                "checkpoint_sha256": checkpoint_hash, "selected_epochs": selection["selected_epochs"]},
               output / "decoder.pt")
    (output / "results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"event": "complete", "output": str(output / "results.json"),
                      "validation": metrics}), flush=True)


if __name__ == "__main__":
    main()
