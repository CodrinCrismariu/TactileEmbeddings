"""Evaluate image-conditioned latent and pressure uncertainty without training."""
import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import RawEgoTouch
from .losses import metric_counts, summarize
from .model import DinoBackbone, EgoTouchLatentModel, MODEL_ID, REVISION, load_vae_weights
from .train import features


LEVELS = (.5, .8, .9, .95)


def _new_stats(device):
    return {
        "point": torch.zeros(7, dtype=torch.float64, device=device),
        "valid": 0, "crps_sum": 0., "gaussian_nll_sum": 0.,
        "sample_mae_sum": 0., "best_window_mae_sum": 0., "windows": 0,
        "pressure_std_sum": 0., "latent_std_sum": 0., "latent_count": 0,
        "corr_n": 0, "corr_x": 0., "corr_y": 0., "corr_x2": 0.,
        "corr_y2": 0., "corr_xy": 0.,
        "pit": torch.zeros(10, dtype=torch.float64, device=device),
        "coverage": {str(level): [0., 0.] for level in LEVELS},
    }


def _update(stats, samples, target, mask, latent_logvar, threshold):
    k, b = samples.shape[:2]
    mean = samples.mean(0)
    std = samples.std(0, unbiased=False)
    stats["point"] += metric_counts(mean, target, mask, threshold)
    y = target[mask]
    s = samples[:, mask]
    m, u = mean[mask], std[mask]
    n = y.numel()
    stats["valid"] += n
    stats["pressure_std_sum"] += u.double().sum().item()
    stats["sample_mae_sum"] += (s - y).abs().double().sum().item() / k

    # Ensemble CRPS: E|X-y| - 0.5 E|X-X'|, using sorted samples in O(K log K).
    ordered = s.sort(dim=0).values
    weights = (2 * torch.arange(1, k + 1, device=s.device) - k - 1).to(s.dtype)[:, None]
    crps = (s - y).abs().mean(0) - (weights * ordered).sum(0) / (k * k)
    stats["crps_sum"] += crps.double().sum().item()
    variance = u.square().clamp_min(1e-6)
    nll = .5 * (torch.log(2 * torch.pi * variance) + (y - m).square() / variance)
    stats["gaussian_nll_sum"] += nll.double().sum().item()

    # Oracle metric: did any draw cover the observed pressure map for each clip?
    expanded_mask = mask.unsqueeze(0)
    per_sample_window = ((samples - target).abs() * expanded_mask).flatten(2).sum(2)
    denominator = mask.flatten(1).sum(1).clamp_min(1)
    stats["best_window_mae_sum"] += (per_sample_window / denominator).min(0).values.double().sum().item()
    stats["windows"] += b

    error = (m - y).abs().double(); uncertainty = u.double()
    stats["corr_n"] += n
    stats["corr_x"] += uncertainty.sum().item(); stats["corr_y"] += error.sum().item()
    stats["corr_x2"] += uncertainty.square().sum().item(); stats["corr_y2"] += error.square().sum().item()
    stats["corr_xy"] += (uncertainty * error).sum().item()
    pit_bins = ((s <= y).float().mean(0) * 10).long().clamp(max=9)
    stats["pit"] += torch.bincount(pit_bins, minlength=10)
    for level in LEVELS:
        tail = (1 - level) / 2
        lo, hi = torch.quantile(s, torch.tensor([tail, 1-tail], device=s.device), dim=0)
        covered = ((y >= lo) & (y <= hi)).sum().item()
        stats["coverage"][str(level)][0] += covered
        stats["coverage"][str(level)][1] += (hi - lo).double().sum().item()

    latent_std = (.5 * latent_logvar).exp()
    stats["latent_std_sum"] += latent_std.double().sum().item()
    stats["latent_count"] += latent_std.numel()


def _finish(stats, sample_count):
    n = max(stats["valid"], 1)
    cn = max(stats["corr_n"], 1)
    cov = stats["corr_xy"] - stats["corr_x"] * stats["corr_y"] / cn
    vx = stats["corr_x2"] - stats["corr_x"] ** 2 / cn
    vy = stats["corr_y2"] - stats["corr_y"] ** 2 / cn
    correlation = cov / math.sqrt(max(vx * vy, 1e-24))
    pit = (stats["pit"] / stats["pit"].sum().clamp_min(1)).cpu().tolist()
    result = {
        "samples": sample_count,
        "ensemble_mean_metrics": summarize(stats["point"]),
        "crps": stats["crps_sum"] / n,
        "gaussian_approx_nll": stats["gaussian_nll_sum"] / n,
        "expected_sample_mae": stats["sample_mae_sum"] / n,
        "best_of_samples_window_mae": stats["best_window_mae_sum"] / max(stats["windows"], 1),
        "mean_pressure_std": stats["pressure_std_sum"] / n,
        "mean_latent_std": stats["latent_std_sum"] / max(stats["latent_count"], 1),
        "pressure_std_absolute_error_pearson": correlation,
        "pit_histogram_10_bins": pit,
        "pit_uniform_l1_error": sum(abs(value - .1) for value in pit) / 2,
        "central_interval_calibration": {},
    }
    for level in LEVELS:
        covered, width = stats["coverage"][str(level)]
        result["central_interval_calibration"][str(level)] = {
            "empirical_coverage": covered / n,
            "coverage_error": covered / n - level,
            "mean_width": width / n,
        }
    return result


@torch.no_grad()
def evaluate(model, backbone, loader, device, view_mode, threshold, sample_count, seed):
    model.eval(); stats = _new_stats(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    for step, batch in enumerate(loader):
        mask = batch["view_mask"].to(device)
        if view_mode == "ego": mask[:, :, 1:] = False
        target = batch["target"].to(device); target_mask = batch["target_mask"].to(device)
        with torch.autocast("cuda", torch.bfloat16, enabled=device.type == "cuda"):
            visual = features(backbone, batch, device)
            out = model(visual, mask, sample_latent=False)
            mu, logvar = out["mu"], out["logvar"]
            noise = torch.randn((sample_count, *mu.shape),
                                generator=generator, device=device, dtype=mu.dtype)
            z = mu.unsqueeze(0) + noise * (.5*logvar).exp().unsqueeze(0)
            samples = model.decode_latent(z).float()
        _update(stats, samples, target, target_mask, logvar.float(), threshold)
        if step % 100 == 0:
            print(json.dumps({"event":"progress", "views":view_mode,
                              "batch":step, "batches":len(loader)}), flush=True)
    return _finish(stats, sample_count)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="test_unseen")
    parser.add_argument("--views", nargs="+", choices=("ego","all"), default=("ego","all"))
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.samples < 2: raise ValueError("Uncertainty evaluation requires at least two samples")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False); cfg = state["config"]
    if cfg.get("latent_objective") != "vae": raise ValueError("This evaluator supports image-only VAE checkpoints")
    if state["backbone"] != MODEL_ID or state["backbone_revision"] != REVISION:
        raise ValueError("Backbone mismatch")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone=DinoBackbone().to(device)
    keys=("width","heads","temporal_layers","cross_view_layers","latent_dim","dropout","stochastic")
    model=EgoTouchLatentModel(input_dim=backbone.dim,**{key:cfg[key] for key in keys}).to(device)
    load_vae_weights(model,state["model"])
    dataset=RawEgoTouch(cfg["root"],split=args.split,frames=cfg["frames"],
                        stride=cfg["stride"],size=cfg["size"])
    loader=DataLoader(dataset,batch_size=cfg["batch_size"],num_workers=cfg["workers"],pin_memory=True)
    results={"checkpoint":args.checkpoint,"checkpoint_epoch":state["epoch"],
             "model_kind":"image_only_vae",
             "split":args.split,"windows":len(dataset),"views":{}}
    for index, view_mode in enumerate(args.views):
        results["views"][view_mode]=evaluate(model,backbone,loader,device,view_mode,
            cfg["evaluation_threshold"],args.samples,args.seed+index)
    destination=Path(args.output); destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_suffix(destination.suffix+".tmp")
    temporary.write_text(json.dumps(results,indent=2)); temporary.replace(destination)
    print(json.dumps({"event":"complete","output":str(destination)}),flush=True)


if __name__ == "__main__": main()
