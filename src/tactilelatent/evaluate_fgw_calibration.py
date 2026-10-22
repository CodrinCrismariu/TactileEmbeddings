"""Separate episode-conformal evaluation using cached held-out sensor latents.

Heads are retrained with calibration episodes excluded, so this cannot reuse
the point-benchmark head that was refitted on all training episodes.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from .aligned_frames import manifest_signature, write_json
from .evaluate_holdout_decoder import frame_table, regression_loss
from .frozen_probe import fit_head, head, latent_draw, seed_all, sha256


def calibration_partitions(episodes, seed):
    unique = sorted(set(episodes), key=lambda e: hashlib.sha256(f"calibration:{seed}:{e}".encode()).digest())
    if len(unique) < 6:
        raise ValueError("At least six training episodes required for disjoint fit/tune/calibration")
    n = max(1, math.ceil(.15 * len(unique)))
    calibration, tune, fit = set(unique[:n]), set(unique[n:2 * n]), set(unique[2 * n:])
    return {name: [i for i, e in enumerate(episodes) if e in values]
            for name, values in (("fit", fit), ("tune", tune), ("calibration", calibration))}


def conformal_quantile(scores, level):
    values = np.sort(np.asarray(scores, dtype=np.float64))
    if not len(values) or not 0 < level < 1 or not np.isfinite(values).all():
        raise ValueError("Finite calibration scores and a probability level required")
    position = math.ceil((len(values) + 1) * level)
    return float(values[position - 1]) if position <= len(values) else float("inf")


@torch.no_grad()
def predictions(network, cache, indices, samples, seed, device):
    network.eval(); points, spreads = [], []
    rng = torch.Generator(device=device).manual_seed(seed)
    for lo in range(0, len(indices), 8):
        ids = indices[lo:lo + 8]
        mu, logvar = cache["mu"][ids].to(device), cache["logvar"][ids].to(device)
        point = network(mu)
        z = latent_draw(mu.unsqueeze(0).expand(samples, -1, -1, -1),
                        logvar.unsqueeze(0).expand(samples, -1, -1, -1), rng)
        draws = torch.cat([network(x) for x in z.reshape(-1, z.shape[-1]).split(512)])
        draws = draws.reshape(samples, len(ids), mu.shape[1], -1)
        points.append(point.cpu()); spreads.append(draws.std(0, unbiased=False).clamp_min(.001).cpu())
    return torch.cat(points), torch.cat(spreads)


def episode_scores(point, spread, truth, episodes):
    residual = ((point - truth).abs() / spread).numpy()
    return [float(residual[[i for i, e in enumerate(episodes) if e == episode]].max())
            for episode in sorted(set(episodes))]


def calibrated_metrics(point, spread, truth, episodes, scores):
    result = {"deterministic_mae": float((point - truth).abs().mean()), "levels": {}}
    for level in (.9, .95):
        q = conformal_quantile(scores, level)
        width = spread * q
        lo, hi = (point - width).clamp(0, 1), (point + width).clamp(0, 1)
        covered = (truth >= lo) & (truth <= hi)
        whole_episode = [bool(covered[[i for i, e in enumerate(episodes) if e == episode]].all())
                         for episode in sorted(set(episodes))]
        result["levels"][str(level)] = {"quantile": q if np.isfinite(q) else None,
            "full_range_due_to_small_calibration_set": not np.isfinite(q),
            "joint_episode_coverage": float(np.mean(whole_episode)),
            "scalar_coverage": float(covered.float().mean()), "mean_width": float((hi - lo).mean())}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[512])
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args(); seed_all(args.seed)
    if args.samples < 2 or any(width < 1 for width in args.hidden_dims):
        parser.error("At least two samples and positive hidden widths required")
    output = Path(args.output)
    if (output / "results.json").exists():
        raise FileExistsError("Calibration evaluation already completed")
    digest, manifest_digest = sha256(args.checkpoint), sha256(args.manifest)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    if (state["config"]["heldout_source"] != "ftp_motiontrans"
            or state["provenance"]["data"] != manifest_signature(args.manifest)):
        raise ValueError("Calibration requires the source-disjoint canonical visual encoder")
    data = {}
    for split in ("train", "val"):
        cache = torch.load(Path(args.cache_dir) / f"{split}_features.pt", map_location="cpu", weights_only=False)
        signature = cache["signature"]
        if (signature["checkpoint_sha256"] != digest or signature["manifest_sha256"] != manifest_digest
                or signature["source"] != "ftp_motiontrans" or signature["split"] != split
                or not signature["with_targets"] or not cache["valid"].all()):
            raise ValueError("Incompatible calibration latent cache")
        data[split] = cache
    train, validation = data["train"], data["val"]
    if set(train["episode_id"]) & set(validation["episode_id"]):
        raise ValueError("Calibration/train episodes overlap validation")
    partitions = calibration_partitions(train["episode_id"], args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = head(32, 6, hidden_dims=args.hidden_dims).to(device)
    selection = fit_head(network, frame_table(train, partitions["fit"], device),
        frame_table(train, partitions["tune"], device), regression_loss,
        epochs=args.epochs, patience=10, seed=args.seed, device=device)
    seed_all(args.seed); network = head(32, 6, hidden_dims=args.hidden_dims).to(device)
    fit_head(network, frame_table(train, sorted(partitions["fit"] + partitions["tune"]), device),
        None, regression_loss, fixed_epochs=selection["selected_epochs"], seed=args.seed, device=device)
    ids = partitions["calibration"]
    point, spread = predictions(network, train, ids, args.samples, args.seed + 1000, device)
    truth = train["target"][ids].flatten(2)
    scores = episode_scores(point, spread, truth, [train["episode_id"][i] for i in ids])
    point, spread = predictions(network, validation, list(range(len(validation["mu"]))),
                                args.samples, args.seed + 2026, device)
    metrics = calibrated_metrics(point, spread, validation["target"].flatten(2), validation["episode_id"], scores)
    result = {"config": vars(args), "checkpoint_sha256": digest,
        "checkpoint_epoch": state["epoch"], "head": [32, *args.hidden_dims, 6],
        "selected_epochs": selection["selected_epochs"], "validation": metrics,
        "calibration_episode_count": len(scores),
        "partition_episode_ids": {k: sorted({train["episode_id"][i] for i in v}) for k, v in partitions.items()},
        "coverage_unit": "entire episode, all observed times and six channels jointly",
        "guarantee_condition": "Exchangeability of calibration and evaluation episodes; no OOD guarantee",
        "point_benchmark_note": "Separate head retrained without calibration episodes; do not mix with full-label point benchmarks"}
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"head": network.cpu().state_dict(), "config": vars(args), "scores": scores,
                "checkpoint_sha256": digest}, output / "calibration_head.pt")
    write_json(output / "results.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
