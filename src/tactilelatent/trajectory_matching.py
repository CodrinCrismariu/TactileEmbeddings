"""Fixed tactile-trajectory GW correspondences and detached visual targets.

POT entropic GW compares frozen temporal distance matrices. It is not GDTW:
chronology is checked after solving, and reversed/nonmonotone matches rejected.
Reliability is a bounded heuristic, never a calibrated match probability.
"""
import math
from pathlib import Path
import random
import time
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .aligned_frames import write_json
from .fgw_data import CodeOnlyDataset, source_seed
from .relational import HELDOUT, trajectory_distances
from .train import features


def match_trajectories(a, b, epsilon=.05, max_cost=.25, max_entropy=.98, max_iter=100):
    """Solve detached teacher-only GW; fail closed on bad marginals/solver errors."""
    import ot
    if ot.__version__ != "0.9.6.post1":
        raise ValueError("Trajectory matching requires the pinned POT 0.9.6.post1")
    if epsilon <= 0 or max_cost <= 0 or not 0 < max_entropy <= 1 or max_iter < 1:
        raise ValueError("Invalid entropic trajectory matcher configuration")
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if any(c.ndim != 2 or c.shape[0] != c.shape[1] or len(c) < 2 or
           not np.isfinite(c).all() or (c < 0).any() for c in (a, b)):
        raise ValueError("Finite nonnegative square trajectory distances required")
    p, q = np.full(len(a), 1 / len(a)), np.full(len(b), 1 / len(b))
    def valid_result(result):
        t, log = result
        return (bool(log.get("err")) and log["err"][-1] <= 1e-5 and
                np.isfinite(t).all() and not (t < 0).any() and
                np.allclose(t.sum(1), p, atol=1e-5) and np.allclose(t.sum(0), q, atol=1e-5))
    try:
        with warnings.catch_warnings(record=True) as caught:
            solved, retries = [], 0
            for temperature in (epsilon, epsilon * 1.1):
                initial = None
                for budget in (1000, 10000):
                    result = ot.gromov.entropic_gromov_wasserstein(a, b, p, q,
                        epsilon=temperature, max_iter=max_iter, tol=1e-7, symmetric=True, G0=initial,
                        solver="PGD", numItermax=budget, stopThr=1e-6, log=True)
                    if valid_result(result):
                        break
                    if budget == 1000:
                        retries += 1
                    initial = result[0] if np.isfinite(result[0]).all() and (result[0] > 0).all() else None
                solved.append(result)
        plans = [result[0] for result in solved]
        # Intermediate Sinkhorn warnings can occur despite a valid final plan.
        # Check actual final marginals and outer convergence; log the warnings.
        if not all(valid_result(result) for result in solved):
            return None, {"reason": "solver_failure", "solver_retries": retries,
                          "marginal_error": max(float(np.abs(t.sum(0) - q).max()) for t in plans)}
    except (ValueError, RuntimeError, FloatingPointError):
        return None, {"reason": "solver_failure"}
    plan = plans[0]
    rows = plan / plan.sum(1, keepdims=True)
    entropy = float(-(rows * np.log(np.maximum(rows, 1e-15))).sum(1).mean() / math.log(len(b)))
    cost = float(np.einsum("ikjl,ij,kl->", (a[:, :, None, None] - b[None, None, :, :]) ** 2, plan, plan))
    expected_time = rows @ np.linspace(0, 1, len(b))
    ordered = bool(np.all(np.diff(expected_time) >= -1e-3) and expected_time[-1] > expected_time[0] + .05)
    stability = max(0., 1 - float(np.abs(plan - plans[1]).sum() / 2))
    reliability = max(0., (1 - entropy) * math.exp(-cost / max_cost) * stability)
    stats = {"cost": cost, "entropy": entropy, "temperature_stability": stability,
             "solver_warnings": len(caught), "solver_retries": retries,
             "reliability": reliability, "ordered": ordered}
    if cost > max_cost or entropy > max_entropy or not ordered:
        return None, {**stats, "reason": "ambiguous_or_poor_fit"}
    return torch.from_numpy(plan).float(), {**stats, "reason": "accepted"}


def transport(plan, values):
    """Row-normalized transport preserves constants with unequal bank sizes."""
    mass = plan.sum(-1, keepdim=True)
    return (plan @ values) / mass.clamp_min(1e-12), mass[:, 0] > 1e-12


def trajectory_loss(mu, payload):
    losses = []
    for item in payload:
        predicted = mu[item["batch"], item["positions"]].float()
        target = item["target"].detach()
        loss = (predicted - target).square().mean()
        losses.append(loss * item["reliability"])
    value = torch.stack(losses).mean() if losses else mu.sum() * 0
    return value, {"trajectory_loss": float(value.detach()), "trajectory_matched_clips": len(losses)}


class TrajectoryObjective:
    def __init__(self, cfg, dataset, floors, device, output, provenance=None):
        if HELDOUT in dataset.indices_by_source or any(r["split"] != "train" for r in dataset.episodes):
            raise ValueError("Trajectory banks require source training episodes only")
        self.cfg, self.dataset, self.floors, self.device = cfg, dataset, floors, device
        self.provenance = provenance
        self.output = Path(output) / "trajectory"
        self.output.mkdir(parents=True, exist_ok=True)
        self.targets = {}

    @torch.no_grad()
    def refresh(self, model, backbone, banks, epoch):
        began, settings = time.monotonic(), self.cfg["trajectory"]
        descriptors, by_source, records = {}, {}, {}
        limit = settings.get("clips_per_source", 32)
        for source, indices in sorted(banks.items()):
            if source not in self.floors:
                continue  # Numeric trajectories have no justified AE noise floor.
            by_source[source] = []
            for index in indices[:limit]:
                item = CodeOnlyDataset.__getitem__(self.dataset, index)
                descriptor = trajectory_distances(item["value"], item["valid"], self.floors[source])
                if descriptor is None:
                    continue
                descriptors[index] = descriptor
                by_source[source].append(index)
                records[index] = {"source": source, "episode_id": item["episode_id"],
                                  "frame_indices": item["frame_indices"].tolist()}
        accepted, diagnostics, visited = [], [], set()
        rng = random.Random(source_seed(self.cfg["seed"], epoch, "tactile_trajectories"))
        for source, indices in by_source.items():
            other_sources = [s for s, values in by_source.items() if s != source and values]
            for index in indices:
                if not other_sources:
                    continue
                partner_source = rng.choice(other_sources)
                pool = by_source[partner_source]
                candidates = rng.sample(pool, min(len(pool), settings.get("candidates", 4)))
                best = None
                for partner in candidates:
                    pair = tuple(sorted((index, partner)))
                    if pair in visited:
                        continue
                    visited.add(pair)
                    plan, stats = match_trajectories(descriptors[index][0].numpy(), descriptors[partner][0].numpy(),
                        settings.get("epsilon", .05), settings.get("max_cost", .25), settings.get("max_entropy", .98))
                    diagnostics.append({"first": index, "second": partner, **stats})
                    if plan is not None and (best is None or stats["reliability"] > best[3]):
                        best = (index, partner, plan, stats["reliability"])
                if best is not None:
                    accepted.append(best)
        # Only after fixed-teacher matching do we read the model's coordinates.
        needed = sorted({i for a, b, _, _ in accepted for i in (a, b)})
        means, previous_mode = {}, model.training
        model.eval()
        try:
            for index in needed:
                original = next(iter(DataLoader(Subset(self.dataset, [index]), batch_size=1)))
                with torch.autocast(self.device.type, torch.bfloat16, enabled=self.device.type == "cuda"):
                    mu, _ = model.visual.encode(features(backbone, original, self.device), original["view_mask"].to(self.device))
                means[index] = mu[0].float().cpu()[descriptors[index][1]]
        finally:
            model.train(previous_mode)
        self.targets = {}
        for a, b, plan, reliability in accepted:
            for first, second, coupling in ((a, b, plan), (b, a, plan.T)):
                target, valid = transport(coupling, means[second])
                self.targets.setdefault(first, []).append({"positions": descriptors[first][1][valid],
                    "target": target[valid], "reliability": reliability})
        info = {"candidate_pairs": len(diagnostics), "accepted_pairs": len(accepted),
                "solver_failures": sum(d["reason"] == "solver_failure" for d in diagnostics),
                "informative_clips": {s: len(v) for s, v in by_source.items()},
                "seconds": time.monotonic() - began}
        snapshot = {"bank_records": records, "pairs": accepted, "diagnostics": diagnostics,
                    "targets": self.targets, "settings": settings, "epoch": epoch + 1,
                    "provenance": self.provenance}
        path = self.output / f"epoch_{epoch + 1:03d}.pt"
        tmp = path.with_suffix(".tmp"); torch.save(snapshot, tmp); tmp.replace(path)
        write_json(path.with_suffix(".json"), info)
        return info

    def payload(self, original):
        result = []
        for batch, index in enumerate(original["window_index"].tolist()):
            if not bool(original["alignment_anchor"][batch]):
                continue
            for item in self.targets.get(index, []):
                result.append({"batch": batch, "positions": item["positions"].to(self.device),
                               "target": item["target"].to(self.device), "reliability": item["reliability"]})
        return result
