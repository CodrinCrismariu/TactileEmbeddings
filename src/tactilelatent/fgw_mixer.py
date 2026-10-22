"""Detached, mass-normalized FGW barycenters for unpaired interaction states.

No labels, DINO similarity, or hand-selected correspondences enter this module.
POT alpha is the structural weight (0 = Wasserstein, 1 = GW).
"""
from dataclasses import dataclass
import time
import warnings

import numpy as np
import torch


HELDOUT = "ftp_motiontrans"
VARIANTS = ("none", "gw", "fgw", "stochastic_fgw", "partial_stochastic_fgw")


@dataclass
class MixerConfig:
    variant: str = "stochastic_fgw"
    structure_weight: float = .5
    transported_mass: float = .75
    bank_size: int = 256
    barycenter_size: int = 256
    rounds: int = 5
    max_iter: int = 100
    # POT's partial solver shares this cap with its inner network-flow solve.
    # A 100-iteration EMD budget is insufficient for 256x256 banks.
    partial_max_iter: int = 10000
    tolerance: float = 1e-7
    loss_weight: float = .1
    ramp_epochs: int = 2
    spread_weight: float = .01
    bank_fraction: float = .25

    def __post_init__(self):
        if self.variant not in VARIANTS:
            raise ValueError("Unknown FGW variant")
        if not 0 <= self.structure_weight <= 1 or not 0 < self.transported_mass <= 1:
            raise ValueError("Invalid structure weight or transported mass")
        if min(self.bank_size, self.barycenter_size, self.rounds, self.max_iter, self.partial_max_iter, self.ramp_epochs) < 1:
            raise ValueError("Positive mixer sizes and iteration limits required")
        if not 0 <= self.bank_fraction <= 1 or min(self.loss_weight, self.spread_weight) < 0:
            raise ValueError("Invalid mixer loss or sampling fraction")


def distances(x):
    d = np.linalg.norm(x[:, None] - x[None, :], axis=-1)
    np.fill_diagonal(d, 0.)
    return d


def push_to_support(plan, codes):
    mass = plan.sum(1)
    value = plan @ codes / np.maximum(mass[:, None], 1e-15)
    return value, mass > 1e-12


def pull_to_source(plan, codes, original):
    mass = plan.sum(0)
    value = plan.T @ codes / np.maximum(mass[:, None], 1e-15)
    valid = mass > 1e-12
    value[~valid] = original[~valid]
    return value, valid


def check_plan(plan, p, q, mass, partial, tolerance=2e-6):
    if not np.isfinite(plan).all() or plan.min() < -tolerance:
        raise FloatingPointError("Invalid transport plan")
    if abs(plan.sum() - mass) > tolerance:
        raise FloatingPointError("Transported mass does not match requested mass")
    row, col = plan.sum(1), plan.sum(0)
    if partial:
        valid = (row <= p + tolerance).all() and (col <= q + tolerance).all()
    else:
        valid = np.allclose(row, p, atol=tolerance, rtol=0) and np.allclose(col, q, atol=tolerance, rtol=0)
    if not valid:
        raise FloatingPointError("Transport marginal constraints violated")


def _solve(z, x, cfg, scale, initial=None):
    import ot
    p, q = np.full(len(z), 1 / len(z)), np.full(len(x), 1 / len(x))
    alpha = 1. if cfg.variant == "gw" else cfg.structure_weight
    c1, c2 = distances(z) / scale, distances(x) / scale
    m = np.square(z[:, None] - x[None, :]).sum(-1) / scale**2
    partial = cfg.variant == "partial_stochastic_fgw"
    mass = cfg.transported_mass if partial else 1.
    # Fail closed on solver warnings; a failed solve never becomes a target.
    with warnings.catch_warnings(record=True) as messages:
        warnings.simplefilter("always")
        if partial and mass < 1.:
            plan, log = ot.gromov.partial_fused_gromov_wasserstein(
                m, c1, c2, p, q, alpha=alpha, m=mass, G0=initial,
                nb_dummies=8, symmetric=True, numItermax=cfg.partial_max_iter, tol=cfg.tolerance, log=True)
        else:
            plan, log = ot.gromov.fused_gromov_wasserstein(
                m, c1, c2, p, q, alpha=alpha, G0=initial, symmetric=True,
                max_iter=cfg.max_iter, tol_rel=cfg.tolerance, tol_abs=cfg.tolerance, log=True)
    if messages or log.get("warning"):
        raise FloatingPointError("FGW solver warning: " + str(log.get("warning") or messages[0].message))
    check_plan(plan, p, q, mass, partial)
    if not np.isfinite(log.get("partial_fgw_dist", log.get("fgw_dist", 0.))):
        raise FloatingPointError("Nonfinite FGW objective")
    return plan


def fit_mixer(banks, cfg, seed=45, previous=None, scale=None):
    """Return detached per-source targets plus plans, support and diagnostics.

    Coordinate barycenter updates are an OTM-style alternating approximation,
    not a claim of globally solving the nonconvex FGW barycenter objective.
    """
    began = time.monotonic()
    if HELDOUT in banks or len(banks) < 2:
        raise ValueError("Need at least two source banks, excluding MotionTrans")
    arrays = {s: np.asarray(x.detach().cpu().double() if torch.is_tensor(x) else x, dtype=np.float64)
              for s, x in sorted(banks.items())}
    dims = {x.shape[1] for x in arrays.values() if x.ndim == 2}
    if len(dims) != 1 or any(x.ndim != 2 or len(x) < 1 or not np.isfinite(x).all() for x in arrays.values()):
        raise ValueError("Finite, nonempty, common-dimensional banks required")
    initial_variance = {s: float(x.var(0).mean()) for s, x in arrays.items()}
    if cfg.variant == "none":
        return {"targets": {s: torch.from_numpy(x.copy()).float() for s, x in arrays.items()},
                "valid": {s: torch.zeros(len(x), dtype=torch.bool) for s, x in arrays.items()},
                "banks": {s: torch.from_numpy(x.copy()).float() for s, x in arrays.items()},
                "plans": {}, "barycenter": None, "scale": scale,
                "diagnostics": {"seconds": time.monotonic() - began, "sources": {}, "solver_failures": 0},
                "initial_variance": initial_variance}
    if scale is None:
        medians = []
        for x in arrays.values():
            d = distances(x); positive = d[d > 1e-12]
            if len(positive):
                medians.append(np.median(positive))
        scale = float(np.mean(medians)) if medians else 1.
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Positive fixed distance scale required")
    rng = np.random.default_rng(seed)
    # Balanced pooled initialization avoids permanently privileging source 0.
    sources = list(arrays)
    pooled = np.stack([arrays[sources[i % len(sources)]][rng.integers(len(arrays[sources[i % len(sources)]]))]
                       for i in range(cfg.barycenter_size)])
    z = previous.copy() if previous is not None and previous.shape == pooled.shape else pooled
    plans = {}
    for _ in range(cfg.rounds):
        numerator, denominator = np.zeros_like(z), np.zeros((len(z), 1))
        for source, x in arrays.items():
            t = _solve(z, x, cfg, scale, plans.get(source))
            plans[source] = t
            numerator += t @ x / len(sources)
            denominator += t.sum(1, keepdims=True) / len(sources)
        z = np.where(denominator > 1e-12, numerator / np.maximum(denominator, 1e-15), z)
    # Refit to the final support; all pushed contributions then use these plans.
    plans = {s: _solve(z, x, cfg, scale, plans[s]) for s, x in arrays.items()}
    pushed = [push_to_support(plans[s], arrays[s]) for s in sources]
    if cfg.variant in ("stochastic_fgw", "partial_stochastic_fgw"):
        supported = np.stack([valid for _, valid in pushed])
        scores = rng.random((len(sources), len(z), z.shape[1]))
        scores[~np.broadcast_to(supported[:, :, None], scores.shape)] = -1
        choice = scores.argmax(0)  # Exactly one supported source per coordinate.
        mixed = np.take_along_axis(np.stack([x for x, _ in pushed]), choice[None], axis=0)[0]
        mixed[~supported.any(0)] = z[~supported.any(0)]
    else:
        mixed = z.copy()
    targets, valid, diagnostics = {}, {}, {}
    for source, x in arrays.items():
        target, supported = pull_to_source(plans[source], mixed, x)
        targets[source] = torch.from_numpy(target).float()
        valid[source] = torch.from_numpy(supported)
        row_mass = plans[source].sum(1)
        prob = plans[source] / np.maximum(row_mass[:, None], 1e-15)
        spectrum = np.linalg.eigvalsh(np.atleast_2d(np.cov(x.T, bias=True))).clip(min=0.) if len(x) > 1 else np.zeros(x.shape[1])
        proportions = spectrum / max(spectrum.sum(), 1e-15)
        rank = float(np.exp(-(proportions * np.log(np.maximum(proportions, 1e-15))).sum())) if spectrum.sum() > 1e-15 else 0.
        diagnostics[source] = {"transported_mass": float(plans[source].sum()),
            "matched_fraction": float(supported.mean()), "mean_coordinate_variance": initial_variance[source],
            "effective_rank": rank,
            "mean_row_entropy": float(-(prob * np.log(np.maximum(prob, 1e-15))).sum(1).mean()),
            "mean_target_displacement": float(np.square(target - x).mean())}
    return {"targets": targets, "valid": valid, "plans": plans, "barycenter": z,
            "banks": {s: torch.from_numpy(x.copy()).float() for s, x in arrays.items()},
            "scale": scale, "initial_variance": initial_variance,
            "diagnostics": {"seconds": time.monotonic() - began, "sources": diagnostics, "solver_failures": 0}}


def state_loss(mu, target, valid, bank, bank_positions, initial_variance, cfg, epoch):
    """Gradients enter current codes only; plans, targets, other bank rows detach."""
    difference = (mu.float() - target.detach().float()).square().mean(-1)
    consistency = difference[valid].mean() if valid.any() else mu.sum() * 0
    # Substitute current rows into a detached bank, including duplicate rows once.
    positions, inverse = torch.unique(bank_positions, return_inverse=True)
    values = torch.zeros(len(positions), mu.shape[-1], device=mu.device, dtype=torch.float32)
    values.index_add_(0, inverse, mu.float())
    counts = torch.bincount(inverse, minlength=len(positions)).float().clamp_min(1)
    updated = bank.detach().float().clone().index_copy(0, positions, values / counts[:, None])
    variance = updated.var(0, unbiased=False).mean()
    reference = max(float(initial_variance), 1e-8)
    spread = torch.relu(.5 * reference - variance).square() / reference**2
    ramp = min(1., max(0., (epoch + 1) / cfg.ramp_epochs))
    loss = ramp * cfg.loss_weight * consistency + cfg.spread_weight * spread
    return loss, {"consistency": float(consistency.detach()), "spread": float(spread.detach())}
