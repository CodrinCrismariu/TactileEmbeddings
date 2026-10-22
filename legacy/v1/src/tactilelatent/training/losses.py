"""Pressure loss and evaluation metrics for the image-only VAE experiment.

The measured pressure map is a training target, never an encoder input. The
loss deliberately emphasizes contact cells and lightly regularizes the
Gaussian latent toward a standard normal distribution.
"""

import torch


def standard_normal_kl(mu, logvar):
    """Average KL[N(mu, diag(exp(logvar))) || N(0,I)] per latent coordinate.

    For one coordinate this is 0.5 * (mu² + variance - 1 - log(variance)).
    Taking a mean over batch, time, and latent width keeps its scale stable.
    """
    return 0.5 * (mu.square() + logvar.exp() - 1 - logvar).mean()


def objective(output, target, mask, beta=0., threshold=.1, contact_weight=3., tv_weight=.01,
              normalize_by_valid=False):
    """Combine pressure reconstruction, smoothness, and latent KL penalties.

    Args:
        output: model dictionary with pressure, mu, and logvar.
        target: measured pressure [B,T,2,21,21]. May contain nonfinite cells.
        mask: same shape; True only where target is valid.
        beta: current weight for the Gaussian KL regularizer.
        threshold: pressure above this counts as contact for loss weighting.
        contact_weight: multiplier applied to contact cells (3 in training).
        tv_weight: multiplier for spatial total variation (0.01 in training).
        normalize_by_valid: divide regression by valid-cell count when True;
            this is what the active training loop uses.
    """
    if not mask.any():
        raise ValueError("Batch has no valid targets")

    pred = output["pressure"]
    # Invalid target cells may be NaN. Replacing them with zero BEFORE taking
    # differences avoids NaNs even though they will later have zero weight.
    safe = torch.where(mask, target, torch.zeros_like(target))
    weight = torch.where(safe > threshold, contact_weight, 1.) * mask
    error = pred - safe

    # Regression = weighted (squared error + half absolute error).
    denominator = mask.sum() if normalize_by_valid else weight.sum()
    regression = ((error.square() + .5 * error.abs()) * weight).sum() / denominator

    # Total variation (TV) penalizes jumps between neighboring valid pixels.
    # The two axes are width and height of each 21x21 hand-pressure grid.
    tv = pred.new_zeros(())
    for axis in (-1, -2):
        diff = torch.diff(pred, dim=axis).abs()
        left = mask.narrow(axis, 0, mask.shape[axis] - 1)
        right = mask.narrow(axis, 1, mask.shape[axis] - 1)
        adjacent = left & right
        tv = tv + (diff * adjacent).sum() / adjacent.sum().clamp_min(1)

    kl = standard_normal_kl(output["mu"], output["logvar"])
    loss = regression + tv_weight * tv + beta * kl
    return loss


@torch.no_grad()
def metric_counts(pred, target, mask, threshold=.1):
    """Accumulate sufficient statistics; summarize() converts them to scores.

    Summing counts across batches before dividing prevents small final batches
    from receiving disproportionate weight in the overall evaluation.
    """
    safe = torch.where(mask, target, torch.zeros_like(target))
    p, y = (pred > threshold) & mask, (safe > threshold) & mask
    return torch.stack([((pred - safe).abs() * mask).sum(), mask.sum(),
                        (p & y).sum(), (p & ~y).sum(), (~p & y).sum(),
                        (torch.minimum(pred, safe) * mask).sum(),
                        (torch.maximum(pred, safe) * mask).sum()]).double()


def summarize(counts):
    """Return pixel MAE, thresholded contact IoU/F1, and volumetric IoU."""
    err, n, tp, fp, fn, lo, hi = counts.tolist()
    return {"mae": err / max(n, 1), "contact_iou": tp / max(tp + fp + fn, 1),
            "contact_f1": 2 * tp / max(2 * tp + fp + fn, 1), "volumetric_iou": lo / max(hi, 1e-8)}
