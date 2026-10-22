"""Latent spread/effective-rank diagnostics; no claim of physical invariance."""

import torch


class LatentStats:
    """Accumulate observed latent moments on CPU in float64.

    Inputs to ``add`` are prediction means/log-variance [B, Q, D] and True
    observation flags [B, Q]. ``result`` reports covariance spread and the
    exponentiated entropy of its normalized eigenvalues. Effective rank
    describes occupied dimensions, not sensor invariance or transfer quality.
    """

    def __init__(self, dim):
        self.sum = torch.zeros(dim, dtype=torch.float64)
        self.outer = torch.zeros(dim, dim, dtype=torch.float64)
        self.count, self.variance = 0, 0.0

    def add(self, prediction, valid):
        values = prediction["mu"].detach()[valid].double().cpu()
        self.sum += values.sum(0)
        self.outer += values.T @ values
        self.count += len(values)
        self.variance += prediction["logvar"].detach()[valid].float().exp().sum().item()

    def result(self):
        """Return population-covariance diagnostics over supported frames only."""
        mean = self.sum / max(self.count, 1)
        cov = self.outer / max(self.count, 1) - mean[:, None] * mean[None, :]
        spectrum = torch.linalg.eigvalsh(cov).clamp_min(0)
        p = spectrum / spectrum.sum().clamp_min(1e-12)
        rank = (
            (-(p * p.clamp_min(1e-12).log()).sum()).exp().item() if spectrum.sum() > 1e-12 else 0.0
        )
        return dict(
            effective_rank_mu=rank,
            mean_coordinate_variance_mu=cov.diag().mean().item(),
            mean_predicted_variance=self.variance / max(self.count * len(mean), 1),
            valid_frames=self.count,
        )
