"""Compute the baseline masked supervision objective without dataset branches.

Inputs are model predictions and contract ``ClipBatch`` supervision; the output
is a scalar loss in standardized teacher-code or direct-target coordinates.
Training calls ``forward_batch`` followed by ``supervision_loss``. Native
reconstruction metrics are computed separately after source decoding.
"""

import torch


def valid_frames(batch):
    """Return [B, Q] observation flags for latent diagnostics.

    Code masks already describe frames; direct-target masks describe native
    coordinates, so any observed coordinate makes a frame supported.
    """
    valid = batch.supervision.valid.bool()
    return valid if batch.supervision.kind == "code" else valid.flatten(2).any(-1)


def supervision_loss(prediction, batch):
    """Reduce supported code MSE or direct MSE + 0.5 MAE to one scalar.

    Code loss averages coordinates per supported frame. Direct loss averages
    only observed coordinates. All-missing supervision contributes a
    differentiable zero; missing observations never become target zeros.
    """
    target = batch.supervision.values.detach().float()
    valid = batch.supervision.valid.bool()
    if batch.supervision.kind == "code":
        error = (prediction["sensor_code"].float() - target).square().mean(-1)
        return error[valid].mean() if valid.any() else error.sum() * 0
    if batch.supervision.kind != "direct":
        raise ValueError("A training sample must declare code or direct supervision")
    safe = torch.where(valid, target, 0.0)
    difference = prediction["pred"].float() - safe
    return ((difference.square() + 0.5 * difference.abs()) * valid).sum() / valid.sum().clamp_min(1)


def forward_batch(model, features, batch):
    """Route a uniform batch through its head using deterministic latent means."""
    return model(
        features,
        batch.visual.view_mask,
        batch.source,
        camera_roles=batch.visual.camera_roles,
        sample_latent=False,
    )
