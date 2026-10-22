"""Implement SAM replay and the baseline optimizer-update learning-rate schedule.

Inputs are first-pass gradients, retained batches/features, and saved dropout
RNG states. Outputs are restored unperturbed weights after one AdamW update.
``replay_update`` is the training engine's entry point; frozen V-JEPA features
are reused rather than recomputed during the perturbed forward.
"""

import math

import torch


def rng_snapshot():
    """Capture CPU and available CUDA generators for exact dropout replay."""
    return {
        "cpu": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    torch.set_rng_state(state["cpu"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


class SAMStep:
    """Manage a temporary global-gradient-norm SAM perturbation.

    ``perturb`` saves original trainable weights and clears first-pass
    gradients. ``restore`` must run before the base optimizer steps or a
    checkpoint is saved. Only an unperturbed update boundary is resumable.
    """

    def __init__(self, optimizer, rho=0.05):
        if rho <= 0:
            raise ValueError("SAM rho must be positive")
        self.optimizer, self.rho, self.originals, self.steps = optimizer, float(rho), [], 0

    @torch.no_grad()
    def perturb(self):
        if self.originals:
            raise RuntimeError("Weights are already perturbed")
        parameters = [
            p
            for g in self.optimizer.param_groups
            for p in g["params"]
            if p.requires_grad and p.grad is not None
        ]
        if not parameters:
            raise ValueError("SAM needs a first-pass gradient")
        norm = torch.linalg.vector_norm(
            torch.stack([torch.linalg.vector_norm(p.grad.float()) for p in parameters])
        )
        if not torch.isfinite(norm):
            raise FloatingPointError("Non-finite SAM first-pass gradient")
        self.originals = [(p, p.detach().clone()) for p in parameters]
        for p in parameters:
            p.add_(p.grad * (self.rho / (norm + 1e-12)))
        self.optimizer.zero_grad(set_to_none=True)
        return norm.item()

    @torch.no_grad()
    def restore(self):
        for parameter, original in self.originals:
            parameter.copy_(original)
        self.originals.clear()

    def state_dict(self):
        if self.originals:
            raise RuntimeError("Checkpoint only at an unperturbed update boundary")
        return {"rho": self.rho, "steps": self.steps}

    def load_state_dict(self, state):
        if state["rho"] != self.rho:
            raise ValueError("SAM radius differs from checkpoint")
        self.steps = state["steps"]


def make_scheduler(optimizer, config, batches_per_epoch):
    """Warm up then cosine-decay learning rate once per accumulated update.

    Partial final accumulation groups count as one update. Warmup and total
    durations use the configured epoch budget, not individual minibatch count.
    """
    updates = math.ceil(batches_per_epoch / config["accumulation"])
    total, warmup = updates * config["epochs"], updates * config["warmup_epochs"]

    def schedule(update):
        if update < warmup:
            return (update + 1) / max(1, warmup)
        progress = min(1.0, (update - warmup) / max(1, total - warmup))
        floor = config["min_learning_rate"] / config["learning_rate"]
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def replay_update(model, optimizer, sam, replay, device):
    """Finish one SAM update with the same features, targets, masks, and RNG.

    ``replay`` contains the accumulated minibatches and their first-forward
    dropout states. Gradients are recomputed at perturbed weights, clipped,
    and used by AdamW after original weights are restored. The caller's RNG
    state is also restored, including when the replay raises an exception.
    """
    from .objectives import forward_batch, supervision_loss

    after = rng_snapshot()
    sam.perturb()
    try:
        for item in replay:
            restore_rng(item["rng"])
            with torch.autocast(device.type, torch.bfloat16, enabled=device.type == "cuda"):
                output = forward_batch(model, item["features"], item["batch"])
                loss = supervision_loss(output, item["batch"]) * item.get("weight", 1.0)
            (loss / len(replay)).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    finally:
        sam.restore()
        restore_rng(after)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    sam.steps += 1
