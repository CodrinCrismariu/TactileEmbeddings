"""SAM weight perturbations around an ordinary AdamW optimizer.

The first accumulated gradient estimates the worst local weight direction.
The second gradient is computed at perturbed weights; the ORIGINAL weights
are restored exactly before AdamW updates them. Frozen DINO is not included.
"""

import shutil
import subprocess
from pathlib import Path

import torch


def check_storage_budget(root, additional=0, cap=700_000_000_000, reserve=80_000_000_000):
    """Check the user's dataset+processing cap before large checkpoint writes."""
    root = Path(root)
    # Atomic status/checkpoint updates can remove a temporary file while du
    # walks the directory. Retry that transient race; other errors still fail.
    command=["du", "-sb", str(root / "datasets"), str(root / "processing")]
    for _ in range(3):
        result=subprocess.run(command,text=True,capture_output=True)
        if result.returncode==0:
            usage=result.stdout
            break
        if 'No such file or directory' not in result.stderr:
            raise RuntimeError(f'Cannot measure project storage: {result.stderr.strip()}')
    else:
        raise RuntimeError(f'Project storage kept changing during measurement: {result.stderr.strip()}')
    used = sum(int(line.split()[0]) for line in usage.splitlines())
    if used + additional > cap or shutil.disk_usage(root).free < additional + reserve:
        raise RuntimeError("Dataset/processing cap or free-space reserve would be exceeded")
    return used


def check_checkpoint_budget(state):
    """Reserve room for a temporary serialized state, including optimizer tensors."""
    seen=set();total=0
    def visit(value):
        nonlocal total
        if torch.is_tensor(value):
            storage=value.untyped_storage();key=(str(value.device),storage.data_ptr())
            if key not in seen:seen.add(key);total+=storage.nbytes()
        elif isinstance(value,dict):
            for item in value.values():visit(item)
        elif isinstance(value,(tuple,list)):
            for item in value:visit(item)
    visit(state)
    return check_storage_budget(Path(__file__).resolve().parents[2],additional=int(total*1.1)+1_048_576)


def rng_snapshot():
    return {"cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    torch.set_rng_state(state["cpu"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


class SAMStep:
    def __init__(self, optimizer, rho=.05):
        if rho <= 0:
            raise ValueError("SAM rho must be positive")
        self.optimizer = optimizer
        self.rho = float(rho)
        self.originals = []
        self.steps = 0

    @torch.no_grad()
    def perturb(self):
        if self.originals:
            raise RuntimeError("Weights are already perturbed")
        parameters = [p for group in self.optimizer.param_groups for p in group["params"]
                      if p.requires_grad and p.grad is not None]
        if not parameters:
            raise ValueError("SAM needs a first-pass gradient")
        norm = torch.linalg.vector_norm(torch.stack([
            torch.linalg.vector_norm(p.grad.float()) for p in parameters]))
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
