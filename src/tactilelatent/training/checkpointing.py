"""Persist resumable schema-2 state and separate viewer-only snapshots.

Inputs are unperturbed task weights, optimizer/SAM/scheduler state, all sampling
RNG states, and verified provenance. Outputs are atomic run files; configuration
or scientific-code changes prevent resume rather than silently altering it.
The training engine calls ``save_checkpoint`` only at update boundaries.
"""

import hashlib
import json
import random
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch

from .optim import restore_rng, rng_snapshot


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def model_weight_hash(model):
    """Hash sorted parameter/buffer names, dtypes, shapes, and exact values."""
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def check_storage(project_root, cap=700_000_000_000, reserve=80_000_000_000):
    """Reject a run write when the data cap or free-space reserve is violated.

    ``cap`` and ``reserve`` use bytes. Measurement is read-only and never
    compresses or deletes research data to satisfy the configured limits.
    """
    root = Path(project_root)
    existing = [str(root / name) for name in ("datasets", "processing") if (root / name).exists()]
    used = 0
    if existing:
        for _ in range(3):
            result = subprocess.run(["du", "-sb", *existing], text=True, capture_output=True)
            if not result.returncode:
                used = sum(int(line.split()[0]) for line in result.stdout.splitlines())
                break
        else:
            raise RuntimeError(f"Cannot measure storage: {result.stderr}")
    if used > cap or shutil.disk_usage(root).free < reserve:
        raise RuntimeError("Dataset/processing cap or free-space reserve exceeded")


def atomic_checkpoint(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def save_checkpoint(
    root,
    completed_epoch,
    model,
    model_config,
    experiment,
    provenance,
    optimizer,
    scheduler,
    sam,
    generator,
    best_score,
    validation,
    elapsed,
):
    """Write resumable last/best state plus an inference-only epoch snapshot.

    ``completed_epoch`` is a human-facing completed count, with 0 for actual
    initialization. Stored ``epoch`` is zero-based (-1 at initialization).
    Resume state includes Python, NumPy, Torch, CUDA, and loader generators;
    snapshots omit those optimizer/RNG details and cannot resume training.
    """
    if sam.originals:
        raise RuntimeError("Cannot save perturbed SAM parameters")
    state = dict(
        schema_version=2,
        model_type=model.model_type,
        model_config=model_config,
        model=model.state_dict(),
        config=experiment,
        provenance=provenance,
        epoch=completed_epoch - 1,
        optimizer=optimizer.state_dict(),
        scheduler=scheduler.state_dict(),
        sam=sam.state_dict(),
        rng=rng_snapshot(),
        python_rng=random.getstate(),
        numpy_rng=np.random.get_state(),
        loader_rng=generator.get_state(),
        best_score=best_score,
        validation=validation,
        elapsed_seconds=elapsed,
    )
    atomic_checkpoint(root / "last.pt", state)
    snapshot = {
        k: state[k]
        for k in (
            "schema_version",
            "model_type",
            "model_config",
            "model",
            "config",
            "provenance",
            "epoch",
            "validation",
        )
    }
    snapshot["inference_only"] = True
    atomic_checkpoint(root / "viewer_epochs" / f"epoch_{completed_epoch:03d}.pt", snapshot)
    if validation and np.mean([r["supervision_loss"] for r in validation.values()]) <= best_score:
        atomic_checkpoint(root / "best.pt", state)
    return state


def restore_checkpoint(
    path, model, model_config, experiment, provenance, optimizer, scheduler, sam, generator
):
    """Restore matching schema-2 weights, optimizer, schedule, and RNG states.

    Historical and inference-only files are explicitly rejected. Configuration
    and provenance equality are checked before state is applied; callers must
    select a new run when scientific inputs or implementation bytes change.
    """
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("schema_version") != 2 or state.get("inference_only"):
        raise ValueError(
            "Only resumable research-core-v2 checkpoints may resume; legacy checkpoints remain viewable"
        )
    if (
        state["model_config"] != model_config
        or state["config"] != experiment
        or state["provenance"] != provenance
    ):
        raise ValueError("Resume configuration, model or artifact provenance changed")
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    sam.load_state_dict(state["sam"])
    restore_rng(state["rng"])
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    generator.set_state(state["loader_rng"])
    return state
