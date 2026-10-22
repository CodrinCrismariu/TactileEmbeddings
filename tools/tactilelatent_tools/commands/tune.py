"""Execute finite tuning plans without implicit candidates, seeds or budgets.

Inputs:
    A JSON plan declaring candidates, seeds, epoch/trial budget and output path.
Outputs:
    Dry-run JSON only, or immutable trial configurations, training runs and
    status reports. Resume requires the identical plan and candidate provenance.
Example:
    tl tune --plan PATH_TO_PLAN.json --dry-run
"""

import hashlib
import json
import os
import tempfile
from copy import deepcopy
from pathlib import Path

from ..cli import project_path


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def resolve_plan(path, root):
    from tactilelatent.configuration import load_experiment

    raw = path.read_bytes()
    plan = json.loads(raw)
    candidates, seeds, budget = plan.get("candidates"), plan.get("seeds"), plan.get("budget")
    if (
        not isinstance(candidates, list)
        or not candidates
        or any(not isinstance(c, str) or not c for c in candidates)
    ):
        raise ValueError(
            "Tuning requires a nonempty explicit candidates list of experiment JSON paths"
        )
    if len(set(candidates)) != len(candidates):
        raise ValueError("Tuning candidates must be distinct")
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(isinstance(s, bool) or not isinstance(s, int) or not 0 <= s < 2**32 for s in seeds)
    ):
        raise ValueError("Tuning requires a nonempty explicit seeds list of uint32 integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Tuning seeds must be distinct")
    if not isinstance(budget, dict):
        raise ValueError("Tuning requires an explicit budget object")
    epochs = _positive(budget.get("epochs_per_trial"), "budget.epochs_per_trial")
    maximum = _positive(budget.get("max_trials"), "budget.max_trials")
    count = len(candidates) * len(seeds)
    if count > maximum:
        raise ValueError(f"Candidate/seed product {count} exceeds max_trials={maximum}")
    if not isinstance(plan.get("output"), str) or not plan["output"]:
        raise ValueError("Tuning requires an explicit output directory")
    output = project_path(plan["output"], root)
    if output == root or root not in output.parents:
        raise ValueError(
            "Tuning output must be inside the project, below a separate output directory"
        )
    relative = output.relative_to(root)
    if relative.parts[0] not in {"processing", "outputs", "artifacts"}:
        raise ValueError("Tuning output must be under processing/, outputs/ or artifacts/")
    trials = []
    for number, candidate in enumerate(candidates):
        source = project_path(candidate, root)
        configuration = load_experiment(source, project_root=root)
        for seed in seeds:
            name = f"candidate_{number:03d}_seed_{seed}"
            folder = output / name
            document = {
                key: deepcopy(configuration[key])
                for key in ("model", "datasets", "training", "transfer")
            }
            document["schema"] = "tactilelatent-experiment-v2"
            document["name"] = name
            document["training"].update(seed=seed, epochs=epochs, output=str(folder))
            document["tuning"] = {
                "candidate": str(source),
                "plan_sha256": hashlib.sha256(raw).hexdigest(),
                "candidate_provenance": configuration["provenance"],
            }
            trials.append(
                {
                    "name": name,
                    "experiment": document,
                    "config": str(folder / "experiment.json"),
                    "output": str(folder),
                    "seed": seed,
                    "epochs": epochs,
                }
            )
    return {
        "plan": str(path),
        "plan_sha256": hashlib.sha256(raw).hexdigest(),
        "output": str(output),
        "budget": budget,
        "trial_count": count,
        "total_epoch_budget": count * epochs,
        "trials": trials,
    }


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".tuning-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def run(args, root):
    plan = resolve_plan(project_path(args.plan, root), root)
    if args.dry_run:
        return {**plan, "stage": "planned", "writes": False}
    output = Path(plan["output"])
    lock = output / "plan.json"
    if lock.exists():
        if not args.resume:
            raise ValueError(
                "Tuning output already has a plan; use --resume for the identical plan"
            )
        if json.loads(lock.read_text()) != plan:
            raise ValueError("Tuning plan/configuration changed; select a new output directory")
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("Tuning output is nonempty and has no matching plan")
        _write_json(lock, plan)
    from tactilelatent.training.engine import train_experiment

    completed = []
    for trial in plan["trials"]:
        path = Path(trial["config"])
        if path.exists() and json.loads(path.read_text()) != trial["experiment"]:
            raise ValueError(f"Materialized trial configuration changed: {path}")
        if not path.exists():
            _write_json(path, trial["experiment"])
        folder = Path(trial["output"])
        checkpoint = folder / "last.pt"
        resume = folder if args.resume and checkpoint.exists() else None
        if checkpoint.exists() and resume is None:
            raise ValueError(
                f"Trial already has a checkpoint; explicit --resume required: {folder}"
            )
        _write_json(
            output / "status.json",
            {
                "stage": "training",
                "trial": trial["name"],
                "completed_trials": len(completed),
                "trials": plan["trial_count"],
            },
        )
        try:
            result = train_experiment(path, project_root=root, resume=resume, preflight_only=False)
        except Exception as error:
            _write_json(
                output / "status.json",
                {
                    "stage": "failed",
                    "trial": trial["name"],
                    "error": f"{type(error).__name__}: {error}",
                },
            )
            raise
        completed.append({"trial": trial["name"], "run": str(folder), "result": result})
    result = {
        "stage": "complete",
        "trial_count": len(completed),
        "total_epoch_budget": plan["total_epoch_budget"],
        "results": completed,
    }
    _write_json(output / "status.json", result)
    return result
