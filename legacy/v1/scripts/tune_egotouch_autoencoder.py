"""Sequential 60-epoch screening -> 200-epoch finalists -> seed repetitions.

Launch under nohup. Re-launch with identical arguments to resume. The trainer
preserves optimizer, LR scheduler and RNG across the 60 -> 200 epoch extension.
Validation selects candidates; official test pressure files are never read.
"""

import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def write_json(path, payload):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(path)


def trial_config(plan, trial):
    return {**plan["common"], **{k: v for k, v in trial.items() if k != "name"}}


def read_results(output):
    rows = []
    for run in sorted((output / "trials").glob("*")):
        if not (run / "best_metrics.json").exists() or not (run / "metrics.jsonl").exists():
            continue
        best = json.loads((run / "best_metrics.json").read_text())
        history = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
        if not history:
            continue
        cfg = json.loads((run / "config.json").read_text())
        sizes = json.loads((run / "summary.json").read_text())
        rows.append({"name": run.name, "sensor_code_dim": cfg["latent_dim"], "seed": cfg["seed"],
                     "output_mode": cfg["output_mode"],
                     "channels": str(cfg["model"]["channels"]), "batch_size": cfg["batch_size"],
                     "initial_lr": cfg["learning_rate"], "weight_decay": cfg["weight_decay"],
                     "contact_weight": cfg["contact_weight"], "mae_weight": cfg["mae_weight"],
                     "completed_epochs": history[-1]["epoch"], "best_epoch": best["epoch"],
                     "score": best["selection_score"], "mae": best["val"]["mae"],
                     "contact_mae": best["val"]["contact_mae"], "contact_f1": best["val"]["contact_f1"],
                     "high_pressure_mae": best["val"]["high_pressure_mae"],
                     "exact_zero_recall": best["val"]["exact_zero_recall"],
                     "false_zero_rate_on_nonzero": best["val"]["false_zero_rate_on_nonzero"],
                     "zero_output_miss_rate_on_contact": best["val"]["zero_output_miss_rate_on_contact"],
                     "parameters": sizes["encoder_parameters"] + sizes["decoder_parameters"],
                     "mean_epoch_seconds": sum(r["seconds"] for r in history) / len(history)})
    if rows:
        with (output / "comparison.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(sorted(rows, key=lambda r: r["score"]))
    return rows


def choose_finalists(rows, per_code, output_mode="sigmoid"):
    """Retain multiple code budgets rather than eliminating smaller efficient codes."""
    rows = [r for r in rows if r.get("output_mode", output_mode) == output_mode]
    selected = []
    for code in sorted({row["sensor_code_dim"] for row in rows}):
        candidates = sorted((r for r in rows if r["sensor_code_dim"] == code), key=lambda r: r["score"])
        selected.extend(r["name"] for r in candidates[:per_code])
    return selected


def run_trial(output, name, cfg, epochs, stage):
    run = output / "trials" / name
    command = [sys.executable, "-u", "scripts/pretrain_egotouch_autoencoder.py"]
    for key, value in cfg.items():
        command.append("--" + key.replace("_", "-"))
        command.extend(str(x) for x in (value if isinstance(value, list) else [value]))
    command.extend(["--epochs", str(epochs), "--output", str(run)])
    if (run / "last.pt").exists():
        # Check actual checkpoint progress, not an incomplete log write.
        import torch
        saved = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
        for key, value in cfg.items():
            stored_key = "latent_dim" if key == "sensor_code_dim" else key
            if saved["config"][stored_key] != value:
                raise ValueError(f"Existing trial {name} differs at {key}")
        if saved["epoch"] + 1 >= epochs:
            return
        command.append("--resume")
        del saved
    elif run.exists():
        # Interrupted before any completed epoch: preserve partial files and restart.
        archived = output / "uncheckpointed"
        archived.mkdir(exist_ok=True)
        run.rename(archived / f"{name}_{time.time_ns()}")
    write_json(output / "status.json", {"status": "running", "stage": stage,
               "trial": name, "target_epochs": epochs, "log": str(output / "logs" / f"{name}.log")})
    print(json.dumps({"event": "start", "stage": stage, "trial": name, "epochs": epochs}), flush=True)
    env = {**os.environ, "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}
    with (output / "logs" / f"{name}.log").open("a") as log:
        subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    read_results(output)
    print(json.dumps({"event": "finished", "stage": stage, "trial": name}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default="configs/egotouch_ae_search_v4.json")
    parser.add_argument("--output", default="processing/runs/egotouch_ae_search_v4")
    parser.add_argument("--smoke", action="store_true", help="Tiny CPU search testing all three stages")
    args = parser.parse_args()
    os.chdir(ROOT)
    plan = json.loads(Path(args.plan).read_text())
    if args.smoke:
        args.output += "_smoke"
        plan.update(screen_epochs=1, final_epochs=2, finalists_per_code=1, repeat_top=1, repeat_seeds=[46])
        plan["common"].update(device="cpu", workers=0, batch_size=4, max_train_frames=16, max_val_frames=16)
        plan["trials"] = [{"name": "smoke_a", "sensor_code_dim": 4, "channels": [4, 8]},
                          {"name": "smoke_b", "sensor_code_dim": 4, "channels": [8, 8]}]
    if len({t["name"] for t in plan["trials"]}) != len(plan["trials"]) or any(
            not re.fullmatch(r"[a-z0-9_]+", t["name"]) for t in plan["trials"]):
        raise ValueError("Trial names must be unique lowercase identifiers")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "sequence.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for directory in ("trials", "logs"):
        (output / directory).mkdir(exist_ok=True)
    # Search restart must use the same plan, data manifest and implementation.
    files = [Path(__file__), ROOT / "src/tactilelatent/tactile_autoencoder.py",
             ROOT / "src/tactilelatent/pretrain_egotouch_autoencoder.py",
             Path(plan["common"].get("manifest", "processing/joint_all/manifest.json"))]
    provenance = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    for name, content in (("plan.json", plan), ("provenance.json", provenance)):
        path = output / name
        if path.exists() and json.loads(path.read_text()) != content:
            raise ValueError(f"Search {name} changed; choose a new output directory")
        write_json(path, content)
    trials = {t["name"]: trial_config(plan, t) for t in plan["trials"]}
    try:
        for name, cfg in trials.items():
            run_trial(output, name, cfg, plan["screen_epochs"], "screening")
        selection = output / "screening_selection.json"
        if not selection.exists():
            rows = read_results(output)
            write_json(selection, {"results": rows, "finalists": choose_finalists(
                rows, plan["finalists_per_code"], plan.get("promotion_output_mode", plan["common"]["output_mode"]))})
        finalists = json.loads(selection.read_text())["finalists"]
        for name in finalists:
            run_trial(output, name, trials[name], plan["final_epochs"], "extended_training")
        selection = output / "repeat_selection.json"
        if not selection.exists():
            ranked = sorted((r for r in read_results(output) if r["name"] in finalists), key=lambda r: r["score"])
            write_json(selection, {"results": ranked, "repeat": [r["name"] for r in ranked[:plan["repeat_top"]]]})
        repeated = json.loads(selection.read_text())["repeat"]
        for name in repeated:
            for seed in plan["repeat_seeds"]:
                run_trial(output, f"{name}_seed{seed}", {**trials[name], "seed": seed},
                          plan["final_epochs"], "seed_repetitions")
        rows = read_results(output)
        stability = []
        for name in repeated:
            members = [r for r in rows if r["name"] in [name, *[f"{name}_seed{s}" for s in plan["repeat_seeds"]]]]
            scores = [r["score"] for r in members]
            mean = sum(scores) / len(scores)
            stability.append({"configuration": name, "seeds": [r["seed"] for r in members],
                              "mean_score": mean, "min_score": min(scores), "max_score": max(scores),
                              "sample_std": (sum((s - mean) ** 2 for s in scores) / max(len(scores) - 1, 1)) ** .5})
        write_json(output / "seed_stability.json", sorted(stability, key=lambda r: r["mean_score"]))
        write_json(output / "status.json", {"status": "complete", "trials": len(rows),
                   "note": "Validation-selected candidates; no official test evaluation performed."})
    except BaseException as error:
        status = json.loads((output / "status.json").read_text()) if (output / "status.json").exists() else {}
        write_json(output / "status.json", {**status, "status": "failed", "error": str(error)})
        raise


if __name__ == "__main__":
    main()
