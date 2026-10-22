"""Resumable tactile-only AE searches for all native sensor layouts.

Reuse completed EgoTouch runs. Bootstrap one model per remaining source, then
screen size/code/rate ablations, extend one finalist per code, and repeat the
winner on two more seeds. The viewer discovers checkpoints as they are saved.
"""
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from tactilelatent.joint_all_config import ALL_TARGETS
from tactilelatent.sensor_cache import build_cache, write_json
from tactilelatent.native_optical import native_shape

ROOT = Path(__file__).resolve().parents[1]
ORDER = ["ftp_rdp", "ftp_rdp_bimanual", "ftp_motiontrans", "opengraph", "egotactile",
         "opentouch", "ftp_vla_touch", "ftp_unit", "ftp_unit_bimanual", "ftp_vitamin", "freetacman"]


def make_plan(args):
    sources = {}
    for source in args.sources:
        kind, shape, _ = ALL_TARGETS[source]
        if kind == "optical_rgb":
            codes, architecture, widths = [64, 128, 256], "conv", [[8, 16, 32, 64, 64], [16, 32, 64, 128, 128], [32, 64, 128, 256, 256]]
            row = next(r for r in json.loads(Path(args.manifest).read_text())["episodes"] if r["source"] == source)
            shape = native_shape(row)
        elif source == "opentouch":
            codes, architecture, widths = [16, 64, 128], "conv", [[16, 32], [32, 64], [64, 128]]
        elif source == "egotactile":
            codes, architecture, widths = [16, 64, 128], "mlp", [[128], [256, 128], [512, 256]]
        elif source == "opengraph":
            codes, architecture, widths = [8, 32, 64], "mlp", [[128], [256, 128], [512, 256]]
        else:
            codes = [1] if shape[0] == 1 else [1, 2] if shape[0] == 2 else [1, 2, 4]
            architecture, widths = "mlp", [[32], [128, 64], [256, 128]]
        lr = 1e-3 if kind in ("force_tanh", "torque_tanh") else 1e-4
        key = "channels" if architecture == "conv" else "hidden_dims"
        trials = []
        for label, width in zip(["small", "medium", "large"], widths):
            for code in codes:
                trials.append({"name": f"{label}_code{code}", "architecture": architecture,
                               "sensor_code_dim": code, key: width, "learning_rate": lr})
                if kind == "optical_rgb":
                    trials[-1].update(native_optical=True, amp=True, batch_size=4 if source == "freetacman" else 8)
        anchor_code = codes[len(codes) // 2]
        anchor = next(t for t in trials if t["name"] == f"medium_code{anchor_code}")
        for label, changes in [("lr_low", {"learning_rate": lr * .3}),
                               ("lr_high", {"learning_rate": lr * 3}),
                               ("mae025", {"mae_weight": .25})]:
            trials.append({**anchor, **changes, "name": f"{label}_code{anchor_code}"})
        if args.smoke:
            trials = [{**anchor, key: [4, 8] if architecture == "conv" else [16], "name": "smoke"}]
        sources[source] = {"target_kind": kind, "target_shape": shape, "trials": trials,
                           "bootstrap": trials[0]["name"] if args.smoke else anchor["name"]}
    return {"common": {"manifest": args.manifest, "cache": args.cache, "batch_size": 128,
                       "weight_decay": 1e-4, "mae_weight": .5, "contact_weight": 3.,
                       "frame_stride": 2, "max_train_frames": 100000, "max_val_frames": 0,
                       "workers": 2, "seed": 45, "device": "cuda"},
            "screen_epochs": args.screen_epochs, "final_epochs": args.final_epochs,
            "bootstrap_epochs": min(3, args.screen_epochs), "repeat_seeds": [46, 47],
            "sources": sources, "egotouch": "Reuse preserved v4 and initial grid checkpoints"}


def results(output):
    plan_path = output / "plan.json"
    plan = json.loads(plan_path.read_text()) if plan_path.exists() else None
    candidates = ({source: {trial["name"] for trial in spec["trials"]}
                   for source, spec in plan["sources"].items()} if plan else None)
    if candidates is not None:
        for names in candidates.values():
            names.update(f"{name}_seed{seed}" for name in list(names) for seed in plan["repeat_seeds"])
    rows = []
    for run in (output / "trials").glob("*/*"):
        if not (run / "best_metrics.json").exists(): continue
        best = json.loads((run / "best_metrics.json").read_text())
        cfg = json.loads((run / "config.json").read_text())
        if candidates is not None and run.name not in candidates.get(cfg["source"], set()):
            continue  # Preserve historical starters without promoting excluded architectures.
        history = [json.loads(s) for s in (run / "metrics.jsonl").read_text().splitlines()]
        if not history: continue
        rows.append({"source": cfg["source"], "name": run.name, "sensor_code_dim": cfg["latent_dim"],
                     "architecture": cfg["architecture"], "seed": cfg["seed"],
                     "completed_epochs": history[-1]["epoch"], "best_epoch": best["epoch"],
                     "score": best["selection_score"], "mae": best["val"]["mae"],
                     "contact_mae": best["val"]["contact_mae"], "contact_f1": best["val"]["contact_f1"],
                     "psnr": best["val"]["psnr"],
                     "mean_epoch_seconds": sum(h["seconds"] for h in history) / len(history),
                     "checkpoint": str(run / "best.pt")})
    if rows:
        with (output / "comparison.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader()
            writer.writerows(sorted(rows, key=lambda r: (r["source"], r["score"])))
        registry = {}
        for source in {r["source"] for r in rows}:
            registry[source] = min((r for r in rows if r["source"] == source), key=lambda r: r["score"])
        write_json(output / "best_by_source.json", registry)
    return rows


def select_finalists(rows, mode="per_code"):
    """Promote only validation-screened candidates under the requested policy."""
    if not rows:
        raise ValueError("No screened candidates to promote")
    if mode == "best_one":
        return [min(rows, key=lambda row: row["score"])["name"]]
    if mode != "per_code":
        raise ValueError(f"Unknown finalist selection: {mode}")
    return [min((row for row in rows if row["sensor_code_dim"] == code),
                key=lambda row: row["score"])["name"]
            for code in sorted({row["sensor_code_dim"] for row in rows})]


def screening_results(output, source, names, epochs):
    """Compare the same screening budget, even if an old plan extended a trial."""
    rows = []
    for row in results(output):
        if row["source"] != source or row["name"] not in names:
            continue
        path = output / "trials" / source / row["name"] / "metrics.jsonl"
        history = [json.loads(line) for line in path.read_text().splitlines()]
        screened = [record for record in history if record["epoch"] <= epochs]
        if not screened or max(record["epoch"] for record in screened) != epochs:
            raise ValueError(f"Candidate has not completed screening: {source}/{row['name']}")
        best = min(screened, key=lambda record: record["selection_score"])
        rows.append({**row, "score": best["selection_score"], "screen_best_epoch": best["epoch"]})
    return rows


def run_trial(output, source, name, cfg, epochs, stage):
    run = output / "trials" / source / name
    command = [sys.executable, "-u", "scripts/pretrain_sensor_autoencoder.py", "--source", source]
    for key, value in cfg.items():
        if isinstance(value, bool):
            if value: command.append("--" + key.replace("_", "-"))
            continue
        command.append("--" + key.replace("_", "-"))
        command.extend(str(x) for x in (value if isinstance(value, list) else [value]))
    command.extend(["--epochs", str(epochs), "--output", str(run)])
    if (run / "last.pt").exists():
        import torch
        checkpoint = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
        for key, value in cfg.items():
            stored = "latent_dim" if key == "sensor_code_dim" else key
            if checkpoint["config"][stored] != value: raise ValueError(f"Trial changed at {key}: {run}")
        if checkpoint["epoch"] + 1 >= epochs: return
        command.append("--resume")
        del checkpoint
    elif run.exists():
        archived = output / "uncheckpointed" / source
        archived.mkdir(parents=True, exist_ok=True)
        run.rename(archived / f"{name}_{time.time_ns()}")
    log = output / "logs" / f"{source}_{name}.log"
    write_json(output / "status.json", {"status": "running", "stage": stage, "source": source,
               "trial": name, "target_epochs": epochs, "log": str(log)})
    print(json.dumps({"event": "start", "source": source, "trial": name, "stage": stage, "epochs": epochs}), flush=True)
    with log.open("a") as stream:
        subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                       env={**os.environ, "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}, check=True)
    results(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="processing/joint_all/manifest.json")
    parser.add_argument("--cache", default="processing/sensor_ae_cache_v1")
    parser.add_argument("--output", default="processing/runs/sensor_autoencoders_native_v2")
    parser.add_argument("--plan", help="Optional explicit JSON plan; overrides generated source/budget settings")
    parser.add_argument("--sources", nargs="+", choices=ORDER, default=ORDER)
    parser.add_argument("--screen-epochs", type=int, default=30)
    parser.add_argument("--final-epochs", type=int, default=100)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(); os.chdir(ROOT)
    if args.plan and args.smoke: parser.error("Use generated plans for --smoke")
    if not 1 <= args.screen_epochs <= args.final_epochs: parser.error("Require 1 <= screening <= finalist epochs")
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    lock = (output / "sequence.lock").open("w"); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (output / "logs").mkdir(exist_ok=True); (output / "trials").mkdir(exist_ok=True)
    if args.smoke:
        rows = json.loads(Path(args.manifest).read_text())["episodes"]
        tiny = []
        for source in args.sources:
            for split in ("train", "val"):
                row = next(r for r in rows if r["source"] == source and r["split"] == split)
                tiny.append({**row, "frames": min(16, row["frames"])})
        args.manifest = str(output / "smoke_manifest.json")
        write_json(Path(args.manifest), {"schema_version": 1, "episodes": tiny})
        args.cache = str(output / "cache")
        args.screen_epochs, args.final_epochs = 1, 2
    plan = json.loads(Path(args.plan).read_text()) if args.plan else make_plan(args)
    args.manifest, args.cache = plan["common"]["manifest"], plan["common"]["cache"]
    if not 1 <= plan["bootstrap_epochs"] <= plan["screen_epochs"] <= plan["final_epochs"]:
        raise ValueError("Invalid plan epoch budgets")
    for source, spec in plan["sources"].items():
        if source not in ORDER: raise ValueError(f"Invalid source: {source}")
        row = next(r for r in json.loads(Path(args.manifest).read_text())["episodes"] if r["source"] == source)
        expected = native_shape(row) if ALL_TARGETS[source][0] == "optical_rgb" else ALL_TARGETS[source][1]
        if spec["target_shape"] != expected:
            raise ValueError(f"Invalid source/target layout in plan: {source}")
        if ALL_TARGETS[source][0] == "optical_rgb" and any(not t.get("native_optical") for t in spec["trials"]):
            raise ValueError("All optical trials must explicitly use full source resolution")
    if args.smoke:
        plan["common"].update(device="cpu", workers=0, batch_size=8, max_train_frames=16, max_val_frames=16)
        plan["repeat_seeds"] = [46]
    files = [Path(__file__), Path(args.manifest), *[ROOT / "src/tactilelatent" / name for name in
             ("sensor_data.py", "sensor_cache.py", "sensor_autoencoder.py", "pretrain_sensor_autoencoder.py",
              "joint_data.py", "media.py", "pretrain_egotouch_autoencoder.py", "native_optical.py")]]
    provenance = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    if args.plan: provenance[args.plan] = hashlib.sha256(Path(args.plan).read_bytes()).hexdigest()
    for filename, payload in [("plan.json", plan), ("provenance.json", provenance)]:
        path = output / filename
        if path.exists() and json.loads(path.read_text()) != payload:
            raise ValueError(f"{filename} changed; choose a new output directory")
        write_json(path, payload)
    try:
        import torch
        torch.set_num_threads(4)
        # First make one real model available for every source, before full searches.
        for source, spec in plan["sources"].items():
            write_json(output / "status.json", {"status": "running", "stage": "cache", "source": source})
            if spec["target_kind"] != "optical_rgb":
                build_cache(args.manifest, source, args.cache, plan["common"]["frame_stride"])
            trial = next(t for t in spec["trials"] if t["name"] == spec["bootstrap"])
            cfg = {**plan["common"], **{k:v for k,v in trial.items() if k != "name"}}
            run_trial(output, source, trial["name"], cfg, plan["bootstrap_epochs"], "bootstrap")
        for source, spec in plan["sources"].items():
            trials = {t["name"]: {**plan["common"], **{k:v for k,v in t.items() if k != "name"}} for t in spec["trials"]}
            for name, cfg in trials.items(): run_trial(output, source, name, cfg, plan["screen_epochs"], "screening")
            selection = output / f"{source}_finalists.json"
            if not selection.exists():
                rows = screening_results(output, source, trials, plan["screen_epochs"])
                selected = select_finalists(rows, plan.get("finalist_selection", "per_code"))
                write_json(selection, {"finalists": selected, "screen_epochs": plan["screen_epochs"],
                           "screening_scores": {row["name"]: row["score"] for row in rows}})
            finalists = json.loads(selection.read_text())["finalists"]
            for name in finalists: run_trial(output, source, name, trials[name], plan["final_epochs"], "finalists")
            repeat_path = output / f"{source}_repeat.json"
            if not repeat_path.exists():
                winner = min((r for r in results(output) if r["source"] == source and r["name"] in finalists),key=lambda r:r["score"])["name"]
                write_json(repeat_path, {"winner": winner})
            winner = json.loads(repeat_path.read_text())["winner"]
            for seed in plan["repeat_seeds"]:
                run_trial(output, source, f"{winner}_seed{seed}", {**trials[winner], "seed": seed}, plan["final_epochs"], "seed_repeats")
        rows = results(output); stability = []
        for source in plan["sources"]:
            winner = json.loads((output / f"{source}_repeat.json").read_text())["winner"]
            selected = [r for r in rows if r["source"] == source and r["name"] in [winner, *[f"{winner}_seed{s}" for s in plan["repeat_seeds"]]]]
            values = [r["score"] for r in selected]
            stability.append({"source":source,"configuration":winner,"seeds":[r["seed"] for r in selected],
                              "mean_score":float(np.mean(values)),"sample_std":float(np.std(values,ddof=1)) if len(values)>1 else None})
        write_json(output / "seed_stability.json", stability)
        write_json(output / "status.json", {"status":"complete","sources":len(plan["sources"]),"trials":len(rows),
                   "note":"Tactile-only pretraining; validation-selected. EgoTouch models preserved separately."})
    except BaseException as error:
        state = json.loads((output / "status.json").read_text()) if (output / "status.json").exists() else {}
        write_json(output / "status.json", {**state,"status":"failed","error":str(error)});raise


if __name__ == "__main__":
    main()
