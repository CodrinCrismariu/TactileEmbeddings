"""One seed, one FGW variant, two placements; no screening or finalist stage."""
import argparse
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import torch

from .aligned_frames import file_hash, write_json
from .fgw_mixer import MixerConfig
from .fgw_study import (BASE_CONFIG, DEFAULT_ROOT as FULL_ROOT, EXISTING_PARENT,
                        implementation_signature, save_config)
from .tactile_teacher import source_context
from .train_alignment import extension_config, check_resume_contract


ROOT = "processing/runs/fgw_two_placements_v1"


def signature():
    return {**implementation_signature(), "fgw_pilot.py": file_hash(__file__)}


def joint_config(spec, placement):
    root = Path(spec["root"])
    cfg = {**spec["base"], "seed": spec["seed"], "epochs": spec["completed_epoch"],
           "fork_parent_epochs": 10, "init_checkpoint": spec["parent"],
           "output": str(root / f"fgw_{placement}_partial_seed45"),
           "fgw": {**spec["mixer"], "placement": placement}}
    if placement == "tactile_first":
        checkpoint = root / "tactile_teacher" / "last.pt"
        cfg["fgw"].update(teacher_checkpoint=str(checkpoint), teacher_sha256=file_hash(checkpoint),
            teacher_cache=str(root / "common_codes"), data_signature=spec["data_signature"], teacher_weight=1.)
    return cfg


def prepare(root=ROOT):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    base = json.loads(Path(BASE_CONFIG).read_text())
    data, _, sources, _, _, _, _ = source_context(base)
    state = torch.load(EXISTING_PARENT, map_location="cpu", weights_only=False, mmap=True)
    if state["epoch"] != 9 or state["config"]["seed"] != 45 or state["provenance"]["data"] != data:
        raise ValueError("Expected the canonical seed-45 epoch-10 parent")
    if len(sources) != 11:
        raise ValueError("All eleven sources required")
    spec = {"profile": "two_placements_first_test", "root": str(root), "seed": 45,
            "placements": ["visual", "tactile_first"], "joint_runs": 2,
            "additional_joint_epochs_per_run": 10, "completed_epoch": 20,
            "screening": False, "finalist_selection": False, "additional_seeds": False,
            "parent": EXISTING_PARENT, "parent_sha256": file_hash(EXISTING_PARENT),
            "data_signature": data, "sources": sources, "heldout": "ftp_motiontrans",
            "base": base, "mixer": asdict(MixerConfig(variant="partial_stochastic_fgw")),
            "teacher_warmup_epochs": 20, "teacher_epochs": 70,
            "heads": {"shallow": [512], "deep": [512, 512]}, "label_fraction": 1.,
            "calibration_stage": False, "implementation_sha256": signature(),
            "comparison_note": "Comparison to the existing parent is exploratory; extra training is not controlled by a new no-FGW run"}
    save_config(root / "pilot.json", spec)
    save_config(root / "configs" / "visual.json", joint_config(spec, "visual"))
    teacher = {**base, "seed": 45, "output": str(root / "tactile_teacher"),
               "teacher_epochs": 70, "teacher_warmup_epochs": 20,
               "teacher_batch_size": 256, "teacher_validation_windows": 1024,
               "common_code_cache": str(root / "common_codes"),
               "fgw": {**spec["mixer"], "placement": "tactile_first"}}
    save_config(root / "configs" / "teacher.json", teacher)
    write_json(root / "resource_estimate.json", {
        "joint_epoch_equivalents": 20, "observed_control_seconds_per_epoch": 1529.,
        "base_joint_hours": 20 * 1529 / 3600, "estimated_total_hours": [9, 12],
        "note": "One necessary code-only teacher and four lightweight transfer fits; no ablation sweep"})
    return spec


def completed_training(checkpoint, cfg_path, epoch):
    if not Path(checkpoint).exists():
        return False, False
    state = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    cfg = json.loads(Path(cfg_path).read_text())
    if state["config"] != cfg:
        if not cfg.get("fgw") or cfg["epochs"] <= state["config"]["epochs"]:
            raise ValueError("Existing pilot training configuration differs")
        provenance = {**state["provenance"], "trainer_sha256": file_hash(
            Path(__file__).with_name("train_alignment.py"))}
        check_resume_contract(state, cfg, provenance, extending=True)
    return state["epoch"] + 1 >= epoch, True


def completed_evaluation(path, checkpoint, widths, seed):
    if not Path(path).exists():
        return False
    result = json.loads(Path(path).read_text())
    if (result["checkpoint_sha256"] != file_hash(checkpoint) or result["seed"] != seed
            or result["head"] != "-".join(map(str, [32, *widths, 6]))
            or result.get("train_fraction", 1.) != 1. or result["validation"]["sample_count"] != 64):
        raise ValueError("Existing pilot evaluation configuration differs")
    return True


def report(spec):
    root = Path(spec["root"])
    parent = torch.load(spec["parent"], map_location="cpu", weights_only=False, mmap=True)
    results = {}
    for placement in spec["placements"]:
        folder = root / f"fgw_{placement}_partial_seed45"
        checkpoint = folder / "last.pt"
        if not checkpoint.exists():
            continue
        state = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
        native = {}
        for source, metrics in state["validation"].items():
            baseline = parent["validation"][source]["normalized_mae"]
            native[source] = {"mae": metrics["normalized_mae"], "parent_mae": baseline,
                              "ratio_to_parent": metrics["normalized_mae"] / max(baseline, 1e-12)}
        transfer = {}
        for name in spec["heads"]:
            path = folder / "evaluations" / f"motiontrans_{name}" / "results.json"
            if path.exists():
                transfer[name] = json.loads(path.read_text())["validation"]
        results[placement] = {"completed_epoch": state["epoch"] + 1,
                              "native_reconstruction": native, "MotionTrans": transfer}
    write_json(root / "summary.json", {"results": results, "comparison_note": spec["comparison_note"],
        "uncertainty": "Reported latent-sampling coverage is uncalibrated; calibration is deferred in this first test"})
    return results


def run(spec):
    root = Path(spec["root"])
    lock = (root / "controller.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # Also prevent an accidental restart of the superseded full sweep.
    shared = Path(FULL_ROOT); shared.mkdir(parents=True, exist_ok=True)
    shared_lock = (shared / "controller.lock").open("a")
    fcntl.flock(shared_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use libraries matching the loaded NVIDIA kernel driver")
    if file_hash(spec["parent"]) != spec["parent_sha256"]:
        raise ValueError("Parent checkpoint changed")
    if shutil.disk_usage(root).free < 25 * 2**30:
        raise RuntimeError("Pilot needs 25 GiB of free storage")
    began = time.monotonic()
    status_path = root / "status.json"

    def job(name, command, cfg=None, checkpoint=None, epoch=None):
        if signature() != spec["implementation_sha256"]:
            raise ValueError("Pilot implementation changed")
        if checkpoint is not None:
            if Path(checkpoint).exists():
                state = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
                requested = json.loads(Path(cfg).read_text())
                if requested.get("fgw") and requested["epochs"] > state["config"]["epochs"]:
                    requested = extension_config(requested, state, requested["epochs"])
                    provenance = {**state["provenance"], "trainer_sha256": file_hash(
                        Path(__file__).with_name("train_alignment.py"))}
                    check_resume_contract(state, requested, provenance, extending=True)
                    write_json(cfg, requested)
                    command.extend(["--extend-epochs", str(requested["epochs"])])
                del state
            done, resume = completed_training(checkpoint, cfg, epoch)
            if done:
                return
            if resume:
                command.append("--resume")
        log_path = root / "logs" / f"{name}.log"; log_path.parent.mkdir(exist_ok=True)
        with log_path.open("a") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            current = {"stage": "running", "job": name, "joint_runs": 2, "seed": spec["seed"],
                       "controller_pid": os.getpid(), "worker_pid": process.pid,
                       "command": command, "log": str(log_path), "elapsed_seconds": time.monotonic() - began}
            write_json(status_path, current); print(json.dumps(current), flush=True)
            try:
                code = process.wait()
            except BaseException:
                process.terminate(); process.wait(); raise
        if code:
            raise RuntimeError(f"Pilot job {name} exited {code}; see {log_path}")
        if checkpoint is not None and not completed_training(checkpoint, cfg, epoch)[0]:
            raise RuntimeError("Training ended before its planned epoch")
        with (root / "jobs.jsonl").open("a") as stream:
            stream.write(json.dumps({"job": name, "status": "complete", "elapsed_seconds": time.monotonic() - began}) + "\n")

    try:
        cfg = root / "configs" / "visual.json"
        visual = joint_config(spec, "visual")
        job("visual_fgw", [sys.executable, "-u", "-m", "tactilelatent.train_alignment", "--config", str(cfg)],
            cfg, Path(visual["output"]) / "last.pt", spec["completed_epoch"])
        cfg = root / "configs" / "teacher.json"
        checkpoint = root / "tactile_teacher" / "last.pt"
        job("tactile_teacher", [sys.executable, "-u", "-m", "tactilelatent.tactile_teacher", "--config", str(cfg)],
            cfg, checkpoint, 70)
        job("common_code_cache", [sys.executable, "-u", "-m", "tactilelatent.tactile_teacher", "--config", str(cfg), "--cache-only"])
        tactile = joint_config(spec, "tactile_first")
        cfg = save_config(root / "configs" / "tactile_first.json", tactile)
        job("tactile_first_fgw", [sys.executable, "-u", "-m", "tactilelatent.train_alignment", "--config", str(cfg)],
            cfg, Path(tactile["output"]) / "last.pt", spec["completed_epoch"])
        # Fixed final checkpoints: no validation-driven selection or extension.
        for placement in spec["placements"]:
            folder = Path(joint_config(spec, placement)["output"])
            checkpoint = folder / f"epoch_{spec['completed_epoch']:03d}.pt"
            for name, widths in spec["heads"].items():
                destination = folder / "evaluations" / f"motiontrans_{name}"
                if completed_evaluation(destination / "results.json", checkpoint, widths, spec["seed"]):
                    continue
                command = [sys.executable, "-u", "-m", "tactilelatent.evaluate_holdout_decoder",
                    "--checkpoint", str(checkpoint), "--manifest", spec["base"]["manifest"],
                    "--source", "ftp_motiontrans", "--output", str(destination),
                    "--cache-dir", str(folder / "evaluations" / "motiontrans_features"),
                    "--hidden-dims", *map(str, widths), "--seed", str(spec["seed"]),
                    "--samples", "64", "--epochs", "100", "--patience", "10",
                    "--train-fraction", "1.0", "--workers", str(spec["base"]["workers"])]
                job(f"transfer_{placement}_{name}", command)
        report(spec)
        write_json(status_path, {"stage": "complete", "joint_runs": 2, "elapsed_seconds": time.monotonic() - began})
    except BaseException as error:
        previous = json.loads(status_path.read_text()) if status_path.exists() else {}
        write_json(status_path, {**previous, "stage": "failed", "error": str(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "report"))
    parser.add_argument("--root", default=ROOT)
    args = parser.parse_args()
    spec = prepare(args.root) if args.action == "prepare" else json.loads((Path(args.root) / "pilot.json").read_text())
    if args.action == "run":
        run(spec)
    elif args.action == "report":
        print(json.dumps(report(spec)))


if __name__ == "__main__":
    main()
