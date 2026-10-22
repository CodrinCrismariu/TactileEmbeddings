"""Detached, resumable queue for the five explicitly selected invariance runs."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from tactilelatent.aligned_frames import file_hash, write_json

ROOT = Path("processing/runs/invariance_16d_v1")
ORDER = ("control", "relational", "meta", "meta_control", "trajectory")


def configurations(root):
    configs = {name: json.loads((root / "configs" / f"{name}.json").read_text()) for name in ORDER}
    common = [{k: v for k, v in cfg.items() if k not in ("output", "relational", "trajectory", "meta")}
              for cfg in configs.values()]
    if any(c != common[0] for c in common[1:]):
        raise ValueError("Conditions differ outside their explicit objectives")
    if any(c["latent_dim"] != 16 or c["epochs"] != 20 or c["parent_epochs"] != 10
           or c["heldout_source"] != "ftp_motiontrans" or c.get("private_dim", 0)
           or Path(c["output"]).resolve() != (root / name).resolve() for name, c in configs.items()):
        raise ValueError("Unexpected architecture, budget, holdout, or output directory")
    if configs["meta_control"]["meta"]["weight"] != 0 or not configs["meta_control"]["meta"]["enabled"]:
        raise ValueError("The matched meta control must execute zero-weight query tasks")
    return configs


def prepare(root):
    if (root / "plan.json").exists():
        raise FileExistsError("Study already prepared; use run to resume the queue")
    root.mkdir(parents=True, exist_ok=True)
    for name in ORDER:
        cfg = json.loads(Path(f"configs/invariance_16d_{name}.json").read_text())
        cfg["output"] = str(root / name)
        write_json(root / "configs" / f"{name}.json", cfg)
    configs = configurations(root)
    # Complete all read-only checks before allowing any expensive worker to start.
    from tactilelatent.train_invariance import prepare as preflight
    for name, cfg in configs.items():
        checked = preflight(cfg)
        del checked
        print(json.dumps({"event": "preflight_passed", "condition": name}), flush=True)
    write_json(root / "plan.json", {"order": list(ORDER), "seed": 45, "parent_epoch": 10,
        "target_epoch": 20, "additional_epochs_each": 10, "conditions": {
            name: {"config": str(root / "configs" / f"{name}.json"),
                   "sha256": file_hash(root / "configs" / f"{name}.json")} for name in ORDER},
        "notes": "Five opt-in conditions; two simultaneous by default. No static/private context, new seeds, automatic transfer evaluations, or hyperparameter changes."})


def read_json(path, fallback):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return fallback


def completed_epoch(folder):
    path = folder / "metrics.jsonl"
    if path.exists():
        for line in reversed(path.read_text().splitlines()):
            try:
                return int(json.loads(line)["epoch"])
            except (ValueError, KeyError):
                continue
    return 10


def run(root, parallel):
    if not 1 <= parallel <= len(ORDER):
        raise ValueError("Parallel worker limit must be between one and five")
    lock = (root / "controller.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = json.loads((root / "plan.json").read_text())
    configs = configurations(root)
    for item in plan["conditions"].values():
        if file_hash(item["config"]) != item["sha256"]:
            raise ValueError("Frozen run configuration changed after preflight")
    if shutil.disk_usage(root).free < 20 * 2**30:
        raise RuntimeError("At least 20 GiB free required for inference and optimizer checkpoints")
    stopping = [False]
    def stop(signum, frame):
        stopping[0] = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    processes, streams = {}, {}
    states = {name: {"stage": "queued", "pid": None, "completed_epoch": completed_epoch(root / name),
                    "target_epoch": 20, "output": str(root / name), "log": str(root / "logs" / f"{name}.log")}
              for name in ORDER}
    for name, state in states.items():
        if state["completed_epoch"] >= 20 and (root / name / "last.pt").exists():
            state["stage"] = "complete"
    began, matching_hash = time.monotonic(), None
    def snapshot(stage):
        write_json(root / "status.json", {"stage": stage, "controller_pid": os.getpid(),
            "max_parallel": parallel, "elapsed_seconds": time.monotonic() - began,
            "matched_initialization_sha256": matching_hash, "runs": states})
    try:
        while not stopping[0]:
            for name, process in list(processes.items()):
                code = process.poll()
                row = states[name]
                row["completed_epoch"] = completed_epoch(root / name)
                row["progress"] = read_json(root / name / "status.json", {})
                initialization = read_json(root / name / "initialization.json", {})
                if initialization:
                    current = initialization["model_sha256"]
                    matching_hash = matching_hash or current
                    if current != matching_hash:
                        raise ValueError("Initial model weights differ between conditions")
                if code is not None:
                    row.update(pid=None, exit_code=code,
                        stage="complete" if code == 0 and row["completed_epoch"] == 20 else "failed")
                    streams.pop(name).close(); processes.pop(name)
                    print(json.dumps({"event": "worker_exited", "condition": name, **row}), flush=True)
            for name in ORDER:
                if len(processes) >= parallel:
                    break
                if states[name]["stage"] != "queued":
                    continue
                command = [sys.executable, "-u", "-m", "tactilelatent.train_invariance", "--config", plan["conditions"][name]["config"]]
                if (root / name / "last.pt").exists():
                    command.append("--resume")
                log = Path(states[name]["log"]); log.parent.mkdir(parents=True, exist_ok=True)
                streams[name] = log.open("a")
                process = subprocess.Popen(command, stdout=streams[name], stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL, start_new_session=True)
                processes[name] = process
                states[name].update(stage="starting", pid=process.pid, process_group=process.pid)
                write_json(root / f"{name}.pid", {"pid": process.pid, "process_group": process.pid, "command": command})
                print(json.dumps({"event": "started", "condition": name, "pid": process.pid}), flush=True)
            for name in processes:
                if states[name].get("progress", {}).get("stage") == "training":
                    states[name]["stage"] = "training"
            terminal = all(row["stage"] in ("complete", "failed") for row in states.values())
            snapshot("complete_with_failures" if terminal and any(row["stage"] == "failed" for row in states.values())
                     else "complete" if terminal else "running")
            if terminal:
                return
            time.sleep(5)
    finally:
        for name, process in processes.items():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for name, process in processes.items():
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL); process.wait()
            states[name].update(stage="stopped", pid=None, completed_epoch=completed_epoch(root / name))
        for stream in streams.values():
            stream.close()
        if processes or stopping[0]:
            snapshot("stopped" if stopping[0] else "controller_failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--parallel", type=int, default=2)
    args = parser.parse_args()
    prepare(args.root) if args.action == "prepare" else run(args.root, args.parallel)
