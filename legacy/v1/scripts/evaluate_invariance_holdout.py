"""Matched MotionTrans evaluation while the invariance study remains paused."""
import argparse
import fcntl
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from tactilelatent.aligned_frames import manifest_signature, write_json
from tactilelatent.alignment_data import evaluation_dataset
from tactilelatent.alignment_model import model_from_state
from tactilelatent.frozen_probe import select_indices, sha256
from tactilelatent.model import DinoBackbone, MODEL_ID, REVISION
from tactilelatent.paired_probe_cache import extract_split


def paired_difference(a, b, seed=45):
    x = {r["episode_id"]: r for r in a["validation"]["episode_errors"]}
    y = {r["episode_id"]: r for r in b["validation"]["episode_errors"]}
    if set(x) != set(y) or any(x[e]["count"] != y[e]["count"] for e in x):
        raise ValueError("Validation episode pairing differs")
    episodes = sorted(x)
    difference = np.array([y[e]["absolute_sum"] - x[e]["absolute_sum"] for e in episodes])
    count = np.array([x[e]["count"] for e in episodes])
    draws = np.random.default_rng(seed).integers(len(episodes), size=(5000, len(episodes)))
    sampled = difference[draws].sum(1) / count[draws].sum(1)
    return {"relational_minus_control_mae": float(difference.sum() / count.sum()),
            "paired_episode_bootstrap_ci95": np.quantile(sampled, [.025, .975]).tolist(),
            "episodes": len(episodes), "bootstrap_replicates": 5000}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="processing/runs/invariance_16d_v1")
    parser.add_argument("--epoch", type=int, default=15)
    parser.add_argument("--runs", nargs="+", default=["control", "relational"],
                        choices=["control", "relational", "meta", "meta_control"],
                        help="Completed conditions to compare with the same held-out protocol")
    parser.add_argument("--heads", nargs="+", choices=["shallow", "deep", "deeper", "deepest"],
                        default=["shallow", "deep"])
    parser.add_argument("--report-tag", default="", help="Separate aggregate report folder, e.g. deeper_heads")
    args = parser.parse_args()
    if len(set(args.runs)) != len(args.runs):
        parser.error("Select each run once")
    if len(set(args.heads)) != len(args.heads):
        parser.error("Select each head once")
    if args.report_tag and Path(args.report_tag).name != args.report_tag:
        parser.error("Report tag must be a folder name")
    root = Path(args.root)
    output = root / "evaluations" / (f"epoch_{args.epoch:03d}" + (f"_{args.report_tag}" if args.report_tag else ""))
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "controller.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    began = time.monotonic()
    def status(stage, **details):
        row = {"stage": stage, "epoch": args.epoch, "controller_pid": os.getpid(),
               "elapsed_seconds": time.monotonic() - began, "runs": args.runs, "heads": args.heads, **details}
        write_json(output / "status.json", row)
        print(json.dumps(row), flush=True)
    try:
        paused = json.loads((root / "pause_status.json").read_text())
        for pid in [paused["controller_pid"]] + [r["pid"] for r in paused["paused_groups"]]:
            if "State:\tT" not in Path(f"/proc/{pid}/status").read_text():
                raise ValueError("Training must remain paused for this evaluation")
        torch.set_num_threads(4)
        device = torch.device("cuda")
        models, snapshots, digests = {}, {}, {}
        manifest, data_signature = None, None
        for name in args.runs:
            checkpoint = root / name / "viewer_epochs" / f"epoch_{args.epoch:03d}.pt"
            folder = root / name / "evaluations" / f"motiontrans_epoch_{args.epoch:03d}"
            folder.mkdir(parents=True, exist_ok=True)
            snapshot = folder / "checkpoint.pt"
            if not snapshot.exists():
                os.link(checkpoint, snapshot)
            if sha256(snapshot) != sha256(checkpoint):
                raise ValueError("Immutable evaluation snapshot changed")
            state = torch.load(snapshot, map_location="cpu", weights_only=False, mmap=True)
            if (state["epoch"] + 1 != args.epoch or state["config"]["heldout_source"] != "ftp_motiontrans"
                    or state["backbone"] != MODEL_ID or state["backbone_revision"] != REVISION
                    or state["config"]["latent_dim"] != 16):
                raise ValueError("Invalid held-out checkpoint")
            current_manifest = state["config"]["manifest"]
            if state["provenance"]["data"] != manifest_signature(current_manifest):
                raise ValueError("Canonical data provenance changed")
            if manifest is not None and (manifest != current_manifest or data_signature != state["provenance"]["data"]):
                raise ValueError("Checkpoint data contracts differ")
            manifest, data_signature = current_manifest, state["provenance"]["data"]
            models[name] = model_from_state(state).to(device).eval().requires_grad_(False)
            snapshots[name], digests[name] = snapshot, sha256(snapshot)
            del state
        backbone = DinoBackbone().to(device).eval().requires_grad_(False)
        manifest_hash = sha256(manifest)
        for split in ("train", "val"):
            dataset = evaluation_dataset(manifest, split=split, sources=["ftp_motiontrans"])
            indices = select_indices(dataset)
            destinations = {name: snapshot.parent / "features" / f"{split}_features.pt" for name, snapshot in snapshots.items()}
            extract_split(backbone, models, manifest, "ftp_motiontrans", split, indices, destinations,
                          digests, manifest_hash, device, batch_size=8, workers=4,
                          progress=lambda split, batch, batches: status("extracting_features", split=split, batch=batch, batches=batches))
        del backbone, models, dataset
        gc.collect(); torch.cuda.empty_cache()
        results = {}
        for name in args.runs:
            report = f"motiontrans_epoch_{args.epoch:03d}_{args.report_tag or 'paused'}_evaluation"
            command = [sys.executable, "-u", "scripts/evaluate_alignment_checkpoint_sweep.py",
                       "--run", str(root / name), "--epochs", str(args.epoch), "--report-name", report,
                       "--head-epochs", "100", "--patience", "10", "--samples", "64", "--seed", "45", "--workers", "2",
                       "--checkpoint-kind", "viewer", "--heads", *args.heads]
            with (output / f"{name}.log").open("a") as log:
                process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                status("fitting_heads", run=name, worker_pid=process.pid, command=command)
                if process.wait():
                    raise RuntimeError(f"{name} evaluation failed; see {output}/{name}.log")
            result = json.loads((root / name / "evaluations" / report / "summary.json").read_text())
            if result["completed_evaluations"] != len(args.heads) or result["failures"]:
                raise ValueError("All selected transfer heads must complete")
            results[name] = result["results"]
            write_json(output / "partial_results.json", {"epoch": args.epoch, "results": results})
        comparisons = {}
        for reference, candidate in (("control", "relational"), ("meta_control", "meta")):
            if reference not in results or candidate not in results:
                continue
            pair = {}
            for head in args.heads:
                a = next(r for r in results[reference] if r["head"] == head)
                b = next(r for r in results[candidate] if r["head"] == head)
                full_a = json.loads(Path(a["results"]).read_text()); full_b = json.loads(Path(b["results"]).read_text())
                paired = paired_difference(full_a, full_b)
                if len(args.runs) == 2 and set(args.runs) == {"control", "relational"}:
                    pair[head] = {"control_mae": a["deterministic_mae"], "relational_mae": b["deterministic_mae"],
                        "relative_change_percent": 100 * (b["deterministic_mae"] / a["deterministic_mae"] - 1),
                        "control_coverage90": a["coverage_90"], "relational_coverage90": b["coverage_90"], **paired}
                else:
                    delta = paired.pop("relational_minus_control_mae")
                    pair[head] = {"reference_run": reference, "candidate_run": candidate,
                        "reference_mae": a["deterministic_mae"], "candidate_mae": b["deterministic_mae"],
                        "relative_change_percent": 100 * (b["deterministic_mae"] / a["deterministic_mae"] - 1),
                        "reference_coverage90": a["coverage_90"], "candidate_coverage90": b["coverage_90"],
                        "candidate_minus_reference_mae": delta, **paired}
            if len(args.runs) == 2 and set(args.runs) == {"control", "relational"}:
                comparisons = pair
            else:
                comparisons[f"{candidate}_vs_{reference}"] = pair
        write_json(output / "summary.json", {"epoch": args.epoch, "results": results, "comparisons": comparisons,
            "protocol": "Same full MotionTrans train/validation episodes; 100 head epochs maximum, patience 10, seed 45, 64 MC draws. Shared frozen DINO extraction, separate frozen encoders.",
            "training_state": "Paused; no automatic resume", "note": "One-seed exploratory development benchmark; uncertainty intervals remain uncalibrated."})
        status("complete", summary=str(output / "summary.json"), comparisons=comparisons)
    except Exception as error:
        status("failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
