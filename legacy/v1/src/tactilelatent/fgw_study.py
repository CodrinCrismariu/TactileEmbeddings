"""Prepare, benchmark, run and report the full resumable 3-seed FGW study.

One controller owns the GPU. Finalists are selected from source validation
before any MotionTrans evaluation. All process logs and jobs remain reviewable.
"""
import argparse
import csv
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

import numpy as np
import torch

from .aligned_frames import file_hash, write_json
from .fgw_data import CodeOnlyDataset, sample_banks
from .fgw_mixer import MixerConfig, VARIANTS, fit_mixer
from .tactile_teacher import TactileTeacher, source_context


DEFAULT_ROOT = "processing/runs/fgw_study_v1"
BASE_CONFIG = "configs/joint_alignment_holdout_motiontrans.json"
EXISTING_PARENT = "processing/runs/joint_alignment_holdout_motiontrans/epoch_010.pt"


def implementation_signature():
    return {name: file_hash(Path(__file__).with_name(name)) for name in (
        "fgw_study.py", "fgw_mixer.py", "fgw_data.py", "fgw_objective.py", "tactile_teacher.py",
        "train_alignment.py", "alignment_model.py", "evaluate_holdout_decoder.py", "viewer_checkpoints.py",
        "evaluate_fgw_calibration.py")}


def save_config(path, cfg):
    path = Path(path)
    if path.exists() and json.loads(path.read_text()) != cfg:
        raise ValueError(f"Refusing to reinterpret an existing study configuration: {path}")
    write_json(path, cfg)
    return path


def prepare(root=DEFAULT_ROOT):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    base = json.loads(Path(BASE_CONFIG).read_text())
    signature, _, sources, _, _, _, _ = source_context(base)
    if len(sources) != 11:
        raise ValueError("Expected all 11 source datasets")
    existing = torch.load(EXISTING_PARENT, map_location="cpu", weights_only=False, mmap=True)
    if existing["epoch"] != 9 or existing["config"]["seed"] != 45 or existing["provenance"]["data"] != signature:
        raise ValueError("Existing seed-45 parent differs from the canonical study")
    del existing
    common = asdict(MixerConfig())
    specification = {"schema_version": 1, "root": str(root), "seeds": [45, 46, 47],
        "sources": sources, "heldout": "ftp_motiontrans", "placements": ["visual", "tactile_first"],
        "variants": list(VARIANTS), "screen_completed_epoch": 20, "final_completed_epoch": 40,
        "parent_completed_epoch": 10, "teacher_warmup_epochs": 20, "teacher_epochs": 70,
        "heads": {"shallow": [512], "deep": [512, 512]}, "label_fractions": [.1, .25, 1.],
        "existing_parent": EXISTING_PARENT, "existing_parent_sha256": file_hash(EXISTING_PARENT),
        "data_signature": signature, "base": base, "mixer": common,
        "implementation_sha256": implementation_signature(),
        "selection": "Equal-source native MAE ratios, no source >10% worse than placement-matched control; never MotionTrans",
        "uncertainty": "Unchanged latent-sampling coverage; stochastic masks are augmentation, not predictive uncertainty"}
    save_config(root / "study.json", specification)
    for seed in specification["seeds"]:
        if seed != 45:
            # Match the original parent's 50-epoch LR schedule, then stop at 10.
            cfg = {**base, "seed": seed, "output": str(root / f"parent_seed{seed}")}
            save_config(root / "configs" / f"parent_seed{seed}.json", cfg)
        warm = {**base, "seed": seed, "output": str(root / f"teacher_warmup_seed{seed}"),
                "teacher_warmup_epochs": 20, "teacher_epochs": 70,
                "teacher_batch_size": 256, "teacher_validation_windows": 1024,
                "fgw": {**common, "variant": "none", "placement": "tactile_first"}}
        save_config(root / "configs" / f"teacher_warmup_seed{seed}.json", warm)
        for variant in specification["variants"]:
            teacher = {**warm, "output": str(root / f"teacher_{variant}_seed{seed}"),
                "init_teacher_checkpoint": str(root / f"teacher_warmup_seed{seed}" / "last.pt"),
                "common_code_cache": str(root / "common_codes" / f"{variant}_seed{seed}"),
                "fgw": {**common, "variant": variant, "placement": "tactile_first",
                        "structure_weight": 1. if variant == "gw" else .5}}
            save_config(root / "configs" / f"teacher_{variant}_seed{seed}.json", teacher)
            visual_config(specification, seed, variant, "visual")
    write_json(root / "resource_estimate.json", {"visual_epoch_equivalents": 500,
        "historical_seconds_per_epoch": 1477.56, "base_visual_gpu_hours": 205.22,
        "provisional_days": [10, 14], "teacher_and_evaluation_time": "benchmark then measure separately",
        "available_disk_gib": shutil.disk_usage(root).free / 2**30,
        "estimated_disk_budget_gib": 200})
    return specification


def visual_config(study, seed, variant, placement):
    root = Path(study["root"])
    parent = study["existing_parent"] if seed == 45 else str(root / f"parent_seed{seed}" / "epoch_010.pt")
    cfg = {**study["base"], "epochs": 40, "seed": seed, "init_checkpoint": parent,
        "fork_parent_epochs": 10, "output": str(root / f"{placement}_{variant}_seed{seed}"),
        "fgw": {**study["mixer"], "placement": placement, "variant": variant,
                "structure_weight": 1. if variant == "gw" else .5}}
    if placement == "tactile_first":
        teacher = root / f"teacher_{variant}_seed{seed}" / "last.pt"
        cfg["fgw"].update(teacher_checkpoint=str(teacher), teacher_sha256=file_hash(teacher),
            teacher_cache=str(root / "common_codes" / f"{variant}_seed{seed}"),
            data_signature=study["data_signature"], teacher_weight=1.)
    path = root / "configs" / f"{placement}_{variant}_seed{seed}.json"
    return save_config(path, cfg)


def choose_finalists(study):
    root = Path(study["root"]); records = {}; rejected = []
    for placement in study["placements"]:
        records[placement] = {}
        for variant in study["variants"]:
            ratios, violations = [], []
            for seed in study["seeds"]:
                def validation(name):
                    checkpoint = root / f"{placement}_{name}_seed{seed}" / "epoch_020.pt"
                    return torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)["validation"]
                values, baseline = validation(variant), validation("none")
                for source in study["sources"]:
                    ratio = values[source]["normalized_mae"] / max(baseline[source]["normalized_mae"], 1e-12)
                    ratios.append(ratio)
                    if ratio > 1.1:
                        violations.append({"seed": seed, "source": source, "ratio": ratio})
            records[placement][variant] = {"mean_native_mae_ratio": float(np.mean(ratios)), "violations": violations}
            if violations:
                rejected.append({"placement": placement, "variant": variant, "violations": violations})
    selected = [("visual", "none")]
    for placement in study["placements"]:
        eligible = [(r["mean_native_mae_ratio"], variant) for variant, r in records[placement].items()
                    if variant != "none" and not r["violations"]]
        # If every mixer fails the reconstruction gate, extend its matched
        # control rather than silently relaxing the 10% rule.
        variant = min(eligible)[1] if eligible else "none"
        selected.append((placement, variant))
    selected = sorted(set(selected))
    result = {"selected": selected, "source_validation": records, "rejected": rejected,
              "MotionTrans_used_for_selection": False}
    write_json(root / "finalists.json", result)
    return selected


def run(study):
    root = Path(study["root"])
    lock = (root / "controller.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use the installed libraries matching the kernel driver")
    if file_hash(study["existing_parent"]) != study["existing_parent_sha256"]:
        raise ValueError("Existing parent changed")
    if study["implementation_sha256"] != implementation_signature():
        raise ValueError("Study implementation changed; do not mix software revisions across ablations")
    if shutil.disk_usage(root).free < 200 * 2**30:
        raise RuntimeError("Study requires 200 GiB of free disk for immutable epochs and feature caches")
    began = time.monotonic()
    status_path = root / "status.json"

    def job(identifier, command, checkpoint=None, epoch=None, expected_config=None, result=None):
        if study["implementation_sha256"] != implementation_signature():
            raise ValueError("Software changed during the study; stop rather than mixing implementations")
        if checkpoint is not None and Path(checkpoint).exists():
            state = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
            if expected_config is not None and state["config"] != json.loads(Path(expected_config).read_text()):
                raise ValueError(f"Existing job config differs: {identifier}")
            if state["epoch"] + 1 >= epoch:
                return
            command.append("--resume")
        if result is not None and Path(result).exists():
            saved = json.loads(Path(result).read_text())
            p = command[command.index("--checkpoint") + 1]
            fraction = float(command[command.index("--train-fraction") + 1]) if "--train-fraction" in command else 1.
            if saved["checkpoint_sha256"] != file_hash(p) or saved.get("train_fraction", 1.) != fraction:
                raise ValueError(f"Stale evaluation result: {identifier}")
            return
        log_path = root / "logs" / f"{identifier}.log"; log_path.parent.mkdir(exist_ok=True)
        with log_path.open("a") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            current = {"stage": "running", "job": identifier, "command": command,
                "controller_pid": os.getpid(), "worker_pid": process.pid,
                "log": str(log_path), "elapsed_seconds": time.monotonic() - began}
            write_json(status_path, current); print(json.dumps(current), flush=True)
            try:
                code = process.wait()
            except BaseException:
                process.terminate(); process.wait(); raise
        if code:
            raise RuntimeError(f"Job {identifier} exited {code}; see {log_path}")
        if epoch is not None:
            completed = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
            if completed["epoch"] + 1 < epoch:
                raise RuntimeError(f"Job did not reach its checkpoint: {identifier}")
        with (root / "jobs.jsonl").open("a") as stream:
            stream.write(json.dumps({"job": identifier, "status": "complete", "elapsed_seconds": time.monotonic() - began}) + "\n")

    try:
        for seed in study["seeds"]:
            if seed != 45:
                cfg = root / "configs" / f"parent_seed{seed}.json"
                job(f"parent_seed{seed}", [sys.executable, "-u", "-m", "tactilelatent.train_alignment", "--config", str(cfg),
                    "--stop-after-epoch", "10"],
                    root / f"parent_seed{seed}" / "last.pt", 10, cfg)
            for variant in study["variants"]:
                cfg = visual_config(study, seed, variant, "visual")
                job(f"screen_visual_{variant}_seed{seed}", [sys.executable, "-u", "-m", "tactilelatent.train_alignment",
                    "--config", str(cfg), "--stop-after-epoch", "20"],
                    root / f"visual_{variant}_seed{seed}" / "last.pt", 20, cfg)
            cfg = root / "configs" / f"teacher_warmup_seed{seed}.json"
            job(f"teacher_warmup_seed{seed}", [sys.executable, "-u", "-m", "tactilelatent.tactile_teacher",
                "--config", str(cfg), "--stop-after-epoch", "20"],
                root / f"teacher_warmup_seed{seed}" / "last.pt", 20, cfg)
            for variant in study["variants"]:
                cfg = root / "configs" / f"teacher_{variant}_seed{seed}.json"
                checkpoint = root / f"teacher_{variant}_seed{seed}" / "last.pt"
                job(f"teacher_{variant}_seed{seed}", [sys.executable, "-u", "-m", "tactilelatent.tactile_teacher", "--config", str(cfg)],
                    checkpoint, 70, cfg)
                # Cache construction itself is resumable at episode boundaries.
                job(f"cache_{variant}_seed{seed}", [sys.executable, "-u", "-m", "tactilelatent.tactile_teacher", "--config", str(cfg), "--cache-only"])
                visual = visual_config(study, seed, variant, "tactile_first")
                job(f"screen_tactile_first_{variant}_seed{seed}", [sys.executable, "-u", "-m", "tactilelatent.train_alignment",
                    "--config", str(visual), "--stop-after-epoch", "20"],
                    root / f"tactile_first_{variant}_seed{seed}" / "last.pt", 20, visual)
        selected = choose_finalists(study)
        for placement, variant in selected:
            for seed in study["seeds"]:
                cfg = visual_config(study, seed, variant, placement)
                job(f"final_{placement}_{variant}_seed{seed}", [sys.executable, "-u", "-m", "tactilelatent.train_alignment", "--config", str(cfg)],
                    root / f"{placement}_{variant}_seed{seed}" / "last.pt", 40, cfg)
        # Configurations and finalists are locked before opening MotionTrans.
        write_json(root / "evaluation_protocol.json", {"finalists": selected,
            "heads": study["heads"], "fractions": study["label_fractions"], "samples": 64,
            "note": "Existing MotionTrans validation is a development benchmark, not a pristine test"})
        for placement in study["placements"]:
            for variant in study["variants"]:
                for seed in study["seeds"]:
                    folder = root / f"{placement}_{variant}_seed{seed}"
                    epochs = [20, 40] if (placement, variant) in selected else [20]
                    for epoch in epochs:
                        checkpoint = folder / f"epoch_{epoch:03d}.pt"
                        shared = folder / "evaluations" / f"motiontrans_epoch_{epoch:03d}" / "features"
                        for name, widths in study["heads"].items():
                            for fraction in study["label_fractions"]:
                                destination = folder / "evaluations" / f"motiontrans_epoch_{epoch:03d}_{name}_labels{int(100 * fraction)}"
                                command = [sys.executable, "-u", "-m", "tactilelatent.evaluate_holdout_decoder",
                                    "--checkpoint", str(checkpoint), "--manifest", study["base"]["manifest"],
                                    "--source", "ftp_motiontrans", "--output", str(destination), "--cache-dir", str(shared),
                                    "--allow-intermediate", "--hidden-dims", *map(str, widths), "--samples", "64",
                                    "--seed", str(seed), "--epochs", "100", "--patience", "10",
                                    "--train-fraction", str(fraction), "--workers", str(study["base"]["workers"])]
                                job(f"eval_{placement}_{variant}_seed{seed}_epoch{epoch}_{name}_labels{int(fraction * 100)}",
                                    command, result=destination / "results.json")
        # Calibration is a separately retrained head with calibration episodes
        # excluded from fitting, keeping the primary point comparison intact.
        for placement, variant in selected:
            for seed in study["seeds"]:
                folder = root / f"{placement}_{variant}_seed{seed}"
                checkpoint = folder / "epoch_040.pt"
                cache = folder / "evaluations" / "motiontrans_epoch_040" / "features"
                for name, widths in study["heads"].items():
                    destination = root / "calibration" / f"{placement}_{variant}_seed{seed}_{name}"
                    command = [sys.executable, "-u", "-m", "tactilelatent.evaluate_fgw_calibration",
                        "--checkpoint", str(checkpoint), "--manifest", study["base"]["manifest"],
                        "--cache-dir", str(cache), "--output", str(destination), "--hidden-dims", *map(str, widths),
                        "--samples", "64", "--seed", str(seed)]
                    job(f"calibration_{placement}_{variant}_seed{seed}_{name}", command,
                        result=destination / "results.json")
        report(root)
        write_json(status_path, {"stage": "complete", "elapsed_seconds": time.monotonic() - began})
    except BaseException as error:
        previous_status = json.loads(status_path.read_text()) if status_path.exists() else {}
        write_json(status_path, {**previous_status, "stage": "failed", "error": str(error)})
        raise


def benchmark(study, device="cuda"):
    from torch.utils.data import DataLoader, Subset
    torch.manual_seed(45); np.random.seed(45)
    cfg = {**study["base"], "seed": 45}
    signature, registry, sources, dimensions, numeric, _, _ = source_context(cfg)
    dataset = CodeOnlyDataset(cfg["manifest"], "train", sources, code_cache=cfg["code_cache"],
        registry=registry, data_signature=signature, include_targets=False)
    model = TactileTeacher(dimensions, numeric).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    indices = sample_banks(dataset, 256, 45, 0)
    banks = {}; began = time.monotonic(); batches = 0
    for source, points in indices.items():
        values = []
        for batch in DataLoader(Subset(dataset, points), batch_size=256):
            x = batch["value"].to(device)
            z, reconstructed = model(source, x)
            valid = batch["valid"].to(device)
            loss = (reconstructed - x).square().mean(-1)[valid].mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step(); batches += 1
            with torch.no_grad():
                z = model.encode(source, x)
                from .fgw_data import final_valid_indices
                last, _ = final_valid_indices(batch)
                values.append(z[torch.arange(len(z), device=z.device), last.to(device)].detach().cpu())
        banks[source] = torch.cat(values)
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    teacher_seconds = time.monotonic() - began
    batch_count = sum((min(len(dataset.indices_by_source[s]), cfg["train_windows_per_source_per_epoch"]) + 255) // 256
                      for s in sources)
    timings = {}
    for variant, mass in (("fgw", 1.), ("stochastic_fgw", 1.), ("partial_stochastic_fgw", .5),
                          ("partial_stochastic_fgw", .75), ("partial_stochastic_fgw", 1.)):
        print(json.dumps({"event": "benchmark_solve", "variant": variant, "mass": mass}), flush=True)
        result = fit_mixer(banks, MixerConfig(variant=variant, transported_mass=mass), seed=45)
        timings[f"{variant}_mass{mass}"] = result["diagnostics"]
    output = {"teacher_code_batches": batches, "teacher_code_seconds": teacher_seconds,
        "teacher_seconds_per_256_clip_batch": teacher_seconds / batches,
        "teacher_batches_per_full_epoch": batch_count,
        "projected_teacher_train_seconds_per_epoch": batch_count * teacher_seconds / batches,
        "mixer_timings": timings, "bank_states": sum(map(len, banks.values())),
        "note": "Teacher uses actual cached codes but untrained projections; visual epoch time still uses historical timing until measured"}
    write_json(Path(study["root"]) / "benchmark.json", output)
    print(json.dumps(output), flush=True)
    return output


def report(root):
    root = Path(root); records = []
    study = json.loads((root / "study.json").read_text())
    for placement in study["placements"]:
        for variant in study["variants"]:
            for seed in study["seeds"]:
                folder = root / f"{placement}_{variant}_seed{seed}"
                for path in (folder / "evaluations").glob("motiontrans*/results.json"):
                    value = json.loads(path.read_text()); metrics = value["validation"]
                    records.append({"placement": placement, "variant": variant, "seed": seed,
                        "encoder_epoch": value["checkpoint_epoch"] + 1, "head": value["head"],
                        "train_fraction": value.get("train_fraction", 1.), "mae": metrics["deterministic_mae"],
                        "coverage90": metrics["regions"]["all"]["intervals"]["0.9"]["empirical_coverage"],
                        "bootstrap_ci95": metrics.get("episode_bootstrap_mae_ci95"),
                        "per_channel": metrics.get("per_channel"),
                        "within_episode_correlations": metrics.get("within_episode_channel_correlations"),
                        "episode_errors": metrics.get("episode_errors"),
                        "path": str(path)})
    grouped = {}
    for row in records:
        key = (row["placement"], row["variant"], row["encoder_epoch"], row["head"], row["train_fraction"])
        grouped.setdefault(key, []).append(row["mae"])
    aggregates = [{"placement": k[0], "variant": k[1], "encoder_epoch": k[2], "head": k[3],
        "train_fraction": k[4], "seeds_completed": len(v), "mae_mean": statistics.mean(v),
        "mae_seed_std": statistics.stdev(v) if len(v) > 1 else None} for k, v in sorted(grouped.items())]
    paired = []
    baselines = {(r["seed"], r["encoder_epoch"], r["head"], r["train_fraction"]): r
                 for r in records if r["placement"] == "visual" and r["variant"] == "none"}
    for row in records:
        reference = baselines.get((row["seed"], row["encoder_epoch"], row["head"], row["train_fraction"]))
        if reference is None or not row["episode_errors"] or row is reference:
            continue
        left = {e["episode_id"]: e for e in row["episode_errors"]}
        right = {e["episode_id"]: e for e in reference["episode_errors"]}
        if set(left) != set(right) or any(left[e]["count"] != right[e]["count"] for e in left):
            raise ValueError("Paired bootstrap requires identical validation episode observations")
        ids = sorted(left)
        differences = np.array([left[e]["absolute_sum"] - right[e]["absolute_sum"] for e in ids])
        counts = np.array([left[e]["count"] for e in ids])
        draw = np.random.default_rng(row["seed"]).integers(len(ids), size=(5000, len(ids)))
        estimates = differences[draw].sum(1) / counts[draw].sum(1)
        paired.append({k: row[k] for k in ("placement", "variant", "seed", "encoder_epoch", "head", "train_fraction")} |
            {"baseline": "visual_none", "mae_difference": float(differences.sum() / counts.sum()),
             "episode_bootstrap_difference_ci95": np.quantile(estimates, [.025, .975]).tolist()})
    calibration_results = [json.loads(p.read_text()) for p in (root / "calibration").glob("*/results.json")]
    write_json(root / "summary.json", {"results": records, "seed_aggregates": aggregates,
        "paired_baseline_comparisons": paired,
        "separate_calibration_results": calibration_results,
        "uncertainty": "Latent-sampling intervals are uncalibrated; this study does not change their training objective"})
    if aggregates:
        with (root / "summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(aggregates[0])); writer.writeheader(); writer.writerows(aggregates)
        subprocess.run(["/usr/bin/python3", str(Path(__file__).resolve().parents[2] / "scripts" / "plot_fgw_study.py"),
                        str(root / "summary.json")], check=True)
    return aggregates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "benchmark", "run", "report"))
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    # Bound BLAS workers before the deferred POT import in the solver.
    study = prepare(args.root) if args.action == "prepare" else json.loads((Path(args.root) / "study.json").read_text())
    if args.action == "benchmark":
        benchmark(study, args.device)
    elif args.action == "run":
        run(study)
    elif args.action == "report":
        print(json.dumps(report(args.root)))


if __name__ == "__main__":
    main()
