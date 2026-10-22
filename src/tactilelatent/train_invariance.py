"""Fresh/continued 16D alignment training with Gaussian or deterministic latents.

Control, relational, trajectory, and meta conditions use the existing inference
checkpoint format. No private/static episode context or new sensor decoder.
"""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .aligned_frames import file_hash, manifest_signature, write_json
from .alignment_data import AlignedClipDataset
from .alignment_model import MODEL_TYPE, JointSensorAlignment
from .fgw_data import BankDataset, BankSampler, sample_banks
from .joint_all_config import ALL_TARGETS
from .joint_data import SourceBatchSampler
from .model import DinoBackbone, MODEL_ID, REVISION
from .relational import HELDOUT, extra_loss, read_floors
from .sam import SAMStep, restore_rng, rng_snapshot
from .sensor_code_cache import CodeCache, cache_signature, verify_teachers
from .train import MODEL_CONFIG_KEYS, features, kl_weight, make_scheduler
from .train_alignment import alignment_loss, atomic_checkpoint, model_weight_hash
from .invariance_validation import validate
from .viewer_checkpoints import save_viewer_epoch


def check_config(cfg):
    if (cfg.get("model_type") != MODEL_TYPE or cfg.get("latent_dim") != 16
            or cfg.get("heldout_source") != HELDOUT or cfg.get("train_sources") is not None
            or cfg.get("fgw") or cfg.get("private_dim", 0) or cfg.get("static_context")):
        raise ValueError("Expected unchanged 16D model, all source datasets, and MotionTrans held out")
    if cfg.get("optimizer_method") not in ("adamw", "sam_adamw"):
        raise ValueError("Unsupported optimizer")
    if cfg["epochs"] <= cfg["parent_epochs"] or min(cfg["batch_size"], cfg["accumulation"]) < 1:
        raise ValueError("Invalid continuation budget")
    if cfg["parent_epochs"] < 0 or (cfg["parent_epochs"] == 0 and (
            cfg.get("init_checkpoint") or cfg.get("init_joint_checkpoint") or cfg.get("parent_sha256"))):
        raise ValueError("Fresh training requires no initialization checkpoint or parent hash")
    mode = cfg.get("latent_mode", "gaussian")
    if mode not in ("gaussian", "deterministic"):
        raise ValueError("Unknown latent mode")
    if mode == "deterministic":
        if cfg.get("stochastic") or cfg.get("beta_max") != 0 or cfg.get("latent_objective") != "deterministic":
            raise ValueError("Deterministic training requires sampling off, zero KL weight, and explicit objective")
    elif cfg.get("latent_objective") != "vae":
        raise ValueError("Gaussian training requires the historical VAE objective")
    for name in ("relational", "trajectory", "meta"):
        objective = cfg.get(name, {})
        weight = objective.get("weight", 0)
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"Invalid {name} weight")
        if objective.get("ramp_epochs", 2) <= 0:
            raise ValueError("Positive objective ramp required")
    if cfg.get("meta", {}).get("enabled"):
        m = cfg["meta"]
        if min(m.get("every_updates", 16), m.get("inner_steps", 5),
               m.get("support_episodes", 4), m.get("query_episodes", 4), m.get("inner_lr", .01)) <= 0:
            raise ValueError("Invalid meta-task settings")
    if cfg.get("trajectory", {}).get("weight", 0) and min(
            cfg["trajectory"].get("clips_per_source", 32), cfg["trajectory"].get("candidates", 4)) < 1:
        raise ValueError("Positive trajectory bank and candidate counts required")
    if not math.isfinite(cfg.get("floor_scale", 1.)) or cfg.get("floor_scale", 1.) <= 0:
        raise ValueError("Positive finite explicit floor scale required")


def check_parent(state, cfg, model_config, provenance):
    if (state.get("model_type") != MODEL_TYPE or state["epoch"] + 1 != cfg["parent_epochs"]
            or state["model_config"] != model_config or state.get("backbone_revision") != REVISION
            or state["config"].get("heldout_source") != HELDOUT
            or state["config"].get("seed") != cfg["seed"]
            or any(state["provenance"].get(k) != provenance[k] for k in ("data", "teachers", "code_caches"))):
        raise ValueError("Parent model, seed, teacher, or chronological data contract differs")
    if Path(state["config"]["output"]).resolve() == Path(cfg["output"]).resolve():
        raise ValueError("Continuation must use a fresh run directory")


def prepare(cfg):
    """Read-only preflight; does not allocate DINO/GPU or create a training run."""
    check_config(cfg)
    signature = manifest_signature(cfg["manifest"])
    audit = json.loads(Path(cfg["audit"]).read_text())
    if audit.get("signature") != signature or not audit.get("passed") or not audit.get("VLA_complete_pixels"):
        raise ValueError("A full passing native chronological audit is required")
    registry = json.loads(Path(cfg["teachers"]).read_text())
    if HELDOUT in registry["sources"]:
        raise ValueError("MotionTrans teacher in source registry")
    verify_teachers(registry)
    rows = json.loads(Path(cfg["manifest"]).read_text())["episodes"]
    sources = sorted({r["source"] for r in rows} - {HELDOUT})
    statistics, dimensions, signatures = {}, {}, {}
    for source in registry["sources"]:
        expected = cache_signature(cfg["manifest"], registry, source, signature)
        cache = CodeCache(cfg["code_cache"], source, expected)
        statistics[source], dimensions[source] = cache.index["statistics"], cache.index["code_dim"]
        signatures[source] = expected
    numeric = {s: ALL_TARGETS[s][1] for s in sources if s not in dimensions}
    if set(numeric) != {"ftp_rdp", "ftp_rdp_bimanual"} or len(sources) != 11:
        raise ValueError("Incomplete eleven-source training registry")
    model_config = {k: cfg[k] for k in MODEL_CONFIG_KEYS}
    model_config.update(input_dim=768, sensor_code_dims=dimensions, numeric_shapes=numeric,
                        statistics=statistics, adapter_hidden_dims=cfg["adapter_hidden_dims"])
    if "latent_mode" in cfg:
        model_config["latent_mode"] = cfg["latent_mode"]
    modules = ("train_invariance.py", "invariance_validation.py", "relational.py", "trajectory_matching.py", "meta_learning.py",
               "alignment_data.py", "alignment_model.py", "model.py", "train.py", "train_alignment.py",
               "fgw_data.py", "joint_data.py", "sam.py", "aligned_frames.py", "viewer_checkpoints.py")
    provenance = {"data": signature, "teachers": registry, "code_caches": signatures,
                  "code": {name: file_hash(Path(__file__).with_name(name)) for name in modules},
                  "parent_sha256": file_hash(cfg["init_checkpoint"]) if cfg["parent_epochs"] else None}
    if provenance["parent_sha256"] != cfg["parent_sha256"]:
        raise ValueError("Pinned parent checkpoint changed")
    if cfg.get("trajectory", {}).get("weight", 0):
        import ot
        if ot.__version__ != "0.9.6.post1":
            raise ValueError("Install the pinned alignment dependency: POT 0.9.6.post1")
        provenance["pot_version"] = ot.__version__
    floors = {}
    if cfg.get("relational_floors"):
        floors = read_floors(cfg["relational_floors"], signature, signatures)
        floors = {s: value * cfg.get("floor_scale", 1.) for s, value in floors.items()}
        provenance["relational_floors_sha256"] = file_hash(cfg["relational_floors"])
    parent = None
    if cfg["parent_epochs"]:
        parent = torch.load(cfg["init_checkpoint"], map_location="cpu", weights_only=False, mmap=True)
        check_parent(parent, cfg, model_config, provenance)
    return sources, registry, model_config, provenance, floors, parent


def replay_step(model, optimizer, sam, replay, cfg, floors, device):
    """Replay ordinary and meta terms with identical data and randomness."""
    after = rng_snapshot()
    norm = sam.perturb()
    try:
        for item in replay:
            restore_rng(item["rng"])
            with torch.autocast(device.type, torch.bfloat16, enabled=device.type == "cuda"):
                if item["kind"] == "meta":
                    from .meta_learning import meta_loss
                    loss, _ = meta_loss(model, item["prepared"], cfg["meta"])
                else:
                    predicted = model(item["features"], item["mask"], item["source"])
                    loss, _, _ = alignment_loss(predicted, item["batch"], item["beta"])
                    extra, _ = extra_loss(predicted, item["batch"], item["source"], cfg, floors,
                                          item["epoch"], item["payload"])
                    loss = loss + extra
            (loss * item["scale"]).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    finally:
        sam.restore()
        restore_rng(after)
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    sam.steps += 1
    return norm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/invariance_16d_relational.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--benchmark-updates", type=int, default=0,
                        help="Bounded optimizer-update benchmark; separate output, no production checkpoints")
    args = parser.parse_args()
    if args.benchmark_updates < 0 or (args.benchmark_updates and args.resume):
        raise ValueError("Benchmark must be positive and cannot resume a production run")
    cfg = json.loads(Path(args.config).read_text())
    if args.benchmark_updates:
        cfg = {**cfg, "output": cfg["output"] + "_benchmark"}
    torch.set_num_threads(4)
    sources, registry, model_config, provenance, floors, parent = prepare(cfg)
    if args.preflight:
        print(json.dumps({"preflight": "passed", "sources": sources, "parent_epoch": cfg["parent_epochs"],
                          "total_epochs": cfg["epochs"], "shared_dim": 16, "private_dim": 0,
                          "floor_sources": sorted(floors), "provenance": provenance["parent_sha256"]}))
        return
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    device = torch.device(args.device)
    kwargs = dict(code_cache=cfg["code_cache"], registry=registry, data_signature=provenance["data"])
    train_set = BankDataset(cfg["manifest"], "train", sources, include_targets=False, **kwargs)
    val_set = AlignedClipDataset(cfg["manifest"], "val", sources, include_targets=True, **kwargs)
    sampling = cfg["matched_bank_sampling"]
    sampler = BankSampler(train_set, cfg["batch_size"], cfg["seed"],
                          cfg["train_windows_per_source_per_epoch"], sampling["fraction"])
    train_loader = DataLoader(train_set, batch_sampler=sampler, num_workers=cfg["workers"], pin_memory=device.type == "cuda")
    val_sampler = SourceBatchSampler(val_set, cfg["batch_size"], seed=cfg["seed"], max_windows_per_source=cfg["validation_windows_per_source"])
    val_loader = DataLoader(val_set, batch_sampler=val_sampler, num_workers=cfg["workers"], pin_memory=device.type == "cuda")
    if any(set(ds.indices_by_source) != set(sources) for ds in (train_set, val_set)):
        raise ValueError("Missing train or validation source")
    folder = Path(cfg["output"])
    path = folder / "last.pt"
    if path.exists() and not args.resume:
        raise FileExistsError("Choose a fresh output or explicitly resume")
    folder.mkdir(parents=True, exist_ok=True)
    backbone = DinoBackbone().to(device).eval().requires_grad_(False)
    model = JointSensorAlignment(**model_config).to(device)
    if parent is not None:
        model.load_state_dict(parent["model"], strict=True)
    del parent
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    schedule_cfg = {**cfg, "epochs": cfg["epochs"] - cfg["parent_epochs"]}
    scheduler = make_scheduler(optimizer, schedule_cfg, len(train_loader))
    sam = SAMStep(optimizer, cfg["sam_rho"]) if cfg["optimizer_method"] == "sam_adamw" else None
    start, best, updates, task_number = cfg["parent_epochs"], float("inf"), 0, 0
    if args.resume:
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if saved["config"] != cfg or saved["provenance"] != provenance:
            raise ValueError("Resume configuration or code/data provenance changed")
        model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        if sam:
            sam.load_state_dict(saved["sam"])
        start, best = saved["epoch"] + 1, saved["best_score"]
        updates, task_number = saved["updates"], saved["meta_task_number"]
        restore_rng(saved["rng"]); random.setstate(saved["python_rng"]); np.random.set_state(saved["numpy_rng"])
        del saved
        # An interrupted epoch can leave log rows beyond the last atomic state.
        # Remove only those uncommitted rows before replaying that epoch.
        for filename, field, bound in (("meta_tasks.jsonl", "update", updates),
                                       ("metrics.jsonl", "epoch", start + 1)):
            log = folder / filename
            if log.exists():
                committed = []
                for line in log.read_text().splitlines():
                    try:
                        if json.loads(line)[field] < bound:
                            committed.append(line)
                    except (ValueError, KeyError):
                        continue
                temporary = log.with_suffix(".tmp")
                temporary.write_text("\n".join(committed) + ("\n" if committed else ""))
                temporary.replace(log)
    from .meta_learning import TaskSampler, prepare_task, meta_loss
    task_sampler = TaskSampler(train_set, cfg["seed"], cfg["meta"].get("support_episodes", 4),
        cfg["meta"].get("query_episodes", 4), cfg["meta"].get("group_field")) if cfg.get("meta", {}).get("enabled") else None
    trajectory = None
    if cfg.get("trajectory", {}).get("weight", 0):
        from .trajectory_matching import TrajectoryObjective
        trajectory = TrajectoryObjective(cfg, train_set, floors, device, folder, provenance)
    write_json(folder / "config.json", cfg)
    if not args.resume:
        write_json(folder / "initialization.json", {"parent_epoch": cfg["parent_epochs"],
            "parent_sha256": provenance["parent_sha256"], "model_sha256": model_weight_hash(model),
            "optimizer": "fresh; matched across conditions", "private_dim": 0,
            "latent_mode": cfg.get("latent_mode", "gaussian"),
            "learned_variance_parameters": any('prior_logvar' in name for name,_ in model.named_parameters()),
            "initialization": "random shared encoder/adapters" if cfg["parent_epochs"] == 0 else "parent weights"})
    began = time.monotonic()
    print(json.dumps({"event": "indexed", "start_epoch": start + 1, "epochs": cfg["epochs"],
                      "batches_per_epoch": len(train_loader), "sources": sources}), flush=True)
    for epoch in range(start, cfg["epochs"]):
        banks = sample_banks(train_set, sampling["bank_size"], cfg["seed"], epoch)
        sampler.set_epoch(epoch, banks)
        before = rng_snapshot()
        trajectory_info = trajectory.refresh(model, backbone, banks, epoch) if trajectory else None
        restore_rng(before)
        model.train(); optimizer.zero_grad(set_to_none=True)
        replay, totals, counts = [], defaultdict(lambda: np.zeros(3)), defaultdict(int)
        objective_totals = defaultdict(lambda: defaultdict(float))
        means_sum, outer_sum = torch.zeros(16, dtype=torch.float64), torch.zeros(16, 16, dtype=torch.float64)
        latent_count, predicted_variance = 0, 0.
        group_task = None
        for step, original in enumerate(train_loader):
            group_size = min(cfg["accumulation"], len(train_loader) - (step // cfg["accumulation"]) * cfg["accumulation"])
            if step % cfg["accumulation"] == 0:
                group_task = task_sampler.sample(task_number) if task_sampler and updates % cfg["meta"].get("every_updates", 16) == 0 else None
            source = original["source"][0]
            if any(s != source for s in original["source"]) or source == HELDOUT:
                raise ValueError("Invalid source training batch")
            # In a meta outer task this source contributes only support/query,
            # never its persistent adapter's ordinary supervised gradient.
            if group_task is None or source != group_task["source"]:
                batch = {k: v.to(device) for k, v in original.items() if k in ("code_target", "code_valid", "target", "target_mask")}
                mask = original["view_mask"].to(device).clone()
                if mask[:, :, 0].all():
                    mask[:, :, 1:] &= torch.rand(len(mask), 1, 2, device=device) >= cfg["wrist_dropout"]
                with torch.no_grad(), torch.autocast(device.type, torch.bfloat16, enabled=device.type == "cuda"):
                    encoded = features(backbone, original, device)
                rng = rng_snapshot()
                beta = kl_weight(cfg, epoch + step / len(train_loader))
                payload = trajectory.payload(original) if trajectory else None
                with torch.autocast(device.type, torch.bfloat16, enabled=device.type == "cuda"):
                    predicted = model(encoded, mask, source)
                    loss, task, kl = alignment_loss(predicted, batch, beta)
                    extra, metrics = extra_loss(predicted, batch, source, cfg, floors, epoch - cfg["parent_epochs"], payload)
                    loss = loss + extra
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite ordinary/relational loss")
                (loss / group_size).backward()
                if sam:
                    replay.append({"kind": "ordinary", "features": encoded.detach(), "mask": mask, "source": source,
                        "batch": batch, "beta": beta, "payload": payload, "epoch": epoch - cfg["parent_epochs"],
                        "scale": 1 / group_size, "rng": rng})
                totals[source] += [loss.item(), task.item(), kl.item()]; counts[source] += 1
                for key, value in metrics.items():
                    objective_totals[source][key] += value
                mu = predicted["mu"].detach().double().flatten(0, 1).cpu()
                means_sum += mu.sum(0); outer_sum += mu.T @ mu; latent_count += len(mu)
                predicted_variance += predicted["logvar"].detach().float().exp().sum().item()
                if step % cfg["log_every"] == 0:
                    print(json.dumps({"event": "train", "epoch": epoch + 1, "batch": step, "source": source,
                                      "loss": loss.item(), "task": task.item(), "kl": kl.item(), **metrics}), flush=True)
            if (step + 1) % cfg["accumulation"] == 0 or step + 1 == len(train_loader):
                if group_task is not None:
                    prepared = prepare_task(group_task, train_set, backbone, device)
                    rng = rng_snapshot()
                    with torch.autocast(device.type, torch.bfloat16, enabled=device.type == "cuda"):
                        value, meta_metrics = meta_loss(model, prepared, cfg["meta"])
                    if not torch.isfinite(value):
                        raise FloatingPointError("Nonfinite meta query loss")
                    (cfg["meta"]["weight"] * value).backward()
                    if sam:
                        replay.append({"kind": "meta", "prepared": prepared, "rng": rng, "scale": cfg["meta"]["weight"]})
                    with (folder / "meta_tasks.jsonl").open("a") as stream:
                        stream.write(json.dumps({"epoch": epoch + 1, "update": updates, **group_task, **meta_metrics}) + "\n")
                    print(json.dumps({"event": "meta", **meta_metrics}), flush=True)
                    task_number += 1
                if sam:
                    replay_step(model, optimizer, sam, replay, cfg, floors, device)
                    replay.clear()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                    optimizer.step(); optimizer.zero_grad(set_to_none=True)
                scheduler.step(); updates += 1
                write_json(folder / "status.json", {"stage": "training", "epoch": epoch + 1,
                    "batch": step + 1, "batches": len(train_loader), "updates": updates, "meta_tasks": task_number})
                if args.benchmark_updates and updates >= args.benchmark_updates:
                    seconds = time.monotonic() - began
                    write_json(folder / "benchmark.json", {"updates": updates, "seconds": seconds,
                        "seconds_per_update": seconds / updates, "meta_tasks": task_number,
                        "peak_cuda_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
                        "note": "Includes bank preparation; excludes epoch validation and initial model loading"})
                    write_json(folder / "status.json", {"stage": "benchmark_complete", "updates": updates})
                    return
        validation = validate(model, backbone, val_loader, registry, device, floors)
        score = float(np.mean([v["supervision_loss"] for v in validation.values()]))
        improved, best = score < best, min(best, score)
        mean = means_sum / max(latent_count, 1)
        covariance = outer_sum / max(latent_count, 1) - mean[:, None] * mean[None, :]
        spectrum = torch.linalg.eigvalsh(covariance).clamp_min(0)
        proportion = spectrum / spectrum.sum().clamp_min(1e-12)
        rank = (-(proportion * proportion.clamp_min(1e-12).log()).sum()).exp().item() if spectrum.sum() > 0 else 0.
        record = {"epoch": epoch + 1, "continuation_epoch": epoch + 1 - cfg["parent_epochs"],
                  "validation": validation, "selection_score": score, "seconds": time.monotonic() - began,
                  "train": {s: dict(zip(("loss", "supervision", "kl"), (v / counts[s]).tolist())) for s, v in totals.items()},
                  "objectives_batch_mean": {s: {k: v / counts[s] for k, v in row.items()} for s, row in objective_totals.items()},
                  "latent": {"effective_rank_mu": rank, "mean_coordinate_variance_mu": covariance.diag().mean().item(),
                             "mean_predicted_variance": predicted_variance / max(16 * latent_count, 1)},
                  "trajectory": trajectory_info, "meta_tasks": task_number}
        state = {"model_type": MODEL_TYPE, "model_config": model_config, "model": model.state_dict(), "config": cfg,
                 "provenance": provenance, "teachers": registry, "backbone": MODEL_ID, "backbone_revision": REVISION,
                 "epoch": epoch, "best_score": best, "validation": validation, "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(), "rng": rng_snapshot(), "python_rng": random.getstate(),
                 "numpy_rng": np.random.get_state(), "updates": updates, "meta_task_number": task_number}
        if sam:
            state["sam"] = sam.state_dict()
        atomic_checkpoint(path, state); save_viewer_epoch(path, state)
        if improved:
            atomic_checkpoint(folder / "best.pt", state)
        if cfg["checkpoint_every"] and (epoch + 1) % cfg["checkpoint_every"] == 0:
            (folder / f"epoch_{epoch + 1:03d}.pt").hardlink_to(path)
        with (folder / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
    write_json(folder / "status.json", {"stage": "complete", "completed_epoch": cfg["epochs"], "updates": updates})


if __name__ == "__main__":
    main()
