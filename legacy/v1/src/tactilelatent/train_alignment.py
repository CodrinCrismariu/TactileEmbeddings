"""Train visual Gaussian latents against frozen tactile-encoder codes.

The training step never calls a tactile encoder or decoder. Teacher codes are
cached, detached targets. Native decoder reconstructions are evaluation only.
"""
import argparse
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .aligned_frames import file_hash, manifest_signature, write_json
from .alignment_data import AlignedClipDataset
from .alignment_model import MODEL_TYPE, FrozenSensorDecoders, JointSensorAlignment
from .joint_all_config import ALL_TARGETS
from .joint_data import SourceBatchSampler
from .model import DinoBackbone, MODEL_ID, REVISION
from .sam import SAMStep, restore_rng, rng_snapshot
from .sensor_code_cache import CodeCache, cache_signature, verify_teachers
from .train import MODEL_CONFIG_KEYS, features, kl_weight, make_scheduler


# The deployed 50-epoch trainer; epoch extensions explicitly migrate this
# version while retaining strict checks on data, teachers, model and dataset.
EXTENSION_TRAINER_HASHES = {"1cf54b1ad6dc6673b8cd25f5af8cb9c4dd1717edf9c7da15188bfe0f8c64d331",
                            "5a1601067ae56c7886d6e6675e2187da0a8fd93dbaee394adeb62d55595f1f4b",
                            "6767e471a73ec0ae345c7d439a61688a89bae4ce6ac4772ec30b2c6dff830b8e"}


def extension_config(cfg, state, epochs):
    saved = state["config"]
    if epochs < saved["epochs"] or epochs <= state["epoch"] + 1:
        raise ValueError("Extension must retain the existing epoch budget and add training")
    result = {**cfg, "epochs": epochs}
    if saved.get("lr_extension") and epochs == saved["epochs"]:
        result["lr_extension"] = saved["lr_extension"]
    else:
        result["lr_extension"] = {"anchor_update": state["scheduler"]["last_epoch"],
                                  "anchor_lr": state["optimizer"]["param_groups"][0]["lr"]}
    return result


def check_resume_contract(state, cfg, provenance, extending=False):
    if not extending:
        if state["config"] != cfg or state["provenance"] != provenance:
            raise ValueError("Resume config, code, source, or teacher provenance differs")
        return
    old_cfg = {k: v for k, v in state["config"].items() if k not in ("epochs", "lr_extension")}
    new_cfg = {k: v for k, v in cfg.items() if k not in ("epochs", "lr_extension")}
    before, after = state["provenance"], provenance
    if (old_cfg != new_cfg or
            {k: v for k, v in before.items() if k != "trainer_sha256"} !=
            {k: v for k, v in after.items() if k != "trainer_sha256"} or
            before["trainer_sha256"] not in EXTENSION_TRAINER_HASHES | {after["trainer_sha256"]}):
        raise ValueError("Epoch extension cannot change model, data, teachers or optimization settings")


def alignment_scheduler(optimizer, cfg, batches_per_epoch):
    if cfg.get("fgw") and "lr_extension" not in cfg:
        # All screening/finalist branches share a 30-epoch continuation horizon.
        schedule_cfg = {**cfg, "epochs": cfg["epochs"] - cfg.get("fork_parent_epochs", 0)}
        return make_scheduler(optimizer, schedule_cfg, batches_per_epoch)
    if "lr_extension" not in cfg:
        return make_scheduler(optimizer, cfg, batches_per_epoch)
    anchor = cfg["lr_extension"]
    end = math.ceil(batches_per_epoch / cfg["accumulation"]) * (cfg["epochs"] - cfg.get("fork_parent_epochs", 0))
    if end <= anchor["anchor_update"]:
        raise ValueError("Extended scheduler has no remaining updates")
    floor = cfg["min_learning_rate"] / cfg["learning_rate"]
    initial = anchor["anchor_lr"] / cfg["learning_rate"]
    def schedule(update):
        progress = max(0., min(1., (update - anchor["anchor_update"]) / (end - anchor["anchor_update"])))
        return floor + (initial - floor) * .5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def bank_sampling_config(cfg):
    """Allow a plain model to use exactly the same sampled clips as FGW."""
    sampling = cfg.get("matched_bank_sampling")
    if sampling is None and not cfg.get("fgw"):
        return None
    from .fgw_objective import objective_config
    mixer = objective_config(cfg)
    result = sampling or {"bank_size": mixer.bank_size, "fraction": mixer.bank_fraction}
    if result["bank_size"] < 1 or not 0 <= result["fraction"] <= 1:
        raise ValueError("Invalid matched bank sampling")
    if cfg.get("fgw") and (result["bank_size"] != mixer.bank_size or result["fraction"] != mixer.bank_fraction):
        raise ValueError("Matched sampling must agree with FGW bank settings")
    return result


def model_weight_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def alignment_loss(output, batch, beta=0.):
    if "sensor_code" in output:
        valid = batch["code_valid"].bool()
        target = batch["code_target"].detach().float()
        error = (output["sensor_code"].float() - target).square().mean(-1)
        task = error[valid].mean() if valid.any() else error.sum() * 0
    else:
        valid_cells = batch["target_mask"].bool()
        valid = valid_cells.flatten(2).any(-1)
        safe = torch.where(valid_cells, batch["target"], 0.).detach().float()
        difference = output["pred"].float() - safe
        task = ((difference.square() + .5 * difference.abs()) * valid_cells).sum() / valid_cells.sum().clamp_min(1)
    mu, logvar = output["mu"].float(), output["logvar"].float()
    if output.get("latent_mode") == "deterministic":
        if beta != 0:
            raise ValueError("Deterministic latent cannot have a KL penalty")
        kl = mu.sum() * 0
        return task, task.detach(), kl.detach()
    coordinate_kl = .5 * (mu.square() + logvar.exp() - 1 - logvar).mean(-1)
    kl = coordinate_kl[valid].mean() if valid.any() else coordinate_kl.sum() * 0
    return task + beta * kl, task.detach(), kl.detach()


def sam_replay(model, optimizer, sam, replay, device):
    after = rng_snapshot()
    norm = sam.perturb()
    try:
        for item in replay:
            restore_rng(item["rng"])
            with torch.autocast(device.type, torch.bfloat16, enabled=device.type == "cuda"):
                output = model(item["features"], item["mask"], item["source"])
                loss, _, _ = alignment_loss(output, item["batch"], item["beta"])
                if "alignment_payload" in item:
                    from .fgw_objective import visual_extra_loss
                    extra, _ = visual_extra_loss(output, item["alignment_payload"], item["config"], item["alignment_epoch"])
                    loss = loss + extra
            (loss / len(replay)).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    finally:
        sam.restore()
        restore_rng(after)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    sam.steps += 1
    return norm


def check_fork_contract(state, cfg, model_config, provenance):
    """Explicit parent fork; it cannot reinterpret historical checkpoints."""
    if (not cfg.get("fgw") or state.get("model_type") != MODEL_TYPE
            or state["epoch"] + 1 != cfg.get("fork_parent_epochs", 10)
            or state["model_config"] != model_config or state["backbone_revision"] != REVISION
            or state["config"].get("heldout_source") != "ftp_motiontrans"
            or state["config"]["seed"] != cfg["seed"] or model_config["latent_dim"] != cfg["latent_dim"]
            or any(state["provenance"].get(k) != provenance.get(k) for k in ("data", "teachers", "code_caches"))):
        raise ValueError("Alignment parent weights, seed, teachers, or frame provenance differs")
    if Path(cfg["output"]).resolve() == Path(state["config"]["output"]).resolve():
        raise ValueError("A fork requires a separate output directory")


@torch.inference_mode()
def validate(model, backbone, loader, registry, device):
    model.eval()
    decoders = FrozenSensorDecoders(registry, device)
    totals = defaultdict(lambda: defaultdict(float))
    for original in loader:
        source = original["source"][0]
        if any(value != source for value in original["source"]):
            raise ValueError("Mixed source validation batch")
        batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in original.items()}
        with torch.autocast(device.type, torch.bfloat16, enabled=device.type == "cuda"):
            output = model(features(backbone, original, device), batch["view_mask"], source, sample_latent=False)
        target, valid = batch["target"].float(), batch["target_mask"].bool()
        row = totals[source]
        if source in model.adapters:
            code_valid = batch["code_valid"].bool()
            difference = output["sensor_code"].float() - batch["code_target"]
            row["task_sum"] += difference.square().mean(-1)[code_valid].sum().item()
            row["task_count"] += code_valid.sum().item()
            # Evaluate native outputs in frame microbatches, including FreeTacMan.
            predicted_code = model.raw_code(source, output["sensor_code"]).flatten(0, 1)
            true_code = model.raw_code(source, batch["code_target"]).flatten(0, 1)
            zero_code = model.raw_code(source, torch.zeros_like(batch["code_target"])).flatten(0, 1)
            predictions, oracle_values, baseline_values = [], [], []
            for lo in range(0, len(predicted_code), 4):
                predictions.append(decoders.decode(source, predicted_code[lo:lo + 4]))
                oracle_values.append(decoders.decode(source, true_code[lo:lo + 4]))
                baseline_values.append(decoders.decode(source, zero_code[lo:lo + 4]))
            pred = torch.cat(predictions).reshape_as(target)
            oracle = torch.cat(oracle_values).reshape_as(target)
            baseline = torch.cat(baseline_values).reshape_as(target)
            row["oracle_absolute"] += ((oracle - target).abs() * valid).sum().item()
            row["baseline_absolute"] += ((baseline - target).abs() * valid).sum().item()
        else:
            pred = output["pred"].float()
            _, task, _ = alignment_loss(output, batch)
            row["task_sum"] += task.item() * valid.sum().item()
            row["task_count"] += valid.sum().item()
        error = pred - target
        row["absolute"] += (error.abs() * valid).sum().item()
        row["squared"] += (error.square() * valid).sum().item()
        row["cells"] += valid.sum().item()
        adjacent = valid[:, 1:] & valid[:, :-1]
        change_error = ((pred[:, 1:] - pred[:, :-1]) - (target[:, 1:] - target[:, :-1])).abs()
        row["change_error"] += (change_error * adjacent).sum().item()
        row["change_cells"] += adjacent.sum().item()
        if ALL_TARGETS[source][0] in ("pressure_grid", "taxel_vector"):
            contact = valid & (target > .05)
            predicted_contact = valid & (pred > .05)
            row["contact_absolute"] += (error.abs() * contact).sum().item()
            row["contacts"] += contact.sum().item()
            row["tp"] += (contact & predicted_contact).sum().item()
            row["contact_denominator"] += contact.sum().item() + predicted_contact.sum().item()
    result = {}
    for source, row in totals.items():
        cells = max(row["cells"], 1)
        metrics = {"supervision_loss": row["task_sum"] / max(row["task_count"], 1),
                   "normalized_mae": row["absolute"] / cells,
                   "temporal_change_mae": row["change_error"] / max(row["change_cells"], 1),
                   "valid_cells": int(row["cells"])}
        if source in model.adapters:
            metrics.update(code_mse=metrics["supervision_loss"],
                           frozen_ae_mae=row["oracle_absolute"] / cells,
                           mean_code_baseline_mae=row["baseline_absolute"] / cells)
        if ALL_TARGETS[source][0] == "optical_rgb":
            metrics["psnr"] = -10 * math.log10(max(row["squared"] / cells, 1e-15))
        elif row["contacts"]:
            metrics.update(contact_mae=row["contact_absolute"] / row["contacts"],
                           contact_f1=2 * row["tp"] / max(row["contact_denominator"], 1))
        result[source] = metrics
    return result


def atomic_checkpoint(path, state):
    from .sam import check_checkpoint_budget
    check_checkpoint_budget(state)
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/joint_alignment_holdout_motiontrans.json")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--extend-epochs", type=int, help="Resume to this total epoch count with continuous cosine decay")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stop-after-epoch", type=int,
                        help="Checkpoint at this completed epoch without changing the planned LR horizon")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    if args.extend_epochs is not None:
        if not args.resume or args.smoke:
            raise ValueError("Epoch extension requires --resume and a production checkpoint")
        previous = torch.load(Path(cfg["output"]) / "last.pt", map_location="cpu", weights_only=False, mmap=True)
        cfg = extension_config(cfg, previous, args.extend_epochs)
        del previous
    if args.smoke:
        cfg.update(epochs=1, train_windows_per_source_per_epoch=2,
                   validation_windows_per_source=2, workers=0, log_every=1,
                   checkpoint_every=0, output=cfg["output"] + "_smoke")
    if cfg.get("model_type") != MODEL_TYPE or cfg.get("heldout_source") != "ftp_motiontrans":
        raise ValueError("Expected sensor-code alignment with MotionTrans held out")
    if (cfg.get("init_checkpoint") or cfg.get("init_joint_checkpoint")) and not cfg.get("fgw"):
        raise ValueError("This experiment starts with fresh joint weights")
    if cfg.get("fgw") and (cfg["latent_dim"] not in (16, 32) or cfg.get("init_joint_checkpoint")):
        raise ValueError("FGW supports 16D or 32D, with explicit init_checkpoint for warm starts")
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    torch.set_num_threads(4)
    device = torch.device(args.device)
    signature = manifest_signature(cfg["manifest"])
    audit = json.loads(Path(cfg["audit"]).read_text())
    if audit.get("signature") != signature or not audit.get("passed") or not audit.get("VLA_complete_pixels"):
        raise ValueError("Full passing native/frame-pairing audit required")
    registry = json.loads(Path(cfg["teachers"]).read_text())
    verify_teachers(registry)
    rows = json.loads(Path(cfg["manifest"]).read_text())["episodes"]
    sources = sorted({r["source"] for r in rows} - {cfg["heldout_source"]})
    statistics, dimensions, cache_signatures = {}, {}, {}
    for source in sources:
        if source in registry["sources"]:
            expected = cache_signature(cfg["manifest"], registry, source, signature)
            cache = CodeCache(cfg["code_cache"], source, expected)
            statistics[source] = cache.index["statistics"]
            dimensions[source] = cache.index["code_dim"]
            cache_signatures[source] = expected
    numeric = {source: ALL_TARGETS[source][1] for source in sources if source not in dimensions}
    if set(numeric) != {"ftp_rdp", "ftp_rdp_bimanual"}:
        raise ValueError("Missing teacher for an autoencoder-backed source")
    kwargs = dict(code_cache=cfg["code_cache"], registry=registry, data_signature=signature)
    sampling = bank_sampling_config(cfg)
    if cfg.get("fgw"):
        from .fgw_objective import VisualObjective, visual_extra_loss
    if sampling:
        from .fgw_data import BankDataset, BankSampler, sample_banks
        train_set = BankDataset(cfg["manifest"], "train", sources, include_targets=False, **kwargs)
    else:
        train_set = AlignedClipDataset(cfg["manifest"], "train", sources, include_targets=False, **kwargs)
    val_set = AlignedClipDataset(cfg["manifest"], "val", sources, include_targets=True, **kwargs)
    if sampling:
        train_sampler = BankSampler(train_set, cfg["batch_size"], cfg["seed"],
            cfg["train_windows_per_source_per_epoch"], sampling["fraction"])
    else:
        train_sampler = SourceBatchSampler(train_set, cfg["batch_size"], seed=cfg["seed"],
            max_windows_per_source=cfg["train_windows_per_source_per_epoch"], sampling_mode="cycle")
    val_sampler = SourceBatchSampler(val_set, cfg["batch_size"], seed=cfg["seed"],
        max_windows_per_source=cfg["validation_windows_per_source"])
    if set(train_set.indices_by_source) != set(sources) or set(val_set.indices_by_source) != set(sources):
        raise ValueError("Every training source must have valid train/validation clips")
    loader_options = dict(num_workers=cfg["workers"], pin_memory=device.type == "cuda")
    train_loader = DataLoader(train_set, batch_sampler=train_sampler, **loader_options)
    val_loader = DataLoader(val_set, batch_sampler=val_sampler, **loader_options)
    model_config = {key: cfg[key] for key in MODEL_CONFIG_KEYS}
    model_config.update(sensor_code_dims=dimensions, numeric_shapes=numeric, statistics=statistics,
                        adapter_hidden_dims=cfg["adapter_hidden_dims"])
    backbone = DinoBackbone().to(device).eval().requires_grad_(False)
    model_config["input_dim"] = backbone.dim
    model = JointSensorAlignment(**model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scheduler = alignment_scheduler(optimizer, cfg, len(train_loader))
    if cfg["optimizer_method"] not in ("sam_adamw", "adamw"):
        raise ValueError("Unsupported optimizer")
    sam = SAMStep(optimizer, cfg["sam_rho"]) if cfg["optimizer_method"] == "sam_adamw" else None
    output_dir = Path(cfg["output"])
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "last.pt"
    start, best = 0, float("inf")
    provenance = {"data": signature, "teachers": registry, "code_caches": cache_signatures,
                  "trainer_sha256": file_hash(__file__),
                  "model_sha256": file_hash(Path(__file__).with_name("alignment_model.py")),
                  "dataset_sha256": file_hash(Path(__file__).with_name("alignment_data.py"))}
    if cfg.get("fgw"):
        provenance["fgw_code"] = {name: file_hash(Path(__file__).with_name(name)) for name in
            ("fgw_mixer.py", "fgw_data.py", "fgw_objective.py", "tactile_teacher.py")}
        if cfg["fgw"]["placement"] == "tactile_first" and cfg["fgw"]["data_signature"] != signature:
            raise ValueError("Common teacher uses a different frame contract")
        if cfg.get("init_checkpoint"):
            provenance["parent_sha256"] = file_hash(cfg["init_checkpoint"])
    if path.exists() and not args.resume:
        raise FileExistsError("Choose a new output or --resume")
    saved_objective = None
    if not args.resume and cfg.get("init_checkpoint"):
        state = torch.load(cfg["init_checkpoint"], map_location="cpu", weights_only=False, mmap=True)
        check_fork_contract(state, cfg, model_config, provenance)
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = cfg["learning_rate"]; group["initial_lr"] = cfg["learning_rate"]
        scheduler = alignment_scheduler(optimizer, cfg, len(train_loader))
        if sam:
            sam.load_state_dict(state["sam"])
        start = state["epoch"] + 1
        # Every condition has the same parent model/optimizer and stochastic seed.
        del state
    if args.resume:
        state = torch.load(path, map_location="cpu", weights_only=False)
        check_resume_contract(state, cfg, provenance, extending=args.extend_epochs is not None)
        if state["backbone_revision"] != REVISION:
            raise ValueError("Resume config, code, source, or teacher provenance differs")
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        if sam:
            sam.load_state_dict(state["sam"])
        start, best = state["epoch"] + 1, state["best_score"]
        saved_objective = state.get("fgw_state")
        restore_rng(state["rng"])
        random.setstate(state["python_rng"]); np.random.set_state(state["numpy_rng"])
    objective_rng = rng_snapshot()
    objective = VisualObjective(cfg, train_set, device, output_dir, saved_objective) if cfg.get("fgw") else None
    restore_rng(objective_rng)  # Loading a frozen tactile teacher must not alter visual random draws.
    write_json(output_dir / "config.json", cfg)
    if cfg.get("log_initial_weight_hash") and start == 0:
        write_json(output_dir / "initialization.json", {"seed": cfg["seed"],
            "latent_dim": cfg["latent_dim"], "model_sha256": model_weight_hash(model),
            "parameters": sum(p.numel() for p in model.parameters()),
            "sensor_code_dims": dimensions})
    elapsed_before = 0.
    if start and (output_dir / "metrics.jsonl").exists():
        for line in (output_dir / "metrics.jsonl").read_text().splitlines():
            try:
                previous = json.loads(line)
            except ValueError:
                continue
            if previous.get("epoch") == start:
                elapsed_before = previous["seconds"]
    began = time.monotonic() - elapsed_before
    print(json.dumps({"event": "indexed", "training_sources": sources,
                      "heldout": cfg["heldout_source"], "batches_per_epoch": len(train_loader),
                      "start_epoch": start + 1, "total_epochs": cfg["epochs"],
                      "lr_extension": cfg.get("lr_extension")}), flush=True)
    for epoch in range(start, min(cfg["epochs"], args.stop_after_epoch or cfg["epochs"])):
        alignment_info = None
        if objective:
            refresh_rng = rng_snapshot() if cfg.get("preserve_refresh_rng") else None
            try:
                alignment_info = objective.refresh(model, backbone, epoch)
            except Exception as error:
                write_json(output_dir / "alignment_failure.json", {"epoch": epoch + 1,
                    "error": str(error), "solver_failures": 1})
                raise
            if refresh_rng is not None:
                restore_rng(refresh_rng)
            train_sampler.set_epoch(epoch, objective.banks)
        elif sampling:
            train_sampler.set_epoch(epoch, sample_banks(train_set, sampling["bank_size"], cfg["seed"], epoch))
        model.train(); train_sampler.set_epoch(epoch); optimizer.zero_grad(set_to_none=True)
        replay = []; totals = defaultdict(lambda: np.zeros(3)); counts = defaultdict(int)
        means_sum = torch.zeros(cfg["latent_dim"], dtype=torch.float64)
        outer_sum = torch.zeros(cfg["latent_dim"], cfg["latent_dim"], dtype=torch.float64)
        latent_count, predicted_variance = 0, 0.
        for step, original in enumerate(train_loader):
            source = original["source"][0]
            if any(value != source for value in original["source"]):
                raise ValueError("Mixed source batch")
            batch = {key: value.to(device) if torch.is_tensor(value) and key not in ("rgb",) else value
                     for key, value in original.items()}
            mask = batch["view_mask"].clone()
            if mask[:, :, 0].all():
                mask[:, :, 1:] &= torch.rand(len(mask), 1, 2, device=device) >= cfg["wrist_dropout"]
            with torch.no_grad(), torch.autocast(device.type, torch.bfloat16, enabled=device.type == "cuda"):
                encoded = features(backbone, original, device)
            rng = rng_snapshot()
            beta = kl_weight(cfg, epoch + step / len(train_loader))
            group_size = min(cfg["accumulation"], len(train_loader) - (step // cfg["accumulation"]) * cfg["accumulation"])
            payload = objective.payload(original, source) if objective else None
            alignment_epoch = epoch - cfg.get("fork_parent_epochs", 0)
            with torch.autocast(device.type, torch.bfloat16, enabled=device.type == "cuda"):
                prediction = model(encoded, mask, source)
                loss, task, kl = alignment_loss(prediction, batch, beta)
                extra_metrics = {}
                if objective:
                    extra, extra_metrics = visual_extra_loss(prediction, payload, cfg, alignment_epoch)
                    loss = loss + extra
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite code-matching loss")
            (loss / group_size).backward()
            if sam:
                replay.append({"features": encoded.detach(), "mask": mask, "source": source,
                    "batch": {key: value.detach() for key, value in batch.items()
                              if key in ("code_target", "code_valid", "target", "target_mask")},
                    "rng": rng, "beta": beta})
                if objective:
                    replay[-1].update(alignment_payload=payload, config=cfg, alignment_epoch=alignment_epoch)
            totals[source] += [loss.item(), task.item(), kl.item()]; counts[source] += 1
            means = prediction["mu"].detach().double().flatten(0, 1).cpu()
            means_sum += means.sum(0); outer_sum += means.T @ means; latent_count += len(means)
            predicted_variance += prediction["logvar"].detach().float().exp().sum().item()
            if (step + 1) % cfg["accumulation"] == 0 or step + 1 == len(train_loader):
                if sam:
                    sam_replay(model, optimizer, sam, replay, device); replay.clear()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                    optimizer.step(); optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            if step % cfg["log_every"] == 0:
                print(json.dumps({"event": "train", "epoch": epoch + 1, "batch": step,
                    "source": source, "loss": loss.item(), "code_or_numeric_loss": task.item(),
                    "kl": kl.item(), "beta": beta, "lr": scheduler.get_last_lr()[0],
                    **extra_metrics}), flush=True)
        validation = validate(model, backbone, val_loader, registry, device)
        score = float(np.mean([m["supervision_loss"] for m in validation.values()]))
        mean = means_sum / latent_count
        covariance = outer_sum / latent_count - mean[:, None] * mean[None, :]
        spectrum = torch.linalg.eigvalsh(covariance).clamp_min(0)
        proportions = spectrum / spectrum.sum().clamp_min(1e-12)
        rank = (-(proportions * proportions.clamp_min(1e-12).log()).sum()).exp().item()
        improved = score < best; best = min(score, best)
        record = {"epoch": epoch + 1, "validation": validation, "selection_score": score,
                  "train": {source: dict(zip(("loss", "supervision", "kl"), (total / counts[source]).tolist()))
                            for source, total in totals.items()},
                  "latent": {"effective_rank_mu": rank, "mean_coordinate_variance_mu": covariance.diag().mean().item(),
                             "mean_predicted_variance": predicted_variance / (latent_count * cfg["latent_dim"])},
                  "seconds": time.monotonic() - began}
        if objective:
            record["alignment"] = alignment_info
        state = {"model_type": MODEL_TYPE, "model_config": model_config, "model": model.state_dict(),
                 "config": cfg, "provenance": provenance, "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(), "epoch": epoch, "best_score": best,
                 "backbone": MODEL_ID, "backbone_revision": REVISION, "rng": rng_snapshot(),
                 "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                 "teachers": registry, "validation": validation}
        if objective:
            state["fgw_state"] = objective.state_dict()
        if sam:
            state["sam"] = sam.state_dict()
        atomic_checkpoint(path, state)
        from .viewer_checkpoints import save_viewer_epoch
        save_viewer_epoch(path, state)
        if improved:
            atomic_checkpoint(output_dir / "best.pt", state)
        if cfg["checkpoint_every"] and (epoch + 1) % cfg["checkpoint_every"] == 0:
            (output_dir / f"epoch_{epoch + 1:03d}.pt").hardlink_to(path)
        with (output_dir / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
