"""Explicit current-frame joint training on the shared dataset contract.

There are no dataset installation hooks, source-count checks, or dataset-name
branches. Data management supplies verified samples and sensor specifications.
Inputs are the selected experiment configuration and registry-backed clips;
outputs are resumable checkpoints, viewer snapshots, and per-source metrics in
the configured run directory. Invoke with ``tl train --preflight-only`` for a
bounded update check, or ``tl train`` for the explicit configured budget.
"""

import fcntl
import json
import os
import random
import shutil
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..configuration import load_experiment, resolve_path
from .checkpointing import (
    atomic_checkpoint,
    check_storage,
    model_weight_hash,
    restore_checkpoint,
    save_checkpoint,
    write_json,
)
from .objectives import forward_batch, supervision_loss, valid_frames
from .optim import SAMStep, make_scheduler, replay_update, rng_snapshot
from .statistics import LatentStats
from .validation import move_batch, paired_validate, validate


def wrist_dropout(batch, probability, seed, epoch, step, device):
    """Mask whole wrist clips using a local, reproducible sampling generator.

    The ego camera remains available. Wrist dropout is applied only when the
    batch has ego coverage, so a wrist-only clip cannot lose its only view.
    RGB pixels and target measurements remain unchanged.
    """
    mask = batch.visual.view_mask.to(device).clone()
    generator = torch.Generator(device=device).manual_seed(seed + epoch * 1000000 + step)
    if mask[:, :, 0].all():
        mask[:, :, 1:] &= (
            torch.rand(len(mask), 1, 2, device=device, generator=generator) >= probability
        )
    return replace(batch, visual=replace(batch.visual, view_mask=mask))


def train_epoch(
    model,
    backbone,
    loader,
    registry,
    optimizer,
    scheduler,
    sam,
    config,
    data_config,
    device,
    epoch,
    root=None,
):
    """Train one sampled epoch and return per-source loss/latent diagnostics.

    ``epoch`` is zero-based. The sampler determines how many windows each
    source contributes, so an epoch need not exhaust all available clips.
    Mirroring is selected once before encoding. Detached frozen features,
    transformed targets, camera masks, and dropout RNG are reused for SAM's
    second forward, preserving the original example and augmentation.
    """
    from ..models.backbone import spatial_video_features

    model.train()
    optimizer.zero_grad(set_to_none=True)
    replay, totals = [], defaultdict(list)
    stats = LatentStats(model.visual.latent_dim)
    started = time.monotonic()
    for step, original in enumerate(loader):
        original = registry.augment(original, seed=config["seed"], epoch=epoch, step=step)
        batch = wrist_dropout(
            move_batch(original, device),
            config["wrist_dropout"],
            config["seed"],
            epoch,
            step,
            device,
        )
        with (
            torch.no_grad(),
            torch.autocast(device.type, torch.bfloat16, enabled=device.type == "cuda"),
        ):
            features = spatial_video_features(
                backbone, original, device, chunk_size=config.get("video_chunk_size", 2)
            )
        before = rng_snapshot()
        with torch.autocast(device.type, torch.bfloat16, enabled=device.type == "cuda"):
            output = forward_batch(model, features, batch)
            task = supervision_loss(output, batch)
            weight = data_config.get("weights", {}).get(batch.source, 1.0)
            loss = task * weight
        if not torch.isfinite(loss):
            raise FloatingPointError(batch.source)
        group = min(
            config["accumulation"],
            len(loader) - (step // config["accumulation"]) * config["accumulation"],
        )
        (loss / group).backward()
        totals[batch.source].append(task.item())
        stats.add(output, valid_frames(batch))
        replay.append(dict(features=features.detach(), batch=batch, rng=before, weight=weight))
        if (step + 1) % config["accumulation"] == 0 or step + 1 == len(loader):
            replay_update(model, optimizer, sam, replay, device)
            replay.clear()
            scheduler.step()
        if root is not None and step % config.get("log_every", 25) == 0:
            write_json(
                root / "status.json",
                dict(
                    stage="training",
                    pid=os.getpid(),
                    epoch=epoch + 1,
                    completed_epochs=epoch,
                    total_epochs=config["epochs"],
                    batch=step + 1,
                    batches_per_epoch=len(loader),
                    source=batch.source,
                    loss=loss.item(),
                    epoch_seconds=time.monotonic() - started,
                    latent_dim=model.visual.latent_dim,
                    prediction_mode="current_frame",
                    visual_cache="none",
                ),
            )
    return {
        s: dict(supervision=float(np.mean(values))) for s, values in totals.items()
    }, stats.result()


def train_experiment(
    experiment,
    *,
    project_root,
    resume=None,
    preflight_only=False,
    import_checkpoint=None,
    import_only=False,
):
    """Execute the selected experiment using explicit project-relative paths.

    Fresh runs write actual epoch-0 weights before training; resumes require
    matching schema-2 optimizer, RNG, configuration, and artifact provenance.
    A legacy current-frame V-JEPA checkpoint requires an explicit verified import
    into a new run. ``import_only`` publishes that state without optimization.
    ``preflight_only`` performs bounded local model updates but never writes a
    production training run.
    """
    from ..dataset_management import (
        SourceBatchSampler,
        build_dataset,
        build_sources,
        collate_samples,
    )
    from ..models import JointModel
    from ..models.backbone import VJEPABackbone

    if import_only and import_checkpoint is None:
        raise ValueError("Import-only mode requires an explicit checkpoint")
    if import_checkpoint is not None and (resume is not None or preflight_only):
        raise ValueError("Checkpoint import cannot be combined with resume or disposable preflight")
    root_project = Path(project_root).resolve()
    cfg = load_experiment(experiment, project_root=root_project)
    tc = cfg["training"]
    validate_configuration(cfg)
    seed = tc["seed"]
    torch.set_num_threads(tc.get("cpu_threads", 4))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    registry = build_sources(cfg["datasets"], root_project)
    if not preflight_only:
        registry.require_imports()
    train, val = build_dataset(registry, "train"), build_dataset(registry, "val")
    mc = registry.model_config(cfg["model"])
    heldout = cfg["datasets"].get("heldout_source", "ftp_motiontrans")
    if heldout in train.indices_by_source or heldout in val.indices_by_source:
        raise ValueError("Held-out data entered source training or validation")
    device = torch.device(tc.get("device", "cuda"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required by this experiment; select CPU explicitly for small tests"
        )
    bc = cfg["model"]["backbone"]
    backbone = (
        VJEPABackbone(
            resolve_path(bc["repo"], root_project), resolve_path(bc["checkpoint"], root_project)
        )
        .to(device)
        .eval()
    )
    torch.manual_seed(seed)
    model = JointModel(**mc).to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=tc["learning_rate"],
        weight_decay=tc["weight_decay"],
    )
    sam = SAMStep(optimizer, tc["sam_rho"])
    sampler = SourceBatchSampler(
        train,
        tc["batch_size"],
        seed=seed,
        max_windows_per_source=tc["train_windows_per_source_per_epoch"],
        sampling_mode="cycle",
    )
    vsampler = SourceBatchSampler(
        val, tc["batch_size"], seed=seed, max_windows_per_source=tc["validation_windows_per_source"]
    )
    generator = torch.Generator().manual_seed(seed)
    options = dict(
        num_workers=tc["workers"],
        pin_memory=device.type == "cuda",
        generator=generator,
        collate_fn=collate_samples,
    )
    loader, vloader = (
        DataLoader(train, batch_sampler=sampler, **options),
        DataLoader(val, batch_sampler=vsampler, **options),
    )
    scheduler = make_scheduler(optimizer, tc, len(loader))
    provenance = dict(
        configuration=cfg["provenance"],
        datasets=registry.provenance,
        backbone=backbone.provenance,
        teachers={
            "sources": {s: spec.teacher for s, spec in registry.specs.items() if spec.teacher}
        },
        scientific_contract="native-current-frame-baseline-v1",
    )
    output = resolve_path(resume or tc["output"], root_project)
    if preflight_only:
        return preflight(model, backbone, train, registry, cfg, device)
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "training.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (output / "last.pt").exists() and resume is None:
        raise FileExistsError("Run already has a checkpoint; use explicit --resume")
    if import_checkpoint is not None and any(
        path.name != "training.lock" for path in output.iterdir()
    ):
        raise FileExistsError("Checkpoint import requires an empty, separate destination run")
    # Pin code as well as data; a cleanup cannot silently alter a resumable run.
    from ..dataset_management.artifacts import scientific_implementation_signature

    provenance["implementation"] = scientific_implementation_signature(Path(__file__).parents[1])
    start, best, elapsed = 0, float("inf"), 0.0
    if import_checkpoint is not None:
        from .legacy_import import import_legacy_checkpoint

        state, receipt = import_legacy_checkpoint(
            import_checkpoint,
            model_config=mc,
            experiment=cfg,
            provenance=provenance,
            registry=registry,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            sam=sam,
            generator=generator,
            batches_per_epoch=len(loader),
        )
        check_storage(root_project)
        start, best, elapsed = state["epoch"] + 1, state["best_score"], state["elapsed_seconds"]
        previous_metrics = Path(import_checkpoint).resolve().parent / "metrics.jsonl"
        if previous_metrics.is_file():
            rows = [
                json.loads(line)
                for line in previous_metrics.read_text().splitlines()
                if line.strip()
            ]
            if [row.get("epoch") for row in rows] != list(range(1, start + 1)):
                raise ValueError("Parent metrics do not match the imported completed epochs")
        atomic_checkpoint(output / "imported_checkpoint.pt", state)
        atomic_checkpoint(output / "last.pt", state)
        snapshot = {
            key: state[key]
            for key in (
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
        atomic_checkpoint(output / "viewer_epochs" / f"epoch_{start:03d}.pt", snapshot)
        write_json(output / "checkpoint_import.json", receipt)
        if previous_metrics.is_file():
            shutil.copy2(previous_metrics, output / "metrics.jsonl")
    elif resume is not None:
        state = restore_checkpoint(
            output / "last.pt", model, mc, cfg, provenance, optimizer, scheduler, sam, generator
        )
        start, best, elapsed = state["epoch"] + 1, state["best_score"], state["elapsed_seconds"]
    else:
        check_storage(root_project)
        save_checkpoint(
            output,
            0,
            model,
            mc,
            cfg,
            provenance,
            optimizer,
            scheduler,
            sam,
            generator,
            best,
            {},
            0.0,
        )
        write_json(
            output / "initialization.json", dict(seed=seed, model_sha256=model_weight_hash(model))
        )
    write_json(output / "config.json", cfg)
    write_json(output / "provenance.json", provenance)
    write_json(
        output / "status.json",
        dict(
            stage="imported" if import_only else "initializing",
            pid=None if import_only else os.getpid(),
            completed_epochs=start,
            epoch=start + 1,
            total_epochs=tc["epochs"],
            batches_per_epoch=len(loader),
            latent_dim=model.visual.latent_dim,
            prediction_mode="current_frame",
        ),
    )
    if import_only:
        return dict(
            output=str(output), completed_epochs=start, next_epoch=start + 1, stage="imported"
        )
    print(
        json.dumps(
            dict(
                event="indexed",
                completed_epochs=start,
                start_epoch=start + 1,
                total_epochs=tc["epochs"],
                batches_per_epoch=len(loader),
                optimizer_updates=sam.steps,
                learning_rate=scheduler.get_last_lr()[0],
            )
        ),
        flush=True,
    )
    began = time.monotonic() - elapsed
    for epoch in range(start, tc["epochs"]):
        epoch_start = time.monotonic()
        sampler.set_epoch(epoch)
        training, latent = train_epoch(
            model,
            backbone,
            loader,
            registry,
            optimizer,
            scheduler,
            sam,
            tc,
            cfg["datasets"],
            device,
            epoch,
            output,
        )
        write_json(
            output / "status.json", dict(stage="validation", epoch=epoch + 1, pid=os.getpid())
        )
        validation = validate(model, backbone, vloader, registry, root_project, device)
        mirror_validation = paired_validate(
            model, backbone, vloader, registry, root_project, device, seed=seed
        )
        score = float(np.mean([row["supervision_loss"] for row in validation.values()]))
        best = min(best, score)
        record = dict(
            epoch=epoch + 1,
            train=training,
            validation=validation,
            mirror_validation=mirror_validation,
            latent=latent,
            selection_score=score,
            seconds=time.monotonic() - began,
            epoch_seconds=time.monotonic() - epoch_start,
            uncertainty="none",
        )
        check_storage(root_project)
        save_checkpoint(
            output,
            epoch + 1,
            model,
            mc,
            cfg,
            provenance,
            optimizer,
            scheduler,
            sam,
            generator,
            best,
            validation,
            record["seconds"],
        )
        with (output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
    write_json(
        output / "status.json",
        dict(stage="complete", completed_epochs=tc["epochs"], total_epochs=tc["epochs"]),
    )
    return dict(output=str(output), completed_epochs=tc["epochs"])


def validate_configuration(cfg):
    """Reject ignored or contradictory controls rather than guessing semantics."""
    bc, mc, tc, dc = cfg["model"]["backbone"], cfg["model"], cfg["training"], cfg["datasets"]
    from ..models.backbone import BACKBONE_ID, CHECKPOINT_SHA256, REVISION

    expected = dict(
        id=BACKBONE_ID,
        revision=REVISION,
        checkpoint_sha256=CHECKPOINT_SHA256,
        rgb_size=384,
        clip_frames=8,
        frame_stride=2,
        frozen=True,
        spatial_pooling=False,
    )
    if any(bc.get(key) != value for key, value in expected.items()):
        raise ValueError("Backbone controls must match the pinned full-token V-JEPA implementation")
    if any(dc.get(key) != expected[key] for key in ("rgb_size", "clip_frames", "frame_stride")):
        raise ValueError("Dataset visual clip configuration differs from the backbone")
    if mc.get("latent_mode") != "deterministic" or mc.get("prediction_mode") != "current_frame":
        raise ValueError("This engine preserves the deterministic current-frame baseline only")
    if (
        mc.get("input_dim") != 768
        or mc.get("grid_size") != 24
        or mc.get("initialization") != "baseline_v1"
    ):
        raise ValueError(
            "Input width, spatial grid and initialization must match the pinned backbone/baseline"
        )
    if tc.get("optimizer_method") != "sam_adamw":
        raise ValueError("The supported optimizer is explicitly SAM AdamW")
    for name in (
        "epochs",
        "batch_size",
        "accumulation",
        "train_windows_per_source_per_epoch",
        "validation_windows_per_source",
    ):
        if not isinstance(tc[name], int) or isinstance(tc[name], bool) or tc[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    selected = set(dc["sources"])
    if set(dc.get("weights", {})) - selected or any(
        value <= 0 for value in dc.get("weights", {}).values()
    ):
        raise ValueError("Source weights must be positive and name selected sources")


def preflight(model, backbone, dataset, registry, cfg, device):
    """Check one complete SAM update per source on the newly constructed model.

    This changes only the in-memory task model, not saved production weights.
    Native sample/latent shapes and the frozen backbone's absent gradients are
    reported. No checkpoint, dataset preparation, or feature cache is created.
    """
    from ..dataset_management import collate_samples
    from ..models.backbone import spatial_video_features

    rows = []
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"])
    sam = SAMStep(optimizer, cfg["training"]["sam_rho"])
    for source, ids in dataset.indices_by_source.items():
        original = collate_samples([dataset[i] for i in ids[:2]])
        batch = move_batch(original, device)
        with (
            torch.no_grad(),
            torch.autocast(device.type, torch.bfloat16, enabled=device.type == "cuda"),
        ):
            features = spatial_video_features(backbone, original, device)
        before = rng_snapshot()
        with torch.autocast(device.type, torch.bfloat16, enabled=device.type == "cuda"):
            output = forward_batch(model, features, batch)
            loss = supervision_loss(output, batch)
        loss.backward()
        replay_update(
            model, optimizer, sam, [dict(features=features, batch=batch, rng=before)], device
        )
        rows.append(
            dict(
                source=source,
                latent_shape=list(output["mu"].shape),
                loss=loss.item(),
                target_shape=list(batch.target.values.shape),
            )
        )
    if any(p.grad is not None for p in backbone.parameters()):
        raise RuntimeError("Frozen backbone acquired gradients")
    return dict(passed=True, production_weights_modified=False, sources=rows)
