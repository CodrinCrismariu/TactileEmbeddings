"""Evaluate declared saved epochs using the uniform source/sample contract.

Only compact final feature vectors are cached here, never RGB or spatial tokens.
Readout fitting is separate from representation training. MotionTrans is allowed
only by the explicit evaluation dataset request and never becomes a source head.
Inputs are an experiment, saved epochs, and a declared evaluation protocol.
Outputs are provenance-pinned feature tables, fitted readouts, status records,
and validation results. For example, ``tl evaluate --run RUN --epochs 10
--protocol motiontrans`` evaluates a completed checkpoint without resuming it.
"""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from ..configuration import load_experiment, resolve_path
from .probes import (
    atomic_save,
    episode_partition,
    fit_head,
    fit_probe_sweep,
    head,
    seed_all,
    write_json,
)


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checkpoint_paths(run, epochs=None):
    """Resolve distinct completed epoch numbers, or the latest saved state."""
    run = Path(run)
    if epochs is None:
        if not (run / "last.pt").is_file():
            raise FileNotFoundError(run / "last.pt")
        return [(None, run / "last.pt")]
    if (
        not epochs
        or len(set(epochs)) != len(epochs)
        or any(isinstance(e, bool) or e < 1 for e in epochs)
    ):
        raise ValueError("Request distinct positive completed epoch numbers")
    result = []
    for epoch in epochs:
        options = [run / "viewer_epochs" / f"epoch_{epoch:03d}.pt", run / f"epoch_{epoch:03d}.pt"]
        path = next((path for path in options if path.is_file()), None)
        if path is None:
            raise FileNotFoundError(f"No saved completed epoch {epoch} in {run}")
        result.append((epoch, path))
    return result


def verify_checkpoint(state, registry, requested_epoch=None):
    """Check format, source exclusions, epoch identity, and producer bytes.

    New-core scientific files and archived historical producers are resolved
    through their explicit provenance paths. A changed implementation fails
    validation; the evaluator never substitutes a recorded hash.
    """
    mc = state["model_config"]
    completed = state["epoch"] + 1
    if completed < 1 or requested_epoch is not None and completed != requested_epoch:
        raise ValueError("Checkpoint does not match the requested completed epoch")
    if mc.get("prediction_mode") != "current_frame" or mc.get("latent_mode") != "deterministic":
        raise ValueError("This evaluator requires deterministic current-frame spatial checkpoints")
    source_heads = set(mc.get("sensor_code_dims", {})) | set(mc.get("numeric_shapes", {}))
    if registry.heldout_source in source_heads or source_heads & registry.excluded_sources:
        raise ValueError("Checkpoint violates held-out/excluded-source membership")
    if not source_heads or not source_heads <= set(registry.sources):
        raise ValueError("Checkpoint sources do not have registered data contracts")
    # Preserve source-hash validation for both new and archived producers.
    package = Path(__file__).resolve().parents[1]
    implementation = state.get("provenance", {}).get("implementation", {})
    if not implementation:
        raise ValueError("Checkpoint has no implementation provenance")
    for name, digest in implementation.items():
        if state.get("schema_version") == 2:
            from ..dataset_management.artifacts import scientific_implementation_path

            if not scientific_implementation_path(name):
                raise ValueError(f"Unexpected scientific implementation path: {name}")
            path = package / name
        else:
            path = registry.paths.producer(name)
        if file_hash(path) != digest:
            raise ValueError(f"Checkpoint implementation changed: {name}")
    return completed


def select_indices(dataset, *, partition=None, cap=None, seed=45, tune_fraction=0.1):
    """Select chronological window indices with episode-disjoint partitions.

    An optional deterministic cap samples windows inside the requested
    partition; selected indices are sorted back into dataset order.
    """
    result = []
    for index, (episode, _) in enumerate(dataset.windows):
        row = dataset.episodes[episode]
        if (
            partition is None
            or episode_partition(row["episode_id"], seed, tune_fraction) == partition
        ):
            result.append(index)
    if cap is not None and len(result) > cap:
        result = sorted(np.random.default_rng(seed).choice(result, cap, replace=False).tolist())
    if not result:
        raise ValueError(f"No supported clips for partition={partition}")
    return result


def _feature_signature(
    checkpoint_hash, registry, dataset, indices, source, split, with_targets=True
):
    selected = [
        (dataset.episodes[dataset.windows[i][0]]["episode_id"], dataset.windows[i][1])
        for i in indices
    ]
    return dict(
        schema="frozen-current-features-v2",
        checkpoint_sha256=checkpoint_hash,
        data=registry.provenance,
        source=source,
        split=split,
        windows_sha256=hashlib.sha256(
            json.dumps(selected, separators=(",", ":")).encode()
        ).hexdigest(),
        count=len(indices),
        with_targets=with_targets,
    )


@torch.no_grad()
def extract_features(
    model,
    backbone,
    registry,
    dataset,
    indices,
    path,
    checkpoint_hash,
    source,
    split,
    device,
    batch_size=2,
    workers=0,
    progress=None,
    with_targets=True,
):
    """Cache one deterministic [D] current-frame vector per supported clip.

    The cache contains CPU values [clips, D], flattened native targets and
    validity [clips, channels], episode IDs, and current-frame indices. Inputs
    remain complete eight-frame, multi-camera clips. Matching cache provenance
    permits reuse; mismatched checkpoint/data/window identities are rejected.
    With ``with_targets=False``, only compact placeholder labels are retained
    for source-identity probing, avoiding pressure-image target caches.
    """
    from ..dataset_management import collate_samples
    from ..models.backbone import spatial_video_features

    signature = _feature_signature(
        checkpoint_hash, registry, dataset, indices, source, split, with_targets
    )
    path = Path(path)
    if path.exists():
        cached = torch.load(path, map_location="cpu", weights_only=False)
        if cached.get("signature") != signature:
            raise ValueError(f"Frozen feature cache provenance changed: {path}")
        return cached
    model.eval().requires_grad_(False)
    backbone.eval().requires_grad_(False)
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        num_workers=workers,
        collate_fn=collate_samples,
        pin_memory=device.type == "cuda",
    )
    values = []
    targets = []
    valids = []
    episodes = []
    frames = []
    for step, batch in enumerate(loader):
        with torch.autocast(device.type, torch.bfloat16, enabled=device.type == "cuda"):
            features = spatial_video_features(backbone, batch, device)
            roles = batch.visual.camera_roles
            if torch.is_tensor(roles):
                roles = roles.to(device)
            mu, _ = model.encode(features, batch.visual.view_mask.to(device), camera_roles=roles)
        if mu.ndim != 3 or mu.shape[1] != 1:
            raise ValueError("Expected exactly one current-frame representation per clip")
        if with_targets:
            y = batch.target.values.flatten(1).float().cpu()
            mask = batch.target.expanded_mask().flatten(1).bool().cpu()
        else:
            # Source identity needs only features and episode IDs. Never cache
            # pressure maps or tactile images as classifier targets.
            y = torch.zeros(len(mu), 1)
            mask = torch.ones(len(mu), 1, dtype=torch.bool)
        keep = mask.any(1)
        values.append(mu[:, 0].float().cpu()[keep])
        targets.append(torch.where(mask, y, 0.0)[keep])
        valids.append(mask[keep])
        episodes.extend(e for e, k in zip(batch.episode_ids, keep.tolist()) if k)
        frames.extend(batch.frame_indices[:, -1].cpu()[keep].tolist())
        if progress and step % 25 == 0:
            progress(
                "extracting_features",
                source=source,
                split=split,
                batches=step + 1,
                total_batches=len(loader),
            )
    if not episodes:
        raise ValueError("No observed targets in feature extraction")
    result = dict(
        values=torch.cat(values),
        target=torch.cat(targets),
        valid=torch.cat(valids),
        episode_id=episodes,
        frame_indices=torch.tensor(frames, dtype=torch.int64),
        signature=signature,
    )
    if not torch.isfinite(result["values"]).all():
        raise ValueError("Nonfinite frozen representation")
    atomic_save(path, result)
    return result


def _balanced(data):
    """Repeat only inside a partition, giving each source equal examples."""
    maximum = max(len(item["values"]) for item in data)
    xs = []
    ys = []
    for label, item in enumerate(data):
        ids = torch.arange(maximum) % len(item["values"])
        xs.append(item["values"][ids])
        ys.append(torch.full((maximum,), label, dtype=torch.long))
    x, y = torch.cat(xs), torch.cat(ys)
    return dict(mu=x, y=y, valid=torch.ones(len(y), dtype=torch.bool))


def fit_dataset_probe(fit, tune, validation, config, output, device=None, progress=None):
    """Fit an episode-disjoint source classifier on frozen visual latents.

    Each partition maps source names to compact feature tables. Sources are
    balanced by repetition within partitions; no held-out transfer source is
    added. Reported source accuracy diagnoses remaining dataset information,
    rather than establishing physical equivalence or calibrated uncertainty.
    """
    sources = sorted(fit)
    if set(tune) != set(sources) or set(validation) != set(sources):
        raise ValueError("Dataset probe source classes differ")
    for source in sources:
        a, b, c = (set(group[source]["episode_id"]) for group in (fit, tune, validation))
        if a & b or (a | b) & c:
            raise ValueError("Dataset-probe episode leakage")
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = config.get("seed", 45)
    hidden = config.get("hidden_dims", [512])
    dimension = fit[sources[0]]["values"].shape[1]
    all_fit = torch.cat([fit[source]["values"] for source in sources])
    normalization = config.get("standardization", "none")
    if normalization == "fit_episode_only":
        mean = all_fit.mean(0)
        std = all_fit.std(0, unbiased=False).clamp_min(1e-6)
    elif normalization == "none":
        mean = torch.zeros(dimension)
        std = torch.ones(dimension)
    else:
        raise ValueError("Unsupported dataset-probe standardization")

    def normalize(groups):
        return {s: {**row, "values": (row["values"] - mean) / std} for s, row in groups.items()}

    fit, tune, validation = map(normalize, (fit, tune, validation))
    fitting = _balanced([fit[s] for s in sources])
    tuning = _balanced([tune[s] for s in sources])
    fitting = {k: v.to(device) for k, v in fitting.items()}
    tuning = {k: v.to(device) for k, v in tuning.items()}
    seed_all(seed)
    network = head(dimension, len(sources), hidden, 0.1, classification=True).to(device)
    options = dict(
        epochs=config.get("epochs", 100),
        patience=config.get("patience", 10),
        seed=seed,
        lr=config.get("learning_rate", 0.001),
        weight_decay=config.get("weight_decay", 0.0001),
        classification=True,
        progress=progress,
    )
    selection = fit_head(network, fitting, tuning, **options)
    combined = []
    for source in sources:
        combined.append({"values": torch.cat([fit[source]["values"], tune[source]["values"]])})
    fitting = {k: v.to(device) for k, v in _balanced(combined).items()}
    seed_all(seed)
    network = head(dimension, len(sources), hidden, 0.1, classification=True).to(device)
    refit = fit_head(network, fitting, None, fixed_epochs=selection["selected_epochs"], **options)
    confusion = torch.zeros(len(sources), len(sources), dtype=torch.int64)
    per_source = {}
    network.eval()
    with torch.no_grad():
        for label, source in enumerate(sources):
            predicted = torch.cat(
                [
                    network(x.to(device)).argmax(-1).cpu()
                    for x in validation[source]["values"].split(512)
                ]
            )
            counts = torch.bincount(predicted, minlength=len(sources))
            confusion[label] = counts
            per_source[source] = dict(
                accuracy=float((predicted == label).float().mean()), clips=len(predicted)
            )
    result = dict(
        sources=sources,
        input_dimension=dimension,
        head=[dimension, *hidden, len(sources)],
        normalization=normalization,
        selected_epochs=selection["selected_epochs"],
        selection=selection,
        refit=refit,
        validation=dict(
            accuracy=float(confusion.diag().sum() / confusion.sum()),
            macro_accuracy=float(np.mean([row["accuracy"] for row in per_source.values()])),
            confusion_matrix=confusion.tolist(),
            per_source=per_source,
            chance_accuracy=1 / len(sources),
        ),
        note="Source identity diagnostic, not proof of invariance or calibrated uncertainty. Fit/tune/refit source counts balanced by repetition within each partition.",
    )
    atomic_save(
        Path(output) / "classifier.pt",
        dict(
            head=network.cpu().state_dict(),
            input_dimension=dimension,
            hidden_dims=hidden,
            output_dim=len(sources),
            classification=True,
            mean=mean,
            std=std,
            sources=sources,
        ),
    )
    return result


def _dataset_probe(
    model, backbone, registry, checkpoint_hash, output, configuration, device, progress
):
    from ..dataset_management import build_dataset

    settings = configuration["transfer"].get("dataset_probe", {})
    sources = sorted(set(model.sensor_code_dims) | set(model.numeric_shapes))
    if registry.heldout_source in sources:
        raise ValueError("Heldout entered dataset classifier")
    groups = {name: {} for name in ("fit", "tune", "val")}
    for number, source in enumerate(sources):
        train = build_dataset(registry, "train", sources=[source], include_targets=False)
        val = build_dataset(registry, "val", sources=[source], include_targets=False)
        seed = settings.get("seed", 45) + number
        for split, cap, data, destination in [
            ("fit", settings.get("train_clips_per_source", 1000), train, "fit"),
            ("tune", settings.get("tune_clips_per_source", 128), train, "tune"),
            ("val", settings.get("validation_clips_per_source"), val, "val"),
        ]:
            ids = select_indices(
                data, partition=split if split != "val" else None, cap=cap, seed=seed
            )
            groups[destination][source] = extract_features(
                model,
                backbone,
                registry,
                data,
                ids,
                output / "features" / f"{source}_{split}.pt",
                checkpoint_hash,
                source,
                split,
                device,
                workers=configuration["training"].get("workers", 0),
                progress=progress,
                with_targets=False,
            )
    return fit_dataset_probe(
        groups["fit"], groups["tune"], groups["val"], settings, output, device, progress
    )


def _evaluate_experiment(experiment, *, project_root, run, epochs=None, protocol="motiontrans"):
    from ..dataset_management import build_dataset, build_sources
    from ..models import VJEPABackbone, model_from_state

    project_root = Path(project_root).resolve()
    cfg = load_experiment(experiment, project_root=project_root)
    if protocol not in {"motiontrans", "dataset-probe"}:
        raise ValueError("Unknown evaluation protocol")
    registry = build_sources(cfg["datasets"], project_root)
    run = resolve_path(run, project_root)
    output = resolve_path(cfg["transfer"]["output"], project_root) / run.name / protocol
    device = torch.device(cfg["transfer"].get("device", cfg["training"].get("device", "cuda")))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; select transfer.device=cpu for small explicit tests")
    requested = checkpoint_paths(run, epochs)
    results = {}
    started = time.monotonic()
    for desired, path in requested:
        checkpoint_hash = file_hash(path)
        state = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        epoch = verify_checkpoint(state, registry, desired)
        folder = output / f"epoch_{epoch:03d}"
        context = {}

        def progress(stage, **fields):
            if "epoch" in fields:
                fields["head_epoch"] = fields.pop("epoch")
            context.update(
                {
                    key: value
                    for key, value in fields.items()
                    if key in ("seed", "learning_rate", "source", "split", "selected_epochs")
                }
            )
            snapshot = dict(
                stage=stage,
                epoch=epoch,
                protocol=protocol,
                elapsed_seconds=time.monotonic() - started,
                **{**context, **fields},
            )
            write_json(folder / "status.json", snapshot)
            write_json(output / "status.json", snapshot)
            with (folder / "progress.jsonl").open("a") as stream:
                stream.write(json.dumps(snapshot, allow_nan=False) + "\n")

        provenance = dict(
            checkpoint=str(path.resolve()),
            checkpoint_sha256=checkpoint_hash,
            epoch=epoch,
            datasets=registry.provenance,
            protocol=protocol,
            configuration=cfg["provenance"],
            checkpoint_teacher_provenance=state.get("provenance", {}).get("teachers", {}),
        )
        provenance = json.loads(json.dumps(provenance))
        pin = folder / "provenance.json"
        if pin.exists() and json.loads(pin.read_text()) != provenance:
            raise ValueError("Evaluation provenance changed; choose another output")
        write_json(pin, provenance)
        if (folder / "results.json").exists():
            results[str(epoch)] = json.loads((folder / "results.json").read_text())
            continue
        progress("loading_frozen_encoder")
        model = model_from_state(state).to(device).eval().requires_grad_(False)
        bc = cfg["model"]["backbone"]
        backbone = (
            VJEPABackbone(
                resolve_path(bc["repo"], project_root), resolve_path(bc["checkpoint"], project_root)
            )
            .to(device)
            .eval()
        )
        recorded = state.get("provenance", {}).get("backbone", {})
        for key in ("id", "revision", "checkpoint_sha256"):
            if key in recorded and recorded[key] != backbone.provenance[key]:
                raise ValueError("Backbone differs from checkpoint provenance")
        if protocol == "motiontrans":
            source = registry.heldout_source
            if source != cfg["transfer"].get("source", source):
                raise ValueError("Transfer source does not match the heldout policy")
            tables = {}
            for split in ("train", "val"):
                data = build_dataset(registry, split, sources=[source], allow_heldout=True)
                ids = select_indices(data)
                tables[split] = extract_features(
                    model,
                    backbone,
                    registry,
                    data,
                    ids,
                    folder / "features" / f"{split}.pt",
                    checkpoint_hash,
                    source,
                    split,
                    device,
                    workers=cfg["training"].get("workers", 0),
                    progress=progress,
                )
            result = fit_probe_sweep(
                tables["train"], tables["val"], folder / "probes", cfg["transfer"], device, progress
            )
            write_json(
                folder / "viewer_registration.json",
                dict(
                    model_type="motiontrans_probe_v2",
                    encoder_checkpoint=str(path.resolve()),
                    checkpoint_sha256=checkpoint_hash,
                    epoch=epoch,
                    source=source,
                    heads=[
                        str((folder / "probes" / f"seed_{seed}" / "head.pt").resolve())
                        for seed in cfg["transfer"]["head_seeds"]
                    ],
                    normalization=str((folder / "probes/normalization.pt").resolve()),
                    output_shape=[1, tables["train"]["target"].shape[1], 1, 1],
                    uncertainty="uncalibrated_head_fit_variation",
                ),
            )
        else:
            result = _dataset_probe(
                model, backbone, registry, checkpoint_hash, folder, cfg, device, progress
            )
        result.update(
            provenance=provenance,
            checkpoint_epoch=epoch,
            protocol_name=protocol,
            frozen_encoder=True,
        )
        write_json(folder / "results.json", result)
        results[str(epoch)] = result
        progress("complete")
        del model, backbone, state
    summary = dict(
        protocol=protocol,
        results=results,
        output=str(output),
        elapsed_seconds=time.monotonic() - started,
    )
    write_json(output / "summary.json", summary)
    return summary


def evaluate_experiment(experiment, *, project_root, run, epochs=None, protocol="motiontrans"):
    """Run the declared frozen-transfer or source-classifier evaluation.

    ``epochs`` contains human-facing completed counts. Results and continuous
    progress go under the configured transfer output, with failures recorded
    before they are raised. This never starts representation training.
    """
    cfg = load_experiment(experiment, project_root=project_root)
    output = resolve_path(cfg["transfer"]["output"], project_root) / Path(run).name / protocol
    try:
        return _evaluate_experiment(
            experiment, project_root=project_root, run=run, epochs=epochs, protocol=protocol
        )
    except Exception as error:
        write_json(
            output / "status.json",
            dict(stage="failed", protocol=protocol, error=f"{type(error).__name__}: {error}"),
        )
        raise


def load_probe(path, device="cpu"):
    """Strictly restore a fitted readout and its saved normalization metadata."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    network = head(
        state["input_dimension"],
        state["output_dim"],
        state["hidden_dims"],
        state.get("dropout", 0.1),
        state.get("classification", False),
    )
    network.load_state_dict(state["head"], strict=True)
    return network.to(device).eval().requires_grad_(False), state


@torch.no_grad()
def predict_probe(network, state, mu):
    """Standardize [..., D] means once and return [..., output_dim] predictions."""
    shape = mu.shape
    values = mu.reshape(-1, shape[-1]).float()
    mean = torch.as_tensor(state["mean"], device=values.device)
    std = torch.as_tensor(state["std"], device=values.device)
    prediction = network((values - mean) / std)
    return prediction.reshape(*shape[:-1], state["output_dim"])
