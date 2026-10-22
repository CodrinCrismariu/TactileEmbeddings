"""Discover runnable current-frame checkpoints; historical workflows stay separate."""

import hashlib
import json
import time
from pathlib import Path

import torch


def identifier(path):
    return hashlib.sha256(str(Path(path).resolve()).encode()).hexdigest()[:20]


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def current_checkpoint(state):
    config = state.get("model_config", {})
    supported = (
        state.get("schema_version") == 2
        or state.get("model_type") == "joint_vjepa21_spatial_marker_codes_v1"
    )
    return (
        supported
        and config.get("prediction_mode") == "current_frame"
        and "sensor_code_dims" in config
        and "numeric_shapes" in config
    )


class ModelCatalog:
    def __init__(self, project_root):
        self.root = Path(project_root).resolve()
        self.entries = {}
        self.last_scan = -float("inf")
        self.metadata = {}
        self.errors = []

    def checkpoint_metadata(self, path):
        stat = path.stat()
        stamp = (stat.st_mtime_ns, stat.st_size)
        cached = self.metadata.get(str(path))
        if cached and cached[0] == stamp:
            return cached[1]
        state = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        if not current_checkpoint(state):
            raise ValueError("Historical architecture belongs on the Historical viewer page")
        config = state["model_config"]
        result = dict(
            epoch=int(state["epoch"]) + 1,
            latent_dim=config.get("latent_dim"),
            sources=list(config["sensor_code_dims"]) + list(config["numeric_shapes"]),
            schema_version=state.get("schema_version", 1),
        )
        self.metadata[str(path)] = (stamp, result)
        return result

    def scan(self, refresh=False):
        if not refresh and time.monotonic() - self.last_scan < 20:
            return
        found = {}
        errors = []
        processing = self.root / "processing"
        runs = processing / "runs"
        for config_path in sorted(runs.rglob("config.json")) if runs.exists() else []:
            folder = config_path.parent
            if any("smoke" in part.lower() for part in folder.relative_to(runs).parts):
                continue
            config = read_json(config_path)
            model_config = config.get("model", config)
            if model_config.get("prediction_mode") != "current_frame":
                continue
            paths = sorted((folder / "viewer_epochs").glob("epoch_*.pt"))
            if not paths:
                paths = sorted(folder.glob("epoch_*.pt"))
            paths = [path for path in paths if path.stem != "epoch_000"]
            if (folder / "last.pt").exists():
                paths.append(folder / "last.pt")
            if not paths:
                continue
            try:
                meta = self.checkpoint_metadata(paths[-1])
                if meta["epoch"] < 1:
                    continue
            except (OSError, ValueError, KeyError, RuntimeError) as error:
                errors.append(dict(path=str(folder), reason=str(error)))
                continue
            key = identifier(folder)
            found[key] = dict(
                id=key,
                name=str(folder.relative_to(runs)),
                kind="joint",
                path=str(folder),
                sources=meta["sources"],
                latent_dim=meta["latent_dim"],
                epochs=[path.name for path in paths],
                modified=max(path.stat().st_mtime for path in paths),
            )
        for registration in (
            sorted(processing.rglob("viewer_registration.json")) if processing.exists() else []
        ):
            document = read_json(registration)
            if document.get("model_type") != "motiontrans_probe_v2":
                continue
            encoder = Path(document.get("encoder_checkpoint", ""))
            if not encoder.is_absolute():
                encoder = self.root / encoder
            if not encoder.is_file():
                continue
            for head in document.get("heads", []):
                path = Path(head)
                if not path.is_absolute():
                    path = registration.parent / path
                if not path.is_file():
                    continue
                key = identifier(path)
                found[key] = dict(
                    id=key,
                    name=str(registration.parent.relative_to(processing))
                    + " / "
                    + path.parent.name,
                    kind="transfer",
                    path=str(encoder),
                    head=str(path),
                    sources=[document["source"]],
                    epochs=[encoder.name],
                    modified=registration.stat().st_mtime,
                )
        self.entries = found
        self.errors = errors
        self.last_scan = time.monotonic()

    def public(self, refresh=False):
        self.scan(refresh)
        return [
            {key: value for key, value in entry.items() if key not in ("path", "head")}
            for entry in sorted(
                self.entries.values(), key=lambda item: item["modified"], reverse=True
            )
        ]

    def resolve(self, key, epoch=None):
        self.scan()
        if key not in self.entries:
            raise ValueError("Select a runnable model from the catalog")
        entry = self.entries[key]
        epoch = epoch or entry["epochs"][-1]
        if epoch not in entry["epochs"] or Path(epoch).name != epoch:
            raise ValueError("Select an available checkpoint epoch")
        if entry["kind"] == "transfer":
            return entry, Path(entry["path"])
        folder = Path(entry["path"])
        path = folder / "viewer_epochs" / epoch
        if not path.exists():
            path = folder / epoch
        if not path.is_file():
            raise ValueError("Selected epoch is no longer available")
        return entry, path
