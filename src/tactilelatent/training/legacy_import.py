"""Import complete historical joint-training state through verified metadata.

This explicit one-time bridge reads the known native current-frame V-JEPA
checkpoint family. It verifies architecture, sampling, teachers, caches and the
archived producer bytes before restoring optimizer and random state. It never
executes archived Python, rewrites old receipts, or writes a production run.
The maintained trainer owns atomic publication of the returned state and receipt.
"""

import copy
import math
import random
from pathlib import Path

import numpy as np
import torch

from ..dataset_management.artifacts import file_hash, json_hash, read_json, verify_implementation
from ..models.joint import MODEL_TYPE
from .optim import restore_rng

TRAINING_CONTROLS = (
    "seed",
    "epochs",
    "batch_size",
    "accumulation",
    "workers",
    "train_windows_per_source_per_epoch",
    "validation_windows_per_source",
    "learning_rate",
    "min_learning_rate",
    "weight_decay",
    "warmup_epochs",
    "optimizer_method",
    "sam_rho",
    "wrist_dropout",
)
REQUIRED_STATE = frozenset(
    {
        "model_type",
        "model_config",
        "model",
        "config",
        "provenance",
        "teachers",
        "epoch",
        "optimizer",
        "scheduler",
        "sam",
        "rng",
        "python_rng",
        "numpy_rng",
        "loader_rng",
        "best_score",
        "validation",
        "elapsed_seconds",
    }
)


def _same_model_config(old, current):
    normalized = copy.deepcopy(old)
    normalized.setdefault("input_dim", 768)
    normalized.setdefault("initialization", "baseline_v1")
    if normalized != current:
        raise ValueError("Historical model architecture, heads or statistics changed")
    for key in ("sensor_code_dims", "numeric_shapes", "statistics", "direct_numeric"):
        if list(normalized.get(key, {})) != list(current.get(key, {})):
            raise ValueError("Historical model head/statistics order changed")


def _check_training_controls(old, experiment, registry):
    if (
        old.get("training_entrypoint") != "scripts/train_all_sources_vjepa.py"
        or old.get("dataset_family") != "joint_all"
        or old.get("model_type") != MODEL_TYPE
        or old.get("latent_mode") != "deterministic"
        or old.get("prediction_mode") != "current_frame"
        or old.get("stochastic") is not False
        or old.get("beta_max") != 0
        or old.get("feature_cache")
        or old.get("lr_extension")
    ):
        raise ValueError("Unsupported historical joint-training family")
    for name in TRAINING_CONTROLS:
        if name not in old or old[name] != experiment["training"].get(name):
            raise ValueError(f"Historical training control changed: {name}")
    data = experiment["datasets"]
    if (
        old.get("heldout_source") != registry.heldout_source
        or old.get("heldout_source") != data.get("heldout_source")
        or set(old.get("excluded_sources", [])) != set(data.get("excluded_sources", []))
        or any(value != 1 for value in data.get("weights", {}).values())
    ):
        raise ValueError("Historical source selection, weights or holdout changed")
    if old.get("rgb_size") != data.get("rgb_size") or (
        data.get("clip_frames"),
        data.get("frame_stride"),
    ) != (8, 2):
        raise ValueError("Historical visual clip/preprocessing controls changed")
    if old.get("video_chunk_size", 2) != experiment["model"]["backbone"].get("video_chunk_size", 2):
        raise ValueError("Historical backbone chunk size changed")
    old_mirror = old.get("mirror_augmentation", {})
    mirror = data.get("mirror", {})
    if (
        old_mirror.get("enabled") != mirror.get("enabled")
        or old_mirror.get("probability") != mirror.get("probability")
        or old_mirror.get("sources") != mirror.get("sources")
    ):
        raise ValueError("Historical mirroring schedule changed")


def _check_file(paths, value, expected):
    path = paths.resolve(value, historical=True)
    if file_hash(path) != expected:
        raise ValueError(f"Historical artifact identity changed: {value}")
    return path


def _verify_producers(value, paths):
    """Verify nested recorded implementation maps without importing old code."""
    verified = {}
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "implementation":
                verified.update(verify_implementation(child, paths))
            else:
                verified.update(_verify_producers(child, paths))
    elif isinstance(value, list):
        for child in value:
            verified.update(_verify_producers(child, paths))
    return verified


def _normalized_teacher(info, paths):
    result = copy.deepcopy(info)
    for key in ("checkpoint", "original_checkpoint"):
        if key in result:
            result[key] = str(paths.resolve(result[key]))
    return result


def _verify_artifacts(state, experiment, provenance, registry):
    paths, old, recorded = registry.paths, state["config"], state["provenance"]
    entries = experiment["datasets"]["registry_paths"]
    if state["teachers"] != recorded.get("teachers"):
        raise ValueError("Historical teacher registry disagrees with checkpoint provenance")
    selected = set(experiment["datasets"]["sources"])
    old_heads = set(state["model_config"]["sensor_code_dims"]) | set(
        state["model_config"]["numeric_shapes"]
    )
    if selected != old_heads or registry.heldout_source in old_heads:
        raise ValueError("Historical source head membership changed")
    teachers = state["teachers"]["sources"]
    current_teachers = {name: spec.teacher for name, spec in registry.specs.items() if spec.teacher}
    if set(teachers) != set(current_teachers):
        raise ValueError("Historical teacher/source membership changed")
    for name, info in teachers.items():
        if _normalized_teacher(info, paths) != _normalized_teacher(current_teachers[name], paths):
            raise ValueError(f"Historical teacher metadata changed: {name}")
        _check_file(paths, info["checkpoint"], info["sha256"])
    expected_paths = {
        "manifest": "manifest",
        "normalization": "storage_normalization",
        "marker_targets": "marker_targets",
        "teachers": "teachers",
    }
    for old_key, new_key in expected_paths.items():
        if paths.resolve(old[old_key]) != paths.resolve(entries[new_key]):
            raise ValueError(f"Historical artifact selection changed: {old_key}")
    _check_file(paths, entries["manifest"], recorded["data"]["manifest_sha256"])
    normalization_path = _check_file(paths, old["normalization"], recorded["normalization_sha256"])
    if read_json(normalization_path)["data"] != recorded["base_data"]:
        raise ValueError("Historical base-data normalization identity changed")
    _check_file(paths, entries["storage_manifest"], recorded["base_data"]["manifest_sha256"])
    marker = recorded["marker_targets"]
    marker_index = _check_file(
        paths, Path(old["marker_targets"]) / "index.json", marker["index_sha256"]
    )
    if read_json(marker_index)["provenance"] != marker["provenance"]:
        raise ValueError("Historical marker extraction identity changed")
    _check_file(paths, old["marker_teachers"], recorded["marker_teachers_sha256"])
    _check_file(paths, Path(old["code_cache"]) / "registry.json", recorded["code_registry_sha256"])
    for name, signature in recorded["code_caches"].items():
        index = read_json(paths.resolve(Path(old["code_cache"]) / name / "index.json"))
        if index["signature"] != signature:
            raise ValueError(f"Historical base code receipt changed: {name}")
    expansion = recorded["expansion"]
    _check_file(paths, Path(old["data_expansion"]) / "teachers.json", expansion["teachers_sha256"])
    ready = read_json(paths.resolve(entries["expansion_ready"]))
    if ready["manifest_sha256"] != expansion["manifest_sha256"]:
        raise ValueError("Historical expansion data changed")
    mirror_sources = set(experiment["datasets"].get("mirror", {}).get("sources", []))
    for name, digest in expansion["code_index_sha256"].items():
        index_path = ready.get("code_index_paths", {}).get(
            name, str(Path(old["data_expansion"]) / "codes" / name / "index.json")
        )
        _check_file(paths, index_path, digest)
        if name not in mirror_sources and registry[name].cache.index_hash != digest:
            raise ValueError(f"Historical active expansion code changed: {name}")
    acquisitions = recorded["acquisitions"]
    acquisition_path = _check_file(
        paths, entries["acquisition_ready"], acquisitions["ready_sha256"]
    )
    if read_json(acquisition_path)["sources"] != acquisitions["sources"]:
        raise ValueError("Historical acquisition preparation changed")
    for name, entry in acquisitions["sources"].items():
        if registry[name].cache.index_hash != entry["index_sha256"]:
            raise ValueError(f"Historical active acquisition code changed: {name}")
    mirror = experiment["datasets"]["mirror"]
    policy = read_json(paths.resolve(mirror["policy"]))
    mirror_ready = read_json(paths.resolve(Path(mirror["preparation"]) / "ready.json"))
    old_mirror = recorded["mirror_augmentation"]
    if old_mirror.get("latent_equality_constraint") is not False:
        raise ValueError("Historical mirror latent constraint changed")
    for binding in old_mirror["bindings"]:
        if binding["policy_sha256"] != json_hash(policy):
            raise ValueError("Historical mirror policy changed")
        for name, entry in binding["sources"].items():
            if entry != mirror_ready["sources"][name]:
                raise ValueError(f"Historical mirror preparation changed: {name}")
            if (
                not entry.get("numeric")
                and registry[name].cache.index_hash != entry["code_index_sha256"]
            ):
                raise ValueError(f"Historical mirrored code changed: {name}")
            if name in binding.get("code_verification_sha256", {}):
                _check_file(
                    paths,
                    Path(entry["code_index"]).with_name("verification.json"),
                    binding["code_verification_sha256"][name],
                )
    before, after = recorded["backbone"], provenance["backbone"]
    for key, value in before.items():
        if key in ("checkpoint", "repo"):
            if paths.resolve(value) != paths.resolve(after[key]):
                raise ValueError(f"Historical backbone artifact changed: {key}")
        elif key == "feature_shape":
            if value != [4, 3, 576, 768]:
                raise ValueError("Historical backbone token layout changed")
        elif after.get(key) != value:
            raise ValueError(f"Historical backbone preprocessing changed: {key}")
    archived = _verify_producers(recorded, paths)
    if not archived:
        raise ValueError("Historical checkpoint lacks producer identities")
    registry.require_imports()
    return archived


def _check_optimizer_state(state, model, optimizer, batches_per_epoch, training):
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    old_names = [name for name in state["model"] if name in dict(named)]
    if old_names != [name for name, _ in named]:
        raise ValueError("Historical parameter order changed")
    groups = state["optimizer"]["param_groups"]
    if len(groups) != 1 or groups[0]["params"] != list(range(len(named))):
        raise ValueError("Historical optimizer parameter ordering is unsupported")
    if len(optimizer.param_groups) != 1 or [id(p) for p in optimizer.param_groups[0]["params"]] != [
        id(p) for _, p in named
    ]:
        raise ValueError("Current optimizer parameter order differs from model")
    slots = state["optimizer"]["state"]
    if set(slots) != set(groups[0]["params"]):
        raise ValueError("Historical optimizer moments are incomplete")
    for index, (name, parameter) in enumerate(named):
        slot = slots[index]
        if not {"step", "exp_avg", "exp_avg_sq"} <= set(slot):
            raise ValueError(f"Historical optimizer moments missing: {name}")
        if any(slot[key].shape != parameter.shape for key in ("exp_avg", "exp_avg_sq")):
            raise ValueError(f"Historical optimizer slot shape changed: {name}")
    completed = state["epoch"] + 1
    updates = completed * math.ceil(batches_per_epoch / training["accumulation"])
    if state["sam"]["steps"] != updates or state["scheduler"]["last_epoch"] != updates:
        raise ValueError("Historical sampled epoch/update schedule changed")
    if state["sam"]["rho"] != training["sam_rho"]:
        raise ValueError("Historical SAM radius changed")
    return [name for name, _ in named]


def import_legacy_checkpoint(
    path,
    *,
    model_config,
    experiment,
    provenance,
    registry,
    model,
    optimizer,
    scheduler,
    sam,
    generator,
    batches_per_epoch,
):
    """Verify and restore a complete historical state without writing any files.

    All passed model/optimizer/scheduler/SAM objects and global RNG generators
    are restored only after metadata validation. Return a schema-2 state plus
    JSON-serializable receipt retaining the original configuration, provenance,
    teacher registry and source checksum. This is an explicit import into a new
    run; ordinary schema-2 checkpoint resume remains strict and unchanged.
    """
    path = Path(path)
    source_hash = file_hash(path)
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not REQUIRED_STATE <= set(state) or state.get("schema_version") is not None:
        raise ValueError("Historical checkpoint is missing complete resumable state")
    if state.get("inference_only") or state["model_type"] != MODEL_TYPE:
        raise ValueError("Unsupported historical checkpoint model type")
    if type(state["epoch"]) is not int or state["epoch"] < 0:
        raise ValueError("Historical import requires a completed epoch")
    if state["epoch"] + 1 >= experiment["training"]["epochs"]:
        raise ValueError("Historical checkpoint has no remaining configured epochs")
    _same_model_config(state["model_config"], model_config)
    _check_training_controls(state["config"], experiment, registry)
    preflight_path = path.parent / "preflight.json"
    preflight = read_json(preflight_path)
    if (
        not preflight.get("passed")
        or preflight.get("config") != state["config"]
        or preflight.get("provenance") != state["provenance"]
    ):
        raise ValueError("Historical production preflight does not match checkpoint")
    if set(state["rng"]) != {"cpu", "cuda"} or not torch.is_tensor(state["loader_rng"]):
        raise ValueError("Historical random state is incomplete")
    names = _check_optimizer_state(
        state, model, optimizer, batches_per_epoch, experiment["training"]
    )
    archived = _verify_artifacts(state, experiment, provenance, registry)
    if file_hash(path) != source_hash:
        raise ValueError("Historical checkpoint changed during verification")
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    sam.load_state_dict(state["sam"])
    expected_lr = experiment["training"]["learning_rate"] * scheduler.lr_lambdas[0](
        scheduler.last_epoch
    )
    if optimizer.param_groups[0]["lr"] != expected_lr:
        raise ValueError("Historical optimizer learning-rate schedule changed")
    receipt = {
        "schema": "historical-joint-checkpoint-import-v1",
        "source_checkpoint": {"path": str(path.resolve()), "sha256": source_hash},
        "source_preflight": {
            "path": str(preflight_path.resolve()),
            "sha256": file_hash(preflight_path),
        },
        "completed_epochs": state["epoch"] + 1,
        "optimizer_updates": state["sam"]["steps"],
        "parameter_order": names,
        "original_metadata": {
            key: copy.deepcopy(state[key])
            for key in ("model_type", "model_config", "config", "provenance", "teachers")
        },
        "archived_producers": archived,
        "optimizer_reinitialized": False,
        "random_state_reinitialized": False,
        "old_checkpoint_rewritten": False,
        "old_receipts_rewritten": False,
    }
    imported = {key: state[key] for key in REQUIRED_STATE - {"teachers"}}
    imported.update(
        schema_version=2,
        model_config=copy.deepcopy(model_config),
        config=copy.deepcopy(experiment),
        provenance=copy.deepcopy(provenance),
        legacy_import=copy.deepcopy(receipt),
    )
    restore_rng(state["rng"])
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    generator.set_state(state["loader_rng"])
    return imported, receipt
