"""Build one source registry directly from immutable prepared artifacts.

There are no dataset wrappers and no historical Python imports. Release-specific
I/O lives in adapters; the dataset, sampler, trainer and viewer use contracts.
Inputs are selected-source configuration and verified manifests, teachers and
caches. Output adapters retain native rows and expose one uniform sample API.
"""

import copy
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from ..contracts import ClipSample, SourceSpec, Supervision
from ..datasets.codecs import SensorCodec
from ..datasets.sources.definitions import source_definition
from ..datasets.sources.mirroring import require_spec
from ..datasets.sources.native import (
    marker_producer_provenance,
    read_target,
    read_visual,
    sensor_indices,
)
from .artifacts import ArtifactPaths, file_hash, json_hash, read_json, verify_implementation
from .cache import SensorCodeStore, verified_path
from .mirroring import augment_samples


class SourceAdapter:
    """Compose visual/target readers, a sensor codec and an optional code store.

    Release-specific chronology and sensor geometry are reader responsibilities.
    This adapter joins them into ``ClipSample`` without teaching the model about
    a dataset name or altering saved native-row identities.
    """

    def __init__(self, spec, paths, rows, storage, marker_root, cache=None, rgb_size=384):
        self.spec, self.paths, self.rows = spec, paths, rows
        self.storage, self.marker_root, self.cache = storage, marker_root, cache
        self.rgb_size = rgb_size
        self.codec = SensorCodec(spec)
        if spec.target_kind == "marker_displacement":
            provenance = marker_producer_provenance(rows[0], paths, marker_root)
            spec.metadata["raw_marker_producer"] = provenance
            if cache is not None:
                cache.raw_extraction_provenance = provenance

    def read_clip(self, row, end, include_targets=True, *, pad_history=False):
        """Read eight history images and supervision for their final frame.

        Return a sample with one target frame, stored sensor IDs and native-row
        provenance. Cached codes use their frozen training statistics; direct
        targets use the source codec. ``include_targets`` is retained in the
        adapter interface; this implementation always supplies the native target.
        """
        current = np.array([end], np.int64)
        visual = read_visual(
            row, end, self.paths, self.storage, self.rgb_size, pad_history=pad_history
        )
        target = read_target(row, current, self.paths, self.marker_root, self.spec.target_kind)
        if self.spec.supervision_kind == "code":
            code, valid = self.cache.read(row, current)
            supervision = Supervision(
                code,
                valid,
                "code",
                "original",
                {"index": str(self.cache.path), "sha256": self.cache.index_hash},
            )
        else:
            supervision = Supervision(
                self.codec.encode_supervision(target.values), target.valid, "direct"
            )
        return ClipSample(
            visual,
            target,
            supervision,
            self.spec.source,
            row["episode_id"],
            row["split"],
            torch.from_numpy(current),
            torch.from_numpy(sensor_indices(row, current)),
            {"native_row": row, "native_row_sha256": json_hash(row), "mirror_applied": False},
        )

    def read_frame(self, row, frame, include_targets=True):
        """Viewer preview only; left-padding does not change training windows."""
        return self.read_clip(row, frame, include_targets, pad_history=True)

    def eligible_ends(self, row):
        """Return supported current-frame IDs in the canonical clip schedule."""
        if self.cache is not None:
            explicit = self.cache.ends(row)
            if explicit is not None:
                return np.asarray(explicit, np.int64)
        ends = np.arange(14, row["frames"], 15, dtype=np.int64)
        if self.cache is not None and len(ends):
            ends = ends[self.cache.valid_frames(row, ends)]
        return ends


class SourceRegistry:
    """Own ordered adapters, source specifications and verified artifact identity.

    Source discovery order, model-head order and held-out membership are explicit.
    Training, evaluation and visualization access the same adapters through this
    registry; construction never downloads data or rebuilds cached arrays.
    """

    def __init__(self, adapters, episode_rows, data_config, paths, model_artifact, provenance):
        self.sources = OrderedDict((adapter.spec.source, adapter) for adapter in adapters)
        self.specs = OrderedDict((name, adapter.spec) for name, adapter in self.sources.items())
        self.episode_rows = episode_rows
        self.data_config = data_config
        self.paths = paths
        self.recorded_model_config = model_artifact
        self.provenance = provenance
        self.heldout_source = data_config.get("heldout_source", "ftp_motiontrans")
        self.excluded_sources = set(data_config.get("excluded_sources", []))
        mirror = data_config.get("mirror", data_config.get("mirror_augmentation", {}))
        self.mirror_probability = (
            mirror.get("probability", mirror.get("p", 0.5)) if mirror.get("enabled", True) else 0.0
        )

    def __getitem__(self, source):
        return self.sources[source]

    def __iter__(self):
        return iter(self.sources)

    def __len__(self):
        return len(self.sources)

    def model_config(self, model_cfg=None):
        """Specs own head semantics; recorded maps supply only stable ordering."""
        allowed = (
            "input_dim",
            "latent_dim",
            "adapter_hidden_dims",
            "latent_mode",
            "width",
            "heads",
            "layers",
            "feedforward",
            "dropout",
            "grid_size",
            "prediction_mode",
            "initialization",
        )
        result = {
            key: copy.deepcopy(value)
            for key, value in self.recorded_model_config.items()
            if key in allowed
        }
        override = dict(model_cfg or {})
        override.update(override.pop("attention", {}))
        result.update(
            {key: copy.deepcopy(value) for key, value in override.items() if key in allowed}
        )
        requested = list(self.data_config["sources"])
        code_order = [
            name
            for name in self.recorded_model_config.get("sensor_code_dims", {})
            if name in requested
        ]
        numeric_order = [
            name
            for name in self.recorded_model_config.get("numeric_shapes", {})
            if name in requested
        ]
        code_order += [
            name
            for name in requested
            if self.specs[name].supervision_kind == "code" and name not in code_order
        ]
        numeric_order += [
            name
            for name in requested
            if self.specs[name].supervision_kind == "direct" and name not in numeric_order
        ]
        code_dims = {name: self.specs[name].code_dim for name in code_order}
        numeric_shapes = {name: list(self.specs[name].native_shape) for name in numeric_order}
        statistics = {name: copy.deepcopy(self.specs[name].statistics) for name in code_order}
        direct_numeric = {
            name: copy.deepcopy(self.specs[name].numeric_transform)
            for name in numeric_order
            if self.specs[name].numeric_transform
        }
        # Recorded statistics have historical insertion order too; preserve that
        # order for exact initialization/config parity, then append new sources.
        stats_order = [
            name for name in self.recorded_model_config.get("statistics", {}) if name in statistics
        ]
        stats_order += [name for name in code_order if name not in stats_order]
        result.update(
            sensor_code_dims=code_dims,
            numeric_shapes=numeric_shapes,
            statistics={name: statistics[name] for name in stats_order},
            direct_numeric=direct_numeric,
        )
        for key in ("sensor_code_dims", "numeric_shapes"):
            if key in override and override[key] != result[key]:
                raise ValueError("Explicit compatibility head map differs from source definitions")
        expected = set(code_dims) | set(numeric_shapes)
        if (
            expected != set(requested)
            or self.heldout_source in expected
            or self.excluded_sources & expected
        ):
            raise ValueError("Model head membership violates source/holdout contract")
        return result

    def augment(self, batch, seed, epoch, step, probability=None, flags=None):
        """Apply the verified native mirror policy and matching cached codes."""
        return augment_samples(batch, self, seed, epoch, step, probability, flags)

    def consumer_files(self):
        """List maintained source consumers pinned by cache-import sidecars."""
        root = Path(__file__).resolve().parents[1]
        files = [root / "contracts.py"]
        files += sorted((root / "datasets").rglob("*.py"))
        files += sorted((root / "dataset_management").glob("*.py"))
        return files

    def import_root(self):
        """Resolve the explicitly configured versioned cache-import directory."""
        value = self.data_config.get("cache_imports")
        if not value:
            raise ValueError(
                "Configure cache_imports and explicitly import old cache receipts before production training"
            )
        return self.paths.resolve(value)

    def require_imports(self):
        """Reject caches without sidecars matching original artifacts/current code."""
        if not any(
            source != self.heldout_source and adapter.cache is not None
            for source, adapter in self.sources.items()
        ):
            return True
        folder = self.import_root()
        consumers = self.consumer_files()
        for source, adapter in self.sources.items():
            if source != self.heldout_source and adapter.cache is not None:
                adapter.cache.require_import(folder / (source + ".json"), consumers)
        return True


def _normalize_config(config):
    """Accept the maintained data section; reading a flat historical config is explicit compatibility only."""
    if "datasets" in config and isinstance(config["datasets"], dict):
        return config["datasets"]
    if "registry_paths" in config:
        return config
    raise ValueError("Expected a datasets configuration with explicit registry_paths and sources")


def build_sources(config, project_root):
    """Build the uniform registry from explicit configuration and frozen receipts.

    Verify manifests, teachers, code statistics and accepted mirror policies
    without rewriting them. Logical source rows retain their saved identities;
    filesystem paths are resolved separately. The held-out source is readable
    for evaluation but cannot acquire a training head.
    """
    cfg = _normalize_config(config)
    selected = list(cfg["sources"])
    heldout = cfg.get("heldout_source", "ftp_motiontrans")
    if (
        len(selected) != len(set(selected))
        or heldout in selected
        or set(cfg.get("excluded_sources", [])) & set(selected)
    ):
        raise ValueError("Invalid source list or held-out leakage")
    from ..configuration import metadata_root

    entries = cfg["registry_paths"]
    meta = metadata_root(project_root)
    archived = entries.get("archived_producers", "legacy/v1/src/tactilelatent")
    archived_path = Path(archived) if Path(archived).is_absolute() else meta / archived
    paths = ArtifactPaths(project_root, meta, archived_path)
    manifest_path = paths.resolve(entries["manifest"])
    manifest = read_json(manifest_path)
    if (
        manifest.get("schema_version") != 2
        or manifest.get("frame_contract") != "native-chronological-v1"
    ):
        raise ValueError("A verified native chronological schema-2 manifest is required")
    raw_rows = manifest["episodes"]
    # Raw rows retain exactly their saved JSON identities. Resolved paths live only in readers.
    episode_rows = [
        (row["source"], row) for row in raw_rows if row["source"] in set(selected) | {heldout}
    ]
    model_path = paths.resolve(entries["model_config"])
    model_cfg = read_json(model_path)
    teachers_path = paths.resolve(entries["teachers"])
    teachers = read_json(teachers_path)["sources"]
    storage_path = (
        paths.resolve(entries.get("scene_receipts", "processing/storage_v1")) / "registry.json"
    )
    storage = read_json(storage_path) if storage_path.exists() else {}
    normalization = read_json(paths.resolve(entries["storage_normalization"]))
    receipts = {
        str(manifest_path): file_hash(manifest_path),
        str(model_path): file_hash(model_path),
        str(teachers_path): file_hash(teachers_path),
    }
    if storage_path.exists():
        receipts[str(storage_path)] = file_hash(storage_path)
    mirror_cfg = cfg.get("mirror", cfg.get("mirror_augmentation", {}))
    mirror_entries = {}
    policy = {}
    if mirror_cfg.get("enabled", True) and mirror_cfg.get("sources"):
        policy_path = paths.resolve(mirror_cfg["policy"])
        policy = read_json(policy_path)
        ready_path = paths.resolve(mirror_cfg["preparation"]) / "ready.json"
        ready = read_json(ready_path)
        if not ready.get("complete_for_verified_sources", ready.get("complete")) or ready[
            "policy_sha256"
        ] != json_hash(policy):
            raise ValueError("Mirror preparation incomplete or policy changed")
        verify_implementation(ready.get("implementation", {}), paths)
        audit = read_json(ready_path.with_name("policy_audit.json"))
        if audit["policy_sha256"] != ready["policy_sha256"]:
            raise ValueError("Mirror geometry evidence belongs to another policy")
        for source in mirror_cfg["sources"]:
            require_spec(policy, source)
            evidence = audit["sources"][source]
            if not evidence.get("verified"):
                raise ValueError("Unverified mirror geometry")
            verify_implementation(evidence["evidence_sha256"], paths)
            entry = ready["sources"][source]
            if not entry.get("ready"):
                raise ValueError(f"Mirror source not ready: {source}")
            mirror_entries[source] = entry
        receipts[str(policy_path)] = file_hash(policy_path)
        receipts[str(ready_path)] = file_hash(ready_path)
    expansion = read_json(paths.resolve(entries["expansion_ready"]))
    if not expansion.get("ready") or expansion["manifest_sha256"] != file_hash(manifest_path):
        raise ValueError("Expanded publication does not match source manifest")
    acquisition_path = paths.resolve(entries["acquisition_ready"])
    acquisition = read_json(acquisition_path)
    if not acquisition.get("ready"):
        raise ValueError("Native acquisition preparation is incomplete")
    receipts[str(acquisition_path)] = file_hash(acquisition_path)
    for source, entry in acquisition["sources"].items():
        if source not in selected:
            continue
        definition = source_definition(source)
        native_path = (
            paths.resolve(entries.get("acquisition_root", "processing/data_queue_v2"))
            / definition.acquisition
            / "manifest.json"
        )
        native = read_json(native_path)
        if native.get("schema") != "acquisition-native-v2":
            raise ValueError("Invalid native acquisition manifest")
        rows = [
            row
            for row in native["episodes"]
            if all(row.get(key) == value for key, value in definition.row_filters.items())
        ]
        if not rows or any(row["targets"]["shape"][1:] != [definition.input_dim] for row in rows):
            raise ValueError("Native source/channel layout changed")
        channels = rows[0]["targets"]["channel_order"]
        if any(row["targets"]["channel_order"] != channels for row in rows):
            raise ValueError("Channel order differs inside native source")
        episode_rows.extend((source, row) for row in rows)
        receipts[str(native_path)] = file_hash(native_path)
    # New prepared sources publish their own immutable manifests. They do not
    # rewrite an earlier expansion receipt or require new manager branches.
    for source, entry in cfg.get("prepared_sources", {}).items():
        if source not in selected or not entry.get("manifest"):
            continue
        if any(name == source for name, _ in episode_rows):
            raise ValueError("A source cannot be owned by two native manifests")
        extra_path = paths.check(entry["manifest"], entry["manifest_sha256"])
        extra = read_json(extra_path)
        if (
            extra.get("schema_version") != 2
            or extra.get("frame_contract") != "native-chronological-v1"
        ):
            raise ValueError("Additional source needs a native chronological manifest")
        additional = extra["episodes"]
        if not additional or any(row.get("source") != source for row in additional):
            raise ValueError("Additional manifest must contain exactly its registered source")
        episode_rows.extend((source, row) for row in additional)
        receipts[str(extra_path)] = file_hash(extra_path)
    adapters = []
    # Registry order follows actual episode/source discovery; model head order is separate above.
    order = list(dict.fromkeys(source for source, _ in episode_rows))
    for source in order:
        rows = [row for name, row in episode_rows if name == source]
        prepared = cfg.get("prepared_sources", {}).get(source, {})
        teacher = copy.deepcopy(prepared.get("teacher", teachers.get(source, {})))
        if teacher:
            verified_path(paths.resolve(teacher["checkpoint"]), teacher["sha256"])
            teacher["checkpoint"] = str(paths.resolve(teacher["checkpoint"]))
        definition = source_definition(source)
        kind = definition.target_kind
        statistics = copy.deepcopy(
            prepared.get("statistics", model_cfg.get("statistics", {}).get(source, {}))
        )
        numeric = copy.deepcopy(
            prepared.get("numeric_transform", model_cfg.get("direct_numeric", {}).get(source, {}))
        )
        shape = tuple(
            model_cfg.get("numeric_shapes", {}).get(
                source, teacher.get("target_shape", rows[0].get("target_shape", ()))
            )
        )
        if definition.native_shape is not None:
            shape = definition.native_shape
        mirror = (
            {**copy.deepcopy(policy["sources"][source]), "enabled": True}
            if source in mirror_entries
            else {}
        )
        spec = SourceSpec(
            source,
            definition.adapter,
            kind,
            shape,
            "code" if teacher else "direct",
            teacher.get("code_dim"),
            teacher,
            statistics,
            numeric,
            mirror,
            copy.deepcopy(definition.metric_groups),
            {
                "heldout": source == heldout,
                "native_metric_reduction": definition.native_metric_reduction,
                "channel_order": rows[0].get("targets", {}).get("channel_order"),
                "calibration": "released coordinates; no new episode-specific zero",
                "sensor_geometry_verified": mirror.get("hardware_geometry_verified"),
                **definition.metadata_for_row(rows[0]),
            },
        )
        cache = None
        if prepared.get("code_index"):
            cache = SensorCodeStore(
                prepared["code_index"],
                paths,
                teacher or None,
                statistics=statistics or None,
                expected_index_hash=prepared.get("code_index_sha256"),
            )
        elif source in mirror_entries and not mirror_entries[source].get("numeric"):
            entry = mirror_entries[source]
            artifact_policy = entry.get("artifact_policy_sha256", json_hash(policy))
            if entry.get("artifact_policy_path"):
                origin_path = paths.resolve(entry["artifact_policy_path"], historical=True)
                old = read_json(origin_path)
                if (
                    json_hash(old) != artifact_policy
                    or old["sources"][source] != policy["sources"][source]
                ):
                    raise ValueError("Inherited mirror mapping changed")
            quality = read_json(paths.resolve(entry["teacher_quality"]))
            if not quality.get("accepted") or quality["teacher"]["sha256"] != teacher["sha256"]:
                raise ValueError("Mirror decoder has no accepted matching quality receipt")
            verification = read_json(
                paths.resolve(entry["code_index"]).with_name("verification.json")
            )
            if (
                not verification.get("passed")
                or verification["index_sha256"] != entry["code_index_sha256"]
            ):
                raise ValueError("Mirror cache has no matching fresh-encoding verification")
            cache = SensorCodeStore(
                entry["code_index"],
                paths,
                teacher,
                artifact_policy,
                statistics,
                entry["code_index_sha256"],
            )
        elif source in acquisition["sources"]:
            entry = acquisition["sources"][source]
            cache = SensorCodeStore(
                entry["index"],
                paths,
                teacher or None,
                statistics=statistics or None,
                expected_index_hash=entry["index_sha256"],
            )
            if (
                cache.index["signature"]["manifest"]["sha256"]
                != receipts[
                    str(
                        paths.resolve(entries.get("acquisition_root", "processing/data_queue_v2"))
                        / definition.acquisition
                        / "manifest.json"
                    )
                ]
            ):
                raise ValueError("Acquisition code index belongs to another manifest")
        elif definition.code_family == "expansion":
            cache = SensorCodeStore(
                expansion["code_index_paths"][source],
                paths,
                teacher,
                statistics=statistics,
                expected_index_hash=expansion["code_index_sha256"][source],
            )
        elif teacher:
            cache = SensorCodeStore(
                paths.resolve(entries["code_cache"]) / source / "index.json",
                paths,
                teacher,
                statistics=statistics,
            )
            if (
                cache.index["signature"]["manifest_sha256"]
                != normalization["data"]["manifest_sha256"]
            ):
                raise ValueError("Historical code cache belongs to another source manifest")
        if cache is not None:
            raw_hash = cache.index.get(
                "rows_sha256", cache.index.get("signature", {}).get("rows_sha256")
            )
            if (
                raw_hash
                and json_hash(sorted(rows, key=lambda row: (row["split"], row["episode_id"])))
                != raw_hash
            ):
                raise ValueError("Prepared code rows differ from the native manifest")
        if cache is not None and spec.supervision_kind == "code" and not spec.statistics:
            spec.statistics = copy.deepcopy(cache.statistics)
        factory = definition.adapter_factory or SourceAdapter
        adapters.append(
            factory(
                spec,
                paths,
                rows,
                storage,
                entries["marker_targets"],
                cache,
                cfg.get("rgb_size", 384),
            )
        )
    if not (set(selected) | {heldout}) <= set(order):
        raise ValueError("Requested training/held-out source absent from prepared manifests")
    provenance = {
        "schema": "uniform-native-sources-v1",
        "artifacts": receipts,
        "sources": {adapter.spec.source: asdict(adapter.spec) for adapter in adapters},
        "frame_contract": manifest["frame_contract"],
        "paths_resolved_after_row_hashing": True,
        "heldout_source": heldout,
        "excluded_sources": cfg.get("excluded_sources", []),
    }
    return SourceRegistry(adapters, episode_rows, cfg, paths, model_cfg, provenance)
