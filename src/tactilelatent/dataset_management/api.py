"""Implement explicit discovery, verification, preparation and import actions.

Inputs are experiment/source selections and requested management options. Outputs
are JSON-compatible reports; only explicit preparation/import/download actions
write artifacts. Constructing or inspecting a dataset never starts a download.
"""

from pathlib import Path

from .artifacts import atomic_json, file_hash, read_json
from .registry import build_sources


def _data_config(config):
    if isinstance(config, (str, Path)):
        path = Path(config)
        config = read_json(path)
        if isinstance(config.get("datasets"), str):
            nested = Path(config["datasets"])
            nested = nested if nested.is_absolute() else path.parent / nested
            config = {**config, "datasets": read_json(nested)}
    return config.get("datasets", config)


def inventory(registry, source=None, clips=False):
    """Report native layouts, split counts and teacher identities from a registry.

    ``clips=True`` also resolves supported clip ends and can read code payloads;
    it does not decode RGB or rebuild artifacts.
    """
    names = [source] if source else list(registry.data_config["sources"])
    if any(name not in registry.sources for name in names):
        raise ValueError("Unknown source requested")
    result = {}
    for name in names:
        adapter = registry[name]
        spec = adapter.spec
        entry = {
            "source": name,
            "target_kind": spec.target_kind,
            "native_shape": list(spec.native_shape),
            "supervision": spec.supervision_kind,
            "code_dim": spec.code_dim,
            "mirror": bool(spec.mirror),
            "splits": {},
            "teacher_sha256": spec.teacher.get("sha256"),
        }
        for row in adapter.rows:
            group = entry["splits"].setdefault(
                row["split"], {"episodes": 0, "frames": 0, "clips": 0}
            )
            group["episodes"] += 1
            group["frames"] += row["frames"]
            if clips:
                group["clips"] += len(adapter.eligible_ends(row))
        result[name] = entry
    return {
        "schema": "uniform-source-inventory-v1",
        "sources": result,
        "heldout_source": registry.heldout_source,
        "excluded_sources": sorted(registry.excluded_sources),
        "artifacts": registry.provenance["artifacts"],
    }


def import_caches(registry, output, source=None):
    """Import receipts, never arrays; all historical receipts remain immutable."""
    folder = Path(output)
    consumers = registry.consumer_files()
    receipts = {}
    names = [source] if source else registry.data_config["sources"]
    for name in names:
        adapter = registry[name]
        if name == registry.heldout_source:
            raise ValueError("Held-out data cannot enter teacher-code imports")
        if adapter.cache is None:
            continue
        receipt = adapter.cache.write_import(folder / (name + ".json"), consumers)
        receipts[name] = {
            "path": str(folder / (name + ".json")),
            "sha256": file_hash(folder / (name + ".json")),
            "old_receipt_sha256": receipt["old_receipt"]["sha256"],
        }
    result = {
        "complete": True,
        "sources": receipts,
        "arrays_regenerated": False,
        "old_receipts_modified": False,
        "heldout_source": registry.heldout_source,
        "consumer_contract": "ClipSample/ClipBatch-v2",
    }
    atomic_json(folder / "index.json", result)
    return result


def verify(registry, source=None, sample=False, full=False):
    """Check requested artifact levels and return an explicit-scope report.

    ``sample`` reads representative native targets and RGB clips per split;
    ``full`` verifies all published code payloads. Neither certifies the full
    dataset's semantic synchronization or regenerates missing data.
    """
    report = inventory(registry, source, clips=True)
    if sample:
        names = [source] if source else registry.data_config["sources"]
        for name in names:
            adapter = registry[name]
            for split in ("train", "val"):
                candidates = [
                    row
                    for row in adapter.rows
                    if row["split"] == split and len(adapter.eligible_ends(row))
                ]
                if not candidates:
                    raise ValueError("Source lacks a supported split")
                row = candidates[0]
                # Decoding validates native shapes/masks even though the sample
                # itself is not retained in this summary report.
                adapter.read_clip(row, int(adapter.eligible_ends(row)[0]))
                report["sources"][name]["splits"][split]["sample_verified"] = True
    report["verification_levels"] = {
        "source_receipts": "checked",
        "code_payloads": "requested by full flag" if full else "lazy on access",
        "native_targets": "representative train/val samples" if sample else "not decoded",
        "rgb": "representative train/val clips" if sample else "not decoded",
    }
    if full:
        names = [source] if source else registry.data_config["sources"]
        for name in names:
            adapter = registry[name]
            report["sources"][name]["code_payload_verification"] = (
                adapter.cache.verify_all(adapter.rows)
                if adapter.cache is not None
                else {"applicable": False, "reason": "direct supervision; no cached code payload"}
            )
        report["verification_levels"]["code_payloads"] = (
            "all published frozen-code payloads and native episode/frame identities"
        )
    report["passed"] = True
    report["scope"] = (
        "Requested receipt/code checks; not an end-to-end native data-quality or synchronization certification"
    )
    return report


def prepare(registry, source=None, output=None, **options):
    """Prepare the common consumer contract for already processed native sources.

    Preparation checks data/split/cache readiness and may explicitly publish new
    consumer sidecars. It does not redo marker extraction or train a teacher.
    """
    result = verify(
        registry,
        source,
        sample=bool(options.get("sample", False)),
        full=bool(options.get("full", False)),
    )
    if output is not None:
        result["imports"] = import_caches(registry, output, source)
    return result


def download(registry, source=None, **options):
    """Existing prepared sources need no redownload; missing acquisitions require a declared provider."""
    from .providers import download_plan

    if source is None:
        raise ValueError("Downloading requires an explicit source identifier")
    plans = registry.data_config.get("acquisition_plans", {})
    if source in plans:
        return download_plan(plans[source], registry.paths.root, **options)
    adapter = registry[source]
    missing = []
    for row in adapter.rows:
        if "targets" in row:
            candidate = registry.paths.resolve(row["targets"]["path"])
            if not candidate.exists():
                missing.append(str(candidate))
    if missing:
        raise FileNotFoundError(
            "Missing prepared inputs; declare acquisition_plans for explicit download: "
            + missing[0]
        )
    return {
        "source": source,
        "already_prepared": True,
        "episodes": len(adapter.rows),
        "network_requests": 0,
        "reason": "Native manifests/codes are already published; no replacement download requested.",
    }


def manage(command, *, config, project_root, source=None, output=None, **options):
    """Dispatch one declared data-management action with an explicit project root.

    Return a JSON-compatible result. Cache/import actions write versioned consumer
    sidecars, not sensor-code arrays; download actions require pinned acquisition
    plans or complete explicit overrides. Discovery and status remain read-only.
    """
    data = _data_config(config)
    if command == "download" and (
        source in data.get("acquisition_plans", {})
        or any(options.get(k) is not None for k in ("url", "sha256", "destination", "revision"))
    ):
        if not source:
            raise ValueError("Downloading requires an explicit source identifier")
        from .providers import download_plan

        return download_plan(
            data.get("acquisition_plans", {}).get(source),
            project_root,
            allow_network=options.pop("allow_network", True),
            **options,
        )
    registry = build_sources(data, project_root)
    if source is not None and source not in registry.sources:
        raise ValueError("Unknown source")
    if command in ("discover", "status"):
        return inventory(registry, source)
    if command == "verify":
        return verify(
            registry,
            source,
            sample=bool(options.get("sample", False)),
            full=bool(options.get("full", False)),
        )
    if command == "prepare":
        return prepare(registry, source, output, **options)
    if command in ("cache", "import"):
        destination = output if output is not None else registry.import_root()
        return import_caches(registry, destination, source)
    if command == "download":
        return download(registry, source, **options)
    raise ValueError(f"Unknown data management command: {command}")
