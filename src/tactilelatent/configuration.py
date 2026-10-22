"""Read explicit experiment components and record their content identities.

Inputs are an experiment mapping/JSON path and an explicit project root. Outputs
are composed settings with component hashes; loading never starts research jobs.
"""

import hashlib
import json
from pathlib import Path

COMPONENTS = ("model", "datasets", "training", "transfer")


def load_experiment(config, *, project_root):
    """Load model, datasets, training and transfer settings with SHA256 receipts.

    Component paths are relative to the experiment file; inline mappings are
    hashed canonically. Artifact paths inside components remain unresolved and
    are interpreted against ``project_root`` by their consuming implementation.
    """
    project_root = Path(project_root).resolve()
    if isinstance(config, dict):
        document, parent, source = config, project_root, None
    else:
        source = Path(config)
        if not source.is_absolute():
            source = project_root / source
        source = source.resolve()
        document, parent = json.loads(source.read_text()), source.parent
    result = {}
    provenance = {
        "schema": "research-core-v2",
        "components": {},
        "experiment": str(source) if source else None,
    }
    for name in COMPONENTS:
        value = document.get(name)
        if isinstance(value, str):
            path = Path(value)
            if not path.is_absolute():
                path = parent / path
            raw = path.read_bytes()
            value = json.loads(raw)
            provenance["components"][name] = {
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        elif isinstance(value, dict):
            raw = json.dumps(value, sort_keys=True).encode()
            provenance["components"][name] = {
                "path": None,
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        else:
            raise ValueError(f"Experiment requires an explicit {name} object or JSON path")
        result[name] = value
    result["project_root"] = str(project_root)
    result["provenance"] = provenance
    return result


def resolve_path(path, project_root):
    """Resolve a project-relative path while retaining explicit absolute paths."""
    path = Path(path)
    return path if path.is_absolute() else Path(project_root) / path


def metadata_root(project_root):
    """Use the checkout archive when staged, or the explicit project when installed."""
    checkout = Path(__file__).resolve().parents[2]
    return checkout if (checkout / "legacy/v1").is_dir() else Path(project_root).resolve()
