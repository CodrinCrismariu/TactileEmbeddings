"""Opt-in acquisition providers with explicit revisions and a storage reserve.

Downloads are a management action, never a side effect of model/dataset imports.
No provider deletes files or silently changes native sensor preprocessing.
Inputs are pinned acquisition plans and explicit project roots. Outputs describe
downloaded/resumed files; invoking a provider requires an explicit network action.
"""

import shutil
import urllib.request
from pathlib import Path

from .artifacts import file_hash


def resolve_download_plan(
    plan, project_root, *, url=None, sha256=None, destination=None, revision=None
):
    """Validate explicit CLI overrides without network or filesystem mutation."""
    root = Path(project_root).resolve()
    configured = dict(plan or {})
    supplied = [url is not None, sha256 is not None, destination is not None]
    if any(supplied) and not all(supplied):
        raise ValueError("URL, SHA256 and destination must be supplied together")
    if all(supplied):
        target = Path(destination)
        target = (root / target).resolve() if not target.is_absolute() else target.resolve()
        if configured:
            if configured.get("provider") != "https_file":
                raise ValueError("HTTPS overrides conflict with configured provider")
            previous = Path(configured["destination"])
            previous = (
                (root / previous).resolve() if not previous.is_absolute() else previous.resolve()
            )
            if (
                configured.get("url") != url
                or configured.get("sha256") != sha256
                or previous != target
            ):
                raise ValueError(
                    "Overrides contradict the pinned acquisition plan; create a new plan"
                )
        else:
            configured = {
                "provider": "https_file",
                "url": url,
                "sha256": sha256,
                "destination": str(target),
            }
    if not configured:
        raise ValueError("Declare an acquisition plan or provide URL, SHA256 and destination")
    if revision is not None:
        if configured.get("provider") != "huggingface":
            raise ValueError("Revision is only valid for a pinned Hugging Face plan")
        if configured.get("revision") != revision:
            raise ValueError("Revision contradicts the pinned acquisition plan")
    if configured.get("provider") not in ("https_file", "huggingface"):
        raise ValueError("Unsupported acquisition provider")
    return configured


def download_plan(
    plan,
    project_root,
    *,
    allow_network=False,
    url=None,
    sha256=None,
    destination=None,
    revision=None,
    **options,
):
    """Execute a pinned, resumable acquisition inside the selected datasets path.

    Verify HTTPS payload hashes or immutable repository revisions, retain partial
    files on failure, and enforce the configured free-space reserve. Existing
    conflicting data raises instead of being replaced or deleted.
    """
    plan = resolve_download_plan(
        plan, project_root, url=url, sha256=sha256, destination=destination, revision=revision
    )
    root = Path(project_root).resolve()
    destination = Path(plan["destination"])
    destination = (
        (root / destination).resolve() if not destination.is_absolute() else destination.resolve()
    )
    if not destination.is_relative_to(root / "datasets") or destination == root / "datasets":
        raise ValueError("Downloads must stay inside a selected dataset path")
    provider = plan["provider"]
    if provider == "https_file":
        import re

        if not plan["url"].startswith("https://") or not re.fullmatch(
            "[0-9a-f]{64}", plan.get("sha256", "")
        ):
            raise ValueError("Pinned HTTPS URL and lowercase SHA256 required")
        if destination.exists():
            if file_hash(destination) != plan["sha256"]:
                raise ValueError("Destination contains different data; choose a new versioned path")
            return {
                "provider": provider,
                "destination": str(destination),
                "resumed": True,
                "deleted_files": 0,
                "network_requests": 0,
            }
    else:
        pinned = plan.get("revision")
        if not pinned or pinned in ("main", "master"):
            raise ValueError("Pin an immutable dataset revision")
        if not plan.get("allow_patterns"):
            raise ValueError("An explicit modality/file selection is required")
    if not allow_network:
        raise PermissionError("Use an explicit allow_network download action")
    reserve = int(plan.get("free_space_reserve_bytes", 100 * 2**30))
    if shutil.disk_usage(root).free - int(plan.get("expected_bytes", 0)) < reserve:
        raise RuntimeError("Download would violate the free-space reserve")
    if provider == "huggingface":
        from huggingface_hub import snapshot_download

        result = snapshot_download(
            repo_id=plan["repo_id"],
            repo_type="dataset",
            revision=plan["revision"],
            allow_patterns=plan["allow_patterns"],
            local_dir=destination,
            max_workers=int(plan.get("workers", 4)),
        )
        return {
            "provider": provider,
            "destination": str(result),
            "revision": plan["revision"],
            "deleted_files": 0,
        }
    url = plan["url"]
    digest = plan["sha256"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    offset = temporary.stat().st_size if temporary.exists() else 0
    request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
    with urllib.request.urlopen(request, timeout=120) as response:
        mode = "ab" if offset and response.status == 206 else "wb"
        with temporary.open(mode) as stream:
            shutil.copyfileobj(response, stream, 1024 * 1024)
    if file_hash(temporary) != digest:
        raise ValueError("Downloaded file checksum mismatch; partial retained for inspection")
    temporary.replace(destination)
    return {
        "provider": provider,
        "destination": str(destination),
        "sha256": digest,
        "deleted_files": 0,
    }
