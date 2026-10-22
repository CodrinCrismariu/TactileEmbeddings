"""Download complete selected FTP-1 archives without exceeding the data cap.

Archives remain compressed; the project reads their Zarr members in place.
This script never removes existing files and pins the upstream revision.
"""

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


REPO = "MJJJJ1064/FTP-1-Dataset"
REVISION = "d287c9c532514a08b2eb29bf1259997bb32f4f3b"
ROOT = Path("datasets/MJJJJ1064__FTP-1-Dataset")
GROUPS = ("VLA_touch", "Unit", "RDP_Bimanual", "RDP", "MotionTrans", "ViTaMIn")
LIMIT = 700_000_000_000


def directory_size(path):
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download(group, item):
    started = time.monotonic()
    print(json.dumps({"event": "start", "group": group, "bytes": item.size}), flush=True)
    path = Path(hf_hub_download(repo_id=REPO, repo_type="dataset", revision=REVISION,
                                filename=item.rfilename, local_dir=ROOT))
    if path.stat().st_size != item.size:
        raise ValueError(f"Wrong size: {path}")
    actual = digest(path)
    if actual != item.lfs.sha256:
        raise ValueError(f"SHA-256 mismatch: {path}")
    result = {"event": "complete", "group": group, "path": str(path),
              "bytes": item.size, "sha256": actual,
              "elapsed_seconds": round(time.monotonic() - started, 1)}
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("groups", nargs="+", choices=GROUPS)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--report", default="processing/audits/ftp_extra_download.json")
    args = parser.parse_args()
    selected = list(dict.fromkeys(args.groups))
    info = HfApi().dataset_info(REPO, revision=REVISION, files_metadata=True)
    if info.sha != REVISION:
        raise ValueError("Repository revision changed")
    files = {s.rfilename: s for s in info.siblings}
    items = {g: files[f"{g}/{g}.tar"] for g in selected}
    if any(x.lfs is None or not x.lfs.sha256 for x in items.values()):
        raise ValueError("Missing LFS hashes")
    current = directory_size(Path("datasets")) + directory_size(Path("processing"))
    extra = sum(x.size for x in items.values()
                if not (ROOT / x.rfilename).is_file())
    projected = current + extra + 10_000_000_000  # reserve processing/headroom
    print(json.dumps({"event": "budget", "current_bytes": current,
                      "new_bytes": extra, "projected_with_reserve": projected,
                      "cap_bytes": LIMIT}), flush=True)
    if projected > LIMIT:
        raise ValueError("Selected downloads would exceed the 700 GB cap")
    results = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_to_group = {pool.submit(download, group, item): group
                           for group, item in items.items()}
        for future in as_completed(future_to_group):
            group = future_to_group[future]
            results[group] = future.result()
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({"revision": REVISION, "groups": results}, indent=2))
    print(json.dumps({"event": "all_complete", "groups": sorted(results)}), flush=True)


if __name__ == "__main__":
    main()
