"""Verify the six selected local dataset scopes against pinned HF revisions.

This checks exact file inventories/sizes and, with --hash, every available
Hugging Face LFS SHA-256. The FTP-1 scope is intentionally Unit_Bimanual only;
the repository as a whole is much larger than the project storage budget.
"""

import argparse
import fnmatch
import hashlib
import json
import time
from pathlib import Path

from huggingface_hub import HfApi


SOURCES = {
    "egotouch": {
        "repo": "zhouzhoujy/EgoTouch",
        "revision": "cfdbb0ac31cc2af4247943820aa250575e7e6637",
        "root": "datasets/EgoTouch",
        "patterns": ["split.json", "*/*/*/chest.mp4", "*/*/*/left.mp4",
                     "*/*/*/right.mp4", "*/*/*/pressure_grids.npz"],
    },
    "egotactile": {
        "repo": "HustleHard/EgoTactile",
        "revision": "bdc81fb73f8c9697f40796d6f01d291c87ca67b1",
        "root": "datasets/HustleHard__EgoTactile",
        "patterns": ["*"],
    },
    "ftp_unit_bimanual": {
        "repo": "MJJJJ1064/FTP-1-Dataset",
        "revision": "d287c9c532514a08b2eb29bf1259997bb32f4f3b",
        "root": "datasets/MJJJJ1064__FTP-1-Dataset",
        "patterns": ["Unit_Bimanual/*"],
    },
    "freetacman": {
        "repo": "OpenDriveLab/FreeTacMan",
        "revision": "030316fb41d6fa1e58cccb4bbe0f6fddbb932671",
        "root": "datasets/OpenDriveLab__FreeTacMan",
        "patterns": ["*"],
    },
    "opengraph": {
        "repo": "OpenGraphLabs-Research/ego-tactile-manipulation",
        "revision": "90c4e304b8e3a9578d5bb938992206358db1a660",
        "root": "datasets/OpenGraphLabs-Research__ego-tactile-manipulation",
        "patterns": ["*"],
    },
    "opentouch": {
        "repo": "rayxsong/opentouch",
        "revision": "16b1d915d6a0f6a6b2434a165ae8354d28fb0fd9",
        "root": "datasets/rayxsong__opentouch",
        "patterns": ["*"],
    },
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_source(api, name, spec, check_hash):
    info = api.dataset_info(spec["repo"], revision=spec["revision"], files_metadata=True)
    if info.sha != spec["revision"]:
        raise ValueError(f"{name}: resolved revision differs from pinned revision")
    files = [f for f in info.siblings if f.size is not None and
             any(fnmatch.fnmatch(f.rfilename, pat) for pat in spec["patterns"])]
    result = {
        "repo": spec["repo"], "revision": spec["revision"],
        "root": spec["root"], "patterns": spec["patterns"],
        "expected_files": len(files), "expected_bytes": sum(f.size for f in files),
        "missing": [], "wrong_size": [], "wrong_hash": [],
        "hashed_files": 0, "hashed_bytes": 0, "size_only_files": 0,
    }
    started = time.monotonic()
    last_report = started
    for number, item in enumerate(files, 1):
        path = Path(spec["root"]) / item.rfilename
        if not path.is_file():
            result["missing"].append(item.rfilename)
        elif path.stat().st_size != item.size:
            result["wrong_size"].append(item.rfilename)
        elif check_hash and item.lfs and item.lfs.sha256:
            if sha256(path) != item.lfs.sha256:
                result["wrong_hash"].append(item.rfilename)
            result["hashed_files"] += 1
            result["hashed_bytes"] += item.size
        else:
            result["size_only_files"] += 1
        if check_hash and time.monotonic() - last_report >= 30:
            print(json.dumps({"event": "progress", "source": name,
                              "files_checked": number, "expected_files": len(files),
                              "hashed_gb": round(result["hashed_bytes"] / 1e9, 2)}), flush=True)
            last_report = time.monotonic()
    result["inventory_complete"] = not any(result[key] for key in ("missing", "wrong_size", "wrong_hash"))
    result["elapsed_seconds"] = round(time.monotonic() - started, 2)
    print(json.dumps({"event": "source_complete", "source": name,
                      "inventory_complete": result["inventory_complete"],
                      "files": len(files), "hashed_gb": round(result["hashed_bytes"] / 1e9, 2),
                      "elapsed_seconds": result["elapsed_seconds"]}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hash", action="store_true", help="hash all LFS files, not just check size")
    parser.add_argument("--output", default="processing/audits/joint_six_integrity.json")
    args = parser.parse_args()
    api = HfApi()
    result = {name: audit_source(api, name, spec, args.hash) for name, spec in SOURCES.items()}
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2))
    temporary.replace(destination)


if __name__ == "__main__":
    main()
