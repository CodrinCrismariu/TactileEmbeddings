"""Index FTP-1 tar members so Zarr can be read without archive extraction."""

import argparse
import json
import tarfile
from pathlib import Path


ROOT = Path("datasets/MJJJJ1064__FTP-1-Dataset")
OUT = Path("processing/indexes")


def index_group(group):
    path = ROOT / group / f"{group}.tar"
    if not path.is_file():
        raise FileNotFoundError(path)
    members = {}
    zarrs = set()
    with tarfile.open(path, "r:") as archive:
        for member in archive:
            if not member.isfile():
                continue
            members[member.name] = [member.offset_data, member.size]
            parts = member.name.split("/")
            for part in parts:
                if part.endswith(".zarr"):
                    zarrs.add(part)
                    break
    stat = path.stat()
    data = {"signature": [str(path.resolve()), stat.st_size, stat.st_mtime_ns],
            "members": members}
    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / f"ftp_{group.lower()}.json"
    temp = destination.with_suffix(".tmp")
    temp.write_text(json.dumps(data))
    temp.replace(destination)
    print(json.dumps({"group": group, "index": str(destination),
                      "members": len(members), "zarrs": sorted(zarrs),
                      "index_bytes": destination.stat().st_size}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("groups", nargs="+")
    args = parser.parse_args()
    for group in args.groups:
        index_group(group)


if __name__ == "__main__":
    main()
