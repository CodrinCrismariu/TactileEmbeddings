"""Summarize paired visual/tactile arrays in indexed FTP-1 Zarr groups."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import zarr

from tactilelatent.tar_zarr import IndexedTarZarrStore


ROOT = Path("datasets/MJJJJ1064__FTP-1-Dataset")
INDEX = Path("processing/indexes")


def inspect(group, sample_frames=False):
    tar_path = ROOT / group / f"{group}.tar"
    index_path = INDEX / f"ftp_{group.lower()}.json"
    index = json.loads(index_path.read_text())
    zarr_names = sorted({name.split("/", 2)[1] for name in index["members"]
                         if "/" in name and name.split("/", 2)[1].endswith(".zarr")})
    records = []
    for name in zarr_names:
        store = IndexedTarZarrStore(tar_path, index_path, f"{group}/{name}")
        z = zarr.open_group(store=store, mode="r")
        if "data" not in z or "meta/episode_ends" not in z:
            records.append({"zarr": name, "usable": False, "reason": "missing data or episode_ends"})
            continue
        arrays = {key: {"shape": list(z[f"data/{key}"].shape),
                        "dtype": str(z[f"data/{key}"].dtype)}
                  for key in z["data"].array_keys()}
        camera = {k: v for k, v in arrays.items()
                  if "camera" in k and "rgb" in k and len(v["shape"]) >= 4}
        tactile = {k: v for k, v in arrays.items()
                   if "tactile_data" in k}
        ends = z["meta/episode_ends"][:]
        frames = int(ends[-1]) if len(ends) else 0
        aligned = bool(camera and tactile and all(v["shape"][0] == frames
                                                   for v in [*camera.values(), *tactile.values()]))
        sample = {}
        if aligned and sample_frames:
            for key in [*camera, *tactile]:
                array = z[f"data/{key}"]
                positions = [0, frames // 2, frames - 1]
                values = array[positions]
                numeric = np.issubdtype(values.dtype, np.number)
                valid = values.shape[0] == 3 and (not numeric or bool(np.isfinite(values).all()))
                sample[key] = {"shape": list(values.shape), "finite": valid}
            aligned = all(v["finite"] for v in sample.values())
        records.append({"zarr": name, "episodes": len(ends), "frames": frames,
                        "camera": camera, "tactile": tactile,
                        "sample": sample,
                        "usable": aligned})
    summary = {"group": group, "zarr_groups": len(records),
               "paired_zarr_groups": sum(r["usable"] for r in records),
               "paired_episodes": sum(r.get("episodes", 0) for r in records if r["usable"]),
               "paired_frames": sum(r.get("frames", 0) for r in records if r["usable"]),
               "cameras": dict(Counter(k for r in records for k in r.get("camera", {}))),
               "tactile_arrays": dict(Counter(k for r in records for k in r.get("tactile", {}))),
               "records": records}
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("groups", nargs="+")
    parser.add_argument("--output", default="processing/audits/ftp_extra_schema.json")
    parser.add_argument("--sample-frames", action="store_true")
    args = parser.parse_args()
    results = {group: inspect(group, sample_frames=args.sample_frames) for group in args.groups}
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
