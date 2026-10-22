"""Add six indexed FTP-1 groups to the existing six-source episode manifest.

No media is copied or extracted. Every row addresses camera/tactile chunks
inside its verified tar. An entire source can later be held out by the trainer.
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import zarr

from .joint_all_config import ALL_TARGETS, FTP_SPECS
from .prepare_joint import split_for
from .tar_zarr import IndexedTarZarrStore


def make_manifest(base_path, schema_path):
    base = json.loads(Path(base_path).read_text())
    schema = json.loads(Path(schema_path).read_text())
    if base["schema_version"] != 1 or len(base["sources"]) != 6:
        raise ValueError("Expected the verified original six-source manifest")
    if set(schema) != {spec["group"] for spec in FTP_SPECS.values()}:
        raise ValueError("FTP schema audit does not cover all six selected groups")
    rows = list(base["episodes"])
    source_summaries = dict(base["sources"])
    root = Path("datasets/MJJJJ1064__FTP-1-Dataset")
    for source, spec in FTP_SPECS.items():
        archive_group = spec["group"]
        audit = schema[archive_group]
        if audit["zarr_groups"] != audit["paired_zarr_groups"]:
            raise ValueError(f"Unpaired Zarr task in {archive_group}")
        tar_path = root / archive_group / f"{archive_group}.tar"
        index_path = Path("processing/indexes") / f"ftp_{archive_group.lower()}.json"
        kind, shape, scale = ALL_TARGETS[source]
        source_rows = []
        for task in audit["records"]:
            if not task["usable"] or not task["sample"]:
                raise ValueError(f"Unsampled or unusable task: {archive_group}/{task['zarr']}")
            if not set(spec["views"].values()) <= set(task["camera"]):
                raise ValueError(f"Missing visual key in {archive_group}/{task['zarr']}")
            if not set(spec["target_keys"]) <= set(task["tactile"]):
                raise ValueError(f"Missing tactile key in {archive_group}/{task['zarr']}")
            cells = 0
            for key in spec["target_keys"]:
                native = task["tactile"][key]["shape"][1:]
                cells += native[0] * (3 if kind == "optical_rgb" else
                                      math.prod(native[1:]))
            if cells != shape[0]:
                raise ValueError(f"Target channels mismatch in {archive_group}/{task['zarr']}")
            store = IndexedTarZarrStore(tar_path, index_path,
                                        f"{archive_group}/{task['zarr']}")
            group = zarr.open_group(store=store, mode="r")
            ends = group["meta/episode_ends"][:]
            if len(ends) != task["episodes"] or int(ends[-1]) != task["frames"]:
                raise ValueError(f"Episode metadata differs from audit: {archive_group}/{task['zarr']}")
            start = 0
            for episode_number, end in enumerate(ends):
                length = int(end) - start
                if length >= 15:
                    episode_id = f"{task['zarr']}:episode-{episode_number}"
                    source_rows.append({
                        "source": source, "episode_id": episode_id, "frames": length,
                        "split": split_for(source, episode_id),
                        "views": spec["views"], "target_kind": kind,
                        "target_shape": shape, "target_scale": scale,
                        "tar_path": str(tar_path), "tar_index": str(index_path),
                        "archive_group": archive_group, "zarr_name": task["zarr"],
                        "target_start": start, "target_keys": spec["target_keys"],
                        **({"target_transform_scale": spec["target_transform_scale"]}
                           if "target_transform_scale" in spec else {}),
                    })
                start = int(end)
        rows.extend(source_rows)
        source_summaries[source] = {"usable_episodes": len(source_rows),
                                    "counts": dict(Counter(r["split"] for r in source_rows)),
                                    "zarr_groups": audit["zarr_groups"],
                                    "frames": audit["paired_frames"]}
    if set(r["source"] for r in rows) != set(ALL_TARGETS):
        raise ValueError("Manifest does not contain exactly twelve expected sources")
    return {"schema_version": 1, "episodes": rows, "sources": source_summaries,
            "integrity_report": base["integrity_report"],
            "extra_schema_report": schema_path,
            "extra_download_reports": ["processing/audits/ftp_extra_download.json",
                                       "processing/audits/ftp_extra_download_second.json"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="processing/joint_six/manifest.json")
    parser.add_argument("--schema", default="processing/audits/ftp_extra_schema.json")
    parser.add_argument("--output", default="processing/joint_all/manifest.json")
    args = parser.parse_args()
    result = make_manifest(args.base, args.schema)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(".tmp")
    temp.write_text(json.dumps(result, indent=2))
    temp.replace(destination)
    print(json.dumps({"manifest": str(destination), "episodes": len(result["episodes"]),
                      "sources": {k: v.get("usable_episodes", sum(v.get("counts", {}).values()))
                                  for k, v in result["sources"].items()}}), flush=True)


if __name__ == "__main__":
    main()
