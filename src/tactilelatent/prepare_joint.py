"""Index six verified visual/tactile datasets without duplicating their media.

Run only after scripts/audit_joint_sources.py --hash succeeds. The output is
an episode-level JSON manifest plus a small aligned OpenGraph tactile array.
Every source keeps its native target shape and a separate target_kind so a
joint trainer can route examples to the correct decoder/loss.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import pyarrow.parquet as pq
import zarr

from .media import video_info
from .tar_zarr import IndexedTarZarrStore


ROOTS = {
    "egotouch": Path("datasets/EgoTouch"),
    "egotactile": Path("datasets/HustleHard__EgoTactile"),
    "opentouch": Path("datasets/rayxsong__opentouch"),
    "opengraph": Path("datasets/OpenGraphLabs-Research__ego-tactile-manipulation"),
    "freetacman": Path("datasets/OpenDriveLab__FreeTacMan"),
    "ftp_unit_bimanual": Path("datasets/MJJJJ1064__FTP-1-Dataset"),
}

TARGETS = {
    "egotouch": ("pressure_grid", [2, 21, 21], 1.0),
    "egotactile": ("taxel_vector", [1, 1, 256], 255.0),
    "opentouch": ("pressure_grid", [1, 16, 16], 3072.0),
    "opengraph": ("taxel_vector", [2, 1, 80], 4095.0),
    "freetacman": ("optical_rgb", [6, 32, 32], 255.0),
    "ftp_unit_bimanual": ("optical_rgb", [6, 32, 32], 255.0),
}


def split_for(source, group):
    """Stable source-local 90/10 episode split; never split one episode."""
    digest = hashlib.sha256(f"joint-six-v1:{source}:{group}".encode()).digest()
    return "val" if int.from_bytes(digest[:4], "big") % 10 == 0 else "train"


def record(source, episode_id, frames, views, **extra):
    kind, shape, scale = TARGETS[source]
    return {
        "source": source, "episode_id": episode_id, "frames": int(frames),
        "split": split_for(source, episode_id), "views": views,
        "target_kind": kind, "target_shape": shape, "target_scale": scale,
        **extra,
    }


def index_egotouch(report):
    root = ROOTS["egotouch"]
    split_map = json.loads((root / "split.json").read_text())
    rows = []
    missing = []
    for split, episode_paths in split_map.items():
        for entry in episode_paths:
            relative = Path(*Path(entry).parts[-3:]).with_suffix("")
            episode = root / relative
            media = [episode / name for name in ("chest.mp4", "left.mp4", "right.mp4")]
            pressure_path = episode / "pressure_grids.npz"
            if not all(p.is_file() for p in (*media, pressure_path)):
                missing.append(str(relative))
                continue
            with np.load(pressure_path) as pressure:
                n = len(pressure["left_pressure_grid"])
                assert pressure["right_pressure_grid"].shape == (n, 21, 21)
            if any(video_info(path)[0] != n for path in media):
                raise ValueError(f"EgoTouch video/pressure length mismatch: {episode}")
            if n < 15:
                continue
            row = record("egotouch", str(relative), n,
                         {"0": str(media[0]), "1": str(media[1]), "2": str(media[2])},
                         target_path=str(pressure_path))
            row["split"] = split  # Preserve the source's published split.
            rows.append(row)
    report["egotouch"] = {
        "published_split_episodes": sum(map(len, split_map.values())),
        "usable_episodes": len(rows), "missing_from_published_split": len(missing),
        "missing_examples": missing[:10],
    }
    return rows


def index_egotactile(report):
    root = ROOTS["egotactile"]
    rows, skipped = [], []
    for label_path in sorted((root / "Raw_data").rglob("data.json")):
        video = label_path.with_name("video.mp4")
        data = json.loads(label_path.read_text())
        n = len(data)
        if not video.is_file() or n < 15 or video_info(video)[0] != n or any(
            "RH" not in frame or len(frame["RH"].get("sensor_256", [])) != 256
            for frame in data
        ):
            skipped.append(str(label_path))
            continue
        rows.append(record("egotactile", str(label_path.parent.relative_to(root)), n,
                           {"0": str(video)}, target_path=str(label_path)))
    report["egotactile"] = {"usable_episodes": len(rows), "skipped": len(skipped),
                             "skipped_examples": skipped[:10]}
    return rows


def index_opentouch(report):
    root = ROOTS["opentouch"]
    rows, skipped = [], []
    for path in sorted(root.glob("*.hdf5")):
        with h5py.File(path) as h5:
            for demo_name, demo in h5["data"].items():
                n = len(demo["rgb_images_jpeg"])
                if "right_pressure" not in demo or demo["right_pressure"].shape != (n, 16, 16) or n < 15:
                    skipped.append(f"{path.name}:{demo_name}")
                    continue
                row = record("opentouch", f"{path.name}:{demo_name}", n,
                             {"0": str(path)}, hdf5_demo=demo_name)
                # Keep every demonstration in one HDF5 recording on the
                # same side of the split to reduce scene/task leakage.
                row["split"] = split_for("opentouch", path.name)
                rows.append(row)
    report["opentouch"] = {"usable_episodes": len(rows), "skipped": len(skipped),
                            "skipped_examples": skipped[:10]}
    return rows


def index_opengraph(report, output_dir):
    root = ROOTS["opengraph"]
    data_path = root / "data/chunk-000/file-000.parquet"
    columns = ["observation.tactile.left", "observation.tactile.right"]
    table = pq.read_table(data_path, columns=columns)
    tactile = np.stack([
        np.asarray(table.column(name).to_pylist(), dtype=np.float32)
        for name in columns
    ], axis=1)  # [all aligned video times, 2 hands, 80 taxels]
    if tactile.shape[1:] != (2, 80) or not np.isfinite(tactile).all():
        raise ValueError("OpenGraph tactile columns are not aligned 2x80 arrays")
    tactile_path = output_dir / "opengraph_tactile.npy"
    np.save(tactile_path, tactile)

    metadata = pq.read_table(
        root / "meta/episodes/chunk-000/file-000.parquet",
        columns=["episode_index", "length", "dataset_from_index", "dataset_to_index",
                 "videos/observation.images.ego/chunk_index",
                 "videos/observation.images.ego/file_index",
                 "videos/observation.images.ego/from_timestamp"],
    )
    rows = []
    for ep in metadata.to_pylist():
        n = ep["length"]
        offset = ep["dataset_from_index"]
        if ep["dataset_to_index"] != offset + n or offset + n > len(tactile):
            raise ValueError(f"OpenGraph episode {ep['episode_index']} target offset mismatch")
        video = root / ("videos/observation.images.ego/"
                        f"chunk-{ep['videos/observation.images.ego/chunk_index']:03d}/"
                        f"file-{ep['videos/observation.images.ego/file_index']:03d}.mp4")
        # The video chunk may contain several episodes. Frame offsets come
        # from the published LeRobot video timestamp at 30 frames per second.
        video_start = round(ep["videos/observation.images.ego/from_timestamp"] * 30)
        if not video.is_file() or video_start + n > video_info(video)[0]:
            raise ValueError(f"OpenGraph video boundary mismatch: {video}")
        rows.append(record("opengraph", f"episode-{ep['episode_index']}", n,
                           {"0": str(video)}, target_path=str(tactile_path),
                           target_start=offset, video_start=video_start))
    report["opengraph"] = {"usable_episodes": len(rows), "target_rows": len(tactile),
                            "target_array_bytes": tactile_path.stat().st_size}
    return rows


def freetacman_camera_roles(episode_id, mapping=None):
    if mapping is None:
        mapping = json.loads((Path(__file__).resolve().parents[2] / 'configs/freetacman_camera_roles.json').read_text())
    return mapping['overrides'].get(episode_id, mapping['default'])


def index_freetacman(report):
    root = ROOTS["freetacman"]
    groups = {}
    for path in root.rglob("*_camera*.mp4"):
        base, suffix = str(path).rsplit("_camera", 1)
        camera = int(suffix.removesuffix(".mp4"))
        groups.setdefault(base, {})[camera] = path
    rows, skipped = [], []
    role_mapping = json.loads((Path(__file__).resolve().parents[2] / 'configs/freetacman_camera_roles.json').read_text())
    for base, cameras in sorted(groups.items()):
        if set(cameras) != {1, 2, 3}:
            skipped.append((base, "missing camera"))
            continue
        counts = [video_info(cameras[i])[0] for i in (1, 2, 3)]
        if min(counts) < 15 or max(counts) != min(counts):
            skipped.append((base, f"frame counts {counts}"))
            continue
        relative = Path(base).relative_to(root)
        # Camera roles vary by episode, including 53 SprinkleSesame swaps.
        roles = freetacman_camera_roles(str(relative), role_mapping)
        row = record("freetacman", str(relative), counts[0],
                     {"1": str(cameras[roles['scene']])},
                     tactile_videos=[str(cameras[roles['surface1']]), str(cameras[roles['surface2']])])
        # Entire task categories stay together in train or validation.
        row["split"] = split_for("freetacman", relative.parts[0])
        rows.append(row)
    report["freetacman"] = {"video_groups": len(groups), "usable_episodes": len(rows),
                             "skipped": len(skipped), "skipped_examples": skipped[:10]}
    return rows


def index_ftp(report):
    root = ROOTS["ftp_unit_bimanual"]
    tar_path = root / "Unit_Bimanual/Unit_Bimanual.tar"
    index_path = Path("processing/indexes/ftp_tar.json")
    rows = []
    groups = {}
    for zarr_name in ("chips_grasping.zarr", "chickenlegs_hanging.zarr"):
        store = IndexedTarZarrStore(tar_path, index_path, f"Unit_Bimanual/{zarr_name}")
        group = zarr.open_group(store=store, mode="r")
        ends = group["meta/episode_ends"][:]
        all_keys = ("camera_main_rgb", "left_wrist_camera_rgb", "right_wrist_camera_rgb",
                    "left_tactile_data_gripper", "right_tactile_data_gripper")
        if any(group[f"data/{key}"].shape[0] != int(ends[-1]) for key in all_keys):
            raise ValueError(f"FTP array/frame count mismatch: {zarr_name}")
        start = 0
        for episode_number, end in enumerate(ends):
            n = int(end) - start
            if n >= 15:
                rows.append(record("ftp_unit_bimanual", f"{zarr_name}:episode-{episode_number}", n,
                                   {"0": "camera_main_rgb", "1": "left_wrist_camera_rgb",
                                    "2": "right_wrist_camera_rgb"},
                                   tar_path=str(tar_path), tar_index=str(index_path),
                                   zarr_name=zarr_name, target_start=start))
            start = int(end)
        groups[zarr_name] = {"episodes": len(ends), "frames": int(ends[-1])}
    report["ftp_unit_bimanual"] = {"usable_episodes": len(rows), "groups": groups}
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--integrity", default="processing/audits/joint_six_integrity.json")
    parser.add_argument("--output-dir", default="processing/joint_six")
    args = parser.parse_args()
    integrity = json.loads(Path(args.integrity).read_text())
    if set(integrity) != set(ROOTS) or not all(x["inventory_complete"] for x in integrity.values()):
        raise ValueError("All six pinned source scopes must pass integrity audit first")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    episodes = []
    for name, scanner in (
        ("egotouch", lambda: index_egotouch(report)),
        ("egotactile", lambda: index_egotactile(report)),
        ("opentouch", lambda: index_opentouch(report)),
        ("opengraph", lambda: index_opengraph(report, output_dir)),
        ("freetacman", lambda: index_freetacman(report)),
        ("ftp_unit_bimanual", lambda: index_ftp(report)),
    ):
        rows = scanner()
        episodes.extend(rows)
        print(json.dumps({"event": "indexed", "source": name, **report[name]}), flush=True)

    summary = {name: dict(counts=Counter(r["split"] for r in episodes if r["source"] == name),
                          target_kind=TARGETS[name][0], target_shape=TARGETS[name][1],
                          target_scale=TARGETS[name][2], **report[name])
               for name in ROOTS}
    payload = {"schema_version": 1, "episodes": episodes, "sources": summary,
               "integrity_report": args.integrity}
    destination = output_dir / "manifest.json"
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(destination)
    print(json.dumps({"event": "complete", "episodes": len(episodes),
                      "manifest": str(destination)}), flush=True)


if __name__ == "__main__":
    main()
