"""Preflight canonical pairing, native sizes, original VLA pixels, and boundaries."""
import io
import json
import re
import time
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import av
import h5py
import numpy as np
from PIL import Image

from .aligned_frames import (manifest_signature, sensor_indices, source_snapshot,
                             targets, write_json)
from .sensor_data import ftp_group


def audit_manifest(manifest, output, full_vla=True):
    signature = manifest_signature(manifest)
    rows = json.loads(Path(manifest).read_text())["episodes"]
    started = time.monotonic()
    videos, hdf5, ftp = defaultdict(list), defaultdict(list), defaultdict(list)
    counts = Counter()
    for row in rows:
        counts[row["source"]] += 1
        if row["source"] in ("egotouch", "egotactile", "opengraph", "freetacman"):
            for path in [*row["views"].values(), *row.get("tactile_videos", [])]:
                videos[path].append(row)
        if row["source"] == "opentouch":
            hdf5[row["views"]["0"]].append(row)
        if "tar_path" in row:
            ftp[(row["source"], row["zarr_name"])].append(row)
        if row["source"] == "ftp_vla_touch" and sorted(row["sensor_frame_order"]) != list(range(row["frames"])):
            raise ValueError("VLA mapping is not a complete episode permutation")

    def check_video(item):
        path, uses = item
        with av.open(path) as container:
            stream = container.streams.video[0]
            for row in uses:
                start = row.get("video_start", 0)
                if start < 0 or start + row["frames"] > stream.frames:
                    raise ValueError(f"Video episode boundary mismatch: {path}")
                if row["source"] != "opengraph" and stream.frames != row["frames"]:
                    raise ValueError(f"Scene/target video frame counts differ: {path}")
                if row["source"] == "opengraph" and float(stream.average_rate) != 30.:
                    raise ValueError("OpenGraph frame-offset timebase differs")
                if path in row.get("tactile_videos", []) and [6, stream.height, stream.width] != row["target_shape"]:
                    raise ValueError("Native FreeTacMan sensor geometry differs")
            return True
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(check_video, videos.items()))

    for path, episodes in hdf5.items():
        with h5py.File(path) as handle:
            for row in episodes:
                demo = handle["data"][row["hdf5_demo"]]
                if (len(demo["rgb_images_jpeg"]) != row["frames"]
                        or tuple(demo["right_pressure"].shape) != (row["frames"], 16, 16)):
                    raise ValueError("OpenTouch scene/pressure indexing differs")
    for episodes in ftp.values():
        first = episodes[0]
        group = ftp_group(first)
        ends = group["meta/episode_ends"][:]
        starts = np.concatenate(([0], ends[:-1]))
        keys = set(first["views"].values()) | set(first.get("target_keys",
                          ["left_tactile_data_gripper", "right_tactile_data_gripper"]))
        for key in keys:
            if group["data/" + key].shape[0] != int(ends[-1]):
                raise ValueError(f"FTP scene/sensor array lengths differ: {key}")
        for row in episodes:
            number = int(row["episode_id"].rsplit("episode-", 1)[1])
            if row["target_start"] != starts[number] or row["frames"] != ends[number] - starts[number]:
                raise ValueError("FTP absolute episode offsets differ")

    # Every episode gets a real native target read, not just registry metadata.
    for number, row in enumerate(rows):
        if row["target_kind"] != "optical_rgb" or row["source"] != "freetacman":
            sample = targets(row, sorted({0, row["frames"] - 1}))
            if sample.shape[1:] != tuple(row["target_shape"]):
                raise ValueError("Native shape verification failed")
        if number % 2000 == 0:
            print(json.dumps({"event": "pairing_audit", "episodes": number,
                              "total": len(rows)}), flush=True)

    tactile_checks, scene_headers = 0, 0
    for archive_path in sorted({r["raw_scene"]["archive"] for r in rows if "raw_scene" in r}):
        with zipfile.ZipFile(archive_path) as archive:
            members = defaultdict(list)
            pattern = re.compile(r"^(.*)/(camera[12])/rgb_(\d+)\.jpg$")
            for name in archive.namelist():
                match = pattern.match(name)
                if match:
                    members[(match[1], match[2])].append(int(match[3]))
            for row in (r for r in rows if r.get("raw_scene", {}).get("archive") == archive_path):
                info = row["raw_scene"]
                for folder in info["folders"].values():
                    if sorted(members[(info["prefix"], folder)]) != list(range(row["frames"])):
                        raise ValueError("Original scene chronology has missing/duplicate frame numbers")
                frames = np.arange(row["frames"]) if full_vla else np.array(sorted({0, row["frames"] - 1}))
                from .native_optical import native_images
                stored = native_images(row, sensor_indices(row, frames))[0]
                for i, frame in enumerate(frames):
                    original = np.asarray(Image.open(io.BytesIO(archive.read(
                        f"{info['prefix']}/gelsight/gel_{int(frame)}.jpg"))).convert("RGB"))
                    if not np.array_equal(original, stored[i]):
                        raise ValueError(f"VLA tactile pixels differ: {row['episode_id']}:{frame}")
                    tactile_checks += 1
                    for folder in info["folders"].values():
                        with Image.open(io.BytesIO(archive.read(
                                f"{info['prefix']}/{folder}/rgb_{int(frame)}.jpg"))) as image:
                            if image.size != (640, 480):
                                raise ValueError("Original VLA scene geometry changed")
                        scene_headers += 1
                print(json.dumps({"event": "vla_pairing_audit", "episode": row["episode_id"],
                                  "native_pixel_checks": tactile_checks}), flush=True)
    if manifest_signature(manifest) != signature:
        raise ValueError("Sources changed during preflight")
    result = {"signature": signature, "passed": True, "episodes_by_source": dict(counts),
              "video_headers": len(videos), "FTP_task_groups": len(ftp),
              "OpenTouch_demos": sum(map(len, hdf5.values())),
              "VLA_complete_pixels": full_vla, "VLA_tactile_pixel_checks": tactile_checks,
              "VLA_original_scene_headers": scene_headers,
              "scene_resolution_policy": "original scene source -> explicit224 DINO preprocessing",
              "target_resolution_policy": "native geometry; no resizing, cropping, or rotation",
              "hardware_synchronization": "not independently measured",
              "seconds": time.monotonic() - started}
    write_json(output, result)
    return result
