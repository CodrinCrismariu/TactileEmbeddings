"""Canonical frame pairing. Tactile pixels stay native; only DINO RGB is resized.

Historical loaders and active AE training are deliberately left untouched.
Schema-2 manifests explicitly describe native targets and chronological indices.
"""
import hashlib
import io
import json
import os
import zipfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .media import video_frames
from .native_optical import native_images, native_shape
from .sensor_data import load_targets, scene_frames


CONTRACT = "native-chronological-v1"
MANIFEST = "processing/joint_alignment/manifest.json"


def file_hash(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def validate_frame_order(row, record):
    order = record["chronological_to_stored_index"]
    if (not record.get("exact_pixel_multiset_match")
            or sorted(order) != list(range(row["frames"]))
            or any(record.get(key) != row.get(key)
                   for key in ("target_start", "zarr_name", "tar_path"))):
        raise ValueError(f"Invalid tactile chronology: {row['episode_id']}")
    return order


def sensor_indices(row, indices, order=None):
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1 or not len(indices) or (indices < 0).any() or (indices >= row["frames"]).any():
        raise ValueError("Frame indices must stay inside one episode")
    if row["source"] == "ftp_vla_touch":
        order = row.get("sensor_frame_order") if order is None else order
        if order is None:
            raise ValueError("VLA requires a verified chronological-to-stored lookup")
        return np.asarray(order, dtype=np.int64)[indices]
    return indices


def target_images(row, indices):
    """Native uint8 sensor images in declared sensor order, never scene cameras."""
    return native_images(row, sensor_indices(row, indices))


def targets(row, indices):
    values = load_targets(row, sensor_indices(row, indices))
    if tuple(values.shape[1:]) != tuple(row["target_shape"]):
        raise ValueError(f"Native target shape mismatch: {row['episode_id']}")
    return values


@lru_cache(maxsize=8)
def _archive(pid, path):
    # PID avoids sharing seek state inherited from a parent DataLoader process.
    return zipfile.ZipFile(path)


def raw_scene_images(row, indices, slot):
    info = row["raw_scene"]
    archive = _archive(os.getpid(), info["archive"])
    folder = info["folders"][str(slot)]
    return [np.asarray(Image.open(io.BytesIO(archive.read(
        f"{info['prefix']}/{folder}/rgb_{int(frame)}.jpg"))).convert("RGB"))
        for frame in indices]


def scene_rgb(row, indices, size=224):
    """Scene-only input. Targets cannot accidentally enter DINO camera slots."""
    indices = np.asarray(indices, dtype=np.int64)
    sensor_indices(row, indices)  # Validate logical episode boundaries.
    from .storage import converted_scene
    converted = converted_scene(row, indices, size)
    if converted is not None:
        return converted
    rgb = torch.zeros(len(indices), 3, 3, size, size)
    mask = torch.zeros(len(indices), 3, dtype=torch.bool)
    for slot, path in row["views"].items():
        camera = int(slot)
        if row["source"] == "ftp_vla_touch":
            frames = raw_scene_images(row, indices, slot)
            rotation = row["raw_scene"]["rotation_quarters"][slot]
            # Explicit DINO preprocessing; native tactile frames never use it.
            images = [np.rot90(np.asarray(Image.fromarray(image).resize(
                (size, size), Image.Resampling.BILINEAR)), rotation).copy() for image in frames]
            rgb[:, camera] = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).float() / 255.
        elif row["source"] in ("egotouch", "egotactile", "opengraph", "freetacman"):
            rgb[:, camera] = video_frames(path, indices + row.get("video_start", 0), size=size)
        else:
            rgb[:, camera] = scene_frames(row, indices, slot, size=size)
        mask[:, camera] = True
    return rgb, mask


def source_snapshot(episodes):
    """Detect replaced source files without hashing hundreds of GB every epoch."""
    paths = set()
    for row in episodes:
        for key in ("tar_path", "tar_index", "target_path"):
            if key in row:
                paths.add(row[key])
        paths.update(row.get("tactile_videos", []))
        paths.update(p for p in row["views"].values() if Path(p).is_file())
        if "raw_scene" in row:
            paths.add(row["raw_scene"]["archive"])
    from .storage import original_stat, scene_entry
    for row in episodes:
        entry = scene_entry(row)
        if entry:
            paths.update(view['path'] for view in entry['views'].values())
    result = {}
    for p in sorted(paths):
        path = Path(p)
        if path.exists():
            result[str(path.resolve())] = [path.stat().st_size, path.stat().st_mtime_ns]
        else:
            previous = original_stat(path)
            if previous is None:
                raise FileNotFoundError(path)
            result[str(path.resolve())] = previous
    return result


def manifest_signature(path):
    payload = json.loads(Path(path).read_text())
    if payload.get("schema_version") != 2 or payload.get("frame_contract") != CONTRACT:
        raise ValueError("New joint training requires the verified native schema-2 manifest")
    return {"manifest_sha256": file_hash(path),
            "source_snapshot_sha256": json_hash(source_snapshot(payload["episodes"])),
            "loader_sha256": hashlib.sha256(b"".join(Path(__file__).with_name(name).read_bytes()
                for name in ("aligned_frames.py", "native_optical.py", "sensor_data.py", "media.py"))).hexdigest()}


def prepare_manifest(original, frame_order, roles_path, output):
    payload = json.loads(Path(original).read_text())
    audit = json.loads(Path(frame_order).read_text())
    roles = json.loads(Path(roles_path).read_text())
    if audit.get("source") != "ftp_vla_touch":
        raise ValueError("Wrong frame-order audit source")
    prefixes = {}
    for name, info in audit["archives"].items():
        if file_hash(info["path"]) != info["sha256"]:
            raise ValueError(f"Raw VLA archive changed: {name}")
        with zipfile.ZipFile(info["path"]) as archive:
            for member in archive.namelist():
                if member.endswith("/gelsight/gel_0.jpg"):
                    prefix = member.split("/gelsight/")[0]
                    key = (name, int(prefix.rsplit("/episode_", 1)[1]))
                    if key in prefixes:
                        raise ValueError("Ambiguous raw episode prefix")
                    prefixes[key] = prefix
    ids = set()
    for row in payload["episodes"]:
        identity = row["source"], row["episode_id"]
        if identity in ids:
            raise ValueError("Duplicate episode identity")
        ids.add(identity)
        if row["source"] == "freetacman":
            role = roles["overrides"].get(row["episode_id"], roles["default"])
            root = row["tactile_videos"][0].rsplit("_camera", 1)[0]
            row["views"] = {"1": f"{root}_camera{role['scene']}.mp4"}
            row["tactile_videos"] = [f"{root}_camera{role[key]}.mp4" for key in ("surface1", "surface2")]
        if row["source"] == "ftp_vla_touch":
            record = audit["episodes"][row["episode_id"]]
            row["sensor_frame_order"] = validate_frame_order(row, record)
            row["raw_scene"] = {"archive": audit["archives"][record["raw_archive"]]["path"],
                "prefix": prefixes[(record["raw_archive"], record["raw_episode"])],
                "folders": {"0": "camera1", "2": "camera2"},
                "rotation_quarters": {"0": 0, "2": 1}}
        if row["target_kind"] == "optical_rgb":
            row["target_shape"] = native_shape(row)
    payload.update(schema_version=2, frame_contract=CONTRACT,
                   provenance={"original_manifest_sha256": file_hash(original),
                               "frame_order_sha256": file_hash(frame_order),
                               "camera_roles_sha256": file_hash(roles_path)})
    write_json(output, payload)
    return payload
