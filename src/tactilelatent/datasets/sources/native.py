"""Native chronological readers shared by the uniform source adapters.

The branching here concerns release formats, never the shared training loop.
No tactile image is resized. Marker-only releases never open retired imagery.
Inputs are immutable manifest rows and logical frame IDs; outputs are visual or
sensor contracts with explicit stored-frame mappings and validity.
"""

import hashlib
import json
import zipfile
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import h5py
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
import zarr
from PIL import Image

from ...contracts import SensorTarget, VisualClip
from ...dataset_management.artifacts import file_hash, json_hash, read_json
from .archive import IndexedTarZarrStore
from .video import video_frames


@lru_cache(maxsize=32)
def json_records(path):
    """Read a release's immutable JSON records with a bounded process cache."""
    return json.loads(Path(path).read_text())


@lru_cache(maxsize=32)
def _ftp_cached(tar_path, index_path, prefix, stamp):
    return zarr.open_group(store=IndexedTarZarrStore(tar_path, index_path, prefix), mode="r")


def ftp_group(row, paths):
    """Open the row's indexed native Zarr group without extracting its tar."""
    archive, index = paths.resolve(row["tar_path"]), paths.resolve(row["tar_index"])
    a, b = archive.stat(), index.stat()
    return _ftp_cached(
        str(archive),
        str(index),
        f"{row.get('archive_group', 'Unit_Bimanual')}/{row['zarr_name']}",
        (a.st_size, a.st_mtime_ns, b.st_size, b.st_mtime_ns),
    )


def sensor_indices(row, indices):
    """Map logical episode frames to their verified stored sensor positions.

    Most releases use identity order. VLA Touch requires its saved full
    chronological-to-stored permutation; missing or invalid mappings raise.
    """
    indices = np.asarray(indices, dtype=np.int64)
    if (
        indices.ndim != 1
        or not len(indices)
        or (indices < 0).any()
        or (indices >= row["frames"]).any()
    ):
        raise ValueError("Frame indices must stay inside one episode")
    if row.get("source") == "ftp_vla_touch":
        order = row.get("sensor_frame_order")
        if order is None or sorted(order) != list(range(row["frames"])):
            raise ValueError("VLA requires its verified chronological-to-stored permutation")
        return np.asarray(order, dtype=np.int64)[indices]
    return indices


@lru_cache(maxsize=16)
def native_arrays(path, digest):
    """Return copied NPZ arrays only after verifying the recorded payload hash."""
    if file_hash(path) != digest:
        raise ValueError("Native target hash changed")
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def acquisition_history(row, end, paths, pad_history=False):
    """Resolve eight stride-two frames and their explicit camera timestamps.

    Return logical IDs, camera-specific stored IDs and verified native arrays.
    Missing/ambiguous RGB alignment raises. Viewer-only padding repeats frame
    zero; ordinary training rejects a history crossing an episode boundary.
    """
    indices = np.arange(end - 14, end + 1, 2, dtype=np.int64)
    if pad_history:
        indices = np.maximum(indices, 0)
    if indices[0] < 0 or indices[-1] >= row["frames"]:
        raise ValueError("History crosses episode boundary")
    native = native_arrays(str(paths.resolve(row["targets"]["path"])), row["targets"]["sha256"])
    if np.any(np.diff(native["timestamps"][indices])[np.diff(indices) > 0] <= 0):
        raise ValueError("Nonchronological tactile history")
    if "rgb_frame_valid" in native and not native["rgb_frame_valid"][indices].all():
        raise ValueError("Missing or ambiguous RGB timestamp")
    views = {}
    for slot, view in row["views"].items():
        if view["role"] not in ("ego", "left_wrist", "right_wrist", "fixed_scene"):
            raise ValueError("Unknown camera role")
        if "frame_indices_key" in view:
            if view.get("frame_indices_artifact") != row["targets"]["path"]:
                raise ValueError("External camera mapping")
            mapped = native[view["frame_indices_key"]][indices]
            limit = view.get("frames")
        else:
            offset = int(view.get("frame_offset", 0))
            mapped = indices + offset
            limit = offset + view["frames"] if "frames" in view else None
        if (
            (mapped < 0).any()
            or (limit is not None and (mapped >= limit).any())
            or (np.diff(mapped)[np.diff(indices) > 0] <= 0).any()
        ):
            raise ValueError("Missing/nonchronological camera history")
        views[str(slot)] = mapped
    if not views:
        raise ValueError("No scene camera")
    return indices, views, native


def read_visual(row, end, paths, storage, size=384, pad_history=False):
    """Return eight RGB history frames with fixed camera slots and mappings.

    ``VisualClip.rgb`` is ``[8, 3, 3, size, size]`` in ``[0, 1]``: time, camera,
    RGB channel, height, width. Absent cameras have false availability, zero RGB
    and stored frame ID ``-1``. Compressed-scene receipts and release-specific
    chronology determine reads; only visual data is resized.
    """
    history = np.arange(end - 14, end + 1, 2, dtype=np.int64)
    if pad_history:
        history = np.maximum(history, 0)
    sensor_indices(row, history)
    rgb = torch.zeros(8, 3, 3, size, size)
    mask = torch.zeros(8, 3, dtype=torch.bool)
    mapped = torch.full((8, 3), -1, dtype=torch.int64)
    roles = ["scene_or_main", "left_wrist", "right_wrist"]
    timestamps = None
    key = hashlib.sha256((row.get("source", "") + "\0" + row["episode_id"]).encode()).hexdigest()[
        :24
    ]
    entry = storage.get("episodes", {}).get(key)
    if "targets" in row:
        history, views, native = acquisition_history(row, end, paths, pad_history)
        timestamps = torch.from_numpy(native["timestamps"][history].copy())
        for slot, indices in views.items():
            camera = int(slot)
            view = row["views"][slot]
            rgb[:, camera] = video_frames(paths.resolve(view["path"]), indices, size)
            mask[:, camera] = True
            mapped[:, camera] = torch.from_numpy(indices.copy())
            roles[camera] = view["role"]
    elif entry is not None:
        for slot, info in entry["views"].items():
            path = paths.resolve(info["path"])
            stat = path.stat()
            if [stat.st_size, stat.st_mtime_ns] != info["stat"]:
                raise ValueError("Converted RGB changed outside migration ledger")
            indices = history + info.get("frame_offset", 0)
            camera = int(slot)
            rgb[:, camera] = video_frames(path, indices, size)
            mask[:, camera] = True
            mapped[:, camera] = torch.from_numpy(indices.copy())
    else:
        for slot, value in row["views"].items():
            camera = int(slot)
            indices = history.copy()
            if isinstance(value, dict):
                raise ValueError("Native acquisition views require explicit mapping")
            if row["source"] in ("baihu", "egopressure", "egotouch", "egotactile", "opengraph"):
                indices += row.get("video_start", 0)
                image = video_frames(paths.resolve(value), indices, size)
            elif row["source"] == "opentouch":
                images = []
                with h5py.File(paths.resolve(value), "r") as h5:
                    demo = h5["data"][row["hdf5_demo"]]
                    for index in indices:
                        with Image.open(BytesIO(bytes(demo["rgb_images_jpeg"][int(index)]))) as raw:
                            array = np.asarray(
                                raw.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
                            ).copy()
                            images.append(torch.from_numpy(array).permute(2, 0, 1).float() / 255.0)
                image = torch.stack(images)
            elif "raw_scene" in row:
                info = row["raw_scene"]
                images = []
                with zipfile.ZipFile(paths.resolve(info["archive"])) as archive:
                    folder = info["folders"][slot]
                    rotation = info["rotation_quarters"][slot]
                    for index in indices:
                        name = f"{info['prefix']}/{folder}/rgb_{int(index)}.jpg"
                        with Image.open(BytesIO(archive.read(name))) as raw:
                            array = np.asarray(
                                raw.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
                            )
                        array = np.rot90(array, rotation).copy()
                        images.append(torch.from_numpy(array).permute(2, 0, 1).float() / 255.0)
                image = torch.stack(images)
            elif "tar_path" in row:
                positions = indices + row["target_start"]
                values = ftp_group(row, paths)["data/" + value][positions]
                image = (
                    torch.from_numpy(np.asarray(values).copy()).permute(0, 3, 1, 2).float() / 255.0
                )
                image = F.interpolate(
                    image, size=(size, size), mode="bilinear", align_corners=False
                )
                indices = positions
            else:
                raise ValueError(f"Unsupported scene release: {row['source']}")
            rgb[:, camera] = image
            mask[:, camera] = True
            mapped[:, camera] = torch.from_numpy(indices.copy())
    role_ids = torch.tensor(
        [
            {"scene_or_main": 0, "ego": 0, "fixed_scene": 0, "left_wrist": 1, "right_wrist": 2}[
                role
            ]
            for role in roles
        ]
    )
    return VisualClip(rgb, mask, torch.from_numpy(history), mapped, role_ids, timestamps)


@lru_cache(maxsize=8)
def _pressure_file(path, stamp):
    parquet = pq.ParquetFile(path)
    groups = [
        parquet.read_row_group(i, columns=["frame"]).column("frame").to_pylist()
        for i in range(parquet.num_row_groups)
    ]
    return parquet, groups


@lru_cache(maxsize=4)
def _pressure_group(path, stamp, group, columns):
    parquet, _ = _pressure_file(path, stamp)
    return parquet.read_row_group(group, columns=list(columns))


def pressure_rows(path, frame_ids, columns):
    """Join pressure records by explicit frame ID, retaining missing rows as None.

    Results follow the requested ID order, not Parquet row positions. Duplicate
    frame identities raise rather than silently assigning the wrong target.
    """
    path = Path(path)
    stat = path.stat()
    stamp = (stat.st_size, stat.st_mtime_ns)
    wanted = set(map(int, frame_ids))
    found = {}
    _, groups = _pressure_file(str(path), stamp)
    for group, ids in enumerate(groups):
        if not wanted.intersection(ids):
            continue
        table = _pressure_group(str(path), stamp, group, tuple(columns))
        for position, frame in enumerate(ids):
            frame = int(frame)
            if frame not in wanted:
                continue
            if frame in found:
                raise ValueError("Duplicate explicit pressure frame ID")
            record = {"frame": frame}
            for name in columns:
                cell = table.column(name)[position]
                record[name] = (
                    None
                    if not cell.is_valid
                    else cell.values.to_numpy(zero_copy_only=False)
                    if hasattr(cell, "values")
                    else cell.as_py()
                )
            found[frame] = record
    return [found.get(int(index)) for index in frame_ids]


def pressure_target(row, indices, paths):
    """Read native ``[F, 1, 224, 224]`` UV pressure with frozen normalization.

    Restore the released pressure-map magnitude, then apply the recorded bounded
    transform ``p / (p + training_scale)``. Missing/invalid annotations remain NaN
    here; measured zeros remain zero. The caller converts support to a mask.
    """
    ids = np.asarray(row["source_frame_ids"], np.int64)[indices]
    records = pressure_rows(
        paths.resolve(row["target_path"]), ids, ["pressure_map", "pressure_map_range"]
    )
    values = np.full((len(ids), 1, 224, 224), np.nan, np.float32)
    for index, record in enumerate(records):
        if record is None or record["pressure_map"] is None or record["pressure_map_range"] is None:
            continue
        scale = np.asarray(record["pressure_map_range"], np.float32)
        if scale.shape != (2,) or not np.isfinite(scale).all() or scale[1] < 0:
            continue
        pressure = np.asarray(record["pressure_map"], np.float32).reshape(224, 224) * scale[1]
        values[index, 0] = np.where(np.isfinite(pressure) & (pressure >= 0), pressure, np.nan)
    normalization = read_json(paths.resolve(row["pressure_normalization"]))["scale_counts"]
    if not np.isfinite(normalization) or normalization <= 0:
        raise ValueError("Invalid training pressure scale")
    return values / (values + normalization)


def read_marker_target(row, indices, paths, marker_root):
    """Return signed ``[F, N, 2]`` marker displacement and ``[F, N]`` validity.

    Displacement is divided by each marker's pixel spacing. Geometry preserves
    normalized reference coordinates, fixed surface IDs and pixel spacing; no
    tactile imagery is opened. Row identity, payload hash and chronological
    mappings must match their extraction receipt. Invalid tracks stay masked.
    """
    if row["source"] == "baihu":
        path = paths.resolve(row["marker_path"])
    else:
        key = json_hash([row["source"], row["split"], row["episode_id"]])[:24]
        path = paths.resolve(marker_root) / row["source"] / row["split"] / key / "tracks.npz"
    receipt = read_json(path.with_name("complete.json"))
    if receipt["row_sha256"] != json_hash(row):
        raise ValueError("Marker native row/frame contract changed")
    if receipt.get("version") == "native-marker-displacement-v2":
        for name, field in [
            ("marker_targets.py", "extractor_sha256"),
            ("marker_data.py", "loader_sha256"),
        ]:
            if field in receipt:
                paths.check_producer(name, receipt[field])
    elif receipt.get("version") != "baihu-marker-displacement-v1":
        raise ValueError("Unknown marker artifact contract")
    data = native_arrays(str(path), receipt["tracks_sha256"])
    frames = row["frames"]
    n = len(data["reference_uv"])
    if data["displacement_px"].shape != (frames, n, 2) or data["valid"].shape != (frames, n):
        raise ValueError("Marker array shape changed")
    if not np.array_equal(data["frame_indices"], np.arange(frames)):
        raise ValueError("Marker chronology changed")
    if "sensor_frame_indices" in data and not np.array_equal(
        data["sensor_frame_indices"][indices], sensor_indices(row, indices)
    ):
        raise ValueError("Marker/scene chronology changed")
    if row["source"] == "baihu" and set(data["surface_id"]) != set(range(4)):
        raise ValueError("Baihu requires four fixed surfaces")
    values = torch.from_numpy(data["displacement_px"][indices] / data["spacing_px"][None, :, None])
    valid = torch.from_numpy(data["valid"][indices].copy())
    geometry = {
        "reference": torch.from_numpy(data["reference_uv"].astype(np.float32)),
        "surface": torch.from_numpy(data["surface_id"].copy()),
        "spacing": torch.from_numpy(data["spacing_px"].copy()),
        "count": n,
        "point_mask": torch.ones(n, dtype=torch.bool),
    }
    return SensorTarget(values, valid, "marker_displacement", geometry, "marker_spacing")


def read_target(row, indices, paths, marker_root, target_kind):
    """Read requested native targets without resizing or filling missing support.

    Source-specific channel/hand order and declared normalization stay inside
    this reader. Numeric releases keep released units; image/grid targets keep
    native dimensions and source normalization. For non-marker targets invalid
    values are stored as zero with false validity, distinct from measured zero.
    """
    if target_kind == "marker_displacement":
        return read_marker_target(row, indices, paths, marker_root)
    if "targets" in row:
        data = native_arrays(str(paths.resolve(row["targets"]["path"])), row["targets"]["sha256"])
        values = data["values"][indices].astype(np.float32)
        valid = data["valid"][indices] & np.isfinite(values)
        return SensorTarget(
            torch.from_numpy(np.where(valid, values, 0.0)),
            torch.from_numpy(valid.copy()),
            "native_numeric",
            {},
            "released_native",
        )
    source = row["source"]
    frames = sensor_indices(row, indices)
    if source == "egopressure":
        array = pressure_target(row, frames, paths)
    elif source == "egotouch":
        with np.load(paths.resolve(row["target_path"]), allow_pickle=False) as data:
            array = np.stack(
                [data[f"{hand}_pressure_grid"][frames] for hand in ("left", "right")], axis=1
            ).astype(np.float32)
    elif source == "egotactile":
        records = json_records(str(paths.resolve(row["target_path"])))
        array = np.asarray(
            [records[int(index)]["RH"]["sensor_256"] for index in frames], np.float32
        ).reshape(len(frames), 1, 1, 256)
        array /= row["target_scale"]
    elif source == "opengraph":
        array = (
            np.asarray(
                np.load(paths.resolve(row["target_path"]), mmap_mode="r")[
                    frames + row["target_start"]
                ]
            )
            .copy()
            .astype(np.float32)[:, :, None, :]
        )
        array /= row["target_scale"]
    elif source == "opentouch":
        with h5py.File(paths.resolve(row["views"]["0"]), "r") as h5:
            array = (
                h5["data"][row["hdf5_demo"]]["right_pressure"][frames]
                .copy()
                .astype(np.float32)[:, None]
            )
        array /= row["target_scale"]
    elif row["target_kind"] == "optical_rgb":
        images = []
        group = ftp_group(row, paths)
        for key in row["target_keys"]:
            value = np.asarray(group["data/" + key][frames + row["target_start"]])
            if value.ndim != 5 or value.shape[-1] != 3 or value.dtype != np.uint8:
                raise ValueError("Invalid native optical sensor array")
            images.extend(value[:, sensor] for sensor in range(value.shape[1]))
        array = (
            np.concatenate([image.transpose(0, 3, 1, 2) for image in images], axis=1).astype(
                np.float32
            )
            / 255.0
        )
    elif "tar_path" in row:
        group = ftp_group(row, paths)
        raw = torch.cat(
            [
                torch.from_numpy(
                    np.asarray(group["data/" + key][frames + row["target_start"]]).copy()
                )
                .float()
                .reshape(len(frames), -1)
                for key in row["target_keys"]
            ],
            dim=1,
        )
        array = (0.5 + 0.5 * torch.tanh(raw / row["target_transform_scale"])).reshape(
            len(frames), -1, 1, 1
        ).numpy() / row["target_scale"]
    else:
        raise ValueError(f"Unsupported native sensor source: {source}")
    if tuple(array.shape[1:]) != tuple(row.get("target_shape", array.shape[1:])):
        raise ValueError("Native target shape changed")
    valid = np.isfinite(array)
    if valid.any() and ((array[valid] < 0).any() or (array[valid] > 1).any()):
        raise ValueError("Normalized native target outside [0,1]")
    return SensorTarget(
        torch.from_numpy(np.where(valid, array, 0.0)),
        torch.from_numpy(valid),
        target_kind,
        {},
        "source_normalized",
    )


def marker_producer_provenance(row, paths, marker_root):
    """Keep historical extractor claims separate from verified code producers."""
    if "marker_path" in row:
        path = paths.resolve(row["marker_path"])
    else:
        key = json_hash([row["source"], row["split"], row["episode_id"]])[:24]
        path = paths.resolve(marker_root) / row["source"] / row["split"] / key / "tracks.npz"
    receipt = read_json(path.with_name("complete.json"))
    result = {
        "receipt": str(path.with_name("complete.json")),
        "receipt_sha256": file_hash(path.with_name("complete.json")),
        "schema": receipt["version"],
        "recorded_hashes": {
            key: value
            for key, value in receipt.items()
            if key in ("extractor_sha256", "tracker_sha256", "loader_sha256")
        },
        "tracks_sha256": receipt["tracks_sha256"],
    }
    if receipt["version"] == "baihu-marker-displacement-v1":
        # This release predates the archived v1 source snapshot. Its reader has
        # always validated native row/track checksums, not replayed extraction.
        result.update(
            archived_raw_producer_bytes_verified=False,
            status="original extractor/tracker bytes unavailable in archived snapshots; recorded hashes retained",
            code_producer_boundary="paired frozen-code producers are independently pinned and checked",
        )
    else:
        for name, field in [
            ("marker_targets.py", "extractor_sha256"),
            ("marker_data.py", "loader_sha256"),
        ]:
            if field in receipt:
                paths.check_producer(name, receipt[field])
        result.update(
            archived_raw_producer_bytes_verified=True,
            status="native marker extractor/loader hashes checked when target is read",
        )
    return result
