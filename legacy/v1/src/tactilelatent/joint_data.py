"""A common image-input/native-tactile-target contract for six sources.

Every item has RGB [8,3 camera slots,3,224,224], a camera mask [8,3], a
source-specific normalized tactile target [8,C,H,W], and a valid-cell mask.
The output target shape is NOT forced to be the same across sources: batches
must contain one source, and a future joint model must select that source's
decoder and loss. SourceBatchSampler provides homogeneous batches.
"""

import io
import hashlib
import json
import random
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import zarr
from PIL import Image
from torch.utils.data import Dataset, Sampler

from .media import video_frames
from .tar_zarr import IndexedTarZarrStore


@lru_cache(maxsize=32)
def _json_records(path):
    return json.loads(Path(path).read_text())


@lru_cache(maxsize=64)
def _ftp_group_cached(tar_path, index_path, archive_group, zarr_name, storage_stamp):
    store = IndexedTarZarrStore(tar_path, index_path, f"{archive_group}/{zarr_name}")
    return zarr.open_group(store=store, mode="r")


def _ftp_group(tar_path, index_path, archive_group, zarr_name):
    # A verified RGB migration can atomically replace the mixed archive/index.
    # Never reuse offsets from an earlier physical version of that archive.
    from pathlib import Path
    tar, index = Path(tar_path).stat(), Path(index_path).stat()
    return _ftp_group_cached(tar_path,index_path,archive_group,zarr_name,
                             (tar.st_size,tar.st_mtime_ns,index.st_size,index.st_mtime_ns))


def _from_uint8_images(array):
    """Convert [T,H,W,3] uint8 Zarr images to [T,3,H,W] float in [0,1]."""
    return torch.from_numpy(np.asarray(array).copy()).permute(0, 3, 1, 2).float() / 255.0


class JointClipDataset(Dataset):
    """Lazy 8-frame clips from the verified six-source manifest.

    Use split='train' or 'val'. EgoTouch's published test splits are retained
    in the manifest and can be selected explicitly, but are not mixed into
    training. Each window uses eight source frames at stride two and windows
    advance by the full 15-frame span, avoiding overlap within an episode.
    """

    def __init__(self, manifest="processing/joint_six/manifest.json", split="train",
                 sources=None, frames=8, stride=2):
        if frames < 1 or stride < 1:
            raise ValueError("frames and stride must be positive")
        self.frames = frames
        self.stride = stride
        self.span = (frames - 1) * stride + 1
        payload = json.loads(Path(manifest).read_text())
        if payload["schema_version"] != 1:
            raise ValueError("Unsupported joint manifest schema")
        selected = set(sources) if sources else None
        self.episodes = [row for row in payload["episodes"]
                         if (split is None or row["split"] == split)
                         and (selected is None or row["source"] in selected)]
        self.windows = []
        self.indices_by_source = defaultdict(list)
        for episode_number, row in enumerate(self.episodes):
            for start in range(0, row["frames"] - self.span + 1, self.span):
                window_index = len(self.windows)
                self.windows.append((episode_number, start))
                self.indices_by_source[row["source"]].append(window_index)
        if not self.windows:
            raise ValueError(f"No valid joint clips for split={split}, sources={sources}")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        episode_number, start = self.windows[index]
        row = self.episodes[episode_number]
        indices = np.arange(start, start + self.span, self.stride)
        rgb = torch.zeros(self.frames, 3, 3, 224, 224)
        view_mask = torch.zeros(self.frames, 3, dtype=torch.bool)
        source = row["source"]

        if source == "egotouch":
            for slot, path in row["views"].items():
                rgb[:, int(slot)] = video_frames(path, indices)
                view_mask[:, int(slot)] = True
            with np.load(row["target_path"]) as pressure:
                target = np.stack([pressure["left_pressure_grid"][indices],
                                   pressure["right_pressure_grid"][indices]], axis=1)
            target = torch.from_numpy(target.copy()).float()

        elif source == "egotactile":
            rgb[:, 0] = video_frames(row["views"]["0"], indices)
            view_mask[:, 0] = True
            records = _json_records(row["target_path"])
            target = torch.tensor([records[int(i)]["RH"]["sensor_256"] for i in indices],
                                  dtype=torch.float32).reshape(self.frames, 1, 1, 256)

        elif source == "opentouch":
            with h5py.File(row["views"]["0"], "r") as h5:
                demo = h5["data"][row["hdf5_demo"]]
                frames = []
                for i in indices:
                    jpeg = bytes(demo["rgb_images_jpeg"][int(i)])
                    with Image.open(io.BytesIO(jpeg)) as image:
                        image = image.convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)
                        frames.append(torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1))
                rgb[:, 0] = torch.stack(frames).float() / 255.0
                target = torch.from_numpy(demo["right_pressure"][indices].copy()).float().unsqueeze(1)
            view_mask[:, 0] = True

        elif source == "opengraph":
            video_indices = indices + row["video_start"]
            rgb[:, 0] = video_frames(row["views"]["0"], video_indices)
            view_mask[:, 0] = True
            tactile = np.load(row["target_path"], mmap_mode="r")
            target = torch.from_numpy(np.asarray(tactile[indices + row["target_start"]]).copy())
            target = target.unsqueeze(2)  # [T,2,1,80]

        elif source == "freetacman":
            rgb[:, 1] = video_frames(row["views"]["1"], indices)
            view_mask[:, 1] = True
            left, right = [video_frames(path, indices, size=32)
                           for path in row["tactile_videos"]]
            target = torch.cat((left, right), dim=1)  # [T,6,32,32], already [0,1]

        elif source == "ftp_unit_bimanual":
            group = _ftp_group(row["tar_path"], row["tar_index"], "Unit_Bimanual",
                               row["zarr_name"])
            positions = indices + row["target_start"]
            for slot, key in row["views"].items():
                rgb[:, int(slot)] = _from_uint8_images(group[f"data/{key}"][positions])
                view_mask[:, int(slot)] = True
            optical = []
            for key in ("left_tactile_data_gripper", "right_tactile_data_gripper"):
                # Zarr 2 supports the frame-index array alone; select its
                # singleton sensor dimension after reading those frames.
                image = group[f"data/{key}"][positions][:, 0]  # [T,240,320,3]
                resized = F.interpolate(_from_uint8_images(image), size=(32, 32),
                                        mode="bilinear", align_corners=False)
                optical.append(resized)
            target = torch.cat(optical, dim=1)  # [T,6,32,32], already [0,1]

        elif "archive_group" in row:
            group = _ftp_group(row["tar_path"], row["tar_index"],
                               row["archive_group"], row["zarr_name"])
            positions = indices + row["target_start"]
            for slot, key in row["views"].items():
                image = _from_uint8_images(group[f"data/{key}"][positions])
                if image.shape[-2:] != (224, 224):
                    image = F.interpolate(image, size=(224, 224), mode="bilinear",
                                          align_corners=False)
                rgb[:, int(slot)] = image
                view_mask[:, int(slot)] = True
            if row["target_kind"] == "optical_rgb":
                optical = []
                for key in row["target_keys"]:
                    frames = np.asarray(group[f"data/{key}"][positions])
                    if frames.ndim != 5 or frames.shape[-1] != 3:
                        raise ValueError(f"Unexpected optical target shape: {source}:{key}")
                    time, sensors, height, width, _ = frames.shape
                    image = _from_uint8_images(frames.reshape(time * sensors, height, width, 3))
                    image = F.interpolate(image, size=(32, 32), mode="bilinear",
                                          align_corners=False)
                    optical.append(image.reshape(time, sensors * 3, 32, 32))
                target = torch.cat(optical, dim=1)
            else:
                readings = [torch.from_numpy(np.asarray(group[f"data/{key}"][positions]).copy())
                            .float().reshape(self.frames, -1) for key in row["target_keys"]]
                raw = torch.cat(readings, dim=1)
                # Preserve negative values and avoid hard clipping. These
                # transforms are source-local, not shared physical units.
                target = (.5 + .5 * torch.tanh(raw / row["target_transform_scale"]))
                target = target.reshape(self.frames, -1, 1, 1)

        else:
            raise ValueError(f"Unknown source {source}")

        # Numeric tactile sensors have source-specific ADC ranges. Optical
        # target video readers already return [0,1] RGB, so do not divide twice.
        if row["target_kind"] != "optical_rgb":
            target = target / row["target_scale"]
        valid = torch.isfinite(target)
        if not valid.any() or (target[valid] < 0).any() or (target[valid] > 1).any():
            raise ValueError(f"Out-of-range target in {source}:{row['episode_id']}")
        target = torch.where(valid, target, 0.0)
        if list(target.shape[1:]) != row["target_shape"]:
            raise ValueError(f"Target shape mismatch in {source}:{row['episode_id']}")
        return {
            "rgb": rgb, "view_mask": view_mask,
            "target": target, "target_mask": valid,
            "source": source, "target_kind": row["target_kind"],
            "episode_id": row["episode_id"],
        }


class SourceBatchSampler(Sampler):
    """Mix sources over an epoch, but keep each mini-batch one source.

    This prevents PyTorch's default collate from trying to stack unlike
    targets such as [2,21,21] pressure and [6,32,32] optical tactile RGB.
    Every window is seen once unless drop_last=True.
    """

    def __init__(self, dataset, batch_size, seed=0, drop_last=False,
                 max_windows_per_source=None, sampling_mode="random"):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if max_windows_per_source is not None and max_windows_per_source < 1:
            raise ValueError("max_windows_per_source must be positive")
        if sampling_mode not in ("random", "cycle"):
            raise ValueError("sampling_mode must be random or cycle")
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.drop_last = drop_last
        self.max_windows_per_source = max_windows_per_source
        self.sampling_mode = sampling_mode
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        batches = []
        for source, indices in self.dataset.indices_by_source.items():
            limit = self.max_windows_per_source
            if self.sampling_mode == "cycle" and limit is not None and len(indices) > limit:
                # Walk a deterministic shuffled permutation without repeats
                # until all source clips have been visited. On wrap, make a
                # new permutation. Epoch-number mapping survives a restart.
                position = self.epoch * limit
                shuffled = []
                while len(shuffled) < limit:
                    cycle, offset = divmod(position, len(indices))
                    seed_bytes = hashlib.sha256(
                        f"{self.seed}:{source}:{cycle}".encode()).digest()[:8]
                    source_rng = random.Random(int.from_bytes(seed_bytes, "big"))
                    permutation = list(indices)
                    source_rng.shuffle(permutation)
                    take = min(limit - len(shuffled), len(indices) - offset)
                    shuffled.extend(permutation[offset:offset + take])
                    position += take
            else:
                shuffled = list(indices)
                rng.shuffle(shuffled)
                if limit is not None:
                    shuffled = shuffled[:limit]
            for start in range(0, len(shuffled), self.batch_size):
                batch = shuffled[start:start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)
        rng.shuffle(batches)
        return iter(batches)

    def __len__(self):
        size = self.batch_size
        lengths = [min(len(indices), self.max_windows_per_source)
                   if self.max_windows_per_source is not None else len(indices)
                   for indices in self.dataset.indices_by_source.values()]
        if self.drop_last:
            return sum(length // size for length in lengths)
        return sum((length + size - 1) // size for length in lengths)
