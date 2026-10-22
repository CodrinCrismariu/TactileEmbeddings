"""EgoTouch data loaders.

RawEgoTouch reads the official MP4 videos and pressure NPZ files lazily and
is the ONLY loader used by the active training loop. EgoTouch is an older
HDF5 adapter retained for previous preprocessing workflows.
"""
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .media import video_frames, video_info

VIEWS = ("chest_color", "left_color", "right_color")
PRESSURE = ("left_pressure_grid", "right_pressure_grid")


def resolve_splits(root):
    root = Path(root)
    raw = json.loads((root / "split.json").read_text())
    splits = {}
    owners = {}
    for split, entries in raw.items():
        splits[split] = []
        for entry in entries:
            # Upstream splits use absolute paths from the authors' machine.
            relative = Path(*Path(entry).parts[-3:])
            key = relative.as_posix()
            if key in owners:
                raise ValueError(f"Duplicate episode {key}: {owners[key]} and {split}")
            owners[key] = split
            splits[split].append(root / relative)
    return splits


def inspect_episode(path):
    with h5py.File(path, "r") as f:
        required = ["images/chest_color", "timestamps"] + [f"pressure/{k}" for k in PRESSURE]
        for key in required:
            if key not in f:
                raise ValueError(f"{path}: missing {key}")
        n = len(f["timestamps"])
        ts = np.asarray(f["timestamps"], dtype=np.float64)
        if ts.shape != (n,) or n < 2 or not np.isfinite(ts).all() or np.any(np.diff(ts) <= 0):
            raise ValueError(f"{path}: timestamps must be finite, increasing seconds")
        if not 0.001 <= np.median(np.diff(ts)) <= 1:
            raise ValueError(f"{path}: timestamp scale must be verified as seconds")
        for key in required:
            if len(f[key]) != n:
                raise ValueError(f"{path}: modality length mismatch at {key}")
        for key in PRESSURE:
            if f[f"pressure/{key}"].shape[1:] != (21, 21):
                raise ValueError(f"{path}: unexpected pressure shape")
        for key in VIEWS:
            if f"images/{key}" in f:
                ds = f[f"images/{key}"]
                if len(ds) != n or ds.ndim != 4 or ds.shape[-1] != 3 or ds.dtype != np.uint8:
                    raise ValueError(f"{path}: expected aligned uint8 RGB at {key}")
        return n, ts


class EgoTouch(Dataset):
    def __init__(self, root, split="train", frames=8, stride=2, size=224,
                 episode_limit=None, window_step=8, max_gap=0.15):
        if min(frames, stride, size, window_step) < 1:
            raise ValueError("Frames, stride, size, and window step must be positive")
        self.frames, self.stride, self.size = frames, stride, size
        splits = resolve_splits(root)
        paths = splits[split]
        missing = [str(p) for p in paths if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"{split}: {len(missing)}/{len(paths)} HDF5 episodes missing. First: {missing[0]}")
        self.windows = []
        for path in paths[:episode_limit]:
            n, ts = inspect_episode(path)
            history = (frames - 1) * stride
            for end in range(history, n, window_step):
                indices = np.arange(end - history, end + 1, stride)
                if frames > 1 and np.diff(ts[indices]).max() > max_gap:
                    continue
                self.windows.append((path, indices, ts[indices] - ts[end]))
        if not self.windows:
            raise ValueError(f"No valid windows in {split}")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        path, indices, times = self.windows[index]
        images = torch.zeros(self.frames, 3, 3, self.size, self.size)
        view_mask = torch.zeros(self.frames, 3, dtype=torch.bool)
        # Open inside each call: no inherited HDF5 handles across workers.
        with h5py.File(path, "r") as f:
            for v, key in enumerate(VIEWS):
                if f"images/{key}" not in f:
                    continue
                for t, frame in enumerate(f[f"images/{key}"][indices]):
                    rgb = np.array(Image.fromarray(frame).resize((self.size, self.size), Image.Resampling.BICUBIC))
                    images[t, v] = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255
                view_mask[:, v] = True
            target = torch.from_numpy(np.stack([f[f"pressure/{key}"][indices[-1]] for key in PRESSURE])).float()
        valid = torch.isfinite(target)
        if not valid.any() or ((target[valid] < 0) | (target[valid] > 1)).any():
            raise ValueError(f"{path}: invalid normalized pressure; audit preprocessing")
        return {"rgb": images, "view_mask": view_mask, "times": torch.tensor(times, dtype=torch.float32),
                "target": torch.nan_to_num(target), "target_mask": valid, "episode": str(path)}


def _raw_split_paths(root, split):
    root = Path(root)
    raw = json.loads((root / "split.json").read_text())
    paths = []
    for entry in raw[split]:
        relative = Path(*Path(entry).parts[-3:])
        paths.append(root / relative.with_suffix(""))
    return paths


class RawEgoTouch(Dataset):
    """Lazy image-only-input loader: MP4 plus pressure supervision.

    Clips follow the non-overlapping 8-frame/stride-2 protocol and retain
    pressure targets at every timestep, rather than final-frame only.

    One item contains RGB [T,3 cameras,3 channels,H,W], pressure
    [T,2 hands,21,21], and masks. The model receives RGB/view_mask only;
    pressure/target_mask are for calculating the training loss.
    """
    def __init__(self, root, split="train", frames=8, stride=2, size=224,
                 episode_limit=None, strict=False):
        self.frames, self.stride, self.size = frames, stride, size
        # Eight frames sampled every two source frames occupy indices s..s+14:
        # a span of 15 source-frame positions. Advance by the full span so
        # successive training windows do not overlap.
        span = (frames - 1) * stride + 1
        self.windows = []
        self.missing_episodes = []
        for path in _raw_split_paths(root, split)[:episode_limit]:
            required = [path / name for name in ("chest.mp4", "left.mp4", "right.mp4",
                                                  "pressure_grids.npz")]
            if not all(p.is_file() for p in required):
                if strict:
                    raise FileNotFoundError(f"Incomplete EgoTouch episode: {path}")
                self.missing_episodes.append(str(path))
                continue
            # Only inspect the small target array and video metadata now.
            # Actual video frames are decoded on demand in __getitem__.
            with np.load(path / "pressure_grids.npz") as pressure:
                n = len(pressure["left_pressure_grid"])
                if len(pressure["right_pressure_grid"]) != n:
                    raise ValueError(f"Pressure length mismatch: {path}")
            video_counts = [video_info(p)[0] for p in required[:3]]
            if any(count != n for count in video_counts):
                raise ValueError(f"Video/pressure length mismatch: {path}: {video_counts} vs {n}")
            for start in range(0, n - span + 1, span):
                self.windows.append((path, np.arange(start, start + span, stride)))
        if not self.windows:
            raise ValueError(f"No complete EgoTouch clips in {split}")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        path, indices = self.windows[index]
        views = []
        for name in ("chest.mp4", "left.mp4", "right.mp4"):
            views.append(video_frames(path / name, indices, size=self.size))
        with np.load(path / "pressure_grids.npz") as pressure:
            target = np.stack([pressure["left_pressure_grid"][indices],
                               pressure["right_pressure_grid"][indices]], axis=1)
        target = torch.from_numpy(target).float()
        # Invalid pressure cells are replaced by zero for safe tensor math;
        # target_mask ensures they never contribute to the loss or metrics.
        valid = torch.isfinite(target)
        if not valid.any() or ((target[valid] < 0) | (target[valid] > 1)).any():
            raise ValueError(f"Invalid normalized pressure: {path}")
        result = {"rgb": torch.stack(views, dim=1),
                "view_mask": torch.ones(self.frames, 3, dtype=torch.bool),
                "target": torch.nan_to_num(target), "target_mask": valid,
                "episode": str(path), "frame_indices": torch.from_numpy(indices)}
        return result
