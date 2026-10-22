"""Full-resolution optical targets. Never resize, crop, or use the 32px cache."""
import json
from functools import lru_cache

import av
import numpy as np
import torch
from torch.utils.data import Dataset

from .sensor_data import ftp_group


def native_video_frames(path, indices):
    indices = [int(i) for i in indices]
    if not indices or indices != sorted(set(indices)) or indices[0] < 0:
        raise ValueError("Expected sorted distinct nonnegative frame indices")
    found = {}
    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        rate, origin = float(stream.average_rate), stream.start_time or 0
        container.seek(int(indices[0] / rate / float(stream.time_base)) + origin,
                       stream=stream, backward=True)
        wanted = set(indices)
        for frame in container.decode(video=0):
            index = round((frame.pts - origin) * float(stream.time_base) * rate)
            if index in wanted:
                found[index] = frame.to_ndarray(format="rgb24")
            if index >= indices[-1]:
                break
    if set(found) != wanted:
        raise ValueError(f"Missing requested source frames: {path}")
    return np.stack([found[i] for i in indices])


def native_images(row, indices):
    """Return a list of [T,H,W,3] sensor arrays in stored sensor order."""
    from .marker_storage import marker_only
    if marker_only(row['source']):
        raise ValueError('Tactile images were retired; use the verified marker targets for this source')
    indices = np.asarray(indices, dtype=np.int64)
    if row['source'] == 'freetacman':
        return [native_video_frames(path, indices) for path in row['tactile_videos']]
    group = ftp_group(row)
    images = []
    for key in row.get('target_keys', ['left_tactile_data_gripper', 'right_tactile_data_gripper']):
        value = np.asarray(group['data/' + key][indices + row['target_start']])
        if value.ndim != 5 or value.shape[-1] != 3 or value.dtype != np.uint8:
            raise ValueError(f"Invalid optical sensor: {key}, {value.shape}, {value.dtype}")
        images.extend(value[:, sensor] for sensor in range(value.shape[1]))
    return images


@lru_cache(maxsize=128)
def _shape(serialized_row):
    row = json.loads(serialized_row)
    from .marker_storage import retained_shape
    shape = retained_shape(row)
    if shape is not None:
        return shape
    if row['source'] == 'freetacman':
        sizes = []
        for path in row['tactile_videos']:
            with av.open(path) as container:
                stream = container.streams.video[0]
                sizes.append((stream.height, stream.width))
        channels = 3 * len(sizes)
    else:
        group = ftp_group(row)
        arrays = [group['data/' + k] for k in row.get('target_keys', ['left_tactile_data_gripper', 'right_tactile_data_gripper'])]
        sizes = [tuple(a.shape[2:4]) for a in arrays]
        channels = 3 * sum(a.shape[1] for a in arrays)
    if len(set(sizes)) != 1:
        raise ValueError("Unequal sensor image sizes require separate native decoders")
    return [channels, *sizes[0]]


def native_shape(row):
    return _shape(json.dumps(row, sort_keys=True))


def native_targets(row, indices):
    images = native_images(row, indices)
    target = np.concatenate([image.transpose(0, 3, 1, 2) for image in images], axis=1).astype(np.float32) / 255.
    if list(target.shape[1:]) != native_shape(row):
        raise ValueError("Native target shape changed")
    return target


class NativeOpticalFrames(Dataset):
    """Decode episode-local batches from source archives/videos, without disk copies."""
    def __init__(self, manifest, source, split, stride=2):
        self.episodes = [r for r in json.loads(open(manifest).read())['episodes']
                         if r['source'] == source and r['split'] == split]
        if not self.episodes or stride < 1:
            raise ValueError("Missing split or invalid stride")
        self.stride = stride
        self.shape = native_shape(self.episodes[0])
        if any(native_shape(row) != self.shape for row in self.episodes):
            raise ValueError("Mixed native resolutions require separate decoders")
        self.offsets = np.concatenate(([0], np.cumsum([(r['frames'] + stride - 1) // stride for r in self.episodes])))

    def __len__(self):
        return int(self.offsets[-1])

    def __getitems__(self, indices):
        groups = {}
        for position, index in enumerate(indices):
            episode = int(np.searchsorted(self.offsets, index, side='right') - 1)
            groups.setdefault(episode, []).append((position, (int(index) - self.offsets[episode]) * self.stride))
        result = [None] * len(indices)
        for episode, items in groups.items():
            frames = sorted({int(frame) for _, frame in items})
            values = native_targets(self.episodes[episode], frames)
            lookup = dict(zip(frames, values))
            for position, frame in items:
                target = torch.from_numpy(lookup[int(frame)].copy())
                result[position] = {'target': target, 'valid': torch.ones_like(target, dtype=torch.bool)}
        return result

    def __getitem__(self, index):
        return self.__getitems__([index])[0]
