"""Construct chronological clip windows and collate uniform native contracts.

Inputs are registered source adapters and explicit split/source selections.
Outputs are ``ClipSample`` or source-homogeneous ``ClipBatch`` objects; native
target dimensions remain intact, with padding only for variable marker counts.
"""

from collections import defaultdict

import torch
from torch.utils.data import Dataset

from ..contracts import ClipBatch, SensorTarget, Supervision, VisualClip


class JointDataset(Dataset):
    """Index canonical eight-image, stride-two windows across selected sources.

    Each window spans 15 logical frames and supervises its final frame. Episode
    discovery order and supported end-frame order are retained. The held-out
    source requires explicit evaluation permission; excluded sources are rejected.
    """

    def __init__(
        self, registry, split="train", sources=None, include_targets=True, allow_heldout=False
    ):
        self.registry = registry
        self.split = split
        self.include_targets = include_targets
        requested = list(registry.data_config["sources"]) if sources is None else list(sources)
        if len(requested) != len(set(requested)) or set(requested) - set(registry.sources):
            raise ValueError("Unknown/duplicate source requested")
        if registry.excluded_sources & set(requested):
            raise ValueError("Explicitly removed source cannot enter this dataset")
        if registry.heldout_source in requested and not allow_heldout:
            raise ValueError("Held-out source requires explicit evaluation permission")
        self.episodes = []
        self.episode_sources = []
        self.windows = []
        self.indices_by_source = defaultdict(list)
        selected = set(requested)
        for source, row in registry.episode_rows:
            if source not in selected or (split is not None and row["split"] != split):
                continue
            episode = len(self.episodes)
            self.episodes.append(row)
            self.episode_sources.append(source)
            for end in registry[source].eligible_ends(row):
                if end < 14 or end >= row["frames"] or (int(end) - 14) % 15:
                    raise ValueError(
                        "Prepared clip schedule is outside canonical chronological windows"
                    )
                self.indices_by_source[source].append(len(self.windows))
                self.windows.append((episode, int(end) - 14))
        if not self.windows or set(self.indices_by_source) != selected:
            raise ValueError("Every requested source needs supported clips in the selected split")
        self.frames = 8
        self.stride = 2
        self.span = 15
        self.rgb_size = registry.data_config.get("rgb_size", 384)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        episode, start = self.windows[index]
        return self.registry[self.episode_sources[episode]].read_clip(
            self.episodes[episode], start + 14, self.include_targets
        )


def _stack_optional(values):
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError("Mixed presence of a batch field")
    return torch.stack(values)


def collate_samples(samples):
    """Stack one source/split batch while preserving native targets and masks.

    Marker points are padded to the batch maximum with false support/point masks;
    their reference coordinates, surfaces and spacing follow the same padding.
    Other sensor layouts must stack directly. Missing optional fields cannot be
    silently mixed with present fields, and measured zeros remain valid.
    """
    if (
        not samples
        or len({item.source for item in samples}) != 1
        or len({item.split for item in samples}) != 1
    ):
        raise ValueError("Expected nonempty homogeneous source/split batch")
    if (
        len({item.target.kind for item in samples}) != 1
        or len({item.supervision.kind for item in samples}) != 1
    ):
        raise ValueError("Sensor representation differs inside a batch")
    geometry = {}
    values = [item.target.values for item in samples]
    masks = [item.target.valid for item in samples]
    if samples[0].target.kind == "marker_displacement":
        counts = [len(item.target.geometry["reference"]) for item in samples]
        maximum = max(counts)
        padded_values = []
        padded_valid = []
        for item, count in zip(samples, counts):
            value = torch.zeros(
                (*item.target.values.shape[:-2], maximum, 2), dtype=item.target.values.dtype
            )
            mask = torch.zeros((*item.target.valid.shape[:-1], maximum), dtype=torch.bool)
            value[..., :count, :] = item.target.values
            mask[..., :count] = item.target.valid
            padded_values.append(value)
            padded_valid.append(mask)
        values = padded_values
        masks = padded_valid
        geometry = {"count": torch.tensor(counts, dtype=torch.int64)}
        for name in ("reference", "surface", "spacing", "point_mask"):
            arrays = []
            for item, count in zip(samples, counts):
                value = item.target.geometry[name]
                shape = (maximum, *value.shape[1:])
                out = (
                    torch.ones(shape, dtype=value.dtype)
                    if name == "spacing"
                    else torch.zeros(shape, dtype=value.dtype)
                )
                out[:count] = value
                arrays.append(out)
            geometry[name] = torch.stack(arrays)
    else:
        if any(item.target.geometry for item in samples):
            raise ValueError("Nonmarker geometry requires an explicit collator contract")
    visual = VisualClip(
        torch.stack([item.visual.rgb for item in samples]),
        torch.stack([item.visual.view_mask for item in samples]),
        torch.stack([item.visual.history_frame_indices for item in samples]),
        _stack_optional([item.visual.rgb_frame_indices for item in samples]),
        _stack_optional([item.visual.camera_roles for item in samples]),
        _stack_optional([item.visual.timestamps for item in samples]),
    )
    target = SensorTarget(
        torch.stack(values),
        torch.stack(masks),
        samples[0].target.kind,
        geometry,
        samples[0].target.units,
    )
    supervision = Supervision(
        _stack_optional([item.supervision.values for item in samples]),
        _stack_optional([item.supervision.valid for item in samples]),
        samples[0].supervision.kind,
        "original",
        {"episodes": [item.supervision.artifact for item in samples]},
    )
    return ClipBatch(
        visual,
        target,
        supervision,
        samples[0].source,
        [item.episode_id for item in samples],
        samples[0].split,
        torch.stack([item.frame_indices for item in samples]),
        torch.stack([item.sensor_frame_indices for item in samples]),
        [item.provenance for item in samples],
    )


def build_dataset(registry, split="train", sources=None, include_targets=True, allow_heldout=False):
    """Construct the shared dataset with explicit held-out access control."""
    return JointDataset(registry, split, sources, include_targets, allow_heldout)
