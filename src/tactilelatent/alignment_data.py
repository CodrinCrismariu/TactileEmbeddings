"""Chronological scene clips paired with native targets or cached teacher codes."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .aligned_frames import CONTRACT, manifest_signature, scene_rgb, sensor_indices, targets
from .sensor_code_cache import CodeCache, cache_signature


class AlignedClipDataset(Dataset):
    def __init__(self, manifest, split="train", sources=None, frames=8, stride=2,
                 code_cache=None, registry=None, include_targets=True, data_signature=None, rgb_size=224):
        if frames < 1 or stride < 1:
            raise ValueError("Positive clip length and stride required")
        payload = json.loads(Path(manifest).read_text())
        if payload.get("schema_version") != 2 or payload.get("frame_contract") != CONTRACT:
            raise ValueError("Use a native chronological schema-2 manifest")
        selected = set(sources) if sources is not None else None
        self.episodes = [r for r in payload["episodes"] if (split is None or r["split"] == split)
                         and (selected is None or r["source"] in selected)]
        self.frames, self.stride = frames, stride
        if rgb_size < 1:
            raise ValueError("Positive RGB size required")
        self.rgb_size = rgb_size
        self.span = (frames - 1) * stride + 1
        self.include_targets = include_targets
        self.caches = {}
        if code_cache is not None:
            if registry is None:
                raise ValueError("Teacher registry required with code cache")
            signature = data_signature or manifest_signature(manifest)
            for source in {r["source"] for r in self.episodes} & set(registry["sources"]):
                self.caches[source] = CodeCache(code_cache, source,
                    cache_signature(manifest, registry, source, signature))
        self.windows, self.indices_by_source = [], defaultdict(list)
        for episode, row in enumerate(self.episodes):
            # VLA's mapping is mandatory even when extracting only scene features.
            sensor_indices(row, [0])
            if row["source"] == "ftp_vla_touch" and sorted(row["sensor_frame_order"]) != list(range(row["frames"])):
                raise ValueError("VLA chronological frame mapping is not a permutation")
            for start in range(0, row["frames"] - self.span + 1, self.span):
                if row["source"] in self.caches:
                    valid = self.caches[row["source"]].valid_frames(row["episode_id"],
                        np.arange(start, start + self.span, stride))
                    if not valid.any():
                        continue
                index = len(self.windows)
                self.windows.append((episode, start))
                self.indices_by_source[row["source"]].append(index)
        if not self.windows:
            raise ValueError("No valid aligned clips")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        episode, start = self.windows[index]
        row = self.episodes[episode]
        indices = np.arange(start, start + self.span, self.stride)
        rgb, mask = scene_rgb(row, indices, size=self.rgb_size)
        item = {"rgb": rgb, "view_mask": mask, "source": row["source"],
                "episode_id": row["episode_id"], "frame_indices": torch.from_numpy(indices),
                "sensor_frame_indices": torch.from_numpy(sensor_indices(row, indices))}
        if row["source"] in self.caches:
            item["code_target"], item["code_valid"] = self.caches[row["source"]].read(row["episode_id"], indices)
        if self.include_targets or (self.caches and row["source"] not in self.caches):
            value = torch.from_numpy(targets(row, indices))
            valid = torch.isfinite(value)
            item.update(target=torch.where(valid, value, 0.), target_mask=valid)
        return item


def evaluation_dataset(manifest, **kwargs):
    """Dispatch historical evaluations without silently changing their data."""
    payload = json.loads(Path(manifest).read_text())
    if payload.get("schema_version") == 2:
        return AlignedClipDataset(manifest, **kwargs)
    from .joint_data import JointClipDataset
    kwargs.pop("include_targets", None)
    return JointClipDataset(manifest, **kwargs)
