"""Episode-balanced banks and code-only datasets using the canonical frame map."""
from collections import defaultdict
import hashlib
import math
import random

import numpy as np
import torch
from torch.utils.data import Sampler

from .aligned_frames import targets
from .alignment_data import AlignedClipDataset
from .fgw_mixer import HELDOUT
from .joint_data import SourceBatchSampler


def source_seed(seed, epoch, source):
    return int.from_bytes(hashlib.sha256(f"{seed}:{epoch}:{source}".encode()).digest()[:8], "big")


class BankDataset(AlignedClipDataset):
    def __getitem__(self, key):
        index, anchor = key if isinstance(key, tuple) else (key, False)
        result = super().__getitem__(index)
        result.update(window_index=index, alignment_anchor=anchor)
        return result


class CodeOnlyDataset(AlignedClipDataset):
    """Read standardized cached codes or native numeric targets, never videos."""
    def __getitem__(self, key):
        index, anchor = key if isinstance(key, tuple) else (key, False)
        episode, start = self.windows[index]
        row = self.episodes[episode]
        if row["source"] == HELDOUT:
            raise ValueError("MotionTrans cannot enter the tactile teacher")
        frames = np.arange(start, start + self.span, self.stride)
        if row["source"] in self.caches:
            value, valid = self.caches[row["source"]].read(row["episode_id"], frames)
        else:
            array = targets(row, frames).reshape(len(frames), -1)
            mask = np.isfinite(array)
            # Numeric projections require all their one/two channels present.
            valid = torch.from_numpy(mask.all(1))
            value = torch.from_numpy(np.where(mask, array, 0.)).float()
        return {"source": row["source"], "episode_id": row["episode_id"],
                "frame_indices": torch.from_numpy(frames), "value": value,
                "valid": valid, "window_index": index, "alignment_anchor": anchor}


def code_point(dataset, index):
    # Calling this explicitly bypasses video decoding even for BankDataset.
    item = CodeOnlyDataset.__getitem__(dataset, index)
    frames = item["valid"].nonzero().flatten()
    if not len(frames):
        return None
    return int(frames[-1]), item["value"][frames[-1]]


def sample_banks(dataset, size, seed, epoch):
    if HELDOUT in dataset.indices_by_source:
        raise ValueError("Held-out source in alignment dataset")
    banks = {}
    for source, indices in sorted(dataset.indices_by_source.items()):
        rng = random.Random(source_seed(seed, epoch, source))
        episodes = defaultdict(list)
        for index in indices:
            episodes[dataset.windows[index][0]].append(index)
        for choices in episodes.values():
            rng.shuffle(choices)
        selected, keys = [], list(episodes)
        while keys and len(selected) < size:
            rng.shuffle(keys)
            remaining = []
            for episode in keys:
                choices = episodes[episode]
                while choices:
                    index = choices.pop()
                    if code_point(dataset, index) is not None:
                        selected.append(index)
                        break
                if choices:
                    remaining.append(episode)
                if len(selected) == size:
                    break
            keys = remaining
        if not selected:
            raise ValueError(f"No valid bank states: {source}")
        banks[source] = selected
    return banks


class BankSampler(Sampler):
    """Keep ordinary source budgets and replace exactly 25% with bank states.

    Tuple keys distinguish deliberate anchors from ordinary clips that happen
    to be bank members. Sampling is identical across mixer conditions.
    """
    def __init__(self, dataset, batch_size, seed, max_windows_per_source, fraction=.25):
        self.base = SourceBatchSampler(dataset, batch_size, seed=seed,
            max_windows_per_source=max_windows_per_source, sampling_mode="cycle")
        self.dataset, self.seed, self.fraction = dataset, seed, fraction
        self.banks, self.epoch = {}, 0

    def set_epoch(self, epoch, banks=None):
        self.epoch = epoch
        self.base.set_epoch(epoch)
        if banks is not None:
            self.banks = banks

    def __len__(self):
        return len(self.base)

    def __iter__(self):
        batches = list(self.base)
        by_source = defaultdict(list)
        for b, batch in enumerate(batches):
            source = self.dataset.episodes[self.dataset.windows[batch[0]][0]]["source"]
            by_source[source].extend((b, k) for k in range(len(batch)))
        result = [[(index, False) for index in batch] for batch in batches]
        for source, positions in by_source.items():
            rng = random.Random(source_seed(self.seed, self.epoch, source))
            rng.shuffle(positions)
            pool = list(self.banks[source]); rng.shuffle(pool)
            for k, (b, j) in enumerate(positions[:math.ceil(self.fraction * len(positions))]):
                result[b][j] = (pool[k % len(pool)], True)
        return iter(result)


def final_valid_indices(batch):
    valid = batch["code_valid"] if "code_valid" in batch else (
        batch["valid"] if "valid" in batch else batch["target_mask"].flatten(2).all(-1))
    positions = torch.arange(valid.shape[1], device=valid.device).expand_as(valid)
    last = torch.where(valid, positions, -1).amax(1)
    return last.clamp_min(0), last >= 0
