"""Choose source-homogeneous batches using reproducible epoch schedules.

Inputs are dataset source indices, a seed, an epoch and optional per-source cap.
Outputs are batches of dataset indices; sampling changes no target or frame map.
"""

import hashlib
import random

from torch.utils.data import Sampler


class SourceBatchSampler(Sampler):
    """Mix sources over an epoch, but keep each mini-batch one source.

    This prevents PyTorch's default collate from trying to stack unlike
    targets such as [2,21,21] pressure and [6,32,32] optical tactile RGB.
    Without a per-source cap, each window appears once per epoch before optional
    remainder dropping. A cap limits each source's contribution; cycle mode walks
    a deterministic shuffled source permutation across epochs before reshuffling.
    """

    def __init__(
        self,
        dataset,
        batch_size,
        seed=0,
        drop_last=False,
        max_windows_per_source=None,
        sampling_mode="random",
    ):
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
        """Select the epoch schedule without consuming process-global RNG state."""
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
                    seed_bytes = hashlib.sha256(f"{self.seed}:{source}:{cycle}".encode()).digest()[
                        :8
                    ]
                    source_rng = random.Random(int.from_bytes(seed_bytes, "big"))
                    permutation = list(indices)
                    source_rng.shuffle(permutation)
                    take = min(limit - len(shuffled), len(indices) - offset)
                    shuffled.extend(permutation[offset : offset + take])
                    position += take
            else:
                shuffled = list(indices)
                rng.shuffle(shuffled)
                if limit is not None:
                    shuffled = shuffled[:limit]
            for start in range(0, len(shuffled), self.batch_size):
                batch = shuffled[start : start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)
        rng.shuffle(batches)
        return iter(batches)

    def __len__(self):
        size = self.batch_size
        lengths = [
            min(len(indices), self.max_windows_per_source)
            if self.max_windows_per_source is not None
            else len(indices)
            for indices in self.dataset.indices_by_source.values()
        ]
        if self.drop_last:
            return sum(length // size for length in lengths)
        return sum((length + size - 1) // size for length in lengths)
