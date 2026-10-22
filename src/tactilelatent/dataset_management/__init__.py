"""Discovery, verification and explicit preparation of versioned native data."""

from .joint import JointDataset, build_dataset, collate_samples
from .registry import SourceRegistry, build_sources
from .sampling import SourceBatchSampler

__all__ = [
    "build_sources",
    "SourceRegistry",
    "build_dataset",
    "JointDataset",
    "collate_samples",
    "SourceBatchSampler",
]
from .api import manage as manage
