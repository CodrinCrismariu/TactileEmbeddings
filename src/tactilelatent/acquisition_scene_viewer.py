"""Scene inputs shared by prediction and attention for expansion datasets."""
import numpy as np
import torch

from .acquisition_data import _targets
from .acquisition_viewer import mapped_frame
from .media import video_frames


def joint_row(row):
    """Resolve FEEL's episode-specific layout without changing its manifest."""
    if row['source'] != 'feel':
        return row
    source = {(5,): 'feel_lab', (12,): 'feel_kitchen'}.get(tuple(row['targets']['shape'][1:]))
    if source is None:
        raise ValueError('Unknown FEEL sensor layout')
    return {**row, 'source': source}


def mapped_scene_rgb(row, indices, views, size=384):
    """Decode logical frames using the same offsets/maps as native training.

    Repeated frames are allowed only for repeated logical history positions,
    such as left-padding at an episode boundary. Missing mappings stay errors.
    """
    indices = np.asarray(indices, dtype=np.int64)
    if (indices.ndim != 1 or not len(indices) or (indices < 0).any()
            or (indices >= row['frames']).any() or (np.diff(indices) < 0).any()):
        raise ValueError('Scene history must be chronological and inside its episode')
    if views not in ('ego', 'all'):
        raise ValueError('Choose ego or all scene cameras')
    if views == 'ego' and '0' not in row['views']:
        raise ValueError('This episode has no ego camera; choose all available scene cameras')
    native = _targets(row['targets']['path'], row['targets']['sha256']) if 'targets' in row else None
    changing = np.diff(indices) > 0
    if native is not None and 'timestamps' in native:
        times = native['timestamps'][indices]
        if not np.isfinite(times).all() or (np.diff(times)[changing] <= 0).any():
            raise ValueError('Nonchronological tactile history')
    rgb = torch.zeros(len(indices), 3, 3, size, size)
    mask = torch.zeros(len(indices), 3, dtype=torch.bool)
    for slot, view in row['views'].items():
        if views == 'ego' and slot != '0':
            continue
        camera = int(slot)
        if camera not in range(3):
            raise ValueError('Unknown scene camera slot')
        if isinstance(view, str):
            path, mapped = view, indices
        else:
            if native is None:
                raise ValueError('Mapped scene cameras require the native frame artifact')
            path = view['path']
            mapped = np.asarray([mapped_frame(row, native, int(i), slot) for i in indices])
        if (np.diff(mapped)[changing] <= 0).any():
            raise ValueError('Nonchronological camera history')
        rgb[:, camera] = video_frames(path, mapped.tolist(), size=size)
        mask[:, camera] = True
    if not mask.any():
        raise ValueError('This episode has no scene camera')
    return rgb, mask


def add_attention_sources(models, episodes):
    """Advertise only displayed sensors that have a head in that checkpoint."""
    available = {row['source'] for row in episodes}
    for model in models:
        if model.get('attention_supported'):
            model['attention_sources'] = sorted(available.intersection(model['source_shapes']))
