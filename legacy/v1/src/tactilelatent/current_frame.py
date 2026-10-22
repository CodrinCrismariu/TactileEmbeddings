"""Eight past/current RGB frames, one current tactile target, no future input."""
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset

from .aligned_frames import scene_rgb, sensor_indices, targets
from .marker_targets import MARKER_SOURCES


def causal_clip_indices(frame, length, frames=8, stride=2):
    """Left-pad short histories with the first frame; never read future frames.

    The returned output position is zero: a current-frame model has one query,
    even though its target input occupies the final slot of its RGB clip.
    """
    if length<1 or not 0<=frame<length or frames<1 or stride<1:
        raise ValueError('Invalid current frame or history')
    return np.maximum(frame-(np.arange(frames-1,-1,-1)*stride),0),0


class CurrentFrameCodeClips(Dataset):
    """Reuse canonical clip windows and cached codes, decoding only one target.

    Sensor dimensions, marker geometry and chronological mappings are retained.
    Only clips whose current sensor code is unsupported are omitted; history
    validity cannot make an invalid current target into a valid training label.
    """
    def __init__(self,dataset):
        if not hasattr(dataset,'code_caches'):
            raise ValueError('Current-frame training requires the verified sensor-code dataset')
        self.dataset=dataset
        self.base=dataset.dataset
        self.episodes=dataset.episodes
        self.windows=[];self.indices_by_source=defaultdict(list)
        for episode,start in dataset.windows:
            row=self.episodes[episode];cache=dataset.code_caches.get(row['source'])
            current=np.array([start+self.base.span-1])
            if cache is not None and not cache.valid_frames(row['episode_id'],current).all():
                continue
            self.indices_by_source[row['source']].append(len(self.windows))
            self.windows.append((episode,start))
        if set(self.indices_by_source)!=set(dataset.indices_by_source):
            raise ValueError('A source has no supported current-frame targets')

    def __len__(self):
        return len(self.windows)

    def __getitem__(self,index):
        episode,start=self.windows[index];row=self.episodes[episode]
        history=np.arange(start,start+self.base.span,self.base.stride)
        current=history[-1:]
        rgb,mask=scene_rgb(row,history,size=self.base.rgb_size)
        item=dict(rgb=rgb,view_mask=mask,source=row['source'],episode_id=row['episode_id'],
            history_frame_indices=torch.from_numpy(history),frame_indices=torch.from_numpy(current),
            sensor_frame_indices=torch.from_numpy(sensor_indices(row,current)))
        if row['source'] in MARKER_SOURCES:
            item.update(self.dataset.cache.read(row,current))
        else:
            value=torch.from_numpy(targets(row,current));valid=torch.isfinite(value)
            item.update(target=torch.where(valid,value,0.),target_mask=valid)
        if row['source'] in self.dataset.code_caches:
            item['code_target'],item['code_valid']=self.dataset.code_caches[row['source']].read(row['episode_id'],current)
        return item
