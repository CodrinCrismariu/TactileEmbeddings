"""Bounded decoding and a common visual sample contract."""
import av
import numpy as np
import torch
from PIL import Image


def video_info(path):
    with av.open(str(path)) as f:
        s = f.streams.video[0]
        rate = float(s.average_rate)
        if not s.frames or rate <= 0:
            raise ValueError(f'Unknown frame count/rate: {path}')
        return s.frames, rate


def video_frames(path, indices, size=224):
    """Read CFR frames at exact frame indices; fails rather than repeats frames."""
    indices = [int(i) for i in indices]
    result = {}
    with av.open(str(path)) as f:
        s = f.streams.video[0]
        s.thread_type = 'AUTO'
        rate = float(s.average_rate)
        origin = s.start_time or 0
        f.seek(int(indices[0] / rate / float(s.time_base)) + origin, stream=s, backward=True)
        wanted = set(indices)
        for frame in f.decode(video=0):
            idx = round((frame.pts - origin) * float(s.time_base) * rate)
            if idx in wanted:
                a = frame.reformat(width=size,height=size,format='rgb24').to_ndarray()
                result[idx] = torch.from_numpy(a.copy()).permute(2,0,1).float()/255.
            if idx >= indices[-1]:
                break
    if set(result) != set(indices):
        raise ValueError(f'Missing requested video frames: {path}')
    return torch.stack([result[i] for i in indices])


def image_tensor(array, size=32):
    rgb = np.asarray(array, dtype=np.uint8)
    a = np.array(Image.fromarray(rgb).resize((size,size),Image.Resampling.BILINEAR))
    return torch.from_numpy(a).permute(2,0,1).float()/255.


def make_sample(rgb_views, times, target, valid, episode, size=224):
    images = torch.zeros(len(times),3,3,size,size)
    mask = torch.zeros(len(times),3,dtype=torch.bool)
    for role, data in rgb_views.items():
        images[:,role] = data
        mask[:,role] = True
    if not valid.any():
        raise ValueError(f'No valid targets: {episode}')
    return {'rgb':images,'view_mask':mask,'times':torch.tensor(times,dtype=torch.float32),
            'target':torch.where(valid,target,0.),'target_mask':valid,'episode':episode}


def window_indices(n, frames, stride, limit=4):
    first = (frames-1)*stride
    if first >= n:
        return []
    return [np.arange(end-first,end+1,stride) for end in
            np.unique(np.linspace(first,n-1,min(limit,n-first),dtype=int))]
