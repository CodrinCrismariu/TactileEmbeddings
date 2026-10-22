"""Decode requested chronological scene frames with bounded CFR video reads.

Inputs are a video path, ordered frame IDs and visual size. Outputs are RGB
tensors; resizing applies only to scene inputs and never to tactile targets.
"""

import av
import torch


def video_frames(path, indices, size=384):
    """Return ``[T, 3, size, size]`` float RGB in requested chronological order.

    Repeated IDs support padded viewer history. A missing requested frame raises
    instead of substituting another frame or hiding a synchronization failure.
    """
    indices = [int(i) for i in indices]
    if not indices or min(indices) < 0 or indices != sorted(indices):
        raise ValueError("Expected chronological nonnegative video indices")
    found = {}
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        rate = float(stream.average_rate)
        origin = stream.start_time or 0
        container.seek(
            int(indices[0] / rate / float(stream.time_base)) + origin, stream=stream, backward=True
        )
        wanted = set(indices)
        for frame in container.decode(video=0):
            index = round((frame.pts - origin) * float(stream.time_base) * rate)
            if index in wanted:
                array = frame.reformat(width=size, height=size, format="rgb24").to_ndarray()
                found[index] = torch.from_numpy(array.copy()).permute(2, 0, 1).float() / 255.0
            if index >= indices[-1]:
                break
    if set(found) != set(indices):
        raise ValueError(f"Missing requested video frames: {path}")
    return torch.stack([found[index] for index in indices])
