"""Check native targets, converted camera masks, and indexed video offsets.

This is safe during conversion: only committed ledger entries are inspected.
No training, original-data writes, or cache substitution is performed.
"""
import hashlib
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import av
import numpy as np
import torch

from tactilelatent import storage
from tactilelatent.aligned_frames import scene_rgb, targets, write_json

ROOT = Path('processing/storage_v1')


def main():
    torch.set_num_threads(4)
    rows = json.loads((ROOT / 'manifest.json').read_text())['episodes']
    lookup = {(r['source'], r['episode_id']): r for r in rows}
    native = []
    for check in json.loads((ROOT / 'native_target_checks.json').read_text()):
        value = targets(lookup[(check['source'], check['episode_id'])], np.array(check['indices']))
        assert list(value.shape) == check['target_shape'], check['source']
        assert hashlib.sha256(value.tobytes()).hexdigest() == check['target_sha256'], check['source']
        native.append(dict(source=check['source'], unchanged=True, shape=list(value.shape)))

    converted = []
    ledger = storage.registry()
    seen = set()
    for row in rows:
        entry = ledger.get('episodes', {}).get(storage.episode_key(row))
        if entry is None or row['source'] in seen:
            continue
        seen.add(row['source'])
        indices = np.unique([0, row['frames'] // 2, row['frames'] - 1])
        rgb, mask = scene_rgb(row, indices, size=384)
        assert list(rgb.shape) == [len(indices), 3, 3, 384, 384]
        expected = torch.zeros(len(indices), 3, dtype=torch.bool)
        for slot in row['views']:
            expected[:, int(slot)] = True
        assert torch.equal(mask, expected), row['source']
        assert (rgb[~mask] == 0).all() and torch.isfinite(rgb).all()
        converted.append(dict(source=row['source'], frames=indices.tolist(), slots=list(entry['views'])))

    # Adjacent episodes share one underlying camera video. Its frame offset
    # must be applied before decoding, while absent cameras stay masked.
    with tempfile.TemporaryDirectory(dir=ROOT) as directory:
        video = Path(directory) / 'offset.mp4'
        with av.open(str(video), 'w') as container:
            stream = container.add_stream('libx264', rate=24)
            stream.width = stream.height = 384
            stream.pix_fmt = 'yuv420p'
            stream.options = {'crf': '18', 'preset': 'fast'}
            stream.codec_context.thread_count = 2
            for index in range(16):
                frame = av.VideoFrame.from_ndarray(np.full((384, 384, 3), index * 12, np.uint8), format='rgb24')
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        with av.open(str(video)) as container:
            assert container.streams.video[0].frames == 16
            assert float(container.streams.video[0].average_rate) == 24
        row = dict(source='offset_fixture', episode_id='episode', frames=12)
        stat = video.stat()
        entry = dict(views={'2': dict(path=str(video), stat=[stat.st_size, stat.st_mtime_ns], frame_offset=3)})
        with patch.object(storage, 'registry', return_value={'episodes': {storage.episode_key(row): entry}}):
            rgb, mask = storage.converted_scene(row, np.array([0, 4, 8]), 384)
        levels = rgb[:, 2].mean((1, 2, 3)) * 255
        assert torch.allclose(levels, torch.tensor([36., 84., 132.]), atol=4.)
        assert mask[:, 2].all() and not mask[:, :2].any()
        assert (rgb[:, :2] == 0).all()
    result = dict(passed=True, native_targets=native, converted_sources=converted,
                  synthetic_offset_camera_mask_and_24fps=True)
    write_json(ROOT / 'validation.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
