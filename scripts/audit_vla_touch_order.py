#!/usr/bin/env python3
"""Match raw numbered GelSight JPEGs to the authors' processed H5 frames.

Read-only diagnosis: downloads byte ranges, never edits datasets or training.
Pixel matching tries RGB and BGR because OpenCV saves H5 images in BGR order.
"""
import argparse
import hashlib
import io
import json
import re
import urllib.request
import zipfile
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


class RemoteZip(io.RawIOBase):
    def __init__(self, url, size):
        self.url, self.size, self.pos = url, size, 0
        self.bytes_read = 0
        self.blocks = []

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        return self.pos

    def fetch(self, start, end):
        for lo, hi, data in self.blocks:
            if lo <= start and end <= hi:
                return data[start - lo:end - lo + 1]
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
        with urllib.request.urlopen(request, timeout=45) as response:
            if response.status != 206:
                raise ValueError("Server did not honor HTTP range")
            data = response.read(end - start + 2)
        if len(data) != end - start + 1:
            raise ValueError("Incorrect range length")
        self.bytes_read += len(data)
        self.blocks.append((start, end, data))
        return data

    def read(self, count=-1):
        if count < 0:
            count = self.size - self.pos
        count = min(count, self.size - self.pos)
        if count == 0:
            return b""
        if count > 32 * 1024 * 1024:
            raise ValueError("Unexpected large download; use a local archive")
        data = self.fetch(self.pos, self.pos + count - 1)
        self.pos += len(data)
        return data


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def compare(archive, h5_path, episode, output, remote=None):
    pattern = re.compile(rf"/episode_{episode}/gelsight/gel_(\d+)\.jpg$")
    files = [(int(pattern.search(info.filename).group(1)), info)
             for info in archive.infolist() if pattern.search(info.filename)]
    files.sort(key=lambda entry: entry[0])
    if not files:
        raise ValueError("No matching raw episode")
    if remote:
        # Coalesce adjacent archive members to avoid dozens of HTTP requests.
        first = min(info.header_offset for _, info in files)
        last = max(info.header_offset + info.compress_size + len(info.filename.encode()) + 4096
                   for _, info in files)
        if last - first < 32 * 1024 * 1024:
            remote.fetch(first, min(last, remote.size - 1))
    with h5py.File(h5_path) as h5:
        processed = h5["gelsight/gelsight"][:]
    lookup = {}
    for index, image in enumerate(processed):
        lookup.setdefault(digest(image), []).append(index)
    records = []
    native_frames = []
    for number, info in files:
        image = np.asarray(Image.open(io.BytesIO(archive.read(info))).convert("RGB"))
        rgb_matches = lookup.get(digest(image), [])
        bgr_matches = lookup.get(digest(image[:, :, ::-1]), [])
        records.append({"raw_frame": number, "filename": info.filename,
                        "processed_rgb_matches": rgb_matches, "processed_bgr_matches": bgr_matches})
        native_frames.append(image)
    exact = all(len(r["processed_rgb_matches"]) == 1 for r in records)
    exact_bgr = all(len(r["processed_bgr_matches"]) == 1 for r in records)
    mapping = [r["processed_rgb_matches"][0] for r in records] if exact else (
        [r["processed_bgr_matches"][0] for r in records] if exact_bgr else None)
    result = {"raw_episode": episode, "h5_path": str(h5_path),
              "raw_frames": len(files), "processed_frames": len(processed),
              "exact_rgb_bijection": exact and len(set(mapping)) == len(processed),
              "exact_bgr_bijection": exact_bgr and len(set(mapping)) == len(processed),
              "chronological_to_processed_index": mapping,
              "out_of_order_positions": (sum(i != j for i, j in enumerate(mapping)) if mapping else None),
              "records": records, "downloaded_bytes": remote.bytes_read if remote else None}
    output.mkdir(parents=True, exist_ok=True)
    (output / "raw_order_comparison.json").write_text(json.dumps(result, indent=2))
    np.save(output / "raw_chronological_rgb.npy", np.stack(native_frames))
    print(json.dumps({k: v for k, v in result.items() if k != "records"}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="https://huggingface.co/datasets/AllenBi21/VLA-Touch/resolve/main/raw_data/water_cup.zip")
    parser.add_argument("--size", type=int, default=1286454671)
    parser.add_argument("--local-zip", type=Path)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--h5", type=Path, default=Path("processing/audits/vla_touch_native_dynamics/upstream_watercup_episode_0.h5"))
    parser.add_argument("--output", type=Path, default=Path("processing/audits/vla_touch_native_dynamics"))
    args = parser.parse_args()
    remote = None if args.local_zip else RemoteZip(args.url, args.size)
    with zipfile.ZipFile(args.local_zip or remote) as archive:
        compare(archive, args.h5, args.episode, args.output, remote)


if __name__ == "__main__":
    main()
