"""Build float32, memory-mapped tactile-only caches once for AE ablations."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import Dataset

from .sensor_data import load_targets, SENSORS


def write_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(path)


def cache_signature(manifest, source, stride):
    return {"schema_version": 1, "source": source, "stride": stride,
            "manifest_sha256": hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
            "loader_sha256": hashlib.sha256(b"".join(Path(__file__).with_name(name).read_bytes()
                for name in ("sensor_data.py", "joint_data.py", "media.py"))).hexdigest(),
            "dtype": "float32", "invalid": "NaN", "normalization": "joint-native-v1"}


def build_cache(manifest, source, root="processing/sensor_ae_cache_v1", stride=2, chunk_frames=128):
    from .joint_all_config import ALL_TARGETS
    if ALL_TARGETS[source][0] == "optical_rgb":
        raise ValueError("Optical targets must use the full-resolution direct loader; reduced caches are disabled")
    if stride < 1 or chunk_frames < 1:
        raise ValueError("Stride and chunk size must be positive")
    folder = Path(root) / source
    folder.mkdir(parents=True, exist_ok=True)
    lock = (folder / "build.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX)
    signature = cache_signature(manifest, source, stride)
    index_path = folder / "index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        if index["signature"] != signature:
            raise ValueError(f"Cache signature differs: {folder}; choose a new cache root")
        return index
    rows = [r for r in json.loads(Path(manifest).read_text())["episodes"]
            if r["source"] == source and r["split"] in ("train", "val")]
    ids = [r["episode_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate episodes across splits: {source}")
    if not rows:
        raise ValueError(f"No episodes for {source}")
    shape = rows[0]["target_shape"]
    if any(r["target_shape"] != shape for r in rows):
        raise ValueError(f"Mixed sensor layouts in {source}")
    progress_path = folder / "progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {"signature": signature, "splits": {}}
    if progress["signature"] != signature:
        raise ValueError("Partial cache has a different signature")
    began = time.monotonic()
    index = {"signature": signature, "target_shape": shape, "target_kind": rows[0]["target_kind"], "splits": {}}
    for split in ("train", "val"):
        episodes = [r for r in rows if r["split"] == split]
        lengths = [(r["frames"] + stride - 1) // stride for r in episodes]
        if not episodes:
            raise ValueError(f"Missing {source} {split} split")
        offsets = np.concatenate(([0], np.cumsum(lengths))).tolist()
        state = progress["splits"].setdefault(split, {"completed_episodes": 0})
        final = folder / f"{split}.npy"
        partial = folder / f"{split}.partial.npy"
        if not final.exists():
            if not partial.exists():
                state["completed_episodes"] = 0
            data = (np.load(partial, mmap_mode="r+") if partial.exists() else
                    np.lib.format.open_memmap(partial, mode="w+", dtype=np.float32,
                                              shape=(offsets[-1], *shape)))
            if data.shape != (offsets[-1], *shape):
                raise ValueError("Partial cache shape differs")
            for episode in range(state["completed_episodes"], len(episodes)):
                row = episodes[episode]
                frames = np.arange(0, row["frames"], stride)
                for first in range(0, len(frames), chunk_frames):
                    selected = frames[first:first + chunk_frames]
                    data[offsets[episode] + first:offsets[episode] + first + len(selected)] = load_targets(row, selected)
                data.flush()
                state["completed_episodes"] = episode + 1
                write_json(progress_path, progress)
                if episode % 25 == 0 or episode + 1 == len(episodes):
                    print(json.dumps({"event": "cache", "source": source, "split": split,
                                      "episodes": episode + 1, "total_episodes": len(episodes),
                                      "frames": offsets[episode + 1], "seconds": time.monotonic() - began}), flush=True)
            del data
            partial.replace(final)
        index["splits"][split] = {"episodes": episodes, "offsets": offsets, "frames": offsets[-1]}
    write_json(index_path, index)
    return index


class CachedSensorFrames(Dataset):
    def __init__(self, cache, source, split):
        self.folder = Path(cache) / source
        self.index = json.loads((self.folder / "index.json").read_text())
        selected = self.index["splits"][split]
        self.episodes, self.offsets = selected["episodes"], np.asarray(selected["offsets"])
        self.path, self.data = self.folder / f"{split}.npy", None

    def __len__(self):
        return int(self.offsets[-1])

    def __getitem__(self, index):
        if self.data is None:
            self.data = np.load(self.path, mmap_mode="r")
        target = torch.from_numpy(np.asarray(self.data[index]).copy())
        valid = torch.isfinite(target)
        return {"target": torch.where(valid, target, 0.), "valid": valid}

    def __getstate__(self):
        return {**self.__dict__, "data": None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="processing/joint_all/manifest.json")
    parser.add_argument("--sources", nargs="+", default=list(SENSORS))
    parser.add_argument("--cache", default="processing/sensor_ae_cache_v1")
    parser.add_argument("--stride", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(4)
    for source in args.sources:
        build_cache(args.manifest, source, args.cache, args.stride)


if __name__ == "__main__":
    main()
