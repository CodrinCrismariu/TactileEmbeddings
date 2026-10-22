"""Convert a bounded original-release subset, requiring exact stream alignment."""
import argparse
import json
from pathlib import Path

import av
import h5py
import numpy as np

from .data import PRESSURE, inspect_episode
from .train import check_storage


def convert(episode, destination, size=224):
    if destination.exists():
        inspect_episode(destination)
        return
    with (episode / "jq_pressure.json").open() as f:
        rows = [json.loads(line) for line in f if line.strip()]
    ts = np.array([r["ts"] for r in rows])
    if [r["frame_index"] for r in rows] != list(range(len(rows))):
        raise ValueError("Nonconsecutive frame indices; synchronization needs manual audit")
    with np.load(episode / "pressure_grids.npz", allow_pickle=False) as grids:
        pressure = {key: grids[key] for key in PRESSURE}
        metadata = {key: grids[key].item() for key in grids.files if grids[key].ndim == 0}
    if any(len(p) != len(ts) for p in pressure.values()):
        raise ValueError("Pressure/JSON alignment mismatch")
    check_storage(reserve=len(ts) * size * size * 3 + 1_000_000_000)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".partial")
    with h5py.File(temporary, "w") as out:
        out["timestamps"] = ts
        for key, values in pressure.items():
            out[f"pressure/{key}"] = values
        meta = out.create_group("metadata")
        meta.attrs["source_episode"] = str(episode)
        meta.attrs["target_preprocessing"] = json.dumps(metadata)
        meta.attrs["note"] = "Pilot: original-release grids retained; pressure/bending semantics need audit"
        images = out.create_dataset("images/chest_color", (len(ts), size, size, 3), dtype="u1",
                                    chunks=(1, size, size, 3), compression="gzip", compression_opts=1)
        count = 0
        with av.open(str(episode / "chest.mp4")) as video:
            for count, frame in enumerate(video.decode(video=0), 1):
                if count > len(ts):
                    raise ValueError("RGB has more frames than pressure")
                images[count - 1] = frame.reformat(width=size, height=size, format="rgb24").to_ndarray()
        if count != len(ts):
            raise ValueError(f"RGB/pressure mismatch: {count} vs {len(ts)}")
    inspect_episode(temporary)
    temporary.replace(destination)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="datasets/zhouzhoujy__EgoTouch_pilot")
    parser.add_argument("--output", default="processing/egotouch_pilot")
    parser.add_argument("--split", default="datasets/zhenyuxie-zhzh__EgoTouch_hdf5/split.json")
    args = parser.parse_args()
    original = json.loads(Path(args.split).read_text())
    output = Path(args.output)
    selected = {}
    for split, entries in original.items():
        selected[split] = []
        for entry in entries:
            relative = Path(*Path(entry).parts[-3:])
            episode = Path(args.source) / relative.with_suffix("")
            if (episode / "pressure_grids.npz").exists():
                convert(episode, output / relative)
                selected[split].append(relative.as_posix())
    output.mkdir(parents=True, exist_ok=True)
    (output / "split.json").write_text(json.dumps(selected, indent=2))
    print({s: len(entries) for s, entries in selected.items()})


if __name__ == "__main__":
    main()
