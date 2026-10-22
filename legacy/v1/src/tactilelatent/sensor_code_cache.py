"""Pinned tactile teachers and resumable, native-frame float32 code caches.

No DINO, reconstruction decoder, gradients, or spatial resizing is used here.
"""
import argparse
import fcntl
import json
import shutil
from collections import defaultdict
from pathlib import Path
import time

import av
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

from .aligned_frames import (MANIFEST, file_hash, json_hash, manifest_signature,
                             targets, write_json)
from .sensor_autoencoder import make_autoencoder


SOURCES = ("egotouch", "opengraph", "egotactile", "opentouch", "freetacman",
           "ftp_unit", "ftp_unit_bimanual", "ftp_vla_touch", "ftp_vitamin")
ROOT = "processing/sensor_code_cache_v1"
REGISTRY = "processing/joint_alignment/teachers.json"


def pin_teachers(registry=REGISTRY, search="processing/runs/sensor_autoencoders_native_v2"):
    """Copy completed winners once; never follow live best.pt files during training."""
    registry = Path(registry)
    if registry.exists():
        saved = json.loads(registry.read_text())
        verify_teachers(saved)
        return saved
    winners = json.loads((Path(search) / "best_by_source.json").read_text())
    teachers = {}
    for source in SOURCES:
        checkpoint = (Path("processing/runs/egotouch_ae_search_v4/trials/wide64_code256/best.pt")
                      if source == "egotouch" else Path(winners[source]["checkpoint"]))
        history = [json.loads(line) for line in checkpoint.with_name("metrics.jsonl").read_text().splitlines()]
        minimum = 200 if source == "egotouch" else 100
        if history[-1]["epoch"] < minimum:
            raise ValueError(f"Teacher finalist still running: {source}")
        digest = file_hash(checkpoint)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        config = saved["model_config"]
        destination = registry.parent / "teachers" / f"{source}_{digest[:16]}.pt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            temporary = destination.with_suffix(".tmp")
            shutil.copyfile(checkpoint, temporary)
            if file_hash(temporary) != digest:
                raise ValueError("Teacher changed during pinning")
            temporary.replace(destination)
        teachers[source] = {"checkpoint": str(destination.resolve()), "sha256": digest,
                            "model_config": config, "code_dim": config["latent_dim"],
                            "target_shape": config.get("target_shape", [2, 21, 21]),
                            "original_checkpoint": str(checkpoint),
                            "completed_epochs": history[-1]["epoch"]}
    result = {"schema_version": 1, "sources": teachers}
    write_json(registry, result)
    return result


def verify_teachers(registry):
    if registry.get("schema_version") != 1 or not registry.get("sources"):
        raise ValueError("Invalid teacher registry")
    for source, info in registry["sources"].items():
        if file_hash(info["checkpoint"]) != info["sha256"]:
            raise ValueError(f"Pinned teacher changed: {source}")


class EpisodeChunks(IterableDataset):
    """Each episode goes to exactly one worker; queues contain bounded chunks."""
    def __init__(self, episodes, completed=(), chunk_frames=32):
        self.episodes = episodes
        self.completed = set(completed)
        self.chunk_frames = chunk_frames

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        worker_id, workers = (worker.id, worker.num_workers) if worker else (0, 1)
        for number, row in enumerate(self.episodes):
            if number % workers != worker_id or number in self.completed:
                continue
            if row["source"] == "freetacman":
                # Open each video ONCE, not once per small training microbatch.
                from contextlib import ExitStack
                with ExitStack() as stack:
                    containers = [stack.enter_context(av.open(path)) for path in row["tactile_videos"]]
                    for container in containers:
                        stream = container.streams.video[0]
                        stream.thread_type = "AUTO"
                        stream.thread_count = 2
                    decoders = [container.decode(video=0) for container in containers]
                    chunk, start, count = [], 0, 0
                    for frames in zip(*decoders):
                        for frame, container in zip(frames, containers):
                            stream = container.streams.video[0]
                            position = round((frame.pts - (stream.start_time or 0)) *
                                             float(stream.time_base) * float(stream.average_rate))
                            if position != count:
                                raise ValueError("Video timestamps disagree with chronological frame index")
                        chunk.append(np.concatenate([frame.to_ndarray(format="rgb24").transpose(2, 0, 1)
                                                     for frame in frames], axis=0))
                        count += 1
                        if len(chunk) == self.chunk_frames and count < row["frames"]:
                            array = np.stack(chunk).astype(np.float32)
                            array /= 255.
                            yield number, start, array, False
                            start = count
                            chunk.clear()
                    if count != row["frames"]:
                        raise ValueError(f"Missing or excess native video frames: {row['episode_id']}")
                    if chunk:
                        array = np.stack(chunk).astype(np.float32)
                        array /= 255.
                        yield number, start, array, True
            else:
                # Compact sensors fit one episode in RAM. Decompress NPZ/HDF5
                # once per episode rather than once per 32-frame chunk.
                compact = targets(row, np.arange(row["frames"])) if row["target_kind"] != "optical_rgb" else None
                for start in range(0, row["frames"], self.chunk_frames):
                    indices = np.arange(start, min(row["frames"], start + self.chunk_frames))
                    yield number, start, compact[indices] if compact is not None else targets(row, indices), indices[-1] + 1 == row["frames"]


def cache_signature(manifest, registry, source, data_signature=None):
    return {**(data_signature or manifest_signature(manifest)),
            "teacher_sha256": registry["sources"][source]["sha256"],
            "code_builder_sha256": file_hash(__file__), "dtype": "float32",
            "source": source, "encoder_mode": "eval-fp32-no-grad"}


def train_statistics(codes, valid, episodes, offsets, chunk=65536):
    total = np.zeros(codes.shape[1], np.float64)
    squared = total.copy()
    count = 0
    for episode, row in enumerate(episodes):
        if row["split"] != "train":
            continue
        lo, hi = offsets[episode:episode + 2]
        for start in range(lo, hi, chunk):
            end = min(hi, start + chunk)
            selected = np.asarray(codes[start:end])[np.asarray(valid[start:end], dtype=bool)].astype(np.float64)
            total += selected.sum(0)
            squared += np.square(selected).sum(0)
            count += len(selected)
    if not count:
        raise ValueError("No valid training frames for teacher-code statistics")
    mean = total / count
    std = np.sqrt(np.maximum(0., squared / count - mean**2)).clip(min=1e-6)
    return {"mean": mean.tolist(), "std": std.tolist(), "train_frames": count}


def build_source(manifest, registry, source, root=ROOT, device="cuda", workers=2,
                 chunk_frames=32, batch_size=None, data_signature=None):
    if source not in registry["sources"] or workers < 0 or chunk_frames < 1:
        raise ValueError("Unknown teacher or invalid cache-builder settings")
    folder = Path(root) / source
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if file_hash(registry["sources"][source]["checkpoint"]) != registry["sources"][source]["sha256"]:
            raise ValueError("Pinned teacher changed before cache access")
        signature = cache_signature(manifest, registry, source, data_signature)
        index_path = folder / "index.json"
        if index_path.exists():
            index = json.loads(index_path.read_text())
            if index["signature"] != signature:
                raise ValueError(f"Stale code cache; choose a fresh root: {source}")
            return index
        episodes = [r for r in json.loads(Path(manifest).read_text())["episodes"] if r["source"] == source]
        info = registry["sources"][source]
        if any(row["target_shape"] != info["target_shape"] for row in episodes):
            raise ValueError(f"Teacher/native target shape mismatch: {source}")
        offsets = np.concatenate(([0], np.cumsum([r["frames"] for r in episodes]))).tolist()
        dimension = info["code_dim"]
        needed = offsets[-1] * (4 * dimension + 1)
        if shutil.disk_usage(folder).free < needed + 512 * 2**20:
            raise RuntimeError("Insufficient free disk for complete code cache")
        progress_path = folder / "progress.json"
        progress = json.loads(progress_path.read_text()) if progress_path.exists() else {
            "signature": signature, "completed_episodes": []}
        if progress["signature"] != signature:
            raise ValueError("Partial cache provenance differs")
        completed = set(progress["completed_episodes"])
        partials = [folder / "codes.partial.npy", folder / "valid.partial.npy"]
        shapes = [(offsets[-1], dimension), (offsets[-1],)]
        arrays = []
        for path, shape, dtype in zip(partials, shapes, (np.float32, np.bool_)):
            # Recovery also tolerates a crash between the two final renames.
            final = path.with_name(path.name.replace(".partial", ""))
            existing = path if path.exists() else final if final.exists() else None
            if existing is None:
                if completed:
                    raise ValueError("Partial code storage is missing")
                array = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
            else:
                array = np.load(existing, mmap_mode="r+")
            if array.shape != shape or array.dtype != dtype:
                raise ValueError("Partial code-cache shape/dtype mismatch")
            arrays.append(array)
        codes, valid = arrays
        if file_hash(info["checkpoint"]) != info["sha256"]:
            raise ValueError("Teacher hash differs before encoding")
        checkpoint = torch.load(info["checkpoint"], map_location="cpu", weights_only=False)
        model = make_autoencoder(checkpoint["model_config"])
        model.load_state_dict(checkpoint["model"])
        model.eval().requires_grad_(False)
        model.encoder.to(device)  # Decoder stays on CPU and is never invoked.
        del checkpoint
        batch_size = batch_size or (8 if source == "freetacman" else 32 if dimension == 256 else 512)
        if batch_size < 1:
            raise ValueError("Positive encoder batch size required")
        kwargs = {"prefetch_factor": 1, "multiprocessing_context": "spawn"} if workers else {}
        loader = DataLoader(EpisodeChunks(episodes, completed, chunk_frames), batch_size=None,
                            num_workers=workers, **kwargs)
        began = time.monotonic()
        with torch.inference_mode():
            for number, start, values, last in loader:
                number, start = int(number), int(start)
                values = torch.as_tensor(values)
                if tuple(values.shape[1:]) != tuple(info["target_shape"]):
                    raise ValueError("Streaming loader changed native target shape")
                if episodes[number]["target_kind"] == "optical_rgb":
                    # Native uint8 RGB /255 is finite. Avoid boolean-indexing
                    # hundreds of MB of pixels just to check its known range.
                    finite = torch.ones(values.shape, dtype=torch.bool)
                    frame_valid = torch.ones(len(values), dtype=torch.bool)
                    if values.amin() < 0 or values.amax() > 1:
                        raise ValueError("Native optical normalization outside [0,1]")
                else:
                    finite = torch.isfinite(values)
                    frame_valid = finite.flatten(1).any(1)
                    if ((values[finite] < 0) | (values[finite] > 1)).any():
                        raise ValueError("Native target normalization outside [0,1]")
                where = offsets[number] + start
                for local in range(0, len(values), batch_size):
                    target = values[local:local + batch_size].to(device, dtype=torch.float32)
                    mask = finite[local:local + batch_size].to(device)
                    code = model.encode(target, mask).float().cpu().numpy()
                    if not np.isfinite(code).all():
                        raise ValueError("Nonfinite teacher codes")
                    codes[where + local:where + local + len(code)] = code
                valid[where:where + len(values)] = frame_valid.numpy()
                if last:
                    codes.flush(); valid.flush()
                    completed.add(number)
                    progress["completed_episodes"] = sorted(completed)
                    write_json(progress_path, progress)
                    if len(completed) % 25 == 0 or len(completed) == len(episodes):
                        print(json.dumps({"event": "code_cache", "source": source,
                            "completed_episodes": len(completed), "episodes": len(episodes),
                            "seconds": time.monotonic() - began}), flush=True)
        if len(completed) != len(episodes):
            raise ValueError("Not every episode was cached")
        statistics = train_statistics(codes, valid, episodes, offsets)
        for partial in partials:
            if partial.exists():
                partial.replace(partial.with_name(partial.name.replace(".partial", "")))
        index = {"signature": signature, "episodes": episodes, "offsets": offsets,
                 "frames": offsets[-1], "code_dim": dimension,
                 "target_shape": info["target_shape"], "statistics": statistics}
        write_json(index_path, index)
        return index


class CodeCache:
    def __init__(self, root, source, expected=None):
        self.folder = Path(root) / source
        self.index = json.loads((self.folder / "index.json").read_text())
        if expected is not None and self.index["signature"] != expected:
            raise ValueError(f"Stale cached teacher codes: {source}")
        self.offsets = {row["episode_id"]: self.index["offsets"][i]
                        for i, row in enumerate(self.index["episodes"])}
        self.lengths = {row["episode_id"]: row["frames"] for row in self.index["episodes"]}
        self.codes = self.valid = None

    def read(self, episode_id, frames, normalized=True):
        frames = np.asarray(frames, dtype=np.int64)
        if (frames < 0).any() or (frames >= self.lengths[episode_id]).any():
            raise ValueError("Cached frame index outside episode")
        if self.codes is None:
            self.codes = np.load(self.folder / "codes.npy", mmap_mode="r")
            self.valid = np.load(self.folder / "valid.npy", mmap_mode="r")
        positions = self.offsets[episode_id] + frames
        codes = np.asarray(self.codes[positions]).copy()
        if normalized:
            stats = self.index["statistics"]
            codes = (codes - np.asarray(stats["mean"], np.float32)) / np.asarray(stats["std"], np.float32)
        return torch.from_numpy(codes), torch.from_numpy(np.asarray(self.valid[positions]).copy())

    def valid_frames(self, episode_id, frames):
        if self.valid is None:
            self.valid = np.load(self.folder / "valid.npy", mmap_mode="r")
        return np.asarray(self.valid[self.offsets[episode_id] + np.asarray(frames)])

    def __getstate__(self):
        return {**self.__dict__, "codes": None, "valid": None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=MANIFEST)
    parser.add_argument("--registry", default=REGISTRY)
    parser.add_argument("--output", default=ROOT)
    parser.add_argument("--sources", nargs="+", default=list(SOURCES))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--chunk-frames", type=int, default=32)
    parser.add_argument("--audit", default="processing/joint_alignment/audit.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    registry = json.loads(Path(args.registry).read_text())
    verify_teachers(registry)
    signature = manifest_signature(args.manifest)
    audit = json.loads(Path(args.audit).read_text())
    if audit.get("signature") != signature or not audit.get("passed"):
        raise ValueError("A passing audit of this manifest and these source files is required")
    for source in args.sources:
        build_source(args.manifest, registry, source, args.output, args.device,
                     args.workers, args.chunk_frames, data_signature=signature)
    write_json(Path(args.output) / "registry.json", {"teachers": registry, "data_signature": signature})


if __name__ == "__main__":
    main()
