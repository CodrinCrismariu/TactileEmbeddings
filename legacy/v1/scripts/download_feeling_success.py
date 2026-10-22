"""Acquire all Feeling of Success grasps as a compact, evaluation-only copy.

The raw ZIP would exceed the project's 700 GB data+processing cap. Download
one compressed HDF5 ZIP member at a time, verify its source CRC, retain native
scene RGB losslessly and all scalar metadata, then remove only this script's
temporary raw member. Tactile images are not retained: the frozen encoder
consumes ordinary scene RGB. Never change a training manifest or trainer.
"""

import argparse
import io
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import time
import urllib.request
import zipfile
import zlib

import h5py
import numpy as np

from tactilelatent.frozen_probe import sha256


ROOT = Path(__file__).resolve().parents[1]
SOURCE = "https://opara.zih.tu-dresden.de/items/d5c56810-fc26-4b53-8201-acb75ff055ac"
API = "https://opara.zih.tu-dresden.de/server/api/core/bitstreams/4ef8383c-bcae-4bf6-9bcd-1242488624b5"
CAP = 700_000_000_000
RESERVE = 5_000_000_000
RGB_FIELDS = ("kinectA_rgb_before", "kinectA_rgb_during", "kinectA_rgb_after")


def emit(event, **fields):
    print(json.dumps({"event": event, **fields}), flush=True)


def atomic_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def usage():
    """Apparent bytes conservatively include directories and partial files."""
    result = subprocess.check_output(["du", "-sb", str(ROOT / "datasets"),
                                      str(ROOT / "processing")], text=True)
    return sum(int(line.split()[0]) for line in result.splitlines())


def check_budget(additional=0):
    used = usage()
    if used + additional + RESERVE > CAP:
        raise RuntimeError(f"Storage cap: {used} existing + {additional} additional + {RESERVE} reserve exceeds {CAP}")
    if shutil.disk_usage(ROOT).free < additional + RESERVE:
        raise RuntimeError("Insufficient free disk space")
    return used


def request_range(start, end):
    req = urllib.request.Request(API + "/content",
        headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"})
    response = urllib.request.urlopen(req, timeout=45)
    content_range = response.headers.get("Content-Range", "")
    if response.status != 206 or not content_range.startswith(f"bytes {start}-{end}/"):
        response.close()
        raise ValueError("Archive server ignored or altered the requested byte range")
    return response


def range_bytes(start, end):
    for attempt in range(5):
        try:
            with request_range(start, end) as response:
                data = response.read(end - start + 2)
            if len(data) != end - start + 1:
                raise IOError("Truncated HTTP range")
            return data
        except (OSError, ValueError):
            if attempt == 4:
                raise
            time.sleep(min(2 ** attempt, 8))


def zip_inventory(size):
    # ZIP64 central directory is small. Present its tail at the original
    # offsets so stdlib zipfile parses it without downloading the whole ZIP.
    start = max(0, size - 131072)
    tail = range_bytes(start, size - 1)

    class Tail(io.BytesIO):
        def seek(self, offset, whence=0):
            return super().seek(offset - start if whence == 0 else offset, whence)

        def tell(self):
            return super().tell() + start

    with zipfile.ZipFile(Tail(tail)) as archive:
        members = [entry for entry in archive.infolist() if entry.filename.endswith(".h5")]
    if len(members) != 38:
        raise ValueError(f"Unexpected upstream shard count: {len(members)}")
    return members


def download_member(entry, destination, status, ordinal, total):
    local_header = range_bytes(entry.header_offset, entry.header_offset + 29)
    if local_header[:4] != b"PK\x03\x04":
        raise ValueError("Invalid ZIP local header")
    name_size, extra_size = struct.unpack_from("<HH", local_header, 26)
    offset = entry.header_offset + 30 + name_size + extra_size
    downloaded = destination.stat().st_size if destination.exists() else 0
    if downloaded > entry.compress_size:
        raise ValueError("Partial member exceeds upstream size")
    check_budget(entry.compress_size - downloaded + entry.file_size)
    emit("download_member", shard=entry.filename, compressed_bytes=entry.compress_size,
         resumed_bytes=downloaded)
    # Bounded requests make range resumption safe even on interrupted transfers.
    # A failure appends only bytes actually received and resumes that exact byte.
    while downloaded < entry.compress_size:
        end = min(downloaded + 64 * 1024 * 1024, entry.compress_size) - 1
        check_budget(end - downloaded + 1)
        chunk = range_bytes(offset + downloaded, offset + end)
        with destination.open("ab") as stream:
            stream.write(chunk)
        downloaded += len(chunk)
        atomic_json(status, {"stage": "downloading", "shard": entry.filename,
            "shard_number": ordinal, "total_shards": total,
            "compressed_bytes_done": downloaded, "compressed_bytes_total": entry.compress_size,
            "shard_percent": round(100 * downloaded / entry.compress_size, 2)})
        emit("download_progress", shard=entry.filename, bytes_done=downloaded,
             bytes_total=entry.compress_size, percent=round(100 * downloaded / entry.compress_size, 2))


def inflate_member(entry, compressed, raw):
    if entry.compress_type != zipfile.ZIP_DEFLATED:
        raise ValueError("Only source DEFLATE members are supported")
    check_budget(entry.file_size)
    decoder = zlib.decompressobj(-15)
    crc, count = 0, 0
    with compressed.open("rb") as source, raw.open("wb") as destination:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            block = decoder.decompress(chunk)
            destination.write(block)
            crc = zlib.crc32(block, crc)
            count += len(block)
        block = decoder.flush()
        destination.write(block)
        crc = zlib.crc32(block, crc)
        count += len(block)
    if not decoder.eof or count != entry.file_size or crc != entry.CRC:
        raise ValueError("Source HDF5 size or ZIP CRC verification failed")


def scalar(value):
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8")
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def compact_member(raw, compact, source_member, budget_check=check_budget):
    """Keep all trials and scene RGB, without resizing or lossy recoding."""
    trials, objects, shape_bytes = [], set(), 0
    with h5py.File(raw, "r") as source:
        group = source["data"]
        names = sorted(group.keys(), key=lambda name: int(name.removeprefix("i")))
        # Budget raw retained arrays up front; gzip output should be smaller,
        # but do not rely on compression to stay within the cap.
        raw_size = sum(group[name][field].size * group[name][field].dtype.itemsize
                       for name in names for field in RGB_FIELDS)
        budget_check(raw_size + 1_000_000)
        with h5py.File(compact, "w") as target:
            target.attrs["schema_version"] = 1
            target.attrs["role"] = "evaluation_only"
            target.attrs["source_member"] = source_member
            target.attrs["input_fields"] = json.dumps(list(RGB_FIELDS[:2]))
            target.attrs["outcome_only_fields"] = json.dumps([RGB_FIELDS[2]])
            for name in names:
                original = group[name]
                metadata = {key: scalar(value) for key, value in original.attrs.items()
                            if key not in ("CLASS", "VERSION", "TITLE")}
                if "is_gripping" not in metadata or "object_name" not in metadata:
                    raise ValueError(f"Missing grasp outcome/object metadata: {source_member}:{name}")
                if metadata["is_gripping"] not in (0, 1, False, True):
                    raise ValueError("Invalid grasp success label")
                objects.add(metadata["object_name"])
                trial = target.create_group(name)
                trial.attrs["metadata_json"] = json.dumps(metadata)
                trial.attrs["is_gripping"] = bool(metadata["is_gripping"])
                trial.attrs["object_name"] = metadata["object_name"]
                shapes = {}
                for field in RGB_FIELDS:
                    image = original[field][...]
                    if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
                        raise ValueError(f"Unexpected scene RGB: {source_member}:{name}:{field}")
                    trial.create_dataset(field, data=image, compression="gzip",
                                         compression_opts=1, shuffle=True)
                    shapes[field] = list(image.shape)
                    shape_bytes += image.nbytes
                trials.append({"trial_id": f"{Path(source_member).stem}:{name}",
                               "group": name, "object_name": metadata["object_name"],
                               "success": bool(metadata["is_gripping"]), "rgb_shapes": shapes})
    return {"trials": trials, "objects": sorted(objects), "raw_rgb_bytes": shape_bytes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="datasets/evaluation_only/feeling_of_success")
    args = parser.parse_args()
    os.chdir(ROOT)
    output = (ROOT / args.output).resolve()
    # Temporary cleanup is restricted to this exact new evaluation directory.
    if output != ROOT / "datasets/evaluation_only/feeling_of_success":
        raise ValueError("Use the dedicated evaluation-only output directory")
    output.mkdir(parents=True, exist_ok=True)
    (output / "shards").mkdir(exist_ok=True)
    (output / "temporary").mkdir(exist_ok=True)
    status = output / "status.json"
    manifest_path = output / "manifest.json"
    with urllib.request.urlopen(API, timeout=30) as response:
        upstream = json.load(response)
    identity = {"url": SOURCE, "archive_content_url": API + "/content",
                "archive_bytes": upstream["sizeBytes"], "archive_checksum": upstream["checkSum"]}
    identity_path = output / "source.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Upstream archive identity changed; refusing to mix releases")
    atomic_json(identity_path, identity)
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "schema_version": 1, "dataset": "feeling_of_success", "role": "evaluation_only",
        "encoder_training_excluded": True, "primary_holdout": "ftp_motiontrans",
        "source": identity, "representation": "native scene RGB lossless HDF5 + grasp labels",
        "retained_inputs": list(RGB_FIELDS[:2]), "retained_outcome_only": [RGB_FIELDS[2]],
        "discarded_modalities": ["gelsightA", "gelsightB"],
        "split_policy": "Unassigned; choose object-disjoint fit/tune/test for frozen-head evaluation.",
        "shards": {}, "complete": False}
    members = zip_inventory(upstream["sizeBytes"])
    emit("start", current_bytes=check_budget(), cap_bytes=CAP,
         source_archive_bytes=upstream["sizeBytes"], total_shards=len(members))
    try:
        for ordinal, entry in enumerate(members, 1):
            name = Path(entry.filename).name
            destination = output / "shards" / name
            if name in manifest["shards"]:
                record = manifest["shards"][name]
                if not destination.exists() or sha256(destination) != record["sha256"]:
                    raise ValueError(f"Completed compact shard failed integrity: {destination}")
                emit("skip_verified", shard=name)
                continue
            if destination.exists():
                raise FileExistsError(f"Unregistered compact shard; inspect before resuming: {destination}")
            compressed = output / "temporary" / (name + ".deflate.part")
            raw = output / "temporary" / name
            compact = output / "temporary" / (name + ".compact.tmp")
            download_member(entry, compressed, status, ordinal, len(members))
            atomic_json(status, {"stage": "converting", "shard": name,
                                "shard_number": ordinal, "total_shards": len(members)})
            inflate_member(entry, compressed, raw)
            record = compact_member(raw, compact, entry.filename)
            # Calculate and validate metadata before committing the shard.
            record.update({"path": str(destination.relative_to(ROOT)),
                           "bytes": compact.stat().st_size, "sha256": sha256(compact),
                           "source_crc32": f"{entry.CRC:08x}", "source_bytes": entry.file_size})
            check_budget()
            compact.replace(destination)
            manifest["shards"][name] = record
            atomic_json(manifest_path, manifest)
            # These are only this downloader's own verified, disposable members.
            raw.unlink()
            compressed.unlink()
            emit("shard_complete", shard=name, trials=len(record["trials"]),
                 compact_bytes=record["bytes"], completed_shards=len(manifest["shards"]))
        manifest["trial_count"] = sum(len(row["trials"]) for row in manifest["shards"].values())
        manifest["object_count"] = len({obj for row in manifest["shards"].values() for obj in row["objects"]})
        # The released archive contains 9,296 trials, although the published
        # headline says 9,269. Archive completeness, not that headline, is
        # authoritative: never drop real records to force the expected count.
        expected_members = {Path(entry.filename).name for entry in members}
        trial_ids = [trial["trial_id"] for row in manifest["shards"].values()
                     for trial in row["trials"]]
        if (set(manifest["shards"]) != expected_members or not trial_ids
                or len(set(trial_ids)) != len(trial_ids)):
            raise ValueError("Incomplete archive membership or duplicate trial IDs")
        manifest["published_trial_count"] = 9269
        manifest["count_discrepancy_note"] = (
            "Verified complete release has 9296 unique trials; published headline "
            "is 9269. All release records retained without count-based filtering.")
        manifest["complete"] = True
        manifest["retained_bytes"] = sum(row["bytes"] for row in manifest["shards"].values())
        atomic_json(manifest_path, manifest)
        registry = ROOT / "processing/evaluation_datasets"
        registry.mkdir(exist_ok=True)
        atomic_json(registry / "feeling_of_success.json", {
            "role": "evaluation_only", "primary_holdout": "ftp_motiontrans",
            "manifest": str(manifest_path.relative_to(ROOT)), "trial_count": manifest["trial_count"],
            "object_count": manifest["object_count"], "complete": True,
            "retained_bytes": manifest["retained_bytes"], "encoder_training_excluded": True})
        atomic_json(status, {"stage": "complete", "completed_shards": len(members),
                            "total_shards": len(members), "trial_count": manifest["trial_count"],
                            "retained_bytes": manifest["retained_bytes"], "current_project_bytes": check_budget()})
        emit("complete", trials=manifest["trial_count"], objects=manifest["object_count"],
             retained_bytes=manifest["retained_bytes"])
    except BaseException as error:
        previous = json.loads(status.read_text()) if status.exists() else {}
        atomic_json(status, {**previous, "error": repr(error)})
        raise


if __name__ == "__main__":
    main()
