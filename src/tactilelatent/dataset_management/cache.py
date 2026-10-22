"""Read contiguous, per-episode and paired frozen-sensor code artifacts.

Inputs are original code indexes, immutable producer/teacher identities and
training statistics. Outputs are chronological code tensors with explicit
validity; original receipts and arrays are never rewritten by reads or imports.
"""

from collections import OrderedDict
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from .artifacts import atomic_json, file_hash, json_hash, read_json, verify_implementation


@lru_cache(maxsize=4096)
def _verified(path, size, mtime_ns, digest):
    if file_hash(path) != digest:
        raise ValueError(f"Cached artifact checksum mismatch: {path}")
    return True


def verified_path(path, digest):
    """Return a payload path after checking its recorded SHA256 identity."""
    path = Path(path)
    stat = path.stat()
    _verified(str(path), stat.st_size, stat.st_mtime_ns, digest)
    return path


class SensorCodeStore:
    """Bind cached sensor codes to their teacher, source rows and mirror policy.

    Codes retain the original teacher's width. Paired caches contain separately
    encoded original/mirrored targets; latent coordinates are never reflected.
    Access verifies payload hashes and chronology, then optionally applies the
    frozen training-only mean/std. Missing code support remains an explicit mask.
    """

    def __init__(
        self,
        index_path,
        paths,
        teacher=None,
        policy_hash=None,
        statistics=None,
        expected_index_hash=None,
    ):
        self.paths = paths
        self.path = paths.resolve(index_path)
        if expected_index_hash is not None:
            verified_path(self.path, expected_index_hash)
        self.index = read_json(self.path)
        self.index_hash = file_hash(self.path)
        self.statistics = statistics or self.index.get("statistics")
        recorded_statistics = self.index.get("statistics")
        if (
            statistics
            and recorded_statistics
            and any(
                not np.array_equal(
                    np.asarray(statistics[key]), np.asarray(recorded_statistics[key])
                )
                for key in ("mean", "std")
            )
        ):
            raise ValueError("Configured code statistics differ from the frozen cache")
        self.teacher = teacher or {}
        self.loaded = OrderedDict()
        self.codes = self.valid = None
        signature = self.index.get("signature", {})
        if self.index.get("complete") is False:
            raise ValueError("Code cache is incomplete")
        actual_teacher = signature.get("teacher_sha256", self.index.get("teacher_sha256"))
        if teacher and actual_teacher != teacher["sha256"]:
            raise ValueError("Cached code teacher differs from selected decoder")
        if teacher:
            verified_path(paths.resolve(teacher["checkpoint"]), teacher["sha256"])
        if policy_hash is not None and signature.get("policy_sha256") != policy_hash:
            raise ValueError("Paired code policy changed")
        self.paired = signature.get("variants") == ["original", "mirrored"]
        if policy_hash is not None and not self.paired:
            raise ValueError("A mirror source needs separately encoded paired codes")
        self.rows = {row["episode_id"]: row for row in self.index["episodes"]}
        if len(self.rows) != len(self.index["episodes"]):
            raise ValueError("Duplicate code episode identity")
        self.offsets = (
            {
                row["episode_id"]: self.index["offsets"][i]
                for i, row in enumerate(self.index["episodes"])
            }
            if "offsets" in self.index
            else None
        )
        if self.statistics is not None:
            mean, std = np.asarray(self.statistics["mean"]), np.asarray(self.statistics["std"])
            if (
                mean.ndim != 1
                or mean.shape != std.shape
                or not np.isfinite(mean).all()
                or not np.isfinite(std).all()
                or (std <= 0).any()
            ):
                raise ValueError("Invalid frozen training code statistics")
            if self.statistics.get("fit_split", "train") != "train":
                raise ValueError("Code statistics must be fitted on training only")
            if teacher and len(mean) != teacher["code_dim"]:
                raise ValueError("Code dimension differs from selected decoder")
        self.producers = self._verify_producers(signature)

    def _verify_producers(self, signature):
        mapping = dict(signature.get("implementation", {}))
        if "code_builder_sha256" in signature:
            name = (
                "marker_code_cache.py"
                if signature.get("representation") == "marker_displacement"
                else "sensor_code_cache.py"
            )
            mapping[name] = signature["code_builder_sha256"]
        if "encoder_sha256" in signature:
            mapping["marker_autoencoder.py"] = signature["encoder_sha256"]
        return verify_implementation(mapping, self.paths)

    def _episode(self, row):
        record = self.rows[row["episode_id"]]
        if record.get("row_sha256") and record["row_sha256"] != json_hash(row):
            raise ValueError("Cached native episode/frame contract changed")
        if "frames" in record and record["frames"] != row["frames"]:
            raise ValueError("Cached frame count changed")
        if "split" in record and record["split"] != row["split"]:
            raise ValueError("Cached episode split changed")
        if self.offsets is not None:
            # These entries are entire original manifest rows, not rewritten copies.
            if record != row:
                raise ValueError("Cached original manifest row differs")
            return record
        key = row["episode_id"]
        if key not in self.loaded:
            path = verified_path(self.paths.resolve(record["path"]), record["sha256"])
            with np.load(path, allow_pickle=False) as archive:
                data = {name: archive[name].copy() for name in archive.files}
            if not np.array_equal(data["frame_indices"], np.arange(row["frames"])):
                raise ValueError("Cached code chronology differs")
            dimension = len(self.statistics["mean"]) if self.statistics else 0
            shape = (2, row["frames"], dimension) if self.paired else (row["frames"], dimension)
            valid_shape = shape[:-1]
            if (
                data["codes"].shape != shape
                or data["codes"].dtype != np.float32
                or not np.isfinite(data["codes"]).all()
            ):
                raise ValueError("Invalid cached code shape/type/values")
            if data["valid"].shape != valid_shape or data["valid"].dtype != np.bool_:
                raise ValueError("Invalid cached code validity")
            if self.paired and not np.array_equal(data["valid"][0], data["valid"][1]):
                raise ValueError("Mirroring changed measured support")
            self.loaded[key] = data
            while len(self.loaded) > 4:
                self.loaded.popitem(last=False)
        self.loaded.move_to_end(key)
        return self.loaded[key]

    def _array(self, name):
        value = getattr(self, name)
        if value is None:
            path = self.path.parent / (name + ".npy")
            expected = self.index.get("array_sha256", {}).get(name + ".npy")
            if expected is None:
                raise ValueError("Contiguous cache lacks immutable array checksum")
            verified_path(path, expected)
            value = np.load(path, mmap_mode="r")
            expected_shape = (
                (self.index["frames"], self.index["code_dim"])
                if name == "codes"
                else (self.index["frames"],)
            )
            if value.shape != expected_shape or value.dtype != (
                np.float32 if name == "codes" else np.bool_
            ):
                raise ValueError("Contiguous cache shape/type differs")
            setattr(self, name, value)
        return value

    def read(self, row, frames, mirrored=False, normalized=True):
        """Return ``[F, C]`` float codes and ``[F]`` validity for logical frames.

        ``mirrored=True`` requires a paired cache. Standardization uses the
        recorded training statistics, with no episode-specific fitting.
        """
        frames = np.asarray(frames, np.int64)
        if (
            frames.ndim != 1
            or not len(frames)
            or (frames < 0).any()
            or (frames >= row["frames"]).any()
        ):
            raise ValueError("Cached code frame index outside episode")
        data = self._episode(row)
        if self.offsets is not None:
            if mirrored:
                raise ValueError("Original-only cache has no mirrored target")
            indices = self.offsets[row["episode_id"]] + frames
            codes = np.asarray(self._array("codes")[indices]).copy()
            valid = np.asarray(self._array("valid")[indices]).copy()
        else:
            select = (int(mirrored), frames) if self.paired else frames
            if mirrored and not self.paired:
                raise ValueError("Original-only cache has no mirrored target")
            codes = data["codes"][select].copy()
            valid = data["valid"][select].copy()
        if normalized and self.statistics is not None:
            codes = (codes - np.asarray(self.statistics["mean"], np.float32)) / np.asarray(
                self.statistics["std"], np.float32
            )
        return torch.from_numpy(codes), torch.from_numpy(valid)

    def valid_frames(self, row, frames):
        """Return frozen support for the requested logical episode frames."""
        frames = np.asarray(frames, np.int64)
        data = self._episode(row)
        if self.offsets is not None:
            return np.asarray(self._array("valid")[self.offsets[row["episode_id"]] + frames])
        return data["valid"][0, frames] if self.paired else data["valid"][frames]

    def ends(self, row):
        """Return explicit prepared clip ends when the cache declares them."""
        data = self._episode(row)
        return data.get("ends") if isinstance(data, dict) else None

    def verify_all(self, rows):
        """Hash every frozen-code payload; never decode RGB or regenerate arrays."""
        if file_hash(self.path) != self.index_hash:
            raise ValueError("Old code receipt changed")
        rows = list(rows)
        if {row["episode_id"] for row in rows} != set(self.rows):
            raise ValueError("Full code verification requires the complete native row set")
        for row in rows:
            record = self.rows[row["episode_id"]]
            if self.offsets is None:
                verified_path(self.paths.resolve(record["path"]), record["sha256"])
            self._episode(row)
        if self.offsets is not None:
            for name in ("codes.npy", "valid.npy"):
                verified_path(self.path.parent / name, self.index["array_sha256"][name])
            self._array("codes")
            self._array("valid")
            payloads = 2
        else:
            payloads = len(self.rows)
        return {
            "episodes": len(rows),
            "payload_files": payloads,
            "payload_hashes": "all",
            "frame_contracts": "all",
            "rgb_decoded": False,
            "native_target_quality_evaluated": False,
        }

    def import_receipt(self, consumer_files):
        """Describe original artifacts and current consumers without editing either.

        The new sidecar pins original index, teacher, archived producers, code
        statistics and exact maintained consumer bytes. It does not regenerate
        arrays or replace historical hashes with current implementation hashes.
        """
        if file_hash(self.path) != self.index_hash:
            raise ValueError("Old receipt changed after source binding")
        return {
            "version": "uniform-sensor-code-import-v1",
            "old_receipt": {"path": str(self.path), "sha256": self.index_hash},
            "teacher_sha256": self.teacher.get("sha256"),
            "archived_producers": self.producers,
            "raw_extraction_provenance": getattr(self, "raw_extraction_provenance", {}),
            "new_consumers": {str(path): file_hash(path) for path in consumer_files},
            "statistics_sha256": json_hash(self.statistics),
            "arrays_regenerated": False,
            "old_receipt_rewritten": False,
        }

    def write_import(self, path, consumer_files):
        """Write a new import sidecar, rejecting a changed existing version."""
        receipt = self.import_receipt(consumer_files)
        destination = Path(path)
        if destination.exists() and read_json(destination) != receipt:
            raise ValueError("Import consumer/receipt changed; choose a new sidecar version")
        atomic_json(destination, receipt)
        return receipt

    def require_import(self, path, consumer_files):
        """Require exact agreement between a sidecar and current artifact binding."""
        if read_json(path) != self.import_receipt(consumer_files):
            raise ValueError("Cache import sidecar does not match old artifacts/new consumer")

    def __getstate__(self):
        return {**self.__dict__, "codes": None, "valid": None, "loaded": OrderedDict()}
