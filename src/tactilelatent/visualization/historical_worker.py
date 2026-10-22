"""Run the historical viewer in an isolated, read-only compatibility process.

Inputs:
    Archive and project roots, an immutable relocation manifest, and forwarded
    viewer arguments. The maintained viewer launches this internal worker.
Outputs:
    A local historical inference HTTP service. Original source/config paths map
    only to byte-identical recorded files; data and receipts are never rewritten.
Example:
    tl view --host 127.0.0.1 --port 8080

This module is executed as a child script, never imported into the core viewer.
"""

import argparse
import builtins
import hashlib
import io
import json
import os
import runpy
import stat
import sys
from pathlib import Path


class VerifiedRelocations:
    def __init__(self, manifest, archive_root):
        self.archive = Path(archive_root).resolve()
        self.manifest = Path(manifest)
        document = json.loads(self.manifest.read_text())
        if document.get("schema") != "historical-viewer-relocations-v1":
            raise ValueError("Unsupported historical relocation manifest")
        self.original = Path(document["original_root"])
        if not self.original.is_absolute():
            raise ValueError("Relocation source root must be absolute")
        self.protected = tuple(str(self.original / name) for name in ("src", "scripts", "configs"))
        self.mapping = {}
        self.archived = set()
        for recorded, entry in document["files"].items():
            source = Path(recorded)
            target = self.archive / entry["archived_path"]
            resolved = target.resolve()
            if (
                not source.is_absolute()
                or not self._protected(str(source))
                or not resolved.is_relative_to(self.archive)
                or target.is_symlink()
            ):
                raise ValueError("Relocation path escapes its declared source/archive roots")
            if not resolved.is_file():
                raise ValueError("Missing immutable historical file: " + str(target))
            digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
            if digest != entry["sha256"]:
                raise ValueError("Historical archived source hash mismatch: " + recorded)
            if stat.S_IMODE(resolved.stat().st_mode) != entry["mode"]:
                raise ValueError("Historical archived source permissions changed: " + recorded)
            self.mapping[os.path.normpath(recorded)] = str(resolved)
            self.archived.add(str(resolved))
        self.saved = None

    def _protected(self, path):
        return any(path == root or path.startswith(root + os.sep) for root in self.protected)

    def translate(self, path, *, write=False):
        if isinstance(path, int):
            return path
        try:
            value = os.fsdecode(os.fspath(path))
        except TypeError:
            return path
        if not os.path.isabs(value):
            return path
        normalized = os.path.normpath(value)
        if normalized in self.mapping:
            if write:
                raise PermissionError("Historical source/config mappings are read-only")
            return self.mapping[normalized]
        if normalized in self.archived and write:
            raise PermissionError("Archived historical source/config files are read-only")
        if self._protected(normalized):
            # Existing directories remain real directories. Only individual
            # recorded files are relocated; an unrecorded file is never read
            # from the newly edited core source/config tree.
            original_stat = self.saved[2] if self.saved is not None else os.stat
            try:
                if stat.S_ISDIR(original_stat(normalized).st_mode):
                    return path
            except FileNotFoundError:
                pass
            raise FileNotFoundError("No verified historical relocation record: " + normalized)
        return path

    def install(self):
        if self.saved is not None:
            return self
        self.saved = (builtins.open, io.open, os.stat, os.lstat)
        original_open, original_io, original_stat, original_lstat = self.saved

        def opening(function):
            def open_verified(file, mode="r", *args, **kwargs):
                destination = self.translate(file, write=any(flag in mode for flag in "wax+"))
                return function(destination, mode, *args, **kwargs)

            return open_verified

        def stat_verified(path, *args, **kwargs):
            return original_stat(self.translate(path), *args, **kwargs)

        def lstat_verified(path, *args, **kwargs):
            return original_lstat(self.translate(path), *args, **kwargs)

        builtins.open = opening(original_open)
        io.open = opening(original_io)
        os.stat = stat_verified
        os.lstat = lstat_verified
        return self

    def close(self):
        if self.saved is not None:
            builtins.open, io.open, os.stat, os.lstat = self.saved
            self.saved = None

    def __enter__(self):
        return self.install()

    def __exit__(self, *error):
        self.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--relocations", type=Path)
    options, viewer_arguments = parser.parse_known_args()
    archive = options.archive.resolve()
    manifest = options.relocations or archive / "artifact_relocations.json"
    relocations = VerifiedRelocations(manifest, archive)
    if relocations.original.resolve() != options.project_root.resolve():
        raise ValueError("Historical relocation manifest belongs to a different original project")
    sys.path.insert(0, str(archive / "src"))
    os.chdir(archive)
    launcher = archive / "scripts/view_egotouch_autoencoder.py"
    sys.argv = [str(launcher), *viewer_arguments]
    with relocations:
        runpy.run_path(str(launcher), run_name="__main__")


if __name__ == "__main__":
    main()
