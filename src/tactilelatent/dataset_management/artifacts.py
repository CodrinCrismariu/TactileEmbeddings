"""Immutable artifact identities and explicit legacy-producer imports.

JSON rows are hashed exactly as stored. Resolving paths is a separate operation:
absolute paths must never be inserted into an old row before checking its hash.
Inputs are saved artifact paths, contents and recorded checksums. Outputs are
verified locations or new explicit metadata receipts; old identities stay intact.
"""

import hashlib
import json
import re
import stat
from pathlib import Path


def file_hash(path):
    """Return SHA256 over the file's exact bytes without loading it all at once."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_hash(value):
    """Hash JSON content using the project's recorded sorted-key serialization."""
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read_json(path):
    """Read JSON without resolving or rewriting paths inside its saved rows."""
    return json.loads(Path(path).read_text())


class ArtifactPaths:
    """Resolve current artifacts and explicitly verified historical producers.

    Native data resolves against ``project_root``. Archived implementation paths
    resolve through recorded relocation identities; matching a filename alone
    never establishes compatibility.
    """

    def __init__(self, project_root, metadata_root=None, archived_producers=None):
        self.root = Path(project_root).resolve()
        self.metadata_root = Path(metadata_root or Path(__file__).resolve().parents[3]).resolve()
        self.archived_producers = Path(
            archived_producers or self.metadata_root / "legacy/v1/src/tactilelatent"
        ).resolve()

    def _is_original_source(self, path):
        if not path.is_absolute() or not path.is_relative_to(self.root):
            return False
        relative = path.relative_to(self.root)
        return bool(relative.parts) and relative.parts[0] in ("configs", "src", "scripts")

    def _relocate(self, original):
        """Read only a recorded, byte-identical historical source/config file.

        This is deliberately not a filesystem fallback: an identically named
        archive file without a relocation record is not sufficient evidence.
        The original receipt and its recorded path/hash remain untouched.
        """
        archive = self.metadata_root / "legacy/v1"
        manifest = archive / "artifact_relocations.json"
        if not manifest.is_file():
            raise FileNotFoundError(f"No historical relocation manifest: {manifest}")
        document = read_json(manifest)
        if (
            not isinstance(document, dict)
            or document.get("schema") != "historical-viewer-relocations-v1"
            or document.get("original_root") != str(self.root)
            or not isinstance(document.get("files"), dict)
        ):
            raise ValueError("Invalid historical relocation manifest or original project root")
        entry = document["files"].get(str(original))
        if entry is None:
            raise FileNotFoundError(f"No verified historical relocation record: {original}")
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid historical relocation record: {original}")
        relative = entry.get("archived_path")
        expected = entry.get("sha256")
        if (
            not isinstance(relative, str)
            or relative != str(original.relative_to(self.root))
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not isinstance(expected, str)
            or re.fullmatch("[0-9a-f]{64}", expected) is None
        ):
            raise ValueError(f"Invalid historical relocation path/checksum: {original}")
        candidate = archive / relative
        if candidate.is_symlink() or not candidate.resolve().is_relative_to(archive.resolve()):
            raise ValueError(f"Historical relocation escapes archive: {original}")
        if not candidate.is_file():
            raise FileNotFoundError(f"Missing immutable historical file: {candidate}")
        if file_hash(candidate) != expected:
            raise ValueError(f"Historical archived source hash mismatch: {original}")
        if "mode" in entry and stat.S_IMODE(candidate.stat().st_mode) != entry["mode"]:
            raise ValueError(f"Historical archived source permissions changed: {original}")
        return candidate

    def resolve(self, value, *, historical=False):
        """Resolve artifact paths; current files win unless explicitly historical.

        Missing absolute source/config paths may use the verified relocation
        manifest. Historical receipt readers opt in to overriding an existing
        current file at the same path; ordinary current configuration does not.
        """
        path = Path(value)
        if path.is_absolute():
            if self._is_original_source(path) and (historical or not path.exists()):
                return self._relocate(path)
            return path
        if path.parts and path.parts[0] == "legacy":
            return self.metadata_root / path
        resolved = self.root / path
        if historical and self._is_original_source(resolved):
            return self._relocate(resolved)
        return resolved

    def producer(self, value):
        """Original receipt paths identify archived producer bytes, never new code."""
        path = Path(value)
        if self._is_original_source(path) or (
            not path.is_absolute() and path.parts and path.parts[0] in ("src", "scripts", "configs")
        ):
            return self.resolve(value, historical=True)
        if "tactilelatent" in path.parts or (len(path.parts) == 1 and path.suffix == ".py"):
            candidates = [
                self.archived_producers / path.name,
                self.archived_producers / "training" / path.name,
                self.archived_producers / "visualization" / path.name,
            ]
            for candidate in candidates:
                if candidate.exists():
                    return candidate
        return self.resolve(value)

    def check(self, value, expected):
        """Return a historical-capable artifact path only if its hash matches."""
        path = self.resolve(value, historical=True)
        actual = file_hash(path)
        if actual != expected:
            raise ValueError(f"Artifact checksum mismatch: {value}")
        return path

    def check_producer(self, value, expected):
        """Require the archived producer's exact original implementation bytes."""
        path = self.producer(value)
        if file_hash(path) != expected:
            raise ValueError(f"Archived producer checksum mismatch: {value}")
        return path


def verify_implementation(mapping, paths):
    """Verify all recorded producer hashes and return their archived locations."""
    return {
        str(name): {"archived_path": str(paths.check_producer(name, digest)), "sha256": digest}
        for name, digest in mapping.items()
    }


def atomic_json(path, payload):
    """Only called by explicit preparation/import commands, never dataset reads."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def implementation_signature(package_root):
    """Pin maintained consumer implementations/assets without executing imports."""
    root = Path(package_root).resolve()
    return {
        str(path.relative_to(root)): file_hash(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.suffix in (".py", ".html", ".js", ".json")
        and "__pycache__" not in path.parts
    }


SCIENTIFIC_DIRECTORIES = frozenset({"models", "datasets", "dataset_management", "training"})
SCIENTIFIC_FILES = frozenset({"configuration.py", "contracts.py"})


def scientific_implementation_path(name):
    """Accept only relative Python consumers that can affect source training.

    Evaluators, serving code, assets and operational commands are independent
    consumers. Their edits must not invalidate an unrelated training resume.
    Configuration/data/policy contents are pinned separately in run provenance.
    """
    if not isinstance(name, str):
        return False
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or str(path) != name:
        return False
    if path.suffix != ".py" or "__pycache__" in path.parts:
        return False
    return (
        name in SCIENTIFIC_FILES or len(path.parts) > 1 and path.parts[0] in SCIENTIFIC_DIRECTORIES
    )


def scientific_implementation_signature(package_root):
    """Hash only maintained Python consumers that can affect source training."""
    root = Path(package_root).resolve()
    return {
        str(path.relative_to(root)): file_hash(path)
        for path in sorted(root.rglob("*.py"))
        if path.is_file() and scientific_implementation_path(str(path.relative_to(root)))
    }
