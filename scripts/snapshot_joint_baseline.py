"""Make a non-overwriting, checksummed reference for future VLA_touch runs."""

import hashlib
import json
import shutil
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "processing/evaluations/baselines/joint_50ep_vla_touch_v1"
FILES = (
    "processing/runs/joint_all_holdout_vla_8k_50ep/last.pt",
    "processing/runs/joint_all_holdout_vla_8k_50ep/config.json",
    "processing/runs/joint_all_holdout_vla_8k_50ep/metrics.jsonl",
    "processing/joint_all/manifest.json",
    "processing/evaluations/vla_touch_frozen_decoder/decoder.pt",
    "processing/evaluations/vla_touch_frozen_decoder/results.json",
    "processing/evaluations/vla_touch_frozen_decoder/examples/first.png",
    "processing/evaluations/vla_touch_frozen_decoder/examples/worst_error.png",
    "processing/evaluations/vla_touch_frozen_decoder/examples/most_uncertain.png",
    "processing/evaluations/dataset_identity_probe/classifier.pt",
    "processing/evaluations/dataset_identity_probe/results.json",
    "src/tactilelatent/frozen_probe.py",
    "src/tactilelatent/evaluate_holdout_decoder.py",
    "src/tactilelatent/evaluate_dataset_classifier.py",
    "src/tactilelatent/train_joint.py",
    "src/tactilelatent/joint_model.py",
)


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def verify(snapshot):
    inventory = json.loads((snapshot / "baseline.json").read_text())
    for name, expected in inventory["files"].items():
        path = snapshot / name
        if not path.is_file() or digest(path) != expected["sha256"]:
            raise ValueError(f"Baseline file missing or changed: {path}")
    return inventory


def main():
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    if DESTINATION.exists():
        inventory = verify(DESTINATION)
        print(json.dumps({"event": "verified_existing", "baseline": str(DESTINATION),
                          "files": len(inventory["files"])}))
        return
    staging = Path(tempfile.mkdtemp(prefix=".joint_baseline_", dir=DESTINATION.parent))
    try:
        inventory = {"baseline_id": "joint_50ep_vla_touch_v1",
                     "description": "Original 50-epoch joint model and frozen-probe reference",
                     "files": {}}
        for name in FILES:
            source = ROOT / name
            if not source.is_file():
                raise FileNotFoundError(source)
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            checksum = digest(target)
            if checksum != digest(source):
                raise IOError(f"Copy changed bytes: {source}")
            inventory["files"][name] = {"sha256": checksum, "bytes": target.stat().st_size}
        (staging / "baseline.json").write_text(json.dumps(inventory, indent=2))
        staging.rename(DESTINATION)
        verify(DESTINATION)
        print(json.dumps({"event": "created", "baseline": str(DESTINATION),
                          "files": len(inventory["files"])}))
    finally:
        if staging.exists():
            shutil.rmtree(staging)


if __name__ == "__main__":
    main()
