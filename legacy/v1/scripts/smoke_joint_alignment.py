#!/usr/bin/env python3
"""Real-data smoke: native pairing -> teacher codes -> DINO -> joint SAM step.

Uses one existing train and validation episode per source, isolated caches,
and a small spatial/temporal network. Never writes production cache outputs.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch

from tactilelatent.aligned_frames import MANIFEST, manifest_signature, write_json
from tactilelatent.audit_alignment import audit_manifest
from tactilelatent.sensor_code_cache import REGISTRY, SOURCES, build_source, verify_teachers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="processing/joint_alignment/smoke")
    args = parser.parse_args()
    torch.set_num_threads(4)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    original = json.loads(Path(MANIFEST).read_text())
    selected = {}
    for row in original["episodes"]:
        if row["split"] in ("train", "val"):
            selected.setdefault((row["source"], row["split"]), row)
    if len(selected) != 24:
        raise ValueError("Smoke requires both splits of every source")
    original["episodes"] = list(selected.values())
    manifest = root / "manifest.json"
    if not manifest.exists():
        write_json(manifest, original)
    audit = root / "audit.json"
    audit_manifest(manifest, audit)
    registry = json.loads(Path(REGISTRY).read_text())
    verify_teachers(registry)
    signature = manifest_signature(manifest)
    cache = root / "codes"
    for source in SOURCES:
        build_source(manifest, registry, source, cache, args.device, workers=2, data_signature=signature)
    cfg = json.loads(Path("configs/joint_alignment_holdout_motiontrans.json").read_text())
    cfg.update(manifest=str(manifest), audit=str(audit), code_cache=str(cache),
               output=str(root / "training"), width=32, heads=4,
               temporal_layers=1, cross_view_layers=1, accumulation=4)
    path = root / "config.json"
    write_json(path, cfg)
    subprocess.run([sys.executable, "scripts/train_joint_alignment.py", "--config", str(path),
                    "--smoke", "--device", args.device], check=True)
    state = torch.load(root / "training_smoke" / "last.pt", map_location="cpu", weights_only=False)
    if state["sam"]["steps"] < 1 or len(state["validation"]) != 11:
        raise ValueError("Incomplete real-data joint smoke")
    verify_teachers(registry)
    write_json(root / "result.json", {"passed": True, "SAM_updates": state["sam"]["steps"],
               "validation_sources": sorted(state["validation"]), "heldout": "ftp_motiontrans"})


if __name__ == "__main__":
    main()
