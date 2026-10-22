"""Estimate train-only code-distance floors using frozen AE round trips.

These are teacher consistency proxies, not calibrated sensor noise estimates.
Native tactile dimensions and the original pinned code caches are unchanged.
"""
import argparse
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

from .aligned_frames import file_hash, manifest_signature, write_json
from .fgw_data import source_seed
from .relational import HELDOUT
from .sensor_autoencoder import make_autoencoder
from .sensor_code_cache import CodeCache, cache_signature, verify_teachers


@torch.no_grad()
def estimate_source(dataset_rows, cache, teacher, device, episodes=32, frames=4, seed=45):
    if episodes < 1 or frames < 1 or any(r["split"] != "train" or r["source"] == HELDOUT for r in dataset_rows):
        raise ValueError("Floor estimation requires source training episodes")
    rows = sorted(dataset_rows, key=lambda r: r["episode_id"])
    source = rows[0]["source"]
    random.Random(source_seed(seed, 0, source)).shuffle(rows)
    residuals, records, clip_magnitudes = [], [], []
    std = torch.tensor(cache.index["statistics"]["std"], device=device)
    for row in rows:
        available = np.flatnonzero(cache.valid_frames(row["episode_id"], np.arange(row["frames"])))
        if not len(available):
            continue
        indices = available[np.linspace(0, len(available) - 1, min(frames, len(available)), dtype=int)]
        raw, valid = cache.read(row["episode_id"], indices, normalized=False)
        raw = raw.to(device)
        for chunk in raw.split(2):
            reconstructed = teacher.decode(chunk)
            if tuple(reconstructed.shape[1:]) != tuple(row["target_shape"]):
                raise ValueError("Teacher decode changed native tactile shape")
            error = (teacher.encode(reconstructed) - chunk) / std
            if not torch.isfinite(error).all():
                raise FloatingPointError("Nonfinite round-trip code residual")
            residuals.extend(error.square().sum(-1).cpu().tolist())
        records.append({"episode_id": row["episode_id"], "frames": indices.tolist()})
        if row["frames"] >= 15:
            for start in np.linspace(0, row["frames"] - 15, 4, dtype=int):
                codes, mask = cache.read(row["episode_id"], start + np.arange(8) * 2)
                selected = codes[mask.bool()]
                if len(selected) >= 2:
                    distance = torch.cdist(selected, selected)
                    pairs = ~torch.eye(len(selected), dtype=torch.bool)
                    clip_magnitudes.append(float(distance[pairs].square().mean().sqrt()))
        if len(records) >= episodes:
            break
    if not residuals:
        raise ValueError(f"No valid round-trip samples: {source}")
    rms = math.sqrt(float(np.mean(residuals)))
    floor = max(1e-6, math.sqrt(2) * rms)
    return {"distance_floor": floor, "residual_vector_rms": rms,
            "samples": records, "frames": len(residuals),
            "diagnostic_clips": len(clip_magnitudes),
            "informative_clip_fraction": float(np.mean(np.asarray(clip_magnitudes) > floor)) if clip_magnitudes else None,
            "estimator": "sqrt(2) * RMS standardized code round-trip residual norm; minimum 1e-6",
            "interpretation": "Teacher-consistency proxy, not calibrated sensor noise"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/invariance_16d_relational.json")
    parser.add_argument("--output", default="processing/invariance/relational_floors.json")
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(4)
    cfg = json.loads(Path(args.config).read_text())
    signature = manifest_signature(cfg["manifest"])
    audit = json.loads(Path(cfg["audit"]).read_text())
    if audit.get("signature") != signature or not audit.get("passed"):
        raise ValueError("Matching passing native data audit required")
    registry = json.loads(Path(cfg["teachers"]).read_text())
    if HELDOUT in registry["sources"] or cfg.get("heldout_source") != HELDOUT:
        raise ValueError("MotionTrans must remain excluded")
    verify_teachers(registry)
    rows = json.loads(Path(cfg["manifest"]).read_text())["episodes"]
    signatures = {s: cache_signature(cfg["manifest"], registry, s, signature) for s in registry["sources"]}
    header = {"schema_version": 1, "data": signature, "cache_signatures": signatures, "split": "train",
              "seed": cfg["seed"], "episodes_per_source": args.episodes, "frames_per_episode": args.frames,
              "builder_sha256": file_hash(__file__), "numeric_policy": "Original loss only; no numeric AE floor"}
    path = Path(args.output)
    partial = path.with_suffix(".partial.json")
    state = {**header, "sources": {}}
    if path.exists() or partial.exists():
        state = json.loads((path if path.exists() else partial).read_text())
        if {k: state.get(k) for k in header} != header:
            raise ValueError("Floor settings/provenance differ; choose a fresh output")
    for source, info in sorted(registry["sources"].items()):
        if source in state["sources"]:
            continue
        began = time.monotonic()
        saved = torch.load(info["checkpoint"], map_location="cpu", weights_only=False)
        teacher = make_autoencoder(saved["model_config"]).to(args.device).eval().requires_grad_(False)
        teacher.load_state_dict(saved["model"])
        cache = CodeCache(cfg["code_cache"], source, signatures[source])
        state["sources"][source] = estimate_source(
            [r for r in rows if r["source"] == source and r["split"] == "train"], cache, teacher,
            args.device, args.episodes, args.frames, cfg["seed"])
        write_json(partial, state)
        print(json.dumps({"source": source, "floor": state["sources"][source]["distance_floor"],
                          "seconds": time.monotonic() - began}), flush=True)
        del teacher, saved, cache
    write_json(path, state)
    partial.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
