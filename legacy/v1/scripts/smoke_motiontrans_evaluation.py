"""Exercise both MotionTrans evaluation paths with real smoke checkpoints.

Only a few episode-disjoint clips are used here. These artifacts are plumbing
checks, never final benchmark results or initializations for full training.
"""

import json
from pathlib import Path

import torch

from tactilelatent.evaluate_direct_vla_touch import load_direct_model
from tactilelatent.evaluate_holdout_decoder import evaluate, frame_table, regression_loss
from tactilelatent.frozen_probe import (cached_latents, fit_head, head, load_encoder,
                                       seed_all, select_indices, sha256)
from tactilelatent.joint_data import JointClipDataset


def main():
    seed_all(45)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = "processing/joint_all/manifest.json"
    source = "ftp_motiontrans"
    train = JointClipDataset(manifest, sources=[source])
    val = JointClipDataset(manifest, split="val", sources=[source])
    indices = {"fit": select_indices(train, "fit", cap=4),
               "tune": select_indices(train, "tune", cap=2),
               "val": select_indices(val, cap=2)}
    results = {}
    for name, direct in (("joint_all_holdout_motiontrans_smoke", False),
                         ("motiontrans_direct_smoke", True)):
        checkpoint = f"processing/runs/{name}/last.pt"
        output = Path(f"processing/runs/{name}/evaluation_smoke")
        output.mkdir(parents=True, exist_ok=True)
        if direct:
            backbone, model, _ = load_direct_model(checkpoint, device, source)
        else:
            backbone, model, _ = load_encoder(checkpoint, device, source)
            assert source not in model.targets and "ftp_vla_touch" in model.targets
        assert not model.training and not backbone.training
        assert all(not p.requires_grad for module in (backbone, model) for p in module.parameters())
        versions = [p._version for module in (backbone, model) for p in module.parameters()]
        data = {}
        for partition, ids in indices.items():
            data[partition] = cached_latents(backbone, model, manifest, source,
                "val" if partition == "val" else "train", ids,
                output / f"{partition}_features.pt", sha256(checkpoint), sha256(manifest),
                device, with_targets=True, workers=0)
        assert set(data["fit"]["episode_id"]).isdisjoint(data["tune"]["episode_id"])
        assert set(data["fit"]["episode_id"]).isdisjoint(data["val"]["episode_id"])
        if direct:
            decoder = model.decoders[source]
        else:
            decoder = head(32, 6).to(device)
            fit_head(decoder, frame_table(data["fit"], range(4), device),
                     frame_table(data["tune"], range(2), device), regression_loss,
                     epochs=2, patience=1, batch_size=16, device=device)
        metrics = evaluate(decoder, data["fit"], data["val"], device, 64, 2071, output, (6, 1, 1))
        assert len(metrics["per_channel"]) == 6
        assert metrics["validation_clips"] == 2 and metrics["sample_count"] == 64
        assert versions == [p._version for module in (backbone, model) for p in module.parameters()]
        assert all(p.grad is None for module in (backbone, model) for p in module.parameters())
        results[name] = {"smoke_only": True, "metrics": metrics}
        del backbone, model, decoder, data
        if device.type == "cuda":
            torch.cuda.empty_cache()
    destination = Path("processing/runs/motiontrans_baseline_sequence/evaluation_smoke.json")
    destination.write_text(json.dumps(results, indent=2))
    print(json.dumps({"event": "evaluation_smoke_complete", "output": str(destination)}), flush=True)


if __name__ == "__main__":
    main()
