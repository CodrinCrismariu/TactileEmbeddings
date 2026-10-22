"""Decode representative train/val clips and run one six-head VAE forward.

This is a readiness check, not a training run and not a meaningful accuracy
evaluation: five of the six sensor-specific heads are newly initialized.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from tactilelatent.joint_data import JointClipDataset
from tactilelatent.joint_model import JointTactileVAE
from tactilelatent.model import DinoBackbone
from tactilelatent.prepare_joint import TARGETS
from tactilelatent.train import features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-per-source", type=int, default=20)
    parser.add_argument("--output", default="processing/audits/joint_six_smoke.json")
    args = parser.parse_args()
    if args.samples_per_source < 1:
        raise ValueError("samples-per-source must be positive")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = DinoBackbone().to(device)
    model = JointTactileVAE(input_dim=backbone.dim).to(device).eval()
    source = torch.load("processing/runs/egotouch_dinov3_vae_image_only_latent32/last.pt",
                        map_location="cpu", weights_only=False)
    model.initialize_from_egotouch(source["model"])
    report = {}
    for split in ("train", "val"):
        report[split] = {}
        for name, (_, expected_shape, _) in TARGETS.items():
            dataset = JointClipDataset(split=split, sources=[name])
            indices = np.linspace(0, len(dataset) - 1,
                                  min(args.samples_per_source, len(dataset)), dtype=int)
            target_sum = 0.0
            target_cells = 0
            zero_cells = 0
            full_scale_cells = 0
            for index in indices:
                item = dataset[int(index)]
                if list(item["target"].shape[1:]) != expected_shape:
                    raise ValueError(f"{name}: unexpected target shape")
                if not torch.isfinite(item["rgb"]).all() or not torch.isfinite(item["target"]).all():
                    raise ValueError(f"{name}: nonfinite sample")
                if not item["view_mask"].any(dim=-1).all():
                    raise ValueError(f"{name}: timestep without a camera")
                values = item["target"][item["target_mask"]]
                target_sum += values.double().sum().item()
                target_cells += values.numel()
                zero_cells += (values == 0).sum().item()
                full_scale_cells += (values >= .999).sum().item()

            # One real DINO forward + sensor-specific decoder per source.
            item = dataset[int(indices[0])]
            batch = {key: value.unsqueeze(0) for key, value in item.items()
                     if isinstance(value, torch.Tensor)}
            with torch.no_grad(), torch.autocast("cuda", torch.bfloat16,
                                                enabled=device.type == "cuda"):
                output = model(features(backbone, batch, device),
                               batch["view_mask"].to(device), name, sample_latent=False)
            if output["pred"].shape != (1, 8, *expected_shape):
                raise ValueError(f"{name}: decoder output shape mismatch")
            if not torch.isfinite(output["pred"]).all():
                raise ValueError(f"{name}: nonfinite model output")
            report[split][name] = {
                "clips_available": len(dataset), "clips_decoded": len(indices),
                "rgb_shape": list(item["rgb"].shape),
                "target_shape": list(item["target"].shape),
                "view_mask": item["view_mask"][0].tolist(),
                "model_output_shape": list(output["pred"].shape),
                "sample_mean_target": target_sum / target_cells,
                "sample_zero_fraction": zero_cells / target_cells,
                "sample_full_scale_fraction": full_scale_cells / target_cells,
            }
            print(json.dumps({"event": "passed", "split": split, "source": name,
                              **report[split][name]}), flush=True)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2))
    temporary.replace(destination)


if __name__ == "__main__":
    main()
