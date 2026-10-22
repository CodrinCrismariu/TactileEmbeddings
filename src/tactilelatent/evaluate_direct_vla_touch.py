"""Evaluate a directly trained encoder and native sensor decoder.

Legacy defaults preserve VLA_touch; --source ftp_motiontrans selects the
current torque control. Unlike frozen-transfer evaluation, this decoder was
trained jointly with the visual encoder on the chosen source's train clips.
"""

import argparse
import json
from pathlib import Path

import torch

from .evaluate_holdout_decoder import evaluate
from .frozen_probe import cached_latents, seed_all, sha256
from .joint_all_config import ALL_TARGETS
from .joint_data import JointClipDataset
from .joint_model import JointTactileVAE
from .model import DinoBackbone, MODEL_ID, REVISION
from .train import MODEL_CONFIG_KEYS


def load_direct_model(checkpoint, device, source="ftp_vla_touch"):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = state["config"]
    if state["backbone"] != MODEL_ID or state["backbone_revision"] != REVISION:
        raise ValueError("Checkpoint DINO version differs from this code")
    if (source not in ("ftp_vla_touch", "ftp_motiontrans") or cfg.get("train_sources") != [source] or cfg.get("heldout_source")
            or cfg.get("latent_objective") != "vae"):
        raise ValueError(f"Expected a {source}-only image VAE checkpoint")
    if state["epoch"] != cfg["epochs"] - 1:
        raise ValueError("Direct-training checkpoint has not completed its configured epochs")
    backbone = DinoBackbone().to(device).eval().requires_grad_(False)
    model = JointTactileVAE(input_dim=backbone.dim,
        targets={source: ALL_TARGETS[source]},
        **{key: cfg[key] for key in MODEL_CONFIG_KEYS}).to(device).eval()
    model.load_state_dict(state["model"], strict=True)
    model.requires_grad_(False)
    return backbone, model, state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="processing/runs/vla_touch_direct_vae32_50ep/last.pt")
    parser.add_argument("--source", default="ftp_vla_touch", choices=("ftp_vla_touch", "ftp_motiontrans"))
    parser.add_argument("--manifest", default="processing/joint_all/manifest.json")
    parser.add_argument("--output", default="processing/runs/vla_touch_direct_vae32_50ep/evaluation")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.samples < 2:
        parser.error("--samples must be at least 2")
    output = Path(args.output)
    if (output / "results.json").exists() and not args.overwrite:
        raise FileExistsError(f"Completed evaluation exists: {output / 'results.json'}")
    output.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_hash, manifest_hash = sha256(args.checkpoint), sha256(args.manifest)
    backbone, model, state = load_direct_model(args.checkpoint, device, args.source)
    data = {}
    for split in ("train", "val"):
        dataset = JointClipDataset(args.manifest, split=split, sources=[args.source])
        indices = list(range(len(dataset)))
        data[split] = cached_latents(backbone, model, args.manifest, args.source,
            split, indices, output / f"{split}_features.pt", checkpoint_hash, manifest_hash,
            device, with_targets=True, workers=args.workers)
    if set(data["train"]["episode_id"]) & set(data["val"]["episode_id"]):
        raise ValueError("Training and validation episodes overlap")
    decoder = model.decoders[args.source]
    target_shape = ALL_TARGETS[args.source][1]
    metrics = evaluate(decoder, data["train"], data["val"], device, args.samples,
                       args.seed + 2026, output, target_shape)
    result = {"checkpoint": args.checkpoint, "checkpoint_sha256": checkpoint_hash,
              "checkpoint_epoch": state["epoch"], "manifest_sha256": manifest_hash,
              "source": args.source, "training": "direct_joint_encoder_decoder",
              "frozen_for_evaluation": True, "latent": "mu_logvar",
              "head": f"{state['config']['latent_dim']}-512-{decoder[3].out_features}", "seed": args.seed,
              "target_shape": target_shape,
              "calibration_note": "Empirical calibration of predictions from this decoder; latent samples do not measure decoder-weight uncertainty or guarantee all plausible outcomes.",
              "train_clips": len(data["train"]["mu"]),
              "validation": metrics}
    (output / "results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"event": "complete", "output": str(output / "results.json"),
                      "validation_clips": metrics["validation_clips"],
                      "deterministic_mae": metrics["deterministic_mae"],
                      "ensemble_mean_mae": metrics["ensemble_mean_mae"]}), flush=True)


if __name__ == "__main__":
    main()
