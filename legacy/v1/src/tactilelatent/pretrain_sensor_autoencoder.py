"""Train one native sensor AE from tactile-only caches; never load scene RGB/DINO."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .joint_all_config import ALL_TARGETS
from .pretrain_egotouch_autoencoder import atomic_save, frame_batches, reconstruction_loss
from .sensor_autoencoder import SensorAutoencoder
from .sensor_cache import build_cache, CachedSensorFrames, cache_signature, write_json
from .sensor_data import SENSORS
from .native_optical import NativeOpticalFrames, native_shape


def run_epoch(model, loader, device, config, optimizer=None):
    model.train(optimizer is not None)
    sums = torch.zeros(12, dtype=torch.float64, device=device)
    frames = 0
    pressure = config["target_kind"] in ("pressure_grid", "taxel_vector")
    with torch.set_grad_enabled(optimizer is not None):
        for batch in loader:
            target, valid = batch["target"].to(device), batch["valid"].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=config.get("amp", False) and device.type == "cuda"):
                prediction = model(target, valid)
                loss = reconstruction_loss(prediction, target, valid, config["contact_weight"] if pressure else 1., .1, config["mae_weight"])
            if not torch.isfinite(loss):
                raise ValueError("Non-finite autoencoder loss")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
            with torch.no_grad():
                error = (prediction.float() - target).abs()
                contact = valid & (target > .05) if pressure else torch.zeros_like(valid)
                predicted = valid & (prediction > .05) if pressure else torch.zeros_like(valid)
                high = valid & (target >= .5) if pressure else torch.zeros_like(valid)
                values = [valid.sum(), error[valid].sum(), error[valid].square().sum(),
                          loss.detach() * valid.sum(), error[contact].sum(), contact.sum(),
                          error[high].sum(), high.sum(), (contact & predicted).sum(),
                          (predicted & ~contact).sum(), (contact & ~predicted).sum(),
                          (target[valid] - config["baseline_value"]).abs().sum()]
                sums += torch.stack(values).double(); frames += len(target)
    cells, absolute, squared, losses, contact_error, contacts, high_error, highs, tp, fp, fn, baseline = sums.tolist()
    if not cells:
        raise ValueError("No valid sensor values")
    mse, mae = squared / cells, absolute / cells
    contact_mae = contact_error / contacts if contacts else None
    high_mae = high_error / highs if highs else None
    score = (mae + (contact_mae if contact_mae is not None else mae) + (high_mae if high_mae is not None else mae)) / 3 if pressure else mae
    return {"frames": frames, "valid_cells": int(cells), "loss": losses / cells, "mae": mae,
            "mse": mse, "contact_mae": contact_mae, "high_pressure_mae": high_mae,
            "contact_f1": 2 * tp / (2 * tp + fp + fn) if pressure and 2 * tp + fp + fn else None,
            "psnr": -10 * np.log10(mse) if config["target_kind"] == "optical_rgb" and mse > 0 else None,
            "constant_baseline_mae": baseline / cells, "selection_score": score}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, choices=list(SENSORS))
    parser.add_argument("--manifest", default="processing/joint_all/manifest.json")
    parser.add_argument("--cache", default="processing/sensor_ae_cache_v1")
    parser.add_argument("--output", required=True)
    parser.add_argument("--architecture", choices=("conv", "mlp"), default="conv")
    parser.add_argument("--sensor-code-dim", dest="latent_dim", type=int, default=64)
    parser.add_argument("--channels", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--hidden-dims", type=int, nargs="*", default=[256, 128],
                        help="MLP hidden widths; an empty list gives a single affine encoder/decoder")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--mae-weight", type=float, default=.5)
    parser.add_argument("--contact-weight", type=float, default=3.)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--max-train-frames", type=int, default=100000)
    parser.add_argument("--max-val-frames", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--native-optical", action="store_true", help="Read every original optical pixel directly; never use reduced caches")
    parser.add_argument("--amp", action="store_true", help="CUDA bfloat16 computation; targets and metrics remain float32")
    args = parser.parse_args(argv)
    if min(args.epochs, args.batch_size, args.latent_dim, args.frame_stride) < 1:
        parser.error("Epochs, batch size, code and stride must be positive")
    if min(args.workers, args.max_train_frames, args.max_val_frames, args.weight_decay, args.mae_weight) < 0 or args.learning_rate <= 0:
        parser.error("Invalid optimizer or data settings")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError(f"Choose a fresh output or --resume: {output}")
    cfg = {k: v for k, v in vars(args).items() if k != "resume"}
    kind, shape, _ = ALL_TARGETS[args.source]
    if kind == "optical_rgb" and not args.native_optical:
        parser.error("Optical training requires --native-optical; reduced-resolution training is disabled")
    if args.native_optical:
        if kind != "optical_rgb": parser.error("--native-optical requires an optical dataset")
        rows = [r for r in json.loads(Path(args.manifest).read_text())["episodes"] if r["source"] == args.source]
        shape = native_shape(rows[0])
    cfg.update(target_kind=kind, baseline_value=.5 if kind in ("force_tanh", "torque_tanh") else 0.,
               cache_signature=cache_signature(args.manifest, args.source, args.frame_stride),
               selection_metric="balanced_mae" if kind in ("pressure_grid", "taxel_vector") else "mae")
    cfg["model"] = {"target_shape": shape, "architecture": args.architecture, "latent_dim": args.latent_dim,
                    "channels": args.channels, "hidden_dims": args.hidden_dims, "output_mode": "sigmoid"}
    if args.native_optical:
        cfg["cache_signature"] = {"format": "original-optical-direct-v1", "stride": args.frame_stride,
            "manifest_sha256": hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
            "loader_sha256": hashlib.sha256(Path(__file__).with_name("native_optical.py").read_bytes()).hexdigest()}
    else:
        build_cache(args.manifest, args.source, args.cache, args.frame_stride)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    model = SensorAutoencoder(**cfg["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=6, factor=.5, min_lr=1e-6)
    train, val = ([NativeOpticalFrames(args.manifest, args.source, split, args.frame_stride) for split in ("train", "val")]
                  if args.native_optical else [CachedSensorFrames(args.cache, args.source, split) for split in ("train", "val")])
    best, start = float("inf"), 0
    if args.resume:
        saved = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
        previous = saved["config"]
        if args.epochs < previous["epochs"] or {k:v for k,v in cfg.items() if k != "epochs"} != {k:v for k,v in previous.items() if k != "epochs"}:
            raise ValueError("Resume may increase only the epoch budget")
        model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"]); torch.set_rng_state(saved["torch_rng"])
        random.setstate(saved["python_rng"]); np.random.set_state(saved["numpy_rng"])
        if device.type == "cuda": torch.cuda.set_rng_state_all(saved["cuda_rng"])
        best, start = saved["best_score"], saved["epoch"] + 1
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", cfg)
    summary = {"source": args.source, "target_kind": kind, "model": cfg["model"],
               "encoder_parameters": sum(p.numel() for p in model.encoder.parameters()),
               "decoder_parameters": sum(p.numel() for p in model.decoder.parameters()),
               "train_frames_available": len(train), "val_frames_available": len(val)}
    write_json(output / "summary.json", summary)
    print(json.dumps({"event": "initialized", **summary}), flush=True)
    for epoch in range(start, args.epochs):
        began = time.monotonic()
        lr = optimizer.param_groups[0]["lr"]
        batches = list(frame_batches(train, args.batch_size, args.max_train_frames, args.seed + epoch, not args.native_optical))
        if args.native_optical: np.random.default_rng(args.seed + epoch).shuffle(batches)
        train_loader = DataLoader(train, batch_sampler=batches, num_workers=args.workers,
            pin_memory=device.type == "cuda")
        val_loader = DataLoader(val, batch_sampler=list(frame_batches(val, args.batch_size,
            args.max_val_frames, args.seed + 100000, False)), num_workers=args.workers,
            pin_memory=device.type == "cuda")
        training = run_epoch(model, train_loader, device, cfg, optimizer)
        validation = run_epoch(model, val_loader, device, cfg)
        score = validation["selection_score"]; improved = score < best; best = min(best, score)
        scheduler.step(score)
        record = {"epoch": epoch + 1, "source": args.source, "train": training, "val": validation,
                  "selection_score": score, "best_score": best, "learning_rate": lr,
                  "seconds": time.monotonic() - began}
        saved = {"epoch": epoch, "config": cfg, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(), "best_score": best, "torch_rng": torch.get_rng_state(),
                 "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                 "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}
        if improved:
            atomic_save({"model_config": cfg["model"], "model": model.state_dict(), "config": cfg,
                         "encoder": model.encoder.state_dict(), "decoder": model.decoder.state_dict(),
                         "epoch": epoch, "validation": validation}, output / "best.pt")
            write_json(output / "best_metrics.json", record)
        atomic_save(saved, output / "last.pt")
        with (output / "metrics.jsonl").open("a") as stream: stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
