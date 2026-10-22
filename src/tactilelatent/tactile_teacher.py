"""Learn a common tactile teacher from existing codes; never decode sensor images."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from .aligned_frames import file_hash, manifest_signature, targets, write_json
from .fgw_data import BankSampler, CodeOnlyDataset, final_valid_indices, sample_banks
from .fgw_mixer import HELDOUT, fit_mixer, state_loss
from .fgw_objective import objective_config
from .joint_all_config import ALL_TARGETS
from .joint_data import SourceBatchSampler
from .sam import restore_rng, rng_snapshot
from .sensor_code_cache import CodeCache, cache_signature, train_statistics, verify_teachers
from .train_alignment import atomic_checkpoint


def source_context(cfg):
    signature = manifest_signature(cfg["manifest"])
    audit = json.loads(Path(cfg["audit"]).read_text())
    if audit.get("signature") != signature or not audit.get("passed") or not audit.get("VLA_complete_pixels"):
        raise ValueError("Passing native chronological audit required")
    registry = json.loads(Path(cfg["teachers"]).read_text())
    verify_teachers(registry)
    rows = json.loads(Path(cfg["manifest"]).read_text())["episodes"]
    sources = sorted({r["source"] for r in rows} - {HELDOUT})
    statistics, dimensions, caches = {}, {}, {}
    for source in sources:
        if source in registry["sources"]:
            expected = cache_signature(cfg["manifest"], registry, source, signature)
            cache = CodeCache(cfg["code_cache"], source, expected)
            dimensions[source] = cache.index["code_dim"]
            statistics[source] = cache.index["statistics"]
            caches[source] = expected
    numeric = {s: ALL_TARGETS[s][1] for s in sources if s not in dimensions}
    if set(numeric) != {"ftp_rdp", "ftp_rdp_bimanual"}:
        raise ValueError("Missing pinned sensor teacher")
    provenance = {"data": signature, "teachers": registry, "code_caches": caches,
                  "teacher_code_sha256": file_hash(__file__),
                  "mixer_sha256": file_hash(Path(__file__).with_name("fgw_mixer.py")),
                  "bank_data_sha256": file_hash(Path(__file__).with_name("fgw_data.py"))}
    return signature, registry, sources, dimensions, numeric, statistics, provenance


class TactileTeacher(nn.Module):
    def __init__(self, sensor_code_dims, numeric_shapes, latent_dim=32, hidden_dim=128):
        super().__init__()
        if HELDOUT in sensor_code_dims or HELDOUT in numeric_shapes:
            raise ValueError("Held-out source cannot enter teacher")
        self.projections, self.adapters = nn.ModuleDict(), nn.ModuleDict()
        self.numeric_sources = set(numeric_shapes)
        dimensions = {**sensor_code_dims, **{s: math.prod(shape) for s, shape in numeric_shapes.items()}}
        for source, dimension in dimensions.items():
            if source in numeric_shapes:
                self.projections[source] = nn.Linear(dimension, latent_dim)
                self.adapters[source] = nn.Sequential(nn.Linear(latent_dim, dimension), nn.Sigmoid())
            else:
                self.projections[source] = nn.Sequential(nn.Linear(dimension, hidden_dim), nn.GELU(),
                                                         nn.Linear(hidden_dim, latent_dim))
                self.adapters[source] = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.GELU(),
                                                      nn.Linear(hidden_dim, dimension))

    def encode(self, source, value):
        return self.projections[source](value)

    def forward(self, source, value):
        z = self.encode(source, value)
        return z, self.adapters[source](z)


def reconstruction_loss(model, source, predicted, batch):
    error = predicted.float() - batch["value"].float()
    cells = error.square() + (.5 * error.abs() if source in model.numeric_sources else 0.)
    error = cells.mean(-1)
    valid = batch["valid"].bool()
    return error[valid].mean() if valid.any() else error.sum() * 0


@torch.no_grad()
def teacher_validate(model, loader, device):
    model.eval(); sums, counts = {}, {}
    for original in loader:
        source = original["source"][0]
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in original.items()}
        _, prediction = model(source, batch["value"])
        loss = reconstruction_loss(model, source, prediction, batch)
        n = int(batch["valid"].sum())
        sums[source] = sums.get(source, 0.) + float(loss) * n
        counts[source] = counts.get(source, 0) + n
    model.train()
    return {s: sums[s] / max(counts[s], 1) for s in sums}


def train_teacher(cfg, device="cuda", resume=False, stop_after_epoch=None):
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    torch.set_num_threads(4); device = torch.device(device)
    signature, registry, sources, dimensions, numeric, _, provenance = source_context(cfg)
    kwargs = dict(code_cache=cfg["code_cache"], registry=registry, data_signature=signature,
                  include_targets=False)
    dataset = CodeOnlyDataset(cfg["manifest"], "train", sources, **kwargs)
    validation = CodeOnlyDataset(cfg["manifest"], "val", sources, **kwargs)
    mixer = objective_config(cfg)
    sampler = BankSampler(dataset, cfg.get("teacher_batch_size", 256), cfg["seed"],
                          cfg["train_windows_per_source_per_epoch"], mixer.bank_fraction)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=cfg["workers"])
    val_sampler = SourceBatchSampler(validation, cfg.get("teacher_batch_size", 256),
        seed=cfg["seed"], max_windows_per_source=cfg.get("teacher_validation_windows", 1024))
    val_loader = DataLoader(validation, batch_sampler=val_sampler, num_workers=cfg["workers"])
    model_cfg = dict(sensor_code_dims=dimensions, numeric_shapes=numeric, latent_dim=32, hidden_dim=128)
    model = TactileTeacher(**model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.get("teacher_learning_rate", 1e-3), weight_decay=.01)
    output = Path(cfg["output"]); output.mkdir(parents=True, exist_ok=True)
    path = output / "last.pt"
    if path.exists() and not resume:
        raise FileExistsError("Teacher output exists; use resume")
    start, initial, previous, scale, elapsed = 0, {}, None, None, 0.
    checkpoint = path if resume else cfg.get("init_teacher_checkpoint")
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if state["model_config"] != model_cfg or state["provenance"] != provenance:
            raise ValueError("Teacher fork/resume provenance mismatch")
        if resume and state["config"] != cfg:
            raise ValueError("Teacher resume configuration mismatch")
        if not resume and state["epoch"] + 1 != cfg.get("teacher_warmup_epochs", 20):
            raise ValueError("Teacher fork must start at reconstruction warmup boundary")
        if not resume and state["config"]["seed"] != cfg["seed"]:
            raise ValueError("Teacher warmup seed differs")
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        start = state["epoch"] + 1
        initial, previous, scale = state["initial_variance"], state["barycenter"], state["scale"]
        if resume:
            elapsed = state.get("seconds", 0.)
        restore_rng(state["rng"]); random.setstate(state["python_rng"]); np.random.set_state(state["numpy_rng"])
    write_json(output / "config.json", cfg)
    began = time.monotonic() - elapsed
    for epoch in range(start, min(cfg.get("teacher_epochs", 70), stop_after_epoch or 10**9)):
        bank_indices = sample_banks(dataset, mixer.bank_size, cfg["seed"], epoch)
        codes, positions, final_frames = {}, {}, {}
        model.eval()
        with torch.no_grad():
            for source, indices in bank_indices.items():
                batches, frames = [], []
                for batch in DataLoader(Subset(dataset, indices), batch_size=256):
                    z = model.encode(source, batch["value"].to(device))
                    last, valid = final_valid_indices(batch)
                    if not valid.all():
                        raise ValueError("Invalid teacher bank state")
                    batches.append(z[torch.arange(len(z), device=device), last.to(device)].float().cpu())
                    frames.extend(last.tolist())
                codes[source] = torch.cat(batches)
                positions[source] = {i: j for j, i in enumerate(indices)}
                final_frames[source] = dict(zip(indices, frames))
        active = epoch >= cfg.get("teacher_warmup_epochs", 20)
        current = mixer if active else type(mixer)(**{**asdict(mixer), "variant": "none"})
        fit = fit_mixer(codes, current, cfg["seed"] + epoch, previous, scale)
        if active:
            scale = fit["scale"]; previous = fit["barycenter"]
        for source, variance in fit["initial_variance"].items():
            initial.setdefault(source, variance)
        model.train(); sampler.set_epoch(epoch, bank_indices)
        totals, counts = {}, {}
        for batch in loader:
            source = batch["source"][0]
            values = batch["value"].to(device)
            valid_batch = {"value": values, "valid": batch["valid"].to(device)}
            z, reconstructed = model(source, values)
            loss = reconstruction_loss(model, source, reconstructed, valid_batch)
            selected = batch["alignment_anchor"].bool().nonzero().flatten()
            if len(selected):
                indices = [int(batch["window_index"][j]) for j in selected]
                where = torch.tensor([positions[source][i] for i in indices], device=device)
                frames = torch.tensor([final_frames[source][i] for i in indices], device=device)
                extra, _ = state_loss(z[selected.to(device), frames], fit["targets"][source].to(device)[where],
                    fit["valid"][source].to(device)[where], fit["banks"][source].to(device), where,
                    initial[source], current, epoch - cfg.get("teacher_warmup_epochs", 20))
                loss = loss + extra
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite tactile teacher loss")
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            totals[source] = totals.get(source, 0.) + float(loss.detach()); counts[source] = counts.get(source, 0) + 1
        val = teacher_validate(model, val_loader, device)
        record = {"epoch": epoch + 1, "train": {s: totals[s] / counts[s] for s in totals},
                  "validation": val, "alignment": fit["diagnostics"], "seconds": time.monotonic() - began}
        state = {"teacher_type": "fgw_tactile_teacher_v1", "config": cfg, "model_config": model_cfg,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
            "provenance": provenance, "initial_variance": initial, "barycenter": previous,
            "scale": scale, "rng": rng_snapshot(), "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(), "seconds": record["seconds"]}
        atomic_checkpoint(path, state)
        alignment = output / "alignment"; alignment.mkdir(exist_ok=True)
        torch.save({"bank_indices": bank_indices, **fit}, alignment / f"epoch_{epoch + 1:03d}.pt")
        with (output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
    return path


@torch.no_grad()
def cache_teacher(cfg, checkpoint, root, device="cuda"):
    signature, registry, sources, _, _, _, provenance = source_context(cfg)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state["provenance"] != provenance or state["epoch"] + 1 != cfg.get("teacher_epochs", 70):
        raise ValueError("Only a complete, compatible tactile teacher can be cached")
    model = TactileTeacher(**state["model_config"]).to(device).eval().requires_grad_(False)
    model.load_state_dict(state["model"])
    all_rows = json.loads(Path(cfg["manifest"]).read_text())["episodes"]
    for source in sources:
        folder = Path(root) / source; folder.mkdir(parents=True, exist_ok=True)
        expected = {"teacher_sha256": file_hash(checkpoint), "data": signature, "source": source,
                    "builder_sha256": file_hash(__file__), "dtype": "float32", "latent_dim": 32}
        index_path = folder / "index.json"
        if index_path.exists():
            if json.loads(index_path.read_text())["signature"] != expected:
                raise ValueError("Common-code cache provenance differs")
            continue
        rows = [r for r in all_rows if r["source"] == source and r["split"] in ("train", "val")]
        offsets = np.concatenate(([0], np.cumsum([r["frames"] for r in rows]))).tolist()
        progress_path = folder / "progress.json"
        progress = json.loads(progress_path.read_text()) if progress_path.exists() else {
            "signature": expected, "completed_episodes": []}
        if progress["signature"] != expected:
            raise ValueError("Partial common-code cache provenance differs")
        arrays = []
        for name, shape, dtype in (("codes", (offsets[-1], 32), np.float32), ("valid", (offsets[-1],), np.bool_)):
            path = folder / f"{name}.partial.npy"
            if not path.exists() and (folder / f"{name}.npy").exists():
                (folder / f"{name}.npy").replace(path)  # Recover interrupted finalization.
            array = np.load(path, mmap_mode="r+") if path.exists() else np.lib.format.open_memmap(
                path, mode="w+", dtype=dtype, shape=shape)
            if array.shape != shape or array.dtype != dtype:
                raise ValueError("Common-code cache shape/dtype differs")
            arrays.append(array)
        codes, valid = arrays
        raw = CodeCache(cfg["code_cache"], source) if source in registry["sources"] else None
        for episode, row in enumerate(rows):
            if episode in progress["completed_episodes"]:
                continue
            for start in range(0, row["frames"], 16384):
                frames = np.arange(start, min(start + 16384, row["frames"]))
                if raw is not None:
                    values, finite = raw.read(row["episode_id"], frames)
                else:
                    data = targets(row, frames).reshape(len(frames), -1)
                    mask = np.isfinite(data)
                    values, finite = torch.from_numpy(np.where(mask, data, 0.)).float(), torch.from_numpy(mask.all(1))
                z = model.encode(source, values.to(device)).float().cpu().numpy()
                if not np.isfinite(z).all():
                    raise FloatingPointError("Nonfinite common teacher codes")
                lo = offsets[episode] + start
                codes[lo:lo + len(z)] = z; valid[lo:lo + len(z)] = finite.numpy()
            codes.flush(); valid.flush()
            progress["completed_episodes"].append(episode); write_json(progress_path, progress)
        stats = train_statistics(codes, valid, rows, offsets)
        codes.flush(); valid.flush()
        for name in ("codes", "valid"):
            (folder / f"{name}.partial.npy").replace(folder / f"{name}.npy")
        write_json(index_path, {"signature": expected, "episodes": rows, "offsets": offsets,
            "frames": offsets[-1], "code_dim": 32, "statistics": stats,
            "normalization": "none-common-coordinates-preserved"})
        print(json.dumps({"event": "common_code_cache", "source": source, "frames": offsets[-1]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-epoch", type=int)
    parser.add_argument("--cache-only", action="store_true")
    args = parser.parse_args(); cfg = json.loads(Path(args.config).read_text())
    checkpoint = Path(cfg["output"]) / "last.pt" if args.cache_only else train_teacher(
        cfg, args.device, args.resume, args.stop_after_epoch)
    if args.cache_only:
        cache_teacher(cfg, checkpoint, cfg["common_code_cache"], args.device)


if __name__ == "__main__":
    main()
