"""Training-only visual bank refresh, frozen tactile supervision and SAM payloads."""
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .aligned_frames import file_hash, write_json
from .fgw_data import CodeOnlyDataset, final_valid_indices, sample_banks
from .fgw_mixer import MixerConfig, fit_mixer, state_loss
from .sensor_code_cache import CodeCache
from .train import features


def objective_config(cfg):
    supplied = cfg.get("fgw", {})
    fields = MixerConfig.__dataclass_fields__
    return MixerConfig(**{k: v for k, v in supplied.items() if k in fields})


class VisualObjective:
    def __init__(self, cfg, dataset, device, output, saved=None):
        self.cfg, self.dataset, self.device = cfg, dataset, device
        self.mixer = objective_config(cfg)
        self.output = Path(output) / "alignment"
        self.output.mkdir(parents=True, exist_ok=True)
        self.placement = cfg["fgw"]["placement"]
        if self.placement not in ("visual", "tactile_first"):
            raise ValueError("Unknown mixer placement")
        self.initial_variance = (saved or {}).get("initial_variance", {})
        self.scale = (saved or {}).get("scale")
        self.previous = (saved or {}).get("barycenter")
        self.teacher = None
        self.common_caches = {}
        self.teacher_weight = cfg["fgw"].get("teacher_weight", 1.)
        if self.placement == "tactile_first":
            from .tactile_teacher import TactileTeacher
            path = cfg["fgw"]["teacher_checkpoint"]
            if file_hash(path) != cfg["fgw"]["teacher_sha256"]:
                raise ValueError("Frozen common teacher changed")
            state = torch.load(path, map_location="cpu", weights_only=False)
            if state["model_config"]["latent_dim"] != cfg["latent_dim"]:
                raise ValueError("Frozen tactile teacher and visual latent dimensions differ")
            if state["provenance"]["data"] != cfg["fgw"]["data_signature"]:
                raise ValueError("Common teacher data provenance differs")
            self.teacher = TactileTeacher(**state["model_config"]).to(device).eval().requires_grad_(False)
            self.teacher.load_state_dict(state["model"])
            for source in dataset.indices_by_source:
                cache = CodeCache(cfg["fgw"]["teacher_cache"], source)
                signature = cache.index["signature"]
                if (signature["teacher_sha256"] != cfg["fgw"]["teacher_sha256"]
                        or signature["data"] != cfg["fgw"]["data_signature"]
                        or signature["source"] != source
                        or signature["builder_sha256"] != file_hash(Path(__file__).with_name("tactile_teacher.py"))):
                    raise ValueError("Stale common-code cache")
                self.common_caches[source] = cache

    def state_dict(self):
        return {"initial_variance": self.initial_variance, "scale": self.scale, "barycenter": self.previous}

    @torch.no_grad()
    def refresh(self, model, backbone, epoch):
        began = time.monotonic()
        self.banks = sample_banks(self.dataset, self.mixer.bank_size, self.cfg["seed"], epoch)
        model.eval()
        codes, self.frames, self.positions = {}, {}, {}
        for source, indices in self.banks.items():
            chunks, frames = [], []
            for original in DataLoader(Subset(self.dataset, indices), batch_size=self.cfg["batch_size"],
                                       num_workers=self.cfg["workers"]):
                batch = {k: v.to(self.device) if torch.is_tensor(v) and k != "rgb" else v for k, v in original.items()}
                last, valid = final_valid_indices(batch)
                if not valid.all():
                    raise ValueError("Invalid state entered bank")
                with torch.autocast(self.device.type, torch.bfloat16, enabled=self.device.type == "cuda"):
                    mu, _ = model.visual.encode(features(backbone, original, self.device), batch["view_mask"])
                chunks.append(mu[torch.arange(len(mu), device=self.device), last].float().cpu())
                frames.extend(last.cpu().tolist())
            codes[source] = torch.cat(chunks)
            self.frames[source] = dict(zip(indices, frames))
            self.positions[source] = {index: j for j, index in enumerate(indices)}
        if self.placement == "visual":
            self.fit = fit_mixer(codes, self.mixer, self.cfg["seed"] + epoch, self.previous, self.scale)
        else:
            # Common teacher codes were aligned in the code-only stage. Do not
            # run a second visual mixer, which would confound placement tests.
            self.fit = fit_mixer(codes, MixerConfig(variant="none"))
        for source, variance in self.fit["initial_variance"].items():
            self.initial_variance.setdefault(source, variance)
        self.scale = self.fit["scale"]
        if self.fit["barycenter"] is not None:
            self.previous = self.fit["barycenter"]
        info = self.fit["diagnostics"]
        info["refresh_seconds"] = time.monotonic() - began
        info["correspondence_stability"] = None
        previous_path = self.output / f"epoch_{epoch:03d}.pt"
        if previous_path.exists() and self.fit["plans"]:
            before = torch.load(previous_path, map_location="cpu", weights_only=False)
            comparisons = []
            for source, indices in self.banks.items():
                if source not in before["plans"]:
                    continue
                old_positions = {i: j for j, i in enumerate(before["bank_indices"][source])}
                old_plan = before["plans"][source]
                for j, index in enumerate(indices):
                    if index in old_positions:
                        comparisons.append(int(old_plan[:, old_positions[index]].argmax()) ==
                                           int(self.fit["plans"][source][:, j].argmax()))
            if comparisons:
                info["correspondence_stability"] = float(np.mean(comparisons))
        snapshot = {"bank_indices": self.banks, "frame_positions": self.frames,
                    "config": asdict(self.mixer), **self.fit}
        temporary = self.output / "refresh.tmp"
        torch.save(snapshot, temporary); temporary.replace(self.output / f"epoch_{epoch + 1:03d}.pt")
        write_json(self.output / f"epoch_{epoch + 1:03d}.json", info)
        model.train()
        return info

    def payload(self, original, source):
        result = {}
        if self.placement == "tactile_first":
            values, valid = [], []
            cache = self.common_caches[source]
            for episode, frames in zip(original["episode_id"], original["frame_indices"]):
                x, mask = cache.read(episode, frames.numpy(), normalized=False)
                values.append(x); valid.append(mask)
            result.update(teacher_target=torch.stack(values).to(self.device),
                          teacher_valid=torch.stack(valid).to(self.device))
        selected = original["alignment_anchor"].bool().nonzero().flatten()
        if len(selected):
            indices = [int(original["window_index"][j]) for j in selected]
            positions = torch.tensor([self.positions[source][i] for i in indices], device=self.device)
            result.update(selected=selected.to(self.device),
                frames=torch.tensor([self.frames[source][i] for i in indices], device=self.device),
                positions=positions, bank=self.fit["banks"][source].to(self.device),
                target=self.fit["targets"][source].to(self.device)[positions],
                valid=self.fit["valid"][source].to(self.device)[positions],
                initial_variance=self.initial_variance[source])
        return result


def visual_extra_loss(output, payload, cfg, epoch):
    mu = output["mu"]
    loss, metrics = mu.sum() * 0, {}
    if "teacher_target" in payload:
        valid = payload["teacher_valid"]
        error = (mu.float() - payload["teacher_target"].detach()).square().mean(-1)
        teacher = error[valid].mean() if valid.any() else mu.sum() * 0
        loss = loss + cfg["fgw"].get("teacher_weight", 1.) * teacher
        metrics["teacher_mse"] = float(teacher.detach())
    if "selected" in payload:
        mixer = objective_config(cfg)
        value = mu[payload["selected"], payload["frames"]]
        extra, stats = state_loss(value, payload["target"], payload["valid"], payload["bank"],
            payload["positions"], payload["initial_variance"], mixer, epoch)
        loss = loss + extra
        metrics.update(stats)
    return loss, metrics
