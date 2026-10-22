"""Episode-disjoint, rotating-source differentiable head adaptation.

Only fast head parameters change in the inner loop. Query gradients pass
through support adaptation to the shared encoder; no cached latent is detached.
"""
from collections import OrderedDict, defaultdict
import random

import torch
from torch import nn
from torch.func import functional_call
from torch.utils.data import default_collate

from .fgw_data import source_seed
from .relational import HELDOUT
from .train import features


def make_head(latent_dim, output_dim, hidden_dims=(512,), numeric=False, dtype=None, device=None):
    if min(latent_dim, output_dim, *hidden_dims) < 1:
        raise ValueError("Positive head dimensions required")
    layers, previous = [], latent_dim
    for width in hidden_dims:
        layers += [nn.Linear(previous, width, device=device, dtype=dtype), nn.GELU()]
        previous = width
    layers.append(nn.Linear(previous, output_dim, device=device, dtype=dtype))
    if numeric:
        layers.append(nn.Sigmoid())
    return nn.Sequential(*layers)


def target_loss(prediction, target, valid, numeric=False):
    target, valid = target.detach(), valid.bool()
    if prediction.shape != target.shape or valid.shape != target.shape:
        raise ValueError("Meta target shapes differ")
    valid = valid & torch.isfinite(target)
    if not valid.any():
        raise ValueError("A support/query set has no valid targets")
    difference = prediction[valid] - target[valid]
    return (difference.square() + (.5 * difference.abs() if numeric else 0.)).mean()


def adapt_head(head, support, target, valid, steps=5, learning_rate=.01, numeric=False):
    """Functional SGD with exact support-to-encoder derivatives (not FOMAML)."""
    if steps < 1 or learning_rate <= 0:
        raise ValueError("Positive inner steps and learning rate required")
    parameters = OrderedDict(head.named_parameters())
    for _ in range(steps):
        prediction = functional_call(head, parameters, (support,))
        loss = target_loss(prediction, target, valid, numeric)
        grads = torch.autograd.grad(loss, tuple(parameters.values()), create_graph=True)
        parameters = OrderedDict((name, p - learning_rate * g)
                                 for (name, p), g in zip(parameters.items(), grads))
    return parameters


class TaskSampler:
    """Stateless schedule: task number determines source, episodes, and clips."""
    def __init__(self, dataset, seed=45, support_episodes=4, query_episodes=4, group_field=None):
        if min(support_episodes, query_episodes) < 1:
            raise ValueError("Support and query episodes must be positive")
        if HELDOUT in dataset.indices_by_source or any(r["split"] != "train" for r in dataset.episodes):
            raise ValueError("Meta tasks require source training episodes only")
        self.dataset, self.seed = dataset, seed
        self.support_count, self.query_count = support_episodes, query_episodes
        self.sources = sorted(dataset.indices_by_source)
        self.episodes = defaultdict(lambda: defaultdict(list))
        self.groups = {}
        for source, indices in dataset.indices_by_source.items():
            for index in indices:
                ep = dataset.windows[index][0]
                row = dataset.episodes[ep]
                self.episodes[source][ep].append(index)
                if group_field and group_field not in row:
                    raise ValueError(f"Requested grouping field missing: {source}:{group_field}")
                self.groups[ep] = str(row[group_field]) if group_field else row["episode_id"]
            if len(set(self.groups[ep] for ep in self.episodes[source])) < support_episodes + query_episodes:
                raise ValueError(f"Insufficient disjoint episode groups: {source}")

    def source(self, number):
        return self.sources[number % len(self.sources)]

    def sample(self, number):
        source = self.source(number)
        rng = random.Random(source_seed(self.seed, number, source))
        groups = defaultdict(list)
        for ep in self.episodes[source]:
            groups[self.groups[ep]].append(ep)
        chosen = rng.sample(sorted(groups), self.support_count + self.query_count)
        episodes = [rng.choice(groups[group]) for group in chosen]
        indices = [rng.choice(self.episodes[source][ep]) for ep in episodes]
        return {"number": number, "source": source,
                "support": indices[:self.support_count], "query": indices[self.support_count:],
                "support_episodes": [self.dataset.episodes[e]["episode_id"] for e in episodes[:self.support_count]],
                "query_episodes": [self.dataset.episodes[e]["episode_id"] for e in episodes[self.support_count:]],
                "support_groups": chosen[:self.support_count], "query_groups": chosen[self.support_count:]}


def prepare_task(task, dataset, backbone, device):
    """Encode frozen DINO once per support/query clip, in bounded microbatches."""
    result = {"task": task}
    for partition in ("support", "query"):
        encoded, masks, targets, valid = [], [], [], []
        for index in task[partition]:
            original = default_collate([dataset[index]])
            with torch.no_grad(), torch.autocast(device.type, torch.bfloat16, enabled=device.type == "cuda"):
                encoded.append(features(backbone, original, device).detach())
            masks.append(original["view_mask"].to(device))
            if "code_target" in original:
                target = original["code_target"]
                mask = original["code_valid"].unsqueeze(-1).expand_as(target)
                numeric = False
            else:
                target = original["target"].flatten(2)
                mask = original["target_mask"].flatten(2)
                numeric = True
            targets.append(target.to(device)); valid.append(mask.to(device))
        result[partition] = {"features": torch.cat(encoded), "mask": torch.cat(masks),
                             "target": torch.cat(targets), "valid": torch.cat(valid)}
    result["numeric"] = numeric
    return result


def meta_loss(model, prepared, cfg):
    """Query supervision on mean and sampled latents, with no persistent adapter."""
    if prepared["task"]["source"] == HELDOUT:
        raise ValueError("Held-out source in meta loss")
    latent = {}
    for name in ("support", "query"):
        data = prepared[name]
        # Microbatch visual forward to bound attention workspace; keep graphs.
        values = [model.visual.encode(x, m) for x, m in zip(data["features"].split(1), data["mask"].split(1))]
        mu = torch.cat([v[0] for v in values]).float()
        logvar = torch.cat([v[1] for v in values]).float()
        if cfg.get("sample_latent", True):
            z = mu + torch.randn_like(mu) * (.5 * logvar).exp()
            latent[name] = torch.cat([mu, z])
        else:
            latent[name] = mu
    support, query = prepared["support"], prepared["query"]
    repeat = 2 if cfg.get("sample_latent", True) else 1
    sy, sm = support["target"].repeat(repeat, 1, 1), support["valid"].repeat(repeat, 1, 1)
    qy, qm = query["target"].repeat(repeat, 1, 1), query["valid"].repeat(repeat, 1, 1)
    # Head arithmetic and differentiable SGD stay FP32, even in AMP training.
    with torch.autocast(latent["support"].device.type, enabled=False):
        head = make_head(latent["support"].shape[-1], sy.shape[-1],
                         cfg.get("hidden_dims", [512]), prepared["numeric"], device=sy.device)
        initial = target_loss(head(latent["query"]), qy, qm, prepared["numeric"]).detach()
        weights = adapt_head(head, latent["support"], sy, sm, cfg.get("inner_steps", 5),
                             cfg.get("inner_lr", .01), prepared["numeric"])
        loss = target_loss(functional_call(head, weights, (latent["query"],)), qy, qm, prepared["numeric"])
    return loss, {"meta_query_before": float(initial), "meta_query_after": float(loss.detach()),
                  "meta_source": prepared["task"]["source"], "meta_task": prepared["task"]["number"]}
