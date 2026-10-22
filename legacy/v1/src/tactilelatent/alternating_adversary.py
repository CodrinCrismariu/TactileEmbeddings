"""Balanced, recent latents for the two-timescale dataset adversary.

The replay bank contains only detached visual latents. It never stores images,
tactile targets, or gradients, and each source contributes equally to a probe
update regardless of its original dataset size.
"""

from collections import deque

import torch
import torch.nn.functional as F

from .train_joint_adversarial import reverse


class BalancedLatentBank:
    def __init__(self, sources, capacity=256, minimum=64):
        if capacity < 1 or not 1 <= minimum <= capacity:
            raise ValueError("Require 1 <= minimum <= capacity")
        self.sources = tuple(sources)
        self.capacity = int(capacity)
        self.minimum = int(minimum)
        self.entries = {source: deque(maxlen=self.capacity) for source in self.sources}

    def add(self, source, mu, sampled):
        if source not in self.entries or mu.shape != sampled.shape or mu.ndim != 3:
            raise ValueError("Expected matching [batch, time, latent] tensors for a known source")
        for mean_clip, sample_clip in zip(mu.detach(), sampled.detach()):
            self.entries[source].append((mean_clip.float().cpu().clone(),
                                         sample_clip.float().cpu().clone()))

    def ready(self):
        return all(len(self.entries[source]) >= self.minimum for source in self.sources)

    def counts(self):
        return {source: len(self.entries[source]) for source in self.sources}

    def sample(self, clips_per_source, device):
        if not self.ready():
            raise RuntimeError("Balanced bank is not ready")
        means, samples, labels = [], [], []
        for label, source in enumerate(self.sources):
            entries = self.entries[source]
            choices = torch.randint(len(entries), (clips_per_source,)).tolist()
            for index in choices:
                mean_clip, sample_clip = entries[index]
                means.append(mean_clip)
                samples.append(sample_clip)
                labels.append(torch.full((len(mean_clip),), label, dtype=torch.long))
        return (torch.cat(means).to(device), torch.cat(samples).to(device),
                torch.cat(labels).to(device))

    def state_dict(self):
        return {"sources": self.sources, "capacity": self.capacity,
                "minimum": self.minimum,
                "entries": {name: list(self.entries[name]) for name in self.sources}}

    def load_state_dict(self, state):
        if (tuple(state["sources"]) != self.sources or state["capacity"] != self.capacity
                or state["minimum"] != self.minimum):
            raise ValueError("Incompatible balanced replay bank")
        self.entries = {name: deque(state["entries"][name], maxlen=self.capacity)
                        for name in self.sources}


def classifier_loss(classifier, means, samples, labels):
    return .5 * (F.cross_entropy(classifier(means), labels) +
                 F.cross_entropy(classifier(samples), labels))


def fit_classifiers(classifiers, optimizer, bank, *, steps, clips_per_source, device):
    if not bank.ready():
        return 0
    for classifier in classifiers:
        classifier.requires_grad_(True).train()
    for _ in range(steps):
        means, samples, labels = bank.sample(clips_per_source, device)
        optimizer.zero_grad(set_to_none=True)
        loss = sum(classifier_loss(classifier, means, samples, labels)
                   for classifier in classifiers)
        loss.backward()
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    for classifier in classifiers:
        classifier.requires_grad_(False).eval()
    return steps


def reset_challenger(challenger, optimizer):
    for layer in challenger.modules():
        if layer is not challenger and hasattr(layer, "reset_parameters"):
            layer.reset_parameters()
    for parameter in challenger.parameters():
        optimizer.state.pop(parameter, None)
    challenger.requires_grad_(False).eval()


def choose_challenger(classifiers, mu, sampled, source_index):
    """Pick the strongest *current* classifier without training either head."""
    labels = torch.full((mu.numel() // mu.shape[-1],), source_index,
                        dtype=torch.long, device=mu.device)
    means, samples = mu.flatten(0, 1), sampled.flatten(0, 1)
    with torch.no_grad():
        losses = [classifier_loss(model, means, samples, labels)
                  for model in classifiers]
        accuracies = [.5 * ((model(means).argmax(-1) == labels).float().mean() +
                            (model(samples).argmax(-1) == labels).float().mean())
                      for model in classifiers]
    selected = min(range(len(classifiers)), key=lambda index: losses[index].item())
    return selected, [x.item() for x in losses], [x.item() for x in accuracies]


def reversed_loss(classifier, mu, sampled, source_index, strength):
    labels = torch.full((mu.numel() // mu.shape[-1],), source_index,
                        dtype=torch.long, device=mu.device)
    return classifier_loss(classifier, reverse(mu.flatten(0, 1), strength),
                           reverse(sampled.flatten(0, 1), strength), labels)
