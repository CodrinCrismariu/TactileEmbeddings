"""Probe dataset identity in the frozen 32D Gaussian latent of the joint model."""

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from .frozen_probe import (CHECKPOINT, MANIFEST, SOURCES, cached_latents,
                           fit_head, head, latent_draw, load_encoder, seed_all,
                           select_indices, sha256)
from .alignment_data import evaluation_dataset as JointClipDataset
from .joint_all_config import ALL_TARGETS


LEVELS = (.5, .8, .9, .95)


def class_table(parts, device, balanced=False):
    if balanced:
        # Repeat only within a partition so each source contributes equally,
        # including sources with fewer available episodes/clips than the cap.
        maximum = max(cache["mu"].numel() // cache["mu"].shape[-1] for cache in parts)
        parts = [{key: cache[key].flatten(0, 1)[
            torch.arange(maximum) % (cache["mu"].numel() // cache["mu"].shape[-1])].unsqueeze(1)
                  for key in ("mu", "logvar")} for cache in parts]
    return {"mu": torch.cat([cache["mu"].flatten(0, 1) for cache in parts]).to(device),
            "logvar": torch.cat([cache["logvar"].flatten(0, 1) for cache in parts]).to(device),
            "y": torch.cat([torch.full((cache["mu"].numel() // cache["mu"].shape[-1],),
                                       index, dtype=torch.long)
                            for index, cache in enumerate(parts)]).to(device)}


def entropy(probabilities):
    return -(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1)


def probability_set(probabilities, truth, level):
    """Smallest top-probability set whose cumulative mass reaches level."""
    values, order = probabilities.sort(dim=-1, descending=True)
    sizes = (values.cumsum(-1) < level).sum(-1).add(1).clamp(max=probabilities.shape[-1])
    selected = torch.arange(probabilities.shape[-1], device=probabilities.device)[None] < sizes[:, None]
    included = (order == truth[:, None]) & selected
    return included.any(-1), sizes, (values * selected).sum(-1)


def metric_state():
    return {"count": 0, "correct": 0, "nll_sum": 0., "brier_sum": 0.,
            "confidence_sum": 0., "calibration_bins": [[0, 0., 0] for _ in range(10)],
            "entropy_sum": 0.,
            "correct_entropy_sum": 0., "wrong_entropy_sum": 0.,
            "disagreement_sum": 0.,
            "sets": {str(level): {"covered": 0, "size": 0, "mass": 0.}
                     for level in LEVELS}}


def update_metrics(state, probabilities, truth, disagreement=None):
    prediction = probabilities.argmax(-1)
    correct = prediction == truth
    state["count"] += len(truth)
    state["correct"] += correct.sum().item()
    state["nll_sum"] += -probabilities.gather(1, truth[:, None]).clamp_min(1e-12).log().sum().item()
    one_hot = F.one_hot(truth, probabilities.shape[-1]).to(probabilities.dtype)
    state["brier_sum"] += (probabilities - one_hot).square().sum(-1).sum().item()
    confidence = probabilities.max(-1).values
    state["confidence_sum"] += confidence.sum().item()
    bin_ids = (confidence * 10).long().clamp(max=9)
    for index in range(10):
        selected = bin_ids == index
        bucket = state["calibration_bins"][index]
        bucket[0] += selected.sum().item()
        bucket[1] += confidence[selected].sum().item()
        bucket[2] += correct[selected].sum().item()
    ent = entropy(probabilities)
    state["entropy_sum"] += ent.sum().item()
    state["correct_entropy_sum"] += ent[correct].sum().item()
    state["wrong_entropy_sum"] += ent[~correct].sum().item()
    if disagreement is not None:
        state["disagreement_sum"] += disagreement.sum().item()
    for level in LEVELS:
        covered, sizes, mass = probability_set(probabilities, truth, level)
        state["sets"][str(level)]["covered"] += covered.sum().item()
        state["sets"][str(level)]["size"] += sizes.sum().item()
        state["sets"][str(level)]["mass"] += mass.sum().item()
    return prediction, ent, ~correct


def finish_metrics(state):
    n = max(state["count"], 1)
    correct = state["correct"]
    wrong = state["count"] - correct
    bins = state["calibration_bins"]
    ece = sum(count / n * abs(hit / count - confidence / count)
              for count, confidence, hit in bins if count)
    return {"clips": state["count"], "accuracy": correct / n,
            "nll": state["nll_sum"] / n, "brier_score": state["brier_sum"] / n,
            "mean_confidence": state["confidence_sum"] / n,
            "expected_calibration_error_10_bins": ece,
            "mean_predictive_entropy": state["entropy_sum"] / n,
            "mean_entropy_correct": state["correct_entropy_sum"] / correct if correct else None,
            "mean_entropy_wrong": state["wrong_entropy_sum"] / wrong if wrong else None,
            "mean_mc_disagreement": state["disagreement_sum"] / n,
            "probability_sets": {str(level): {
                "empirical_coverage": state["sets"][str(level)]["covered"] / n,
                "mean_selected_probability_mass": state["sets"][str(level)]["mass"] / n,
                "coverage_minus_selected_mass": (
                    state["sets"][str(level)]["covered"] -
                    state["sets"][str(level)]["mass"]) / n,
                "mean_set_size": state["sets"][str(level)]["size"] / n,
            } for level in LEVELS}}


def error_by_entropy_quartile(entropies, errors):
    values = torch.tensor(entropies)
    bad = torch.tensor(errors, dtype=torch.float32)
    thresholds = torch.quantile(values, torch.tensor([.25, .5, .75])).tolist()
    buckets = torch.bucketize(values, torch.tensor(thresholds))
    return [{"quartile": i + 1, "clips": int((buckets == i).sum()),
             "error_rate": bad[buckets == i].mean().item() if (buckets == i).any() else None}
            for i in range(4)]


@torch.no_grad()
def evaluate(network, validation, device, sample_count, seed):
    network.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    nclass = len(SOURCES)
    point_state, mc_state = metric_state(), metric_state()
    by_source = {}
    point_confusion = torch.zeros(nclass, nclass, dtype=torch.long)
    mc_confusion = torch.zeros(nclass, nclass, dtype=torch.long)
    point_entropies, point_errors, mc_entropies, mc_errors = [], [], [], []
    for source_index, source in enumerate(SOURCES):
        cache = validation[source]
        source_point, source_mc = metric_state(), metric_state()
        mu_all = cache["mu"].to(device); logvar_all = cache["logvar"].to(device)
        for start in range(0, len(mu_all), 16):
            mu = mu_all[start:start + 16]
            logvar = logvar_all[start:start + 16]
            clips, times, width = mu.shape
            truth = torch.full((clips,), source_index, dtype=torch.long, device=device)
            point = network(mu.reshape(-1, width)).softmax(-1).reshape(clips, times, nclass).mean(1)
            repeated_mu = mu.unsqueeze(0).expand(sample_count, -1, -1, -1)
            repeated_logvar = logvar.unsqueeze(0).expand(sample_count, -1, -1, -1)
            z = latent_draw(repeated_mu, repeated_logvar, generator)
            probs = torch.cat([network(chunk).softmax(-1)
                               for chunk in z.reshape(-1, width).split(2048)])
            probs = probs.reshape(sample_count, clips, times, nclass).mean(2)
            mc = probs.mean(0)
            disagreement = entropy(mc) - entropy(probs).mean(0)
            point_pred, p_ent, p_wrong = update_metrics(point_state, point, truth)
            update_metrics(source_point, point, truth)
            mc_pred, m_ent, m_wrong = update_metrics(mc_state, mc, truth, disagreement)
            update_metrics(source_mc, mc, truth, disagreement)
            for t, p, m in zip(truth.cpu(), point_pred.cpu(), mc_pred.cpu()):
                point_confusion[t, p] += 1
                mc_confusion[t, m] += 1
            point_entropies.extend(p_ent.cpu().tolist()); point_errors.extend(p_wrong.cpu().tolist())
            mc_entropies.extend(m_ent.cpu().tolist()); mc_errors.extend(m_wrong.cpu().tolist())
        by_source[source] = {"deterministic": finish_metrics(source_point),
                             "monte_carlo": finish_metrics(source_mc)}
        print(json.dumps({"event": "class_evaluated", "source": source,
                          "clips": len(cache["mu"]),
                          "deterministic_accuracy": by_source[source]["deterministic"]["accuracy"],
                          "monte_carlo_accuracy": by_source[source]["monte_carlo"]["accuracy"]}), flush=True)
    return {"sources": list(SOURCES), "sample_count": sample_count,
            "chance_balanced_accuracy": 1 / nclass,
            "deterministic": {**finish_metrics(point_state),
                "macro_accuracy": sum(by_source[s]["deterministic"]["accuracy"] for s in SOURCES) / nclass,
                "confusion_matrix": point_confusion.tolist(),
                "error_by_entropy_quartile": error_by_entropy_quartile(point_entropies, point_errors)},
            "monte_carlo": {**finish_metrics(mc_state),
                "macro_accuracy": sum(by_source[s]["monte_carlo"]["accuracy"] for s in SOURCES) / nclass,
                "confusion_matrix": mc_confusion.tolist(),
                "error_by_entropy_quartile": error_by_entropy_quartile(mc_entropies, mc_errors)},
            "per_source": by_source}


def main():
    global SOURCES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--heldout-source", choices=tuple(ALL_TARGETS), default="ftp_vla_touch")
    parser.add_argument("--manifest", default=MANIFEST)
    parser.add_argument("--output", default="processing/evaluations/dataset_identity_probe")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--train-clips-per-source", type=int, default=1000)
    parser.add_argument("--tune-clips-per-source", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--allow-intermediate", action="store_true",
                        help="Explicitly permit a saved completed epoch before the planned final epoch")
    parser.add_argument("--overwrite", action="store_true",
                        help="Explicitly replace a completed evaluation in --output")
    args = parser.parse_args()
    SOURCES = tuple(sorted(set(ALL_TARGETS) - {args.heldout_source}))
    if args.samples < 2:
        parser.error("--samples must be at least 2")
    seed_all(args.seed)
    output = Path(args.output)
    if (output / "results.json").exists() and not args.overwrite:
        raise FileExistsError(f"Completed evaluation exists: {output / 'results.json'}")
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_hash, manifest_hash = sha256(args.checkpoint), sha256(args.manifest)
    backbone, encoder, state = load_encoder(args.checkpoint, device, args.heldout_source)
    if state["epoch"] != state["config"]["epochs"] - 1 and not args.allow_intermediate:
        raise ValueError("Dataset probe requires the final completed checkpoint")
    training, tuning, validation = {}, {}, {}
    for source_index, source in enumerate(SOURCES):
        dataset = JointClipDataset(args.manifest, split="train", sources=[source])
        for partition, cap, destination in (("fit", args.train_clips_per_source, training),
                                             ("tune", args.tune_clips_per_source, tuning)):
            indices = select_indices(dataset, partition=partition, cap=cap,
                                     seed=args.seed + source_index)
            destination[source] = cached_latents(backbone, encoder, args.manifest,
                source, "train", indices, output / "features" / f"{source}_{partition}.pt",
                checkpoint_hash, manifest_hash, device, workers=args.workers)
        val_set = JointClipDataset(args.manifest, split="val", sources=[source])
        indices = select_indices(val_set)
        validation[source] = cached_latents(backbone, encoder, args.manifest,
            source, "val", indices, output / "features" / f"{source}_val.pt",
            checkpoint_hash, manifest_hash, device, workers=args.workers)
        fit_episodes = set(training[source]["episode_id"])
        tune_episodes = set(tuning[source]["episode_id"])
        val_episodes = set(validation[source]["episode_id"])
        if fit_episodes & tune_episodes or (fit_episodes | tune_episodes) & val_episodes:
            raise ValueError("Probe episode partitions overlap")
    del backbone, encoder
    fit = class_table([training[s] for s in SOURCES], device, balanced=True)
    tune = class_table([tuning[s] for s in SOURCES], device, balanced=True)
    classifier = head(32, len(SOURCES), classification=True).to(device)
    selection = fit_head(classifier, fit, tune, F.cross_entropy, epochs=args.epochs,
                         patience=args.patience, seed=args.seed, device=device)
    seed_all(args.seed)
    classifier = head(32, len(SOURCES), classification=True).to(device)
    combined = class_table([{key: torch.cat([training[s][key], tuning[s][key]])
                             for key in ("mu", "logvar")} for s in SOURCES], device, balanced=True)
    refit = fit_head(classifier, combined, None, F.cross_entropy,
                     fixed_epochs=selection["selected_epochs"], seed=args.seed, device=device)
    metrics = evaluate(classifier, validation, device, args.samples, args.seed + 2026)
    result = {"checkpoint": args.checkpoint, "checkpoint_sha256": checkpoint_hash,
              "checkpoint_epoch": state["epoch"], "manifest_sha256": manifest_hash,
              "checkpoint_is_final": state["epoch"] == state["config"]["epochs"] - 1,
              "frozen_encoder": True, "latent": "mu_logvar", "head": "32-512-11",
              "heldout_source": args.heldout_source, "sources": SOURCES,
              "source_balancing": "equal frame count per source, repetition within fit/tune/refit partitions only",
              "seed": args.seed, "train_clips_per_source": {
                  s: len(training[s]["mu"]) + len(tuning[s]["mu"]) for s in SOURCES},
              "selected_epochs": selection["selected_epochs"],
              "selection": selection, "refit": refit, "validation": metrics}
    torch.save({"head": classifier.cpu().state_dict(), "config": vars(args),
                "sources": SOURCES, "checkpoint_sha256": checkpoint_hash,
                "selected_epochs": selection["selected_epochs"]}, output / "classifier.pt")
    (output / "results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"event": "complete", "output": str(output / "results.json"),
                      "macro_accuracy": metrics["monte_carlo"]["macro_accuracy"]}), flush=True)


if __name__ == "__main__":
    main()
