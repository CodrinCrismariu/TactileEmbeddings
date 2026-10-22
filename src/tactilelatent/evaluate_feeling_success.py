"""Frozen joint-encoder evaluation of grasp success on entirely held-out objects.

Only pre-lift scene RGB enters the encoder. GelSight, labels, object names,
and the post-lift RGB are NEVER encoder inputs. This is a new object-disjoint
evaluation protocol, not a reproduction of the original multimodal paper.
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from .evaluate_dataset_classifier import (entropy, finish_metrics, metric_state,
                                          update_metrics)
from .frozen_probe import (fit_head, head, latent_draw, load_encoder, seed_all,
                           sha256)
from .train import features


CHECKPOINT = "processing/runs/joint_all_holdout_motiontrans_fresh_8k_50ep/last.pt"
MANIFEST = "datasets/evaluation_only/feeling_of_success/manifest.json"
INPUT_FIELDS = ("kinectA_rgb_before", "kinectA_rgb_during")
ADAPTER = "two_real_pre_lift_snapshots_main_camera_last_frame_gaussian_v1"


def object_splits(records, seed=45):
    """Assign whole objects once, without looking at success labels.

    Approximately 70/15/15% of OBJECTS (not trials) become fit/tune/test.
    Tune selects head duration; refit uses fit+tune, never test.
    """
    objects = sorted({row["object_name"] for row in records})
    if len(objects) < 7:
        raise ValueError("Need at least seven objects for meaningful disjoint splits")
    objects = np.random.default_rng(seed).permutation(objects).tolist()
    n_test = max(1, round(.15 * len(objects)))
    n_tune = max(1, round(.15 * len(objects)))
    assignment = {obj: ("test" if i < n_test else "tune" if i < n_test + n_tune else "fit")
                  for i, obj in enumerate(objects)}
    ids = {name: [i for i, row in enumerate(records) if assignment[row["object_name"]] == name]
           for name in ("fit", "tune", "test")}
    assert not (set(ids["fit"]) & set(ids["tune"]) | set(ids["fit"]) & set(ids["test"])
                | set(ids["tune"]) & set(ids["test"]))
    return assignment, ids


class SuccessImages(Dataset):
    """Load exactly two real snapshots, not invented repetitions of eight frames."""

    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        rgb = torch.zeros(2, 3, 3, 224, 224)
        with h5py.File(row["path"], "r") as stream:
            group = stream[row["group"]]
            for time, field in enumerate(INPUT_FIELDS):
                image = Image.fromarray(group[field][...]).convert("RGB")
                image = image.resize((224, 224), Image.Resampling.BICUBIC)
                rgb[time, 0] = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1) / 255.
        mask = torch.zeros(2, 3, dtype=torch.bool)
        mask[:, 0] = True
        return {"rgb": rgb, "view_mask": mask}


@torch.no_grad()
def extract(records, checkpoint, manifest, output, batch_size, workers, device):
    signature = {"checkpoint_sha256": sha256(checkpoint), "manifest_sha256": sha256(manifest),
                 "adapter": ADAPTER, "trial_ids": [row["trial_id"] for row in records]}
    path = output / "latents.pt"
    if path.exists():
        cache = torch.load(path, map_location="cpu", weights_only=False)
        if cache["signature"] != signature:
            raise ValueError("Feature cache identity mismatch")
        return cache
    backbone, model, state = load_encoder(checkpoint, device, "ftp_motiontrans")
    if state["epoch"] != state["config"]["epochs"] - 1:
        raise ValueError("Use the completed final joint baseline checkpoint")
    if any(p.requires_grad for module in (backbone, model) for p in module.parameters()):
        raise AssertionError("Encoder is not frozen")
    if backbone.training or model.training:
        raise AssertionError("Encoder must remain in evaluation mode")
    loader = DataLoader(SuccessImages(records), batch_size=batch_size, num_workers=workers,
                        shuffle=False, pin_memory=device.type == "cuda")
    means, variances = [], []
    for step, batch in enumerate(loader):
        with torch.autocast("cuda", torch.bfloat16, enabled=device.type == "cuda"):
            mu, logvar = model.visual.encode(features(backbone, batch, device),
                                             batch["view_mask"].to(device))
        # During-grasp state, with the real before-grasp image as context.
        means.append(mu[:, -1].float().cpu())
        variances.append(logvar[:, -1].float().cpu())
        if step % 25 == 0 or step == len(loader) - 1:
            print(json.dumps({"event": "extract", "batch": step + 1,
                              "batches": len(loader), "percent": 100 * (step + 1) / len(loader)}),
                  flush=True)
    cache = {"signature": signature, "mu": torch.cat(means), "logvar": torch.cat(variances),
             "y": torch.tensor([int(row["success"]) for row in records], dtype=torch.long)}
    if not torch.isfinite(cache["mu"]).all() or not torch.isfinite(cache["logvar"]).all():
        raise ValueError("Non-finite frozen features")
    torch.save(cache, path.with_suffix(".tmp"))
    path.with_suffix(".tmp").replace(path)
    del backbone, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return cache


def ranking_metrics(scores, truth):
    """Tie-correct AUROC and grouped-threshold average precision, no sklearn."""
    scores, order = scores.double().sort(descending=True)
    truth = truth[order].double()
    positives = int(truth.sum())
    negatives = len(truth) - positives
    if not positives or not negatives:
        return {"auroc": None, "average_precision": None}
    _, counts = torch.unique_consecutive(scores, return_counts=True)
    ends = counts.cumsum(0) - 1
    tp = truth.cumsum(0)[ends]
    fp = (1 - truth).cumsum(0)[ends]
    tpr = torch.cat([tp.new_zeros(1), tp / positives])
    fpr = torch.cat([fp.new_zeros(1), fp / negatives])
    recall_increment = tp - torch.cat([tp.new_zeros(1), tp[:-1]])
    return {"auroc": torch.trapezoid(tpr, fpr).item(),
            "average_precision": (recall_increment / positives * tp / (tp + fp)).sum().item()}


def binary_metrics(probabilities, truth):
    probabilities = probabilities.cpu()
    truth = truth.cpu()
    state = metric_state()
    prediction, ent, wrong = update_metrics(state, probabilities, truth)
    result = finish_metrics(state)
    # The generic helper defaults this field to zero when not supplied.
    # We measure probability spread separately; don't report a fake zero.
    result.pop("mean_mc_disagreement", None)
    result["brier_score"] /= 2  # Conventional binary Brier: (p(success)-y)^2.
    confusion = torch.bincount(truth * 2 + prediction, minlength=4).reshape(2, 2)
    tn, fp, fn, tp = confusion.flatten().tolist()
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    result.update(ranking_metrics(probabilities[:, 1], truth))
    result.update({"confusion_matrix_true_rows_failure_success": confusion.tolist(),
                   "success_recall": recall, "failure_recall": specificity,
                   "balanced_accuracy": (recall + specificity) / 2
                   if recall is not None and specificity is not None else None,
                   "success_precision": tp / (tp + fp) if tp + fp else None,
                   "success_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
                   "entropy_error_auroc": ranking_metrics(ent, wrong.long())["auroc"]})
    # Keep the lowest-entropy examples first: is abstention useful?
    order = ent.argsort(stable=True)
    result["risk_coverage"] = {
        str(level): wrong[order[:max(1, round(level * len(truth)))]].float().mean().item()
        for level in (.25, .5, .75, 1.)}
    return result


@torch.no_grad()
def predict(network, table, samples, seed):
    network.eval()
    rng = torch.Generator(device=table["mu"].device).manual_seed(seed)
    deterministic, ensemble, spread = [], [], []
    for ids in torch.arange(len(table["mu"]), device=table["mu"].device).split(256):
        mu, logvar = table["mu"][ids], table["logvar"][ids]
        deterministic.append(network(mu).softmax(-1).cpu())
        draws = torch.stack([network(latent_draw(mu, logvar, rng)).softmax(-1)
                             for _ in range(samples)])
        ensemble.append(draws.mean(0).cpu())
        spread.append(draws[:, :, 1].std(0, unbiased=False).cpu())
    return torch.cat(deterministic), torch.cat(ensemble), torch.cat(spread)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--manifest", default=MANIFEST)
    parser.add_argument("--output", default="processing/runs/joint_all_holdout_motiontrans_fresh_8k_50ep/evaluations/feeling_of_success")
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.samples < 2 or args.epochs < 1:
        raise ValueError("Require at least two samples and one head epoch")
    output = Path(args.output)
    if (output / "metrics.json").exists() or (output / "head.pt").exists():
        raise FileExistsError("Refusing to overwrite completed head/results; use a new directory")
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(args.manifest).read_text())
    if not manifest.get("complete") or not manifest.get("encoder_training_excluded"):
        raise ValueError("Require a complete evaluation-only dataset")
    records = [{**trial, "path": shard["path"]}
               for _, shard in sorted(manifest["shards"].items()) for trial in shard["trials"]]
    if len(records) != manifest["trial_count"] or len({r["trial_id"] for r in records}) != len(records):
        raise ValueError("Manifest count/identity mismatch")
    assignment, partitions = object_splits(records, args.seed)
    for name, ids in partitions.items():
        if len({records[i]["success"] for i in ids}) != 2:
            raise ValueError(f"Split {name} lacks a class; do not evaluate this split")
    splits = {"object_assignment": assignment, "partitions": {
        name: {"trials": len(ids), "objects": len({records[i]["object_name"] for i in ids}),
               "successes": sum(records[i]["success"] for i in ids),
               "trial_ids": [records[i]["trial_id"] for i in ids]}
        for name, ids in partitions.items()}}
    config = {**vars(args), "head": [32, 512, 2], "encoder_frozen": True,
              "adapter": ADAPTER, "input_fields": list(INPUT_FIELDS),
              "post_lift_rgb_excluded": True, "split": "object_disjoint_70_15_15_seeded",
              "head_objective": "equal mixture of mean-latent and sampled-latent cross entropy",
              "refit": "fit+tune for duration selected only on tune objects",
              "brier_convention": "binary (p_success - y)^2",
              "caveat": "Empirical decoder calibration, not decoder-weight uncertainty. "
              "Scene RGB only; new two-snapshot object-disjoint protocol, not paper reproduction."}
    (output / "config.json").write_text(json.dumps(config, indent=2))
    (output / "splits.json").write_text(json.dumps(splits, indent=2))
    seed_all(args.seed)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = extract(records, args.checkpoint, args.manifest, output, args.batch_size, args.workers, device)
    if cache["mu"].shape != (len(records), 32):
        raise ValueError("Expected one 32D during-grasp latent per trial")
    def table(ids):
        return {key: cache[key][ids].to(device) for key in ("mu", "logvar", "y")}
    fit, tune = table(partitions["fit"]), table(partitions["tune"])
    network = head(32, 2, classification=True).to(device)
    selection = fit_head(network, fit, tune, nn.CrossEntropyLoss(), epochs=args.epochs,
                         patience=args.patience, seed=args.seed, device=device)
    (output / "selection.json").write_text(json.dumps(selection, indent=2))
    seed_all(args.seed)
    network = head(32, 2, classification=True).to(device)
    train_ids = partitions["fit"] + partitions["tune"]
    train = table(train_ids)
    refit = fit_head(network, train, None, nn.CrossEntropyLoss(),
                     fixed_epochs=selection["selected_epochs"], seed=args.seed, device=device)
    test = table(partitions["test"])
    deterministic, ensemble, spread = predict(network, test, args.samples, args.seed + 3001)
    truth = test["y"].cpu()
    prior = train["y"].float().mean().item()
    baseline = torch.tensor([1 - prior, prior]).repeat(len(truth), 1)
    metrics = {"checkpoint_sha256": cache["signature"]["checkpoint_sha256"],
               "manifest_sha256": cache["signature"]["manifest_sha256"],
               "selected_epochs": selection["selected_epochs"], "sample_count": args.samples,
               "test_trials": len(truth), "test_objects": splits["partitions"]["test"]["objects"],
               "training_success_prevalence": prior,
               "deterministic": binary_metrics(deterministic, truth),
               "ensemble": binary_metrics(ensemble, truth),
               "training_prevalence_baseline": binary_metrics(baseline, truth),
               "latent_sampling": {"mean_probability_std": spread.mean().item(),
                    "probability_std_error_auroc": ranking_metrics(spread, (ensemble.argmax(-1) != truth).long())["auroc"],
                    "mean_predicted_latent_std": (.5 * test["logvar"]).exp().mean().item()},
               "per_object": {}}
    objects = [records[i]["object_name"] for i in partitions["test"]]
    for obj in sorted(set(objects)):
        ids = torch.tensor([i for i, name in enumerate(objects) if name == obj])
        metrics["per_object"][obj] = {"deterministic": binary_metrics(deterministic[ids], truth[ids]),
                                     "ensemble": binary_metrics(ensemble[ids], truth[ids])}
    metrics["deterministic"]["object_macro_accuracy"] = float(np.mean([
        v["deterministic"]["accuracy"] for v in metrics["per_object"].values()]))
    metrics["ensemble"]["object_macro_accuracy"] = float(np.mean([
        v["ensemble"]["accuracy"] for v in metrics["per_object"].values()]))
    np.savez_compressed(output / "predictions.npz", truth=truth.numpy(),
                        deterministic=deterministic.numpy(), ensemble=ensemble.numpy(),
                        success_probability_std=spread.numpy(), object_name=np.array(objects),
                        trial_id=np.array([records[i]["trial_id"] for i in partitions["test"]]))
    if sha256(args.checkpoint) != cache["signature"]["checkpoint_sha256"]:
        raise AssertionError("Source checkpoint changed during evaluation")
    torch.save({"head": network.cpu().state_dict(), "config": config,
                "signature": cache["signature"], "selected_epochs": selection["selected_epochs"],
                "selection": selection, "refit": refit}, output / "head.pt")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False))
    print(json.dumps({"event": "complete", "output": str(output),
                      "deterministic_accuracy": metrics["deterministic"]["accuracy"],
                      "ensemble_accuracy": metrics["ensemble"]["accuracy"],
                      "baseline_accuracy": metrics["training_prevalence_baseline"]["accuracy"]}), flush=True)


if __name__ == "__main__":
    main()
