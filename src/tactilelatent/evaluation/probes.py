"""Episode-disjoint readout probes; the visual representation is never optimized.

The regression recipe retains the prior deterministic protocol: sigmoid deep MLP,
AdamW, two dropout forwards per batch, MSE +0.5 MAE, fit-only feature statistics,
tune-only LR/duration selection, then all-training refits. Missing targets remain
masked. Episode bootstraps and fitted-head variation are separate quantities.
Inputs are compact feature tables with ``values`` [clips, D], ``target`` and
``valid`` [clips, channels], episode IDs, and current-frame indices. Outputs
are fitted heads, fit-only normalization, tuning records, and held-out metrics.
``fit_probe_sweep`` runs this protocol without changing the visual encoder.
"""

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def episode_partition(episode_id, seed=45, tune_fraction=0.1):
    """Assign a complete episode reproducibly to fit or tune by stable hash."""
    digest = hashlib.sha256(f"probe:{seed}:{episode_id}".encode()).digest()
    return "tune" if int.from_bytes(digest[:8], "big") / 2**64 < tune_fraction else "fit"


def head(dimension, outputs, hidden_dims=(512, 512, 512), dropout=0.1, classification=False):
    """Build a dropout MLP with sigmoid regression outputs or class logits."""
    if dimension < 1 or outputs < 1 or not hidden_dims or any(n < 1 for n in hidden_dims):
        raise ValueError("Probe dimensions must be positive")
    layers = []
    for width in hidden_dims:
        layers.extend([nn.Linear(dimension, width), nn.GELU(), nn.Dropout(dropout)])
        dimension = width
    layers.append(nn.Linear(dimension, outputs))
    if not classification:
        layers.append(nn.Sigmoid())
    return nn.Sequential(*layers)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def atomic_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    torch.save(value, temp)
    temp.replace(path)


def validate_features(train, validation):
    """Reject malformed tables, nonfinite observations, or episode leakage."""
    for table in (train, validation):
        x, y, valid = table["values"], table["target"], table["valid"]
        if (
            x.ndim != 2
            or y.ndim != 2
            or valid.shape != y.shape
            or len(x) != len(y)
            or len(x) != len(table["episode_id"])
        ):
            raise ValueError("Misaligned feature/target/episode table")
        if (
            not len(x)
            or not torch.isfinite(x).all()
            or not torch.isfinite(y[valid]).all()
            or not valid.any()
        ):
            raise ValueError("Empty or nonfinite feature/target table")
    if (
        train["values"].shape[1] != validation["values"].shape[1]
        or train["target"].shape[1] != validation["target"].shape[1]
    ):
        raise ValueError("Train/validation dimensions differ")
    if set(train["episode_id"]) & set(validation["episode_id"]):
        raise ValueError("Episode leakage between train and validation")


def fit_statistics(train, ids):
    """Return per-feature mean/std computed only over declared fit rows."""
    values = train["values"][ids].detach().float()
    if not len(values):
        raise ValueError("No fit observations for feature statistics")
    return values.mean(0), values.std(0, unbiased=False).clamp_min(1e-6)


def table(data, ids, device):
    return dict(
        mu=data["values"][ids].float().to(device),
        y=data["target"][ids].to(device),
        valid=data["valid"][ids].bool().to(device),
    )


def _loss(pred, target, valid, classification):
    if classification:
        return F.cross_entropy(pred, target.long())
    if not valid.any():
        raise ValueError("Probe batch has no observed targets")
    error = (pred - target)[valid]
    return (error.square() + 0.5 * error.abs()).mean()


def fit_head(
    network,
    fit,
    tune,
    *,
    epochs=100,
    patience=10,
    batch_size=256,
    lr=0.001,
    weight_decay=0.0001,
    seed=45,
    fixed_epochs=None,
    classification=False,
    progress=None,
):
    """Fit a readout, selecting duration on tune episodes only.

    Tables contain ``mu`` [rows, D], ``y`` and ``valid`` target tensors.
    Regression uses observed coordinates and averages two independent dropout
    forwards at the same deterministic features. ``fixed_epochs`` supports a
    fresh all-training refit after tune-only duration selection. No validation
    labels or visual-encoder weights participate in optimization.
    """
    if (
        epochs < 1
        or patience < 1
        or batch_size < 1
        or fixed_epochs is not None
        and fixed_epochs < 1
    ):
        raise ValueError("Positive probe epoch/patience/batch budgets required")
    device = next(network.parameters()).device
    optimizer = torch.optim.AdamW(network.parameters(), lr=lr, weight_decay=weight_decay)
    noise = torch.Generator(device=device).manual_seed(seed)
    order = torch.Generator().manual_seed(seed)
    best, best_epoch, best_state, stale = float("inf"), 0, None, 0
    history = []
    for epoch in range(fixed_epochs or epochs):
        network.train()
        total = seen = 0
        for ids in torch.randperm(len(fit["mu"]), generator=order).split(batch_size):
            ids = ids.to(device)
            x, y, valid = (fit[k][ids] for k in ("mu", "y", "valid"))
            # Deterministic means; consume the historical independent latent RNG
            # while retaining two independent dropout masks in the task head.
            if not classification and not valid.any():
                continue
            z = x + torch.randn(x.shape, device=device, generator=noise, dtype=x.dtype) * 0.0
            loss = 0.5 * (
                _loss(network(x), y, valid, classification)
                + _loss(network(z), y, valid, classification)
            )
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite probe loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(ids)
            seen += len(ids)
        if not seen:
            raise ValueError("No observed training targets")
        record = dict(epoch=epoch, train_loss=total / seen)
        if tune is not None:
            network.eval()
            sums = count = 0
            with torch.no_grad():
                for ids in torch.arange(len(tune["mu"]), device=device).split(batch_size):
                    x, y, valid = (tune[k][ids] for k in ("mu", "y", "valid"))
                    value = _loss(network(x), y, valid, classification)
                    sums += float(value) * len(ids)
                    count += len(ids)
            value = sums / count
            record["tune_loss"] = value
            if value < best - 1e-5:
                best, best_epoch, stale = value, epoch + 1, 0
                best_state = {k: v.detach().cpu().clone() for k, v in network.state_dict().items()}
            else:
                stale += 1
        history.append(record)
        if progress:
            progress("head_epoch", **record)
        if fixed_epochs is None and tune is not None and stale >= patience:
            break
    if tune is not None:
        network.load_state_dict(best_state)
    return dict(
        selected_epochs=fixed_epochs or best_epoch,
        history=history,
        best_tune_loss=None if tune is None else best,
    )


def episode_diagnostics(pred, target, valid, episodes, frames, seed=2071, bootstrap=5000):
    """Measure episode errors, within-episode dynamics, and bootstrap intervals.

    Predictions, targets, and True observation masks are [clips, channels].
    Differences follow sampled current-frame chronology within each episode;
    successive samples need not be successive native frames. Confidence
    intervals resample complete episodes and are not predictive calibration.
    """
    p, y = pred.detach().cpu().double().numpy(), target.detach().cpu().double().numpy()
    mask = valid.detach().cpu().numpy().astype(bool)
    frames = np.asarray(frames)
    rows = []
    center_p = []
    center_y = []
    delta_error = delta_y = delta_p = 0.0
    delta_count = 0
    for episode in sorted(set(episodes)):
        ids = np.array([i for i, e in enumerate(episodes) if e == episode])
        ids = ids[np.argsort(frames[ids], kind="stable")]
        a, b, m = p[ids], y[ids], mask[ids]
        count = m.sum(0)
        error = np.where(m, np.abs(a - b), 0.0)
        channel = np.divide(error.sum(0), count, out=np.zeros_like(error.sum(0)), where=count > 0)
        rows.append(
            dict(
                episode_id=episode,
                absolute_sum=float(error.sum()),
                count=int(count.sum()),
                per_channel_mae=channel.tolist(),
            )
        )
        am = np.divide(np.where(m, a, 0).sum(0), count, out=np.zeros(a.shape[1]), where=count > 0)
        bm = np.divide(np.where(m, b, 0).sum(0), count, out=np.zeros(b.shape[1]), where=count > 0)
        center_p.append(np.where(m, a - am, 0))
        center_y.append(np.where(m, b - bm, 0))
        if len(ids) > 1:
            adjacent = m[1:] & m[:-1] & (np.diff(frames[ids])[:, None] > 0)
            da, db = np.diff(a, axis=0)[adjacent], np.diff(b, axis=0)[adjacent]
            delta_error += float(np.square(da - db).sum())
            delta_y += float(np.square(db).sum())
            delta_p += float(np.square(da).sum())
            delta_count += len(da)
    a, b = np.concatenate(center_p), np.concatenate(center_y)
    denominator = np.sqrt(np.square(a).sum(0) * np.square(b).sum(0))
    correlation = np.divide(
        (a * b).sum(0), denominator, out=np.zeros(a.shape[1]), where=denominator > 0
    )
    totals = np.array([r["absolute_sum"] for r in rows])
    counts = np.array([r["count"] for r in rows])
    if bootstrap < 1:
        raise ValueError("Positive episode-bootstrap budget required")
    rng = np.random.default_rng(seed)
    estimates = []
    for start in range(0, bootstrap, 256):
        draws = rng.integers(len(rows), size=(min(256, bootstrap - start), len(rows)))
        estimates.extend((totals[draws].sum(1) / np.maximum(counts[draws].sum(1), 1)).tolist())
    return dict(
        episode_errors=rows,
        episode_bootstrap_mae_ci95=np.quantile(estimates, [0.025, 0.975]).tolist(),
        within_episode_channel_correlations=correlation.tolist(),
        bootstrap_unit="episode",
        bootstrap_replicates=bootstrap,
        dynamics=dict(
            adjacent_observed_values=delta_count,
            delta_mse=delta_error / max(delta_count, 1),
            no_change_delta_mse=delta_y / max(delta_count, 1),
            delta_variance_skill=1 - delta_error / delta_y if delta_y > 0 else None,
            temporal_amplitude_ratio=(delta_p / delta_y) ** 0.5 if delta_y > 0 else None,
            note="Differences between chronological sampled current frames, not necessarily adjacent native frames.",
        ),
    )


@torch.no_grad()
def score(network, train, validation, device, seed=2071, bootstrap=5000):
    """Score the frozen fitted head against observed validation coordinates.

    Inputs already use fit-only feature standardization. Target magnitudes stay
    in the benchmark's declared target scale; the training-target mean supplies
    a constant-prediction baseline without consulting validation statistics.
    """
    network.eval()
    x = validation["values"].to(device).float()
    pred = torch.cat([network(part).cpu() for part in x.split(512)])
    target = validation["target"].cpu().float()
    valid = validation["valid"].cpu().bool()
    error = (pred - target).double()
    counts = valid.sum(0)
    absolute = torch.where(valid, error.abs(), 0.0).sum(0)
    train_mask = train["valid"].bool()
    mean = torch.where(train_mask, train["target"], 0.0).sum(0) / train_mask.sum(0).clamp_min(1)
    return dict(
        mae=float(error.abs()[valid].mean()),
        mse=float(error.square()[valid].mean()),
        per_channel_mae=[
            float(absolute[i] / counts[i]) if counts[i] else None for i in range(len(counts))
        ],
        per_channel_valid_counts=counts.tolist(),
        mean_target_baseline_mae=float((target - mean).abs()[valid].double().mean()),
        **episode_diagnostics(
            pred,
            target,
            valid,
            validation["episode_id"],
            validation["frame_indices"],
            seed,
            bootstrap,
        ),
    )


def aggregate_heads(rows, bootstrap=5000, seed=2071):
    """Separate fitted-head seed variation from episode-bootstrap uncertainty.

    Every head must score identical episodes and observation counts. Episode
    bootstrap errors are averaged across fitted heads before resampling;
    variation across head fits is reported separately as a sample deviation.
    """
    reference = {r["episode_id"]: r for r in rows[0]["validation"]["episode_errors"]}
    episodes = []
    for name, first in sorted(reference.items()):
        paired = []
        for row in rows:
            current = {r["episode_id"]: r for r in row["validation"]["episode_errors"]}
            if set(current) != set(reference) or current[name]["count"] != first["count"]:
                raise ValueError("Head fits use different validation episodes")
            paired.append(current[name])
        episodes.append(
            dict(
                episode_id=name,
                count=first["count"],
                absolute_sum=float(np.mean([p["absolute_sum"] for p in paired])),
                per_channel_mae=np.mean([p["per_channel_mae"] for p in paired], axis=0).tolist(),
            )
        )
    totals = np.array([r["absolute_sum"] for r in episodes])
    counts = np.array([r["count"] for r in episodes])
    rng = np.random.default_rng(seed)
    estimates = []
    for start in range(0, bootstrap, 256):
        ids = rng.integers(len(episodes), size=(min(256, bootstrap - start), len(episodes)))
        estimates.extend((totals[ids].sum(1) / np.maximum(counts[ids].sum(1), 1)).tolist())
    maes = [r["validation"]["mae"] for r in rows]
    dynamics = {
        key: float(
            np.mean(
                [
                    r["validation"]["dynamics"][key]
                    for r in rows
                    if r["validation"]["dynamics"][key] is not None
                ]
            )
        )
        if any(r["validation"]["dynamics"][key] is not None for r in rows)
        else None
        for key in (
            "delta_mse",
            "no_change_delta_mse",
            "delta_variance_skill",
            "temporal_amplitude_ratio",
        )
    }
    return dict(
        mae=float(np.mean(maes)),
        mae_head_seed_std=float(np.std(maes, ddof=1)) if len(maes) > 1 else 0.0,
        mse=float(np.mean([r["validation"]["mse"] for r in rows])),
        per_channel_mae=[
            float(
                np.mean(
                    [
                        r["validation"]["per_channel_mae"][i]
                        for r in rows
                        if r["validation"]["per_channel_mae"][i] is not None
                    ]
                )
            )
            if any(r["validation"]["per_channel_mae"][i] is not None for r in rows)
            else None
            for i in range(len(rows[0]["validation"]["per_channel_mae"]))
        ],
        within_episode_channel_correlations=np.mean(
            [r["validation"]["within_episode_channel_correlations"] for r in rows], axis=0
        ).tolist(),
        episode_errors=episodes,
        episode_bootstrap_mae_ci95=np.quantile(estimates, [0.025, 0.975]).tolist(),
        episodes=len(episodes),
        bootstrap_replicates=bootstrap,
        dynamics=dynamics,
        bootstrap_note="Episodes resampled after averaging errors across fitted heads; head-fit variation is reported separately.",
    )


def feature_fingerprint(data):
    digest = hashlib.sha256()
    for key in ("values", "target", "valid", "frame_indices"):
        value = data[key].detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    digest.update(json.dumps(data["episode_id"]).encode())
    return digest.hexdigest()


def validate_probe_config(config):
    for name in ("epochs", "patience", "bootstrap_replicates"):
        value = config.get(name, 5000 if name == "bootstrap_replicates" else None)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"Positive integer {name} required")
    for name in ("head_seeds", "learning_rates", "hidden_dims"):
        if not isinstance(config.get(name), list) or not config[name]:
            raise ValueError(f"Explicit nonempty {name} list required")
    if len(set(config["head_seeds"])) != len(config["head_seeds"]):
        raise ValueError("Distinct head seeds required")
    if any(
        isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32
        for seed in config["head_seeds"]
    ):
        raise ValueError("Head seeds must be nonnegative uint32 integers")
    if any(
        not isinstance(rate, (int, float))
        or isinstance(rate, bool)
        or not np.isfinite(rate)
        or rate <= 0
        for rate in config["learning_rates"]
    ):
        raise ValueError("Finite positive learning rates required")
    if any(
        isinstance(width, bool) or not isinstance(width, int) or width < 1
        for width in config["hidden_dims"]
    ):
        raise ValueError("Positive integer hidden widths required")
    if not 0 < config.get("tune_fraction", 0.1) < 1:
        raise ValueError("Tune fraction must be between zero and one")


def fit_probe_sweep(train, validation, output, config, device=None, progress=None):
    """Tune head LR/duration, refit on training episodes, then score validation.

    Episode-disjoint fit/tune assignments and fit-only z-scoring are fixed
    before optimization. Only tune loss selects learning rate and each seed's
    duration. Saved protocol fingerprints prevent reuse with changed features,
    targets, splits, or settings; existing matching trial files support resume.
    The returned metrics include head variation and episode-bootstrap bounds.
    """
    validate_probe_config(config)
    validate_features(train, validation)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = config["head_seeds"]
    rates = config["learning_rates"]
    if not seeds or not rates:
        raise ValueError("Explicit head seeds and learning rates required")
    partition_seed = config.get("split_seed", 45)
    fraction = config.get("tune_fraction", 0.1)
    fit_ids = [
        i
        for i, e in enumerate(train["episode_id"])
        if episode_partition(e, partition_seed, fraction) == "fit"
    ]
    tune_ids = [
        i
        for i, e in enumerate(train["episode_id"])
        if episode_partition(e, partition_seed, fraction) == "tune"
    ]
    if not fit_ids or not tune_ids:
        raise ValueError("Missing episode-disjoint fit/tune training partitions")
    mean, std = fit_statistics(train, fit_ids)

    def normalized(data):
        return {**data, "values": (data["values"].float() - mean) / std}

    train, validation = normalized(train), normalized(validation)
    fit, tune = table(train, fit_ids, device), table(train, tune_ids, device)
    hidden = config["hidden_dims"]
    dropout = config.get("dropout", 0.1)
    dimension = train["values"].shape[1]
    channels = train["target"].shape[1]
    signature = dict(
        config=config,
        dimension=dimension,
        outputs=channels,
        train_sha256=feature_fingerprint(train),
        validation_sha256=feature_fingerprint(validation),
        fit_episodes=sorted({train["episode_id"][i] for i in fit_ids}),
        tune_episodes=sorted({train["episode_id"][i] for i in tune_ids}),
    )
    pin = output / "protocol.json"
    if pin.exists() and json.loads(pin.read_text()) != signature:
        raise ValueError("Probe protocol changed; choose a new evaluation output")
    write_json(pin, signature)
    atomic_save(
        output / "normalization.pt",
        dict(
            mean=mean,
            std=std,
            fit_episode_ids=sorted({train["episode_id"][i] for i in fit_ids}),
            partition_seed=partition_seed,
        ),
    )
    trials = []
    for rate in rates:
        for seed in seeds:
            path = output / "tuning_trials" / f"lr_{rate:g}_seed_{seed}.json"
            if path.exists():
                row = json.loads(path.read_text())
            else:
                if progress:
                    progress("tuning_head", seed=seed, learning_rate=rate)
                seed_all(seed)
                network = head(dimension, channels, hidden, dropout).to(device)
                selection = fit_head(
                    network,
                    fit,
                    tune,
                    epochs=config["epochs"],
                    patience=config["patience"],
                    batch_size=config.get("batch_size", 256),
                    lr=rate,
                    weight_decay=config.get("weight_decay", 0.0001),
                    seed=seed,
                    progress=progress,
                )
                row = dict(seed=seed, learning_rate=rate, selection=selection)
                write_json(path, row)
            trials.append(row)
    losses = {
        rate: float(
            np.mean(
                [r["selection"]["best_tune_loss"] for r in trials if r["learning_rate"] == rate]
            )
        )
        for rate in rates
    }
    chosen = min(rates, key=lambda rate: losses[rate])
    rows = []
    write_json(
        output / "tuning_selection.json",
        dict(
            selected_learning_rate=chosen,
            mean_best_tune_loss=losses,
            selection="Tune episodes only; validation labels never select LR or duration.",
        ),
    )
    for seed in seeds:
        duration = next(
            r["selection"]["selected_epochs"]
            for r in trials
            if r["seed"] == seed and r["learning_rate"] == chosen
        )
        folder = output / f"seed_{seed}"
        result_path = folder / "results.json"
        if result_path.exists():
            row = json.loads(result_path.read_text())
        else:
            if progress:
                progress(
                    "refitting_head", seed=seed, learning_rate=chosen, selected_epochs=duration
                )
            seed_all(seed)
            network = head(dimension, channels, hidden, dropout).to(device)
            refit = fit_head(
                network,
                table(train, list(range(len(train["values"]))), device),
                None,
                fixed_epochs=duration,
                epochs=config["epochs"],
                patience=config["patience"],
                batch_size=config.get("batch_size", 256),
                lr=chosen,
                weight_decay=config.get("weight_decay", 0.0001),
                seed=seed,
                progress=progress,
            )
            row = dict(
                seed=seed,
                learning_rate=chosen,
                selected_head_epochs=duration,
                refit=refit,
                validation=score(
                    network,
                    train,
                    validation,
                    device,
                    seed + 2026,
                    config.get("bootstrap_replicates", 5000),
                ),
            )
            atomic_save(
                folder / "head.pt",
                dict(
                    head=network.cpu().state_dict(),
                    input_dimension=dimension,
                    output_dim=channels,
                    hidden_dims=hidden,
                    dropout=dropout,
                    mean=mean,
                    std=std,
                    seed=seed,
                    selected_epochs=duration,
                ),
            )
            write_json(result_path, row)
        rows.append(row)
    dimensions = [dimension, *hidden, channels]
    parameters = sum((a + 1) * b for a, b in zip(dimensions, dimensions[1:]))
    return dict(
        input_dimension=dimension,
        head=dimensions,
        head_parameters=parameters,
        head_seeds=seeds,
        learning_rates=rates,
        selected_learning_rate=chosen,
        selected_head_epochs=[r["selected_head_epochs"] for r in rows],
        train_clips=len(train["values"]),
        fit_clips=len(fit_ids),
        tune_clips=len(tune_ids),
        validation_clips=len(validation["values"]),
        seed_results=rows,
        tuning_trials=trials,
        validation=aggregate_heads(rows, config.get("bootstrap_replicates", 5000)),
        protocol="Frozen deterministic means; fit-only z-score; episode-disjoint tune-only LR/duration selection; all-training refits.",
    )
