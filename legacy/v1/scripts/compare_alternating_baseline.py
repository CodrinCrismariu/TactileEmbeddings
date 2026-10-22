"""Summarize the completed alternating run against the immutable baseline."""

import json
from pathlib import Path


BASE = Path("processing/evaluations/baselines/joint_50ep_vla_touch_v1")
RUN = Path("processing/runs/joint_all_alternating_fresh_8k_50ep")


def read(path):
    return json.loads(path.read_text())


def comparison(old, new, keys):
    return {key: {"baseline": old[key], "alternating": new[key],
                  "difference": new[key] - old[key]} for key in keys}


def main():
    baseline_vla = read(BASE / "processing/evaluations/vla_touch_frozen_decoder/results.json")
    baseline_identity = read(BASE / "processing/evaluations/dataset_identity_probe/results.json")
    new_vla = read(RUN / "evaluations/vla_touch/results.json")
    new_identity = read(RUN / "evaluations/dataset_identity/results.json")
    old_tactile, new_tactile = baseline_vla["validation"], new_vla["validation"]
    old_source, new_source = baseline_identity["validation"], new_identity["validation"]
    if old_source["sources"] != new_source["sources"]:
        raise ValueError("The source lists differ; identity accuracy is not comparable")
    result = {
        "note": "Outcome comparison only: initialization also differs between runs.",
        "checkpoint_epochs": [baseline_vla["checkpoint_epoch"], new_vla["checkpoint_epoch"]],
        "dataset_identity": {
            mode: comparison(old_source[mode], new_source[mode],
                             ("accuracy", "macro_accuracy", "nll"))
            for mode in ("deterministic", "monte_carlo")},
        "dataset_identity_90pct_set": comparison(
            old_source["monte_carlo"]["probability_sets"]["0.9"],
            new_source["monte_carlo"]["probability_sets"]["0.9"],
            ("empirical_coverage", "mean_set_size")),
        "vla_touch": comparison(old_tactile, new_tactile,
            ("deterministic_mae", "ensemble_mean_mae", "best_of_samples_clip_mae")),
        "vla_touch_distribution": comparison(old_tactile["regions"]["all"],
            new_tactile["regions"]["all"],
            ("crps", "mean_predictive_std", "std_error_pearson")),
        "vla_touch_90pct_interval": comparison(
            old_tactile["regions"]["all"]["intervals"]["0.9"],
            new_tactile["regions"]["all"]["intervals"]["0.9"],
            ("empirical_coverage", "mean_width")),
        "vla_touch_high_error_90pct_interval": comparison(
            old_tactile["regions"]["high_error_quartile"]["intervals"]["0.9"],
            new_tactile["regions"]["high_error_quartile"]["intervals"]["0.9"],
            ("empirical_coverage", "mean_width")),
    }
    path = RUN / "evaluations" / "baseline_comparison.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps({"event": "comparison_complete", "path": str(path)}), flush=True)


if __name__ == "__main__":
    main()
