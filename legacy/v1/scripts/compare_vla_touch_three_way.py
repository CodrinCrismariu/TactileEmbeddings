"""Compare direct VLA_touch training with both frozen-encoder adaptations."""

import json
from pathlib import Path


BASE = Path("processing/evaluations/baselines/joint_50ep_vla_touch_v1")
ALT = Path("processing/runs/joint_all_alternating_fresh_8k_50ep")
DIRECT = Path("processing/runs/vla_touch_direct_vae32_50ep")


def read(path):
    return json.loads(path.read_text())


def main():
    results = {
        "original_frozen_encoder": read(BASE / "processing/evaluations/vla_touch_frozen_decoder/results.json"),
        "alternating_frozen_encoder": read(ALT / "evaluations/vla_touch/results.json"),
        "direct_vla_touch": read(DIRECT / "evaluation/results.json"),
    }
    hashes = {row["manifest_sha256"] for row in results.values()}
    clips = {row["validation"]["validation_clips"] for row in results.values()}
    draws = {row["validation"]["sample_count"] for row in results.values()}
    if len(hashes) != 1 or clips != {183} or draws != {64}:
        raise ValueError("The three evaluations do not use the same 183 clips and 64 draws")
    output = {"manifest_sha256": hashes.pop(), "validation_clips": 183,
              "sample_count": 64,
              "protocol_note": "Direct training updates the encoder on VLA_touch train clips; the other two freeze encoders trained without VLA_touch and fit a new VLA_touch decoder. This is not a zero-shot comparison.",
              "models": {}}
    for name, result in results.items():
        metrics = result["validation"]
        all_pixels = metrics["regions"]["all"]
        high_error = metrics["regions"]["high_error_quartile"]
        output["models"][name] = {
            "checkpoint_sha256": result["checkpoint_sha256"],
            "train_clips": result["train_clips"],
            "deterministic_mae": metrics["deterministic_mae"],
            "ensemble_mean_mae": metrics["ensemble_mean_mae"],
            "best_of_64_clip_mae": metrics["best_of_samples_clip_mae"],
            "mean_image_baseline_mae": metrics["mean_image_baseline_mae"],
            "crps": all_pixels["crps"],
            "predictive_std": all_pixels["mean_predictive_std"],
            "uncertainty_error_correlation": all_pixels["std_error_pearson"],
            "coverage_90": all_pixels["intervals"]["0.9"]["empirical_coverage"],
            "width_90": all_pixels["intervals"]["0.9"]["mean_width"],
            "high_error_coverage_90": high_error["intervals"]["0.9"]["empirical_coverage"],
        }
    path = DIRECT / "evaluation/three_way_comparison.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(output, indent=2))
    print(json.dumps({"event": "comparison_complete", "path": str(path)}), flush=True)


if __name__ == "__main__":
    main()
