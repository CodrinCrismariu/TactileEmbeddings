"""Collect completed and partial autoencoder ablations into one CSV table."""

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="processing/runs/egotouch_ae_sensor_code_grid_v1")
    root = Path(parser.parse_args().root)
    rows = []
    for code_dim in (16, 32, 64, 128):
        for preset in ("small", "medium", "large"):
            run = root / f"{preset}_code{code_dim}"
            row = {"preset": preset, "sensor_code_dim": code_dim, "completed_epochs": 0,
                   "status": "pending", "best_epoch": None, "val_mae": None,
                   "val_contact_mae": None, "val_contact_f1": None,
                   "val_high_pressure_mae": None, "parameters": None, "epoch_seconds_mean": None}
            if (run / "metrics.jsonl").exists():
                history = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
                best = json.loads((run / "best_metrics.json").read_text())
                summary = json.loads((run / "summary.json").read_text())
                config = json.loads((run / "config.json").read_text())
                row.update(completed_epochs=history[-1]["epoch"], best_epoch=best["epoch"],
                           status="complete" if history[-1]["epoch"] == config["epochs"] else "partial",
                           parameters=summary["encoder_parameters"] + summary["decoder_parameters"],
                           epoch_seconds_mean=sum(x["seconds"] for x in history) / len(history))
                for metric in ("mae", "contact_mae", "contact_f1", "high_pressure_mae"):
                    row[f"val_{metric}"] = best["val"][metric]
            rows.append(row)
    with (root / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
