"""Mirror a running joint-training log into TensorBoard without changing training.

Usage:
    .venv/bin/python scripts/tensorboard_joint_live.py \
        processing/runs/joint_all_holdout_vla_8k_50ep --pid 1511362

The batch log contains one sampled dataset every ``log_every`` batches. Exact
per-dataset epoch means and validation errors are read from metrics.jsonl once
an epoch finishes.
"""

import argparse
import json
import os
import time
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


def read_complete_lines(path: Path, offset: int):
    """Read newly appended, newline-terminated records only."""
    if not path.exists():
        return [], offset
    if path.stat().st_size < offset:
        offset = 0
    lines = []
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        stream.seek(offset)
        while True:
            start = stream.tell()
            line = stream.readline()
            if not line or not line.endswith("\n"):
                return lines, start
            lines.append(line)


def parse_record(line: str):
    # tqdm may leave a carriage-return prefix on the next JSON log line.
    brace = line.find("{")
    if brace < 0:
        return None
    try:
        return json.loads(line[brace:])
    except json.JSONDecodeError:
        return None


def running(pid: int | None) -> bool:
    if pid is None:
        return True
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--pid", type=int, help="Exit after this training process exits")
    parser.add_argument("--poll-seconds", type=float, default=2)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"), flush_secs=10)
    log_offset = metrics_offset = 0
    batches_per_epoch = None
    seen_sources = set()
    completed_after_exit = 0

    try:
        while True:
            lines, log_offset = read_complete_lines(run_dir / "nohup.log", log_offset)
            for line in lines:
                record = parse_record(line)
                if not isinstance(record, dict):
                    continue
                if record.get("event") == "indexed":
                    batches_per_epoch = int(record["train_batches_per_epoch"])
                    print(f"Batches per epoch: {batches_per_epoch}", flush=True)
                if {"epoch", "batch", "source"} <= record.keys() and (
                    "loss" in record or "task_loss" in record
                ):
                    if batches_per_epoch is None:
                        continue
                    source = record["source"]
                    step = int(record["epoch"]) * batches_per_epoch + int(record["batch"])
                    writer.add_scalar(f"loss/train_batch/{source}",
                                      record.get("task_loss", record.get("loss")), step)
                    seen_sources.add(source)
                    if "class_loss" in record:
                        writer.add_scalar(f"loss/classifier_batch/{source}",
                                          record["class_loss"], step)
                    if "class_accuracy" in record:
                        writer.add_scalar(f"accuracy/classifier_batch/{source}",
                                          record["class_accuracy"], step)
                    for name in ("persistent_accuracy", "challenger_accuracy"):
                        if name in record:
                            writer.add_scalar(f"accuracy/{name}_batch/{source}",
                                              record[name], step)
                    if "bank_min_count" in record:
                        writer.add_scalar("adversarial/bank_min_count",
                                          record["bank_min_count"], step)
                    for field in ("lr", "beta", "grl_lambda"):
                        if field in record:
                            writer.add_scalar(f"schedule/{field}", record[field], step)

            lines, metrics_offset = read_complete_lines(run_dir / "metrics.jsonl", metrics_offset)
            for line in lines:
                record = parse_record(line)
                if not isinstance(record, dict) or "epoch" not in record:
                    continue
                epoch = int(record["epoch"])
                for source, value in record.get("train_loss_by_source", {}).items():
                    writer.add_scalar(f"loss/train_epoch/{source}", value, epoch)
                for source, value in record.get("classifier_loss_by_source", {}).items():
                    writer.add_scalar(f"loss/classifier_epoch/{source}", value, epoch)
                for source, value in record.get("classifier_accuracy_by_source", {}).items():
                    writer.add_scalar(f"accuracy/classifier_epoch/{source}", value, epoch)
                for name in ("persistent_accuracy_by_source", "challenger_accuracy_by_source"):
                    for source, value in record.get(name, {}).items():
                        writer.add_scalar(f"accuracy/{name}/{source}", value, epoch)
                if "classifier_steps" in record:
                    writer.add_scalar("adversarial/classifier_steps", record["classifier_steps"], epoch)
                for source, values in record.get("validation", {}).items():
                    if isinstance(values, dict) and "normalized_mae" in values:
                        writer.add_scalar(
                            f"error/val_normalized_mae/{source}",
                            values["normalized_mae"], epoch,
                        )
                for field in ("grl_lambda", "next_grl_target", "diagnostic_macro_accuracy"):
                    if record.get(field) is not None:
                        writer.add_scalar(f"adversarial/{field}", record[field], epoch)
                for field in ("sam_rho", "sam_updates", "sam_first_gradient_norm", "sam_perturbed_loss"):
                    if field in record:
                        writer.add_scalar(f"sam/{field}", record[field], epoch)
                for field, value in record.get("latent", {}).items():
                    writer.add_scalar(f"latent/{field}", value, epoch)
                print(f"Logged epoch {epoch} metrics", flush=True)

            if lines or seen_sources:
                writer.flush()
            if not running(args.pid):
                completed_after_exit += 1
                if completed_after_exit >= 2:
                    break
            time.sleep(args.poll_seconds)
    finally:
        writer.close()
        print(f"Stopped; observed {len(seen_sources)} training sources", flush=True)


if __name__ == "__main__":
    main()
