"""Sequential fresh joint baseline, heldout adaptation and direct-sensor control.

Run through train_motiontrans_sequence.sh, which holds an exclusive lock.
Rerunning resumes completed epochs, skips verified completed stages, and
does not overwrite evaluations. Any failed stage stops the sequence.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

from tactilelatent.frozen_probe import sha256


ROOT = Path(__file__).resolve().parents[1]
JOINT = "joint_all_holdout_motiontrans_fresh_8k_50ep"
DIRECT = "motiontrans_direct_vae32_50ep"
PIPELINE = ROOT / "processing/runs/motiontrans_baseline_sequence"


def run(command, log):
    """Append subprocess output and let failures prevent later stages."""
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as stream:
        subprocess.run([sys.executable, *command], cwd=ROOT,
                       stdout=stream, stderr=subprocess.STDOUT, check=True)


def train(name):
    config = ROOT / f"configs/{name}.json"
    cfg = json.loads(config.read_text())
    if cfg["init_checkpoint"] is not None or cfg["init_joint_checkpoint"] is not None:
        raise ValueError("This experiment must start with fresh learned weights")
    run_dir = ROOT / cfg["output"]
    checkpoint = run_dir / "last.pt"
    command = ["-m", "tactilelatent.train_joint", "--config", str(config)]
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if state["config"] != cfg:
            raise ValueError(f"Configuration mismatch: {checkpoint}")
        if state["epoch"] == cfg["epochs"] - 1:
            print(f"Skipping completed training: {name}", flush=True)
            return checkpoint
        del state
        command.append("--resume")
    run_dir.mkdir(parents=True, exist_ok=True)
    # The log mirror runs only during this trainer and uses no GPU.
    with (run_dir / "nohup.log").open("a") as log, (run_dir / "tensorboard_bridge.log").open("a") as bridge:
        trainer = subprocess.Popen([sys.executable, *command], cwd=ROOT,
                                   stdout=log, stderr=subprocess.STDOUT)
        mirror = subprocess.Popen([sys.executable, "scripts/tensorboard_joint_live.py",
                                   str(run_dir), "--pid", str(trainer.pid)], cwd=ROOT,
                                  stdout=bridge, stderr=subprocess.STDOUT)
        try:
            trainer.wait()
            if trainer.returncode:
                raise subprocess.CalledProcessError(trainer.returncode, command)
        except BaseException:
            if trainer.poll() is None:
                trainer.terminate()
                trainer.wait()
            raise
        finally:
            mirror.wait(timeout=30)
    return checkpoint


def evaluation(module, checkpoint, output):
    result = output / "results.json"
    if result.exists():
        row = json.loads(result.read_text())
        if (row["checkpoint_sha256"] != sha256(checkpoint)
                or row["manifest_sha256"] != sha256(ROOT / "processing/joint_all/manifest.json")
                or row["source"] != "ftp_motiontrans"
                or row["validation"]["sample_count"] != 64):
            raise ValueError(f"Completed evaluation belongs to a different experiment: {result}")
        return
    run(["-m", module, "--source", "ftp_motiontrans", "--checkpoint", str(checkpoint),
         "--output", str(output), "--samples", "64"], output / "evaluation.log")


def compare():
    paths = {
        "joint_frozen_encoder": ROOT / f"processing/runs/{JOINT}/evaluations/motiontrans/results.json",
        "direct_motiontrans": ROOT / f"processing/runs/{DIRECT}/evaluation/results.json",
    }
    rows = {name: json.loads(path.read_text()) for name, path in paths.items()}
    if (len({row["manifest_sha256"] for row in rows.values()}) != 1
            or {row["validation"]["validation_clips"] for row in rows.values()} != {1972}
            or {row["validation"]["sample_count"] for row in rows.values()} != {64}
            or any(row["source"] != "ftp_motiontrans" for row in rows.values())):
        raise ValueError("Comparison requires identical MotionTrans validation and sample count")
    result = {"source": "ftp_motiontrans", "validation_clips": 1972, "samples": 64,
              "manifest_sha256": next(iter(rows.values()))["manifest_sha256"],
              "protocol_note": "Frozen joint encoder never trained on MotionTrans; its new head is fitted on MotionTrans train episodes. Direct encoder and head both train on MotionTrans. Not zero-shot and not equal training compute.",
              "models": {name: {"checkpoint_sha256": row["checkpoint_sha256"],
                                 "training_clips": row["train_clips"],
                                 "validation": row["validation"]} for name, row in rows.items()}}
    destination = PIPELINE / "comparison.json"
    if destination.exists():
        if json.loads(destination.read_text()) != result:
            raise FileExistsError(f"Refusing to replace a different comparison: {destination}")
    else:
        destination.write_text(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Only one training/validation clip per source, in separate runs")
    args = parser.parse_args()
    PIPELINE.mkdir(parents=True, exist_ok=True)
    stage = "starting"

    def status(value):
        nonlocal stage
        stage = value
        (PIPELINE / "status.json").write_text(json.dumps({"stage": value, "smoke": args.smoke}, indent=2))
        print(json.dumps({"stage": value, "smoke": args.smoke}), flush=True)

    try:
        if args.smoke:
            for name in ("joint_all_holdout_motiontrans_smoke", "motiontrans_direct_smoke"):
                status(name)
                train(name)
            status("smoke_complete")
            return
        status("joint_baseline_training")
        checkpoint = train(JOINT)
        status("frozen_motiontrans_decoder_evaluation")
        evaluation("tactilelatent.evaluate_holdout_decoder", checkpoint,
                   ROOT / f"processing/runs/{JOINT}/evaluations/motiontrans")
        status("motiontrans_only_training")
        checkpoint = train(DIRECT)
        status("direct_motiontrans_evaluation")
        evaluation("tactilelatent.evaluate_direct_sensor", checkpoint,
                   ROOT / f"processing/runs/{DIRECT}/evaluation")
        compare()
        status("complete")
    except BaseException as error:
        (PIPELINE / "status.json").write_text(json.dumps(
            {"stage": stage, "smoke": args.smoke, "error": repr(error)}, indent=2))
        raise


if __name__ == "__main__":
    main()
