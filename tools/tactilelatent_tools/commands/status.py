"""Read saved run status without dataset, model or optimization imports.

Inputs:
    A run directory or the output path selected by the experiment configuration.
Outputs:
    JSON status and the last complete metrics row on stdout; watch mode repeats
    these read-only snapshots without writing artifacts.
Example:
    tl status --run processing/runs/current --watch --interval 15
"""

import json
import math
import time

from ..cli import config_path, project_path


def _last_metrics(path):
    if not path.exists():
        return None
    # Read at most the final MiB; tolerate one incomplete line written by a live job.
    with path.open("rb") as stream:
        stream.seek(0, 2)
        length = stream.tell()
        offset = max(0, length - 1024 * 1024)
        stream.seek(offset)
        lines = stream.read().splitlines()
    if offset:
        lines = lines[1:]
    for line in reversed(lines):
        if not line.strip():
            continue
        try:
            return json.loads(line)
        except (ValueError, UnicodeDecodeError):
            continue
    return None


def run(args, root):
    if args.run is None:
        from tactilelatent.configuration import load_experiment

        configuration = load_experiment(config_path(args, root), project_root=root)
        directory = project_path(configuration["training"]["output"], root)
    else:
        directory = project_path(args.run, root)
    if (
        not math.isfinite(args.interval)
        or args.interval <= 0
        or args.samples is not None
        and args.samples < 1
    ):
        raise ValueError("Watch interval and finite sample count must be positive")
    if args.samples is not None and not args.watch:
        raise ValueError("--samples is valid only with --watch")

    def snapshot():
        path = directory / "status.json"
        state = json.loads(path.read_text()) if path.exists() else {"stage": "not_started"}
        return {
            "run": str(directory),
            "status": state,
            "last_metrics": _last_metrics(directory / "metrics.jsonl"),
            "writes": False,
        }

    if not args.watch:
        return snapshot()
    count = 0
    while args.samples is None or count < args.samples:
        print(json.dumps(snapshot(), sort_keys=True), flush=True)
        count += 1
        if args.samples is None or count < args.samples:
            time.sleep(args.interval)
    return None
