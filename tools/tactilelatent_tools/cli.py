"""Parse the maintained command menu without loading research dependencies.

Inputs:
    CLI arguments, an optional project root and an explicit experiment config.
Outputs:
    A selected command's report and effects; help performs no GPU, network or
    storage operations. Handlers are imported only after argument parsing.
Example:
    tl --help
"""

import argparse
import importlib
import json
import sys
from pathlib import Path

COMMANDS = {
    "train": "tactilelatent_tools.commands.train",
    "evaluate": "tactilelatent_tools.commands.evaluate",
    "tune": "tactilelatent_tools.commands.tune",
    "data": "tactilelatent_tools.commands.data",
    "view": "tactilelatent_tools.commands.view",
    "status": "tactilelatent_tools.commands.status",
}


def _config(parser):
    parser.add_argument(
        "--config",
        "--experiment",
        default="configs/experiment.json",
        help="Experiment JSON, relative to project root or absolute",
    )


def build_parser():
    parser = argparse.ArgumentParser(prog="tl", description="TactileLatent research operations")
    parser.add_argument(
        "--project-root",
        type=Path,
        help="Data/run project root; defaults to the nearest project above cwd",
    )
    parser.add_argument("--version", action="version", version="TactileLatent tools 0.2.0")
    sub = parser.add_subparsers(dest="command")
    train = sub.add_parser("train", help="Start/resume current-core GPU training")
    _config(train)
    continuation = train.add_mutually_exclusive_group()
    continuation.add_argument(
        "--resume", type=Path, help="Current-core run directory or its resumable last.pt"
    )
    continuation.add_argument(
        "--import-checkpoint",
        type=Path,
        help="Verify and import a legacy current-frame V-JEPA checkpoint into a new run",
    )
    train.add_argument(
        "--import-only",
        action="store_true",
        help="With --import-checkpoint, prepare resumable state without starting optimization",
    )
    train.add_argument(
        "--preflight-only",
        action="store_true",
        help="Disposable native-input/SAM checks; does not alter a saved run",
    )
    evaluate = sub.add_parser("evaluate", help="Fit frozen-representation readout probes")
    _config(evaluate)
    evaluate.add_argument("--run", type=Path, required=True, help="Current-core run to evaluate")
    evaluate.add_argument("--epochs", nargs="+", type=int, help="Explicit completed epoch numbers")
    evaluate.add_argument(
        "--protocol", choices=("motiontrans", "dataset-probe"), default="motiontrans"
    )
    tune = sub.add_parser("tune", help="Run an explicitly bounded candidate/seed plan")
    tune.add_argument(
        "--plan", type=Path, required=True, help="JSON with candidates, seeds, budget and output"
    )
    tune.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve/validate the plan without writing or training",
    )
    tune.add_argument(
        "--resume", action="store_true", help="Resume an identical already-materialized tuning plan"
    )
    data = sub.add_parser("data", help="Manage datasets independently of model training")
    data.add_argument(
        "action", choices=("discover", "verify", "prepare", "import", "cache", "download")
    )
    _config(data)
    data.add_argument(
        "--source", help="Registered source ID; omit only when the operation supports all sources"
    )
    data.add_argument(
        "--output", type=Path, help="Explicit sidecar/output destination; no in-place cache rewrite"
    )
    data.add_argument(
        "--sample",
        action="store_true",
        help="Also verify representative native train/validation clips",
    )
    data.add_argument(
        "--full",
        action="store_true",
        help="Hash complete code payloads/frame contracts; does not audit every RGB frame",
    )
    data.add_argument("--url", help="Explicit download URL when the selected provider supports it")
    data.add_argument("--sha256", help="Required expected digest for an explicit download")
    data.add_argument("--revision", help="Pinned provider revision, if applicable")
    data.add_argument("--destination", type=Path, help="Explicit download destination")
    view = sub.add_parser("view", help="Serve reconstruction, attention and live run history")
    _config(view)
    view.add_argument("--host", default="127.0.0.1")
    view.add_argument("--port", type=int, default=8080)
    status = sub.add_parser(
        "status", help="Read a run status/last metrics row without GPU or writes"
    )
    _config(status)
    status.add_argument(
        "--run", type=Path, help="Run directory; defaults to training.output in the experiment"
    )
    status.add_argument(
        "--watch", action="store_true", help="Print read-only status snapshots continuously"
    )
    status.add_argument(
        "--interval", type=float, default=15.0, help="Positive seconds between watch snapshots"
    )
    status.add_argument(
        "--samples", type=int, help="Optional positive finite snapshot count with --watch"
    )
    return parser


def find_project_root(explicit=None):
    if explicit is not None:
        root = explicit.expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"Project root does not exist: {root}")
        return root
    current = Path.cwd().resolve()
    for root in (current, *current.parents):
        if (root / "pyproject.toml").is_file() and (root / "configs").is_dir():
            return root
    raise ValueError("Cannot locate the project; pass --project-root explicitly")


def config_path(args, root):
    path = Path(args.config).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def project_path(path, root):
    if path is None:
        return None
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    try:
        root = find_project_root(args.project_root)
        if args.command == "view" and not 1 <= args.port <= 65535:
            raise ValueError("Port must be between 1 and 65535")
        if (
            args.command == "evaluate"
            and args.epochs is not None
            and any(epoch < 1 for epoch in args.epochs)
        ):
            raise ValueError("Epochs must be positive completed epoch numbers")
        handler = importlib.import_module(COMMANDS[args.command])
        result = handler.run(args, root)
        if result is not None:
            print(json.dumps(result, indent=2, sort_keys=True, default=str))
        return 0
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"tl {args.command}: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print(f"tl {args.command}: interrupted", file=sys.stderr)
        return 130
