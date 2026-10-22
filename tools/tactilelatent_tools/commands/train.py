"""Start or resume a current-core run through the reusable training API.

Inputs:
    Experiment configuration, verified source artifacts and optional v2 last.pt
    or an explicitly selected legacy current-frame V-JEPA checkpoint to import.
Outputs:
    Run checkpoints, metrics and logs, or disposable preflight results. This
    command never consults a historical training launcher. --import-only writes
    verified continuation state in a new run without starting optimization.
Example:
    tl train --config configs/experiment.json --preflight-only
"""

from ..cli import config_path, project_path


def run(args, root):
    from tactilelatent.training.engine import train_experiment

    resume = project_path(args.resume, root)
    if resume is not None and resume.is_file():
        if resume.name != "last.pt":
            raise ValueError(
                "Resume a run directory or its resumable last.pt, not an inference snapshot"
            )
        resume = resume.parent
    if args.import_only and args.import_checkpoint is None:
        raise ValueError("--import-only requires --import-checkpoint")
    if args.preflight_only and (args.import_only or args.import_checkpoint is not None):
        raise ValueError("Checkpoint import and disposable preflight are separate operations")
    options = {}
    if args.import_checkpoint is not None:
        options["import_checkpoint"] = project_path(args.import_checkpoint, root)
        options["import_only"] = args.import_only
    return train_experiment(
        config_path(args, root),
        project_root=root,
        resume=resume,
        preflight_only=args.preflight_only,
        **options,
    )
