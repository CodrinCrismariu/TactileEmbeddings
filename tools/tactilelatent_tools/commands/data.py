"""Dispatch dataset operations independently of the model trainer.

Inputs:
    Dataset configuration, source ID and explicit verification/preparation or
    acquisition options. Downloads require a pinned plan or URL and digest.
Outputs:
    A discovery/verification report or explicitly requested preparation/download
    artifacts. Cache imports write new sidecars and preserve original arrays.
Example:
    tl data verify --source egopressure
"""

from ..cli import config_path, project_path


def run(args, root):
    from tactilelatent.configuration import load_experiment
    from tactilelatent.dataset_management.api import manage

    configuration = load_experiment(config_path(args, root), project_root=root)
    download_options = {
        "url": args.url,
        "sha256": args.sha256,
        "revision": args.revision,
        "destination": project_path(args.destination, root),
    }
    if args.action != "download" and any(value is not None for value in download_options.values()):
        raise ValueError("Download options are valid only for data download")
    options = {key: value for key, value in download_options.items() if value is not None}
    if args.full:
        if args.action != "verify":
            raise ValueError("--full is valid only for data verify")
        options["full"] = True
    if args.sample:
        if args.action not in ("verify", "prepare"):
            raise ValueError("--sample is valid only for data verify or prepare")
        options["sample"] = True
    return manage(
        args.action,
        config=configuration["datasets"],
        project_root=root,
        source=args.source,
        output=project_path(args.output, root),
        **options,
    )
