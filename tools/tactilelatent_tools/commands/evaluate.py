"""Fit readout heads on frozen representations with an explicit protocol.

Inputs:
    A current-core run, completed epoch numbers and transfer/probe configuration.
Outputs:
    Evaluation features, fitted heads and reports under the run's evaluation path.
Example:
    tl evaluate --run processing/runs/current --epochs 10 --protocol motiontrans
"""

from ..cli import config_path, project_path


def run(args, root):
    from tactilelatent.evaluation.transfer import evaluate_experiment

    return evaluate_experiment(
        config_path(args, root),
        project_root=root,
        run=project_path(args.run, root),
        epochs=args.epochs,
        protocol=args.protocol,
    )
