"""Serve the maintained viewer and historical-checkpoint read adapters.

Inputs:
    Project root, experiment configuration, model artifacts, host and port.
Outputs:
    An HTTP service for reconstruction, attention and training histories;
    explicitly selected inference may use the GPU.
Example:
    tl view --host 127.0.0.1 --port 8080
"""

from ..cli import config_path


def run(args, root):
    from tactilelatent.visualization.server import serve

    return serve(
        project_root=root, host=args.host, port=args.port, experiment=config_path(args, root)
    )
