"""Run the maintained tools through Python's module entrypoint.

Inputs:
    The same arguments accepted by the installed tl console command.
Outputs:
    Command reports and explicitly requested operational effects.
Example:
    python -m tactilelatent_tools --help
"""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
