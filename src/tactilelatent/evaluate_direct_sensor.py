"""Evaluate a directly trained sensor model (MotionTrans or legacy VLA_touch).

This neutral entry point reuses the established deterministic/64-draw
evaluator. --source, --checkpoint and --output should be supplied explicitly.
"""

from .evaluate_direct_vla_touch import main


if __name__ == "__main__":
    main()
