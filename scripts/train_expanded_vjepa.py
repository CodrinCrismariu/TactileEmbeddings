#!/usr/bin/env python3
"""Expanded twelve-source entrypoint; reuse the existing SAM training loop."""
from tactilelatent.expansion_runtime import configure
configure()

from tactilelatent import train_vjepa_markers as trainer
from tactilelatent.expansion_joint import ExpandedTeachers,prepare


def main():
    import sys
    if not any(a=='--config' or a.startswith('--config=') for a in sys.argv[1:]):
        sys.argv.extend(['--config','configs/joint_vjepa21_spatial_expanded_32d.json'])
    # Keep changes local to this process. Historical entrypoints/checkpoints
    # continue to use their original ten-source preparation and teachers.
    original=trainer.prepare
    trainer.prepare=lambda cfg,allow_pilot=False:prepare(cfg,original,trainer,allow_pilot)
    trainer.MixedMarkerTeachers=ExpandedTeachers
    trainer.MARKER_SOURCES=(*trainer.MARKER_SOURCES,'baihu')
    trainer.main()


if __name__=='__main__':main()
