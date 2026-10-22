#!/usr/bin/env python3
"""All seventeen native sensor layouts, verified mirroring, MotionTrans held out."""
from tactilelatent.expansion_runtime import configure
configure()

if __name__=='__main__':
    import sys
    from tactilelatent import train_vjepa_markers as trainer
    from tactilelatent.acquisition_training import install
    if not any(a=='--config' or a.startswith('--config=') for a in sys.argv[1:]):
        sys.argv.extend(['--config','configs/joint_vjepa21_all_sources_32d_mirror_v2.json'])
    install(trainer).main()
