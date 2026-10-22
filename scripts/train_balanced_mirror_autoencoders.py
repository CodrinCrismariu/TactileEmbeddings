#!/usr/bin/env python3
"""Continue all verified mirror AEs with paired native targets and parity checks."""
from tactilelatent.expansion_runtime import configure
configure()

import argparse
from tactilelatent.mirror_balanced_training import run


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/mirror_balanced_training_v3.json')
    args=parser.parse_args()
    result=run(args.config)
    if any(v.get('stage')=='failed' for v in result['sources'].values()):
        raise SystemExit('A source failed; see ready.json. Other sources were still attempted.')
    if result['complete_for_verified_sources']:
        import json,subprocess,sys
        from pathlib import Path
        root=json.loads(Path(args.config).read_text())['output']
        subprocess.run([sys.executable,'scripts/verify_bimanual_mirror.py',
            '--preparation',root,'--base-config','configs/joint_vjepa21_spatial_expanded_32d_mirror_v2.json'],check=True)
