#!/usr/bin/env python3
"""Production preflight, then start/resume the single integrated joint run."""
import json
from pathlib import Path
import subprocess
import sys

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/joint_vjepa21_all_sources_32d_mirror_v2.json')
    config=parser.parse_args().config
    cfg=json.loads(Path(config).read_text());command=[sys.executable,'-u','scripts/train_all_sources_vjepa.py','--config',config]
    subprocess.run(command+['--preflight-only'],check=True)
    subprocess.run(command+(['--resume'] if (Path(cfg['output'])/'last.pt').exists() else []),check=True)
