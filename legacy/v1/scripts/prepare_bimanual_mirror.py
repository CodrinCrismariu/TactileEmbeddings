#!/usr/bin/env python3
"""Prepare verified bilateral teachers/codes; never launch joint training."""
import argparse
from pathlib import Path
from tactilelatent.mirror_preparation import run, ROOT, MANIFEST, POLICY
from tactilelatent.mirror_cache_verification import install as install_verifier


def main():
    install_verifier()
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT)
    p.add_argument('--manifest',default=MANIFEST)
    p.add_argument('--policy',default=POLICY)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--train-frames',type=int,default=100000)
    p.add_argument('--val-frames',type=int,default=10000)
    a=p.parse_args()
    if min(a.train_frames,a.val_frames)<1:p.error('Frame budgets must be positive')
    run(a.output,a.manifest,a.policy,a.device,a.train_frames,a.val_frames)


if __name__=='__main__':main()
