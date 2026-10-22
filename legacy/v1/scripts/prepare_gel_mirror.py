#!/usr/bin/env python3
"""Audit/adapt the confirmed Unit/Baihu mirrors and cache both native variants."""
from tactilelatent.expansion_runtime import configure
configure()

import argparse
from pathlib import Path

from tactilelatent.gel_mirror_preparation import run, ROOT, POLICY


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT)
    p.add_argument('--policy', default=POLICY)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--train-frames', type=int, default=100000)
    p.add_argument('--val-frames', type=int, default=10000)
    args = p.parse_args()
    if min(args.train_frames, args.val_frames) < 1:
        p.error('Frame budgets must be positive')
    ready = run(args.output, args.policy, device=args.device,
                train_frames=args.train_frames, val_frames=args.val_frames)
    if not ready['complete_for_verified_sources']:
        raise SystemExit('Some confirmed sources failed teacher/cache checks; see ready.json')


if __name__ == '__main__':
    main()
