#!/usr/bin/env python3
"""Train/cache missing mirror decoders only. Does not launch joint training."""
from tactilelatent.expansion_runtime import configure
configure()

if __name__=='__main__':
    from tactilelatent.bilateral_mirror_preparation import run
    result=run()
    if not result['complete']:raise SystemExit('Some mirror teachers failed quality; inspect ready.json')
