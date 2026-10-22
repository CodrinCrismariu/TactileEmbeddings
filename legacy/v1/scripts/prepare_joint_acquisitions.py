#!/usr/bin/env python3
"""Prepare frozen numeric codes and valid chronological histories for joint use."""
from tactilelatent.expansion_runtime import configure
configure()

if __name__=='__main__':
    import fcntl
    from tactilelatent.acquisition_joint import ROOT,prepare_acquisitions
    ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'prepare.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        prepare_acquisitions()
