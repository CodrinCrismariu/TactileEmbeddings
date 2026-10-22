#!/usr/bin/env python3
"""Resumable data expansion. No joint-model training is launched."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS','1')
os.environ.setdefault('HF_XET_CHUNK_CACHE_SIZE_BYTES','0')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT','120')

from tactilelatent.expansion_runtime import configure
configure()

from tactilelatent.data_expansion import ROOT,report,setup


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job',choices=['downloads','egopressure','feel','baihu','teachers','publish','all'])
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--pilot-only',action='store_true')
    args=parser.parse_args();setup()
    lock=(ROOT/f'{args.job}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if args.job in ('downloads','all'):
        jobs=['egopressure','feel']+(['baihu','teachers'] if args.job=='all' else [])
        processes=[]
        for job in jobs:
            log=(ROOT/f'{job}.log').open('a')
            p=subprocess.Popen([sys.executable,'-u',__file__,job,'--workers',str(args.workers)],
                stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
            processes.append((job,p,log))
        while any(p.poll() is None for _,p,_ in processes):
            report(args.job,stage='running',jobs={j:{'pid':p.pid,'returncode':p.poll()} for j,p,_ in processes})
            time.sleep(30)
        failed={j:p.returncode for j,p,_ in processes if p.returncode}
        for _,_,log in processes:log.close()
        report(args.job,stage='failed' if failed else 'complete',failures=failed)
        if failed:raise SystemExit(1)
    else:
        try:
            if args.job=='egopressure':
                from tactilelatent.data_expansion import acquire_egopressure
                acquire_egopressure()
            elif args.job=='feel':
                from tactilelatent.data_expansion import acquire_feel
                acquire_feel()
            elif args.job=='baihu':
                from tactilelatent.expansion_baihu import run
                run(args.workers,args.pilot_only)
            elif args.job=='teachers':
                from tactilelatent.expansion_training import run_teachers
                run_teachers()
            elif args.job=='publish':
                from tactilelatent.expansion_training import publish
                publish()
        except Exception as e:
            report(args.job,stage='failed',error=f'{type(e).__name__}: {e}')
            raise


if __name__=='__main__':main()
