#!/usr/bin/env python3
"""Run the approved downloads AND processing; never launches joint training."""
import argparse
import os
import json
from pathlib import Path
os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS','1')
os.environ.setdefault('HF_XET_CHUNK_CACHE_SIZE_BYTES','0')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT','120')
from tactilelatent import acquisition_queue as queue

# Historical v2 configuration remains frozen for existing provenance receipts.
# All normal queue entrypoints (including spawned workers) use the current list.
queue.CONFIG=Path('configs/data_queue_active.json')
original_snapshot=queue.snapshot


def snapshot(config=None):
    result=original_snapshot(config)
    path=queue.ROOT/'STATUS.md'
    text=path.read_text().replace('240 GB acquisition budget',
        f'{result["acquisition_budget_bytes"]/1e9:g} GB acquisition budget')
    text=text.replace('T-Dex and HRDexDB are pilots. TacQuad is an alignment benchmark, excluded from joint visual training.',
        'HRDexDB is a pilot. T-Dex and TacQuad were removed by the user and are not scheduled for acquisition.')
    path.write_text(text)
    return result


queue.snapshot=snapshot

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['run','source','status'])
    p.add_argument('source',nargs='?',choices=[s['id'] for s in json.loads(queue.CONFIG.read_text())['jobs']])
    args=p.parse_args()
    if args.command=='run': queue.run_queue()
    elif args.command=='source':
        if not args.source:p.error('source requires a dataset name')
        queue.run_source(args.source)
    else:
        print(json.dumps(snapshot(),indent=2))
