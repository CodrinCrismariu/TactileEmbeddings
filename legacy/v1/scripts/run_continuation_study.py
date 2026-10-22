"""Detached four-arm queue; clean epoch boundaries alternate training and probes."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from tactilelatent.aligned_frames import file_hash, write_json

RUNS = ('control', 'source_stopping', 'retention', 'combined')
BOUNDARIES = (12, 17, 22, 27)


def completed(folder):
    log=folder/'metrics.jsonl'
    if log.exists():
        for line in reversed(log.read_text().splitlines()):
            try:
                return int(json.loads(line)['epoch'])
            except (ValueError,KeyError):
                pass
    return 7


def prepare(root):
    if (root/'plan.json').exists():
        raise FileExistsError('Study already prepared; use run to resume')
    configs={n:json.loads((root/'configs'/f'{n}.json').read_text()) for n in RUNS}
    common=[{k:v for k,v in c.items() if k not in ('output','source_stopping','retention','beta_max')} for c in configs.values()]
    if any(c != common[0] for c in common[1:]):
        raise ValueError('Arms differ outside the explicit interventions')
    from tactilelatent.train_continuation import prepare as preflight
    for name,cfg in configs.items():
        if Path(cfg['output']).resolve() != (root/name).resolve():
            raise ValueError('Output must match the selected study arm')
        checked=preflight(cfg); del checked
    code_paths=['scripts/run_continuation_study.py','scripts/evaluate_continuation_holdout.py',
        'scripts/evaluate_alignment_checkpoint_sweep.py','scripts/evaluate_invariance_holdout.py',
        'src/tactilelatent/paired_probe_cache.py','src/tactilelatent/frozen_probe.py',
        'src/tactilelatent/evaluate_holdout_decoder.py','src/tactilelatent/train_continuation.py',
        'src/tactilelatent/continuation_objectives.py']
    write_json(root/'plan.json',dict(seed=45,parent_epoch=7,target_epoch=27,additional_epochs_each=20,
        runs=list(RUNS),probe_epochs=[7,*BOUNDARIES],transfer_head=[512,512,512],
        conditions={name:dict(config=str(root/'configs'/f'{name}.json'),sha256=file_hash(root/'configs'/f'{name}.json')) for name in RUNS},
        code={p:file_hash(p) for p in code_paths},
        notes='Same parent weights and fresh optimizer; original cycle sampler; no manual source labels, private context, FGW, extra seeds or screening runs. All four train to epoch 27; transfer probes do not select training duration.'))


def run(root, parallel):
    if parallel not in (1,2):
        raise ValueError('Use at most two GPU training workers')
    lock=(root/'controller.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=json.loads((root/'plan.json').read_text())
    for path,digest in plan['code'].items():
        if file_hash(path)!=digest:
            raise ValueError(f'Pinned study implementation changed: {path}')
    for row in plan['conditions'].values():
        if file_hash(row['config'])!=row['sha256']:
            raise ValueError('Pinned study configuration changed')
    if shutil.disk_usage(root).free < 35*2**30:
        raise RuntimeError('At least 35 GiB free required for checkpoints and probe caches')
    stopping=[False]
    signal.signal(signal.SIGTERM,lambda *_:stopping.__setitem__(0,True))
    signal.signal(signal.SIGINT,lambda *_:stopping.__setitem__(0,True))
    workers,streams={},{}
    began=time.monotonic()
    def status(stage,**details):
        rows={n:dict(completed_epoch=completed(root/n),pid=workers[n].pid if n in workers else None) for n in RUNS}
        write_json(root/'status.json',dict(stage=stage,controller_pid=os.getpid(),elapsed_seconds=time.monotonic()-began,
            max_parallel=parallel,runs=rows,**details))
    def launch(name,command):
        log=root/'logs'/f'{name}.log'; log.parent.mkdir(parents=True,exist_ok=True)
        streams[name]=log.open('a')
        p=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=streams[name],stderr=subprocess.STDOUT,start_new_session=True)
        workers[name]=p
        write_json(root/f'{name}.pid',dict(pid=p.pid,process_group=p.pid,command=command))
        print(json.dumps(dict(event='started',name=name,pid=p.pid,command=command)),flush=True)
    def finish(name):
        p=workers.pop(name); streams.pop(name).close()
        if p.returncode:
            raise RuntimeError(f'{name} failed with exit {p.returncode}; see logs')
    def probe(epoch):
        summary=root/'evaluations'/f'epoch_{epoch:03d}'/'summary.json'
        if summary.exists():
            return
        launch('probe',[sys.executable,'-u','scripts/evaluate_continuation_holdout.py','--root',str(root),'--epoch',str(epoch)])
        while not stopping[0] and workers['probe'].poll() is None:
            status('transfer_probe',epoch=epoch,probe_pid=workers['probe'].pid); time.sleep(5)
        if not stopping[0]:
            finish('probe')
    try:
        probe(7)
        for boundary in BOUNDARIES:
            if stopping[0]:
                break
            pending=[n for n in RUNS if completed(root/n)<boundary]
            while (pending or workers) and not stopping[0]:
                for name,p in list(workers.items()):
                    if p.poll() is not None:
                        finish(name)
                        if completed(root/name)<boundary:
                            raise RuntimeError(f'{name} exited before epoch {boundary}')
                while pending and len(workers)<parallel:
                    name=pending.pop(0)
                    command=[sys.executable,'-u','-m','tactilelatent.train_continuation',
                        '--config',plan['conditions'][name]['config'],'--until-epoch',str(boundary)]
                    if (root/name/'last.pt').exists():
                        command.append('--resume')
                    launch(name,command)
                # Weight hashes verify the four initializations, including queued arms.
                hashes=set()
                for name in RUNS:
                    init=root/name/'initialization.json'
                    if init.exists():
                        hashes.add(json.loads(init.read_text())['model_sha256'])
                if len(hashes)>1:
                    raise ValueError('Arms did not start with identical weights')
                status('training',boundary_epoch=boundary,queued=pending)
                time.sleep(5)
            if not stopping[0]:
                probe(boundary)
        if not stopping[0]:
            status('complete')
    except Exception as error:
        status('failed',error=str(error)); raise
    finally:
        for name,p in workers.items():
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGTERM)
        for name,p in workers.items():
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid,signal.SIGKILL); p.wait()
        for stream in streams.values():
            stream.close()
        if stopping[0]:
            workers.clear(); status('stopped_by_user')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run'))
    parser.add_argument('--root',type=Path,default=Path('processing/runs/continuation_32d_v1'))
    parser.add_argument('--parallel',type=int,default=2)
    args=parser.parse_args()
    prepare(args.root) if args.action=='prepare' else run(args.root,args.parallel)
