"""Honor the epoch-22 stop, finish its probes, then screen frozen representations."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from datetime import datetime,timezone

from tactilelatent.aligned_frames import write_json

RUNS=('control','source_stopping','retention','combined')


def process_matches(pid,start_ticks):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split()[21]==start_ticks
    except FileNotFoundError:
        return False


def epoch_done(root,name):
    folder=root/name
    try:
        status=json.loads((folder/'status.json').read_text())
        rows=[json.loads(r) for r in (folder/'metrics.jsonl').read_text().splitlines() if r.strip()]
        if rows and rows[-1]['epoch']>22:
            raise RuntimeError('A continuation exceeded the requested stopping boundary')
        return (rows and rows[-1]['epoch']==22 and status.get('completed_epoch')==22
                and (folder/'viewer_epochs/epoch_022.pt').exists())
    except (FileNotFoundError,json.JSONDecodeError):
        return False


def continuation_report(root):
    records={epoch:json.loads((root/'evaluations'/f'epoch_{epoch:03d}'/'summary.json').read_text()) for epoch in (7,12,17,22)}
    lines=['# Continuation stopped after epoch 22','',
        'The user stopped the continuation experiment at epoch 22 after its MotionTrans evaluation. No epoch-27 continuation is authorized. All saved encoder and optimizer checkpoints remain available.','',
        '| Arm | Epoch 12 MAE | Epoch 17 MAE | Epoch 22 MAE | Change 17→22 | Epoch 22 nominal 90% coverage |','|---|---:|---:|---:|---:|---:|']
    for name in RUNS:
        a,b,c=(records[e]['results'][name] for e in (12,17,22))
        lines.append(f"| {name} | {a['deterministic_mae']:.6f} | {b['deterministic_mae']:.6f} | {c['deterministic_mae']:.6f} | {c['deterministic_mae']-b['deterministic_mae']:+.6f} | {100*c['coverage_90']:.2f}% |")
    lines+=['',f"The original epoch-7 three-hidden-layer baseline MAE is {records[7]['results']['control']['deterministic_mae']:.6f}. All values above retain the existing Gaussian mean-plus-sampled head-fitting protocol. The separate frozen representation screen uses means-only head fitting for every representation and fits a new matched epoch-7 reference.",'',
        'Slope differences are descriptive results from one run and repeated head fits, not causal estimates. The epoch-22 JSON contains paired episode-bootstrap comparisons against the control and parent. Coverage remains an empirical diagnostic rather than a calibrated uncertainty guarantee.']
    (root/'evaluations/epoch_022/analysis.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('processing/runs/continuation_32d_v1'))
    parser.add_argument('--output',type=Path,default=Path('processing/evaluations/motiontrans_representation_screen_v1'))
    args=parser.parse_args();root=args.root;output=args.output;output.mkdir(parents=True,exist_ok=True)
    request=json.loads((root/'stop_after_epoch_022.json').read_text())
    pid=request['old_controller_pid'];ticks=request['old_controller_start_ticks']
    began=time.monotonic()
    def status(stage,**details):
        row=dict(stage=stage,controller_pid=os.getpid(),elapsed_seconds=time.monotonic()-began,requested_stop_epoch=22,**details)
        write_json(root/'finish_status.json',row);print(json.dumps(row),flush=True)
    def run(command,log_path):
        with log_path.open('a') as log:
            process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            status('evaluating',worker_pid=process.pid,command=command,log=str(log_path))
            if process.wait():
                raise RuntimeError(f'Evaluation failed; see {log_path}')
    try:
        while not all(epoch_done(root,name) for name in RUNS):
            if process_matches(pid,ticks) and 'State:\tT' not in Path(f'/proc/{pid}/status').read_text():
                raise RuntimeError('Original queue unexpectedly resumed before the stopping boundary')
            status('waiting_for_epoch_22',completed=[name for name in RUNS if epoch_done(root,name)])
            time.sleep(5)
        # The old queue is parked while its final worker exits naturally. Its
        # termination handler will skip future probes/launches and clean up.
        if process_matches(pid,ticks):
            os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT)
            for _ in range(60):
                if not process_matches(pid,ticks) or 'State:\tZ' in Path(f'/proc/{pid}/status').read_text():
                    break
                time.sleep(.5)
            else:
                raise RuntimeError('Original queue did not terminate')
        summary=root/'evaluations/epoch_022/summary.json'
        if not summary.exists():
            run([sys.executable,'-u','scripts/evaluate_continuation_holdout.py','--root',str(root),'--epoch','22'],root/'logs/finish_epoch_022.log')
        data=json.loads(summary.read_text())
        if data['epoch']!=22 or set(data['results'])!=set(RUNS):
            raise ValueError('Incomplete epoch-22 transfer evaluation')
        continuation_report(root)
        stopped=dict(stage='stopped_after_epoch_22_evaluation',requested_stop_epoch=22,completed_epoch=22,
            stopped_utc=datetime.now(timezone.utc).isoformat(),controller_pid=None,
            runs={name:dict(completed_epoch=22,pid=None) for name in RUNS},evaluation=str(summary),
            next_action='Frozen representation comparison; no further encoder training')
        write_json(root/'status.json',stopped);write_json(root/'stop_status.json',stopped)
        status('continuation_stopped',evaluation=str(summary))
        screen=output/'summary.json'
        ready=screen.exists() and len(json.loads(screen.read_text()).get('results',{}))==8
        if not ready:
            run([sys.executable,'-u','scripts/evaluate_representation_screen.py','--output',str(output)],output/'nohup.log')
        status('complete',continuation_evaluation=str(summary),representation_summary=str(screen))
    except Exception as error:
        status('failed',error=str(error));raise


if __name__=='__main__':
    main()
