"""Extract native optical dot trajectories without changing source datasets."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

import numpy as np

from tactilelatent.aligned_frames import file_hash, json_hash, manifest_signature, write_json
from tactilelatent.marker_targets import MarkerConfig, MARKER_SOURCES, VERSION
from tactilelatent.marker_data import extract_episode, episode_key


def process(row, root, config):
    began=time.monotonic()
    folder=Path(root)/row['source']/row['split']/episode_key(row)
    cfg=MarkerConfig(**config)
    try:
        complete=folder/'complete.json'
        if complete.exists():
            info=json.loads(complete.read_text())
            if (info['row_sha256']!=json_hash(row) or info['version']!=VERSION
                    or info['extractor_sha256']!=file_hash('src/tactilelatent/marker_targets.py')
                    or info['loader_sha256']!=file_hash('src/tactilelatent/marker_data.py')
                    or any(s['config']!=config for s in info['sensors'])
                    or info['tracks_sha256']!=file_hash(folder/'tracks.npz')):
                raise ValueError('Existing marker artifact is incompatible')
        else:
            info=extract_episode(row,folder,cfg)
        return dict(ok=True,source=row['source'],split=row['split'],episode_id=row['episode_id'],
                    path=str(folder),info=info,seconds=time.monotonic()-began)
    except Exception as error:
        return dict(ok=False,source=row['source'],split=row['split'],episode_id=row['episode_id'],
                    error=str(error),seconds=time.monotonic()-began)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',default='processing/joint_alignment/manifest.json')
    parser.add_argument('--output',type=Path,default=Path('processing/marker_displacement_v2'))
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--pilot',action='store_true',help='First/middle/last episodes per source, including validation')
    args=parser.parse_args();root=args.output;root.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(root/'prepare.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    rows=[r for r in json.loads(Path(args.manifest).read_text())['episodes'] if r['source'] in MARKER_SOURCES]
    rows.sort(key=lambda r:(r['source']=='freetacman',r['source'],r['episode_id']))
    if args.pilot:
        selected=[]
        for source in MARKER_SOURCES:
            subset=[r for r in rows if r['source']==source]
            selected.extend(subset[i] for i in sorted({0,len(subset)//2,len(subset)-1}))
        rows=selected
    configs={s:asdict(MarkerConfig(min_markers=100 if s=='freetacman' else 48)) for s in MARKER_SOURCES}
    provenance=dict(version=VERSION,data=manifest_signature(args.manifest),sources=list(MARKER_SOURCES),
                    configs=configs,pilot=args.pilot,rows_sha256=json_hash(rows),
                    extractor_sha256=file_hash('src/tactilelatent/marker_targets.py'),
                    loader_sha256=file_hash('src/tactilelatent/marker_data.py'))
    pinned=root/'provenance.json'
    if pinned.exists() and json.loads(pinned.read_text())!=provenance:
        raise ValueError('Target extraction provenance differs; use a new output directory')
    write_json(pinned,provenance)
    results=[];began=time.monotonic()
    def status(stage,**details):
        seconds=time.monotonic()-began
        value=dict(stage=stage,pid=os.getpid(),episodes_done=len(results),episodes_total=len(rows),
                   frames_done=sum(r['info']['frames'] for r in results if r['ok']),
                   rejected=sum(not r['ok'] for r in results),seconds=seconds,
                   eta_seconds=seconds/len(results)*(len(rows)-len(results)) if results else None,**details)
        write_json(root/'status.json',value);print(json.dumps(value),flush=True)
    status('extracting')
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        tasks=[pool.submit(process,r,str(root),configs[r['source']]) for r in rows]
        for task in as_completed(tasks):
            result=task.result();results.append(result)
            with (root/'episodes.jsonl').open('a') as stream:stream.write(json.dumps(result)+'\n')
            status('extracting',latest={k:result[k] for k in ('source','episode_id','ok','seconds')})
    # Fail closed: incomplete/untrackable episodes are listed explicitly. A
    # training configuration cannot silently replace missing targets with RGB.
    statistics={}
    for source in MARKER_SOURCES:
        selected=[r['info'] for r in results if r['ok'] and r['source']==source and r['split']=='train']
        count=sum(r['loss_statistics']['coordinate_count'] for r in selected)
        square=sum(r['loss_statistics']['square_sum'] for r in selected)
        statistics[source]=dict(train_coordinate_count=count,
            rms_spacing_units=float(np.sqrt(square/max(count,1))),
            valid_fraction=float(np.mean([r['valid_fraction'] for r in selected])) if selected else None)
    index=dict(provenance=provenance,episodes=[{k:r[k] for k in ('source','split','episode_id','path')} for r in results if r['ok']],
               rejected=[r for r in results if not r['ok']],statistics=statistics,
               all_episodes_complete=all(r['ok'] for r in results))
    write_json(root/'index.json',index)
    write_json(root/'complete.json',dict(provenance=provenance,episodes=len(results),
        rejected=len(index['rejected']),all_episodes_complete=index['all_episodes_complete'],statistics=statistics))
    status('complete' if index['all_episodes_complete'] else 'complete_with_rejected_episodes')


if __name__=='__main__':main()
