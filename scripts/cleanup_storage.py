"""Audited cleanup of retired FreeTacMan data and redundant experiment weights.

Keep configs, metrics, reports, pinned tactile teachers and inference epochs.
Use --apply for deletion; every removed path and reason is recorded first.
"""
import argparse
import gc
import json
from pathlib import Path
import re
import shutil
import time

import torch

from tactilelatent.aligned_frames import file_hash,write_json

ROOT=Path.cwd();REPORT=Path('processing/storage_v1/cleanup.json')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--apply',action='store_true')
    parser.add_argument('--data-only',action='store_true');args=parser.parse_args()
    prior=json.loads(REPORT.read_text()) if REPORT.exists() else {}
    records=prior.get('records',[]);kept=prior.get('kept',[])
    def remove(path,reason):
        path=Path(path)
        if not path.exists() or path.is_symlink():return
        if not path.resolve().is_relative_to(ROOT) or path.resolve()==ROOT:raise ValueError('Unsafe cleanup root')
        size=sum(p.stat().st_size for p in path.rglob('*') if p.is_file() and not p.is_symlink()) if path.is_dir() else path.stat().st_size
        records.append(dict(path=str(path),bytes=size,reason=reason,deleted=False))
        write_json(REPORT,dict(apply=args.apply,records=records,kept=kept,updated_at=time.time()))
        if args.apply:
            if path.is_dir():shutil.rmtree(path)
            else:path.unlink()
            records[-1]['deleted']=True
            print(json.dumps(records[-1]),flush=True)
    # No valid native teacher needs the old 32px/training-sampled tensor cache.
    for path in ['processing/sensor_ae_cache_v1',
        'datasets/OpenDriveLab__FreeTacMan','processing/marker_displacement_v2/targets/freetacman',
        'processing/sensor_code_cache_markers_v1/freetacman','processing/sensor_code_cache_v1/freetacman',
        'processing/vjepa21_clip_cache_v1/freetacman']:
        remove(path,'Obsolete reduced-resolution AE data cache' if 'sensor_ae_cache' in path else 'User-retired FreeTacMan raw data or derived cache')
    if not args.data_only:
        # Pinned teacher exports are tiny and required by historical visual models.
        protected={p.resolve() for p in Path('processing').rglob('*.pt') if 'teachers' in p.parts}
        protected.update(p.resolve() for p in Path('processing/evaluations').rglob('*.pt') if 'seed_46' not in p.parts and 'seed_47' not in p.parts)
        bad=re.compile(r'(smoke|clamp_check|hurdle_check|lr_check|benchmark|uncheckpointed)',re.I)
        for p in sorted(Path('processing/runs').rglob('*.pt')):
            if p.resolve() not in protected and bad.search(str(p)):
                remove(p,'Diagnostic/smoke/abandoned output; logs and configuration retained')
            elif p.resolve() not in protected and any(part.startswith('marker_autoencoders_freetacman_available') for part in p.parts):
                remove(p,'Retired FreeTacMan AE training weights; historical pinned teacher export retained')
        # Discard duplicate refit heads after metrics have been recorded. Keep seed45.
        for parent in [Path('processing/runs'),Path('processing/evaluations')]:
            for p in sorted(parent.rglob('*.pt')):
                if p.resolve() in protected:continue
                if any(re.fullmatch(r'(seed[_-]?(46|47)|.*_seed(46|47))',part) for part in p.parts):
                    remove(p,'Redundant readout/AE seed weights; seed45 or pinned winner and all seed metrics retained')
        # Native AE choices: keep a best checkpoint for each actual architecture/code
        # shape, including wide models; eliminate duplicate hyperparameter trials.
        candidates={}
        for p in Path('processing/runs').rglob('best.pt'):
            cfg_path=p.parent/'config.json'
            if not cfg_path.exists() or p.resolve() in protected:continue
            try:
                cfg=json.loads(cfg_path.read_text());model=cfg.get('model',cfg)
                if model.get('architecture') not in ('conv','mlp'):continue
                key=(cfg.get('source','egotouch'),json.dumps({k:model.get(k) for k in ['architecture','latent_dim','channels','hidden_dims','output_mode']},sort_keys=True))
                metric=p.parent/'best_metrics.json';m=json.loads(metric.read_text()) if metric.exists() else {}
                score=m.get('selection_score',m.get('val',{}).get('mae',float('inf')))
                score=float('inf') if score is None else score
                candidates.setdefault(key,[]).append((score,p))
            except (ValueError,KeyError,TypeError):continue
        for key,rows in candidates.items():
            rows.sort(key=lambda r:(r[0],str(r[1])))
            if key[0]!='freetacman':kept.append(dict(path=str(rows[0][1]),reason='Best weight for this source/architecture/code shape; preserves wide models'))
            for _,p in rows if key[0]=='freetacman' else rows[1:]:
                remove(p,'Retired FreeTacMan AE' if key[0]=='freetacman' else 'Inferior duplicate AE architecture/code shape')
            for _,p in rows:
                last=p.with_name('last.pt')
                if last.exists() and last.resolve() not in protected:remove(last,'Completed AE optimizer state; selected inference weights retained')
        # Drop full epoch checkpoints only when their matching inference snapshot
        # demonstrably contains identical weights. Retain each run's latest state.
        for p in sorted(Path('processing/runs').rglob('epoch_*.pt')):
            if 'viewer_epochs' in p.parts or p.resolve() in protected or not p.exists():continue
            q=p.parent/'viewer_epochs'/p.name
            if not q.exists():continue
            try:
                a=torch.load(p,map_location='cpu',weights_only=False,mmap=True)
                b=torch.load(q,map_location='cpu',weights_only=False,mmap=True)
                same=(a.get('epoch')==b.get('epoch') and a.get('model_config')==b.get('model_config')
                    and a.get('config')==b.get('config') and set(a['model'])==set(b['model'])
                    and all(torch.equal(v,b['model'][k]) for k,v in a['model'].items()))
                del a,b;gc.collect()
                if same:remove(p,'Identical model retained in viewer_epochs; redundant per-epoch optimizer/RNG state')
            except (OSError,ValueError,RuntimeError,KeyError):continue
        # Keep the early-epoch transfer sweep, every tenth epoch and each source's
        # best validation epoch. Remove unevaluated middle epochs of the old DINO
        # baseline and abandoned invariance ablations; metric histories remain.
        archive_roots=['joint_alignment_holdout_motiontrans','fgw_two_placements_v1',
                       'fgw_study_v1','joint_16d_pair_v1','invariance_16d_v1','continuation_32d_v1']
        for name in archive_roots:
            root=Path('processing/runs')/name
            for folder in root.rglob('viewer_epochs'):
                files=sorted(folder.glob('epoch_*.pt'))
                if not files:continue
                keep={int(p.stem.split('_')[-1]) for p in files if int(p.stem.split('_')[-1])<=10 or int(p.stem.split('_')[-1])%10==0}
                keep.add(max(int(p.stem.split('_')[-1]) for p in files))
                history=folder.parent/'metrics.jsonl';best={}
                if history.exists():
                    for line in history.read_text().splitlines():
                        try:
                            row=json.loads(line);epoch=row['epoch']
                            for source,value in row.get('validation',{}).items():
                                if isinstance(value,dict):
                                    score=value.get('supervision_loss',value.get('normalized_mae'))
                                    if score is not None and (source not in best or score<best[source][0]):best[source]=(score,epoch)
                        except (ValueError,KeyError,TypeError):continue
                keep.update(v[1] for v in best.values());keep.update({7,15,17,22})
                for p in files:
                    if int(p.stem.split('_')[-1]) not in keep and p.resolve() not in protected:
                        remove(p,'Unevaluated intermediate epoch of superseded experiment; early sweep, tenth epochs, source-best epochs and final weights retained')
    write_json(REPORT,dict(apply=args.apply,records=records,kept=kept,removed_logical_bytes=sum(r['bytes'] for r in records if r['deleted']),updated_at=time.time()))
    print(json.dumps(dict(paths=len(records),bytes=sum(r['bytes'] for r in records),apply=args.apply)),flush=True)


if __name__=='__main__':main()
