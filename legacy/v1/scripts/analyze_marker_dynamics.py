"""Episode bootstraps and long-trajectory summaries of the frozen audit."""
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from tactilelatent.aligned_frames import write_json
from audit_marker_dynamics import summarize

ROOT=Path('processing/research/marker_dynamics_v1')


def main():
    rows=json.loads((ROOT/'clip_metrics.json').read_text())
    trace=json.loads((ROOT/'freetacman_traces.json').read_text())
    by_episode=defaultdict(list)
    for row in trace:by_episode[row['episode']].append(row)
    envelopes={}
    for name in ('teacher','mean_code','epoch10','epoch20','epoch30'):
        targets=[];predictions=[];episode_correlations=[]
        for ep,clips in by_episode.items():
            clips=sorted(clips,key=lambda r:r['frames'][0])
            y=np.concatenate([r['target'] for r in clips]);p=np.concatenate([r[name] for r in clips])
            y=y-y.mean();p=p-p.mean();targets.append(y);predictions.append(p)
            if y.std()>1e-8 and p.std()>1e-8:episode_correlations.append(float(np.corrcoef(y,p)[0,1]))
        y=np.concatenate(targets);p=np.concatenate(predictions)
        envelopes[name]=dict(episodes=len(by_episode),
            within_episode_correlation=float((y*p).sum()/np.sqrt((y*y).sum()*(p*p).sum())) if (p*p).sum()>1e-15 else None,
            temporal_amplitude_ratio=float(np.sqrt((p*p).sum()/(y*y).sum())),
            temporal_variance_skill=float(1-((p-y)**2).sum()/(y*y).sum()),
            median_episode_correlation=float(np.median(episode_correlations)) if episode_correlations else None,
            note='Envelope = mean absolute displacement of tracked marker coordinates, not pressure/contact force.')
    free=[r for r in rows if r['source']=='freetacman' and r['split']=='val']
    eps=sorted({r['episode'] for r in free});draws=np.random.default_rng(45).integers(len(eps),size=(5000,len(eps)))
    bootstraps={}
    for name in ('teacher','epoch10','epoch20','epoch30'):
        grouped=defaultdict(lambda:defaultdict(float))
        for r in free:
            for k,v in r['metrics'][name].items():grouped[r['episode']][k]+=v
        arrays={k:np.array([grouped[e][k] for e in eps]) for k in grouped[eps[0]]}
        def draw_sum(key):return arrays[key][draws].sum(1)
        difference=(draw_sum('delta_absolute')-draw_sum('delta_baseline_absolute'))/draw_sum('delta_count')
        skill=1-draw_sum('delta_error2')/draw_sum('delta_target2')
        bootstraps[name]=dict(episodes=len(eps),replicates=5000,
            change_mae_minus_no_change_ci95=np.quantile(difference,[.025,.975]).tolist(),
            change_variance_skill_ci95=np.quantile(skill,[.025,.975]).tolist())
    tasks={}
    coverage=json.loads((ROOT/'task_coverage.json').read_text())
    excluded=set(coverage['teacher_missing_by_task'])
    for task in sorted({r['episode'].split('/')[0] for r in rows if r['source']=='freetacman' and r['split']=='train'}):
        rr=[r for r in rows if r['source']=='freetacman' and r['split']=='train' and r['episode'].startswith(task+'/')]
        tasks[task]=dict(partly_or_entirely_missing_from_teacher_training=task in excluded,
            teacher=summarize([r['metrics']['teacher'] for r in rr]),epoch30=summarize([r['metrics']['epoch30'] for r in rr]))
    write_json(ROOT/'extended_summary.json',dict(whole_episode_envelopes=envelopes,episode_bootstraps=bootstraps,training_tasks=tasks))
    lines=['', '## Whole-episode marker-motion envelopes', '',
           'Each curve averages absolute marker displacement across the sensor, then removes its episode mean.',
           'This tests slow interaction variation across independent clips; it does not establish calibrated force accuracy.', '',
           '| Predictor | Correlation | Temporal amplitude ratio | Temporal variance skill |', '|---|---:|---:|---:|']
    fmt=lambda x:'undefined' if x is None else f'{x:.4f}'
    for name,r in envelopes.items():lines.append('| '+' | '.join([name]+[fmt(r[k]) for k in ['within_episode_correlation','temporal_amplitude_ratio','temporal_variance_skill']])+' |')
    lines.extend(['', '## Episode-bootstrap uncertainty for FreeTacMan frame changes', '',
                  'Paired resampling of 186 validation episodes; 5,000 replicates. Positive MAE difference means worse than no change.', '',
                  '| Predictor | Change MAE minus no-change MAE, 95% CI | Dynamics variance skill, 95% CI |', '|---|---|---|'])
    for name,r in bootstraps.items():lines.append(f"| {name} | {r['change_mae_minus_no_change_ci95']} | {r['change_variance_skill_ci95']} |")
    lines.extend(['', '## Teacher training coverage', '',
                  'FreeTacMan teacher training used 5,026 of the 5,439 joint-training episodes. Missing episodes by task:', '',
                  ', '.join(f'{k}: {v}' for k,v in coverage['teacher_missing_by_task'].items())+'.',
                  'Both teacher selection and joint source validation use SprinkleSesame; this is a held-out task, not an untouched final test.', '',
                  '## Task-balanced FreeTacMan training sample', '',
                  '| Task | Teacher MAE px | Visual epoch-30 MAE px | Visual change skill | Visual change correlation |', '|---|---:|---:|---:|---:|'])
    for task,r in tasks.items():
        p=r['epoch30'];lines.append('| '+' | '.join([task,fmt(r['teacher']['mae_px']),fmt(p['mae_px']),fmt(p['delta_variance_skill']),fmt(p['delta_correlation'])])+' |')
    p=ROOT/'REPORT.md';text=p.read_text();heading='\n## Whole-episode marker-motion envelopes'
    if heading in text:text=text.split(heading)[0]
    p.write_text(text+'\n'.join(lines)+'\n')
    print(json.dumps(dict(whole_episode_envelopes=envelopes,episode_bootstraps=bootstraps),indent=2))


if __name__=='__main__':main()
