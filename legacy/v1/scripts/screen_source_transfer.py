"""Correlate saved source validation with held-out transfer; no training."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


def correlation(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or x.std() < 1e-12 or y.std() < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='processing/runs/joint_alignment_holdout_motiontrans')
    parser.add_argument('--output', default='processing/continuation_32d_v1/source_screen')
    args = parser.parse_args()
    run, output = Path(args.run), Path(args.output)
    validation = {r['epoch']: r['validation'] for r in map(json.loads, (run / 'metrics.jsonl').read_text().splitlines())}
    transfer = {}
    for folder in ('motiontrans_early_rank_sweep', 'motiontrans_checkpoint_sweep'):
        for row in json.loads((run / 'evaluations' / folder / 'summary.json').read_text())['results']:
            key = row['encoder_epoch'], row['head']
            if key in transfer and transfer[key] != row['deterministic_mae']:
                raise ValueError('Duplicate epoch with inconsistent transfer results')
            transfer[key] = row['deterministic_mae']
    epochs = sorted({epoch for epoch, _ in transfer} & set(validation))
    sources = sorted(validation[epochs[0]])
    if 'ftp_motiontrans' in sources or len(sources) != 11:
        raise ValueError('Expected exactly eleven source datasets')
    results = []
    for source in sources:
        for metric in ('supervision_loss', 'normalized_mae'):
            for head in ('deep', 'shallow'):
                for window, selected in (('all', epochs), ('post7', [e for e in epochs if e >= 7]),
                                         ('late10', [e for e in epochs if e >= 10]),
                                         ('early1_10', [e for e in epochs if e <= 10])):
                    x = np.array([validation[e][source][metric] for e in selected])
                    y = np.array([transfer[e, head] for e in selected])
                    time = np.array(selected, float)
                    design = np.stack((np.ones(len(time)), np.log(time)), axis=1)
                    residual_x = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
                    residual_y = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
                    all_losses = {e: v[source][metric] for e, v in validation.items()}
                    results.append(dict(source=source, metric=metric, head=head, window=window,
                        n=len(selected), pearson=correlation(x, y), spearman=correlation(rankdata(x), rankdata(y)),
                        log_epoch_residual_pearson=correlation(residual_x, residual_y),
                        slope_difference_pearson=correlation(np.diff(x)/np.diff(time), np.diff(y)/np.diff(time)),
                        source_change_percent=100*(x[-1]/x[0]-1), transfer_change_percent=100*(y[-1]/y[0]-1),
                        source_best_epoch=min(all_losses, key=all_losses.get)))
    output.mkdir(parents=True, exist_ok=True)
    (output / 'results.json').write_text(json.dumps(dict(epochs=epochs, results=results,
        interpretation='Both axes are errors. Negative correlation means source improvement accompanies worse transfer; this does not establish causality. Windows, log-time residuals and finite-difference slopes are descriptive sensitivity checks, not independent evidence.',
        selection='MotionTrans validation guides this development experiment; it is not a pristine final test.'), indent=2)+'\n')
    with (output / 'results.csv').open('w') as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0])); writer.writeheader(); writer.writerows(results)
    ranked = sorted([r for r in results if r['metric']=='supervision_loss' and r['head']=='deep' and r['window']=='post7'], key=lambda r:r['pearson'])
    lines = ['# Source validation versus MotionTrans transfer', '',
        'Nineteen unique checkpoints: epochs 1–10 and 20–100 every ten epochs. Main ranking uses the existing two-hidden-layer transfer head and the thirteen checkpoints from epoch 7 onward. Both quantities are errors: a negative correlation indicates improving source loss with worsening transfer MAE.', '',
        '| Source | Pearson, ≥7 | Spearman, ≥7 | Source loss change, 7→100 | Best source epoch | Pearson after log-time adjustment |',
        '|---|---:|---:|---:|---:|---:|']
    for r in ranked:
        lines.append(f"| {r['source']} | {r['pearson']:+.3f} | {r['spearman']:+.3f} | {r['source_change_percent']:+.1f}% | {r['source_best_epoch']} | {r['log_epoch_residual_pearson']:+.3f} |")
    lines += ['', 'These correlations cannot isolate a source’s gradient effect. The checkpoints are dependent, share an encoder, and are confounded with training time. Time adjustment is only a sensitivity check and does not remove nonlinear or unmeasured confounding. No correlation p-values or causal source labels are reported.', '',
        'The CSV/JSON also include native reconstruction MAE, the shallow head, early/late windows, and correlations between changes per epoch. Source supervision and native reconstruction can disagree because the former measures standardized sensor-code prediction.', '',
        'For the intervention, use an online source-validation stopping rule initialized only from epochs 1–7; do not replay the retrospectively best future epochs as stopping decisions. Keep all eleven sources supervised at their adapters. MotionTrans remains absent from representation training and anchor targets, but its validation has explicitly informed experimental selection.']
    (output / 'analysis.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
