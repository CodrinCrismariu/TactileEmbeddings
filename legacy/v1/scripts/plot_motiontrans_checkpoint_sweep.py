"""Plot the saved sweep report using the system matplotlib installation."""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    path = Path(sys.argv[1])
    summary = json.loads(path.read_text())
    figure, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for name, widths in summary['heads'].items():
        rows = sorted((row for row in summary['results'] if row['head'] == name),
                      key=lambda row: row['encoder_epoch'])
        if not rows:
            continue
        epochs = [row['encoder_epoch'] for row in rows]
        label = rows[0]['architecture'].replace('-', ' → ')
        for axis, metric in zip(axes, ('deterministic_mae', 'coverage_90', 'crps')):
            axis.plot(epochs, [row[metric] for row in rows], 'o-',
                      label=label)
    axes[0].axhline(summary['results'][0]['baseline_mae'], color='gray',
                   linestyle='--', label='Training-mean baseline')
    axes[1].axhline(.9, color='gray', linestyle='--', label='Nominal 90%')
    axes[1].set_ylim(0, 1)
    for axis, label in zip(axes, ('Normalized MAE ↓', '90% interval coverage', 'CRPS ↓')):
        axis.set(xlabel='Frozen encoder epoch', ylabel=label)
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    figure.suptitle('Held-out MotionTrans transfer: accuracy and uncertainty')
    figure.savefig(path.parent / 'checkpoint_comparison.png', dpi=180)
    figure.savefig(path.parent / 'checkpoint_comparison.pdf')
    plt.close(figure)
    if any(row.get('effective_rank_mu') is not None for row in summary['results']):
        figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        for name, widths in summary['heads'].items():
            rows = sorted((row for row in summary['results'] if row['head'] == name
                           and row.get('effective_rank_mu') is not None), key=lambda row: row['encoder_epoch'])
            if not rows:
                continue
            label = rows[0]['architecture'].replace('-', ' → ')
            axes[0].plot([row['encoder_epoch'] for row in rows],
                         [row['deterministic_mae'] for row in rows], 'o-', label=label)
            axes[1].plot([row['effective_rank_mu'] for row in rows],
                         [row['deterministic_mae'] for row in rows], 'o-', label=label)
            for row in rows:
                axes[1].annotate(str(row['encoder_epoch']),
                    (row['effective_rank_mu'], row['deterministic_mae']),
                    xytext=(4, 4), textcoords='offset points', fontsize=8)
        axes[0].set(xlabel='Frozen encoder epoch', ylabel='Normalized MotionTrans MAE ↓')
        axes[1].set(xlabel='Logged effective rank of latent means', ylabel='Normalized MotionTrans MAE ↓')
        for axis in axes:
            axis.grid(alpha=.2)
            axis.legend(fontsize=8)
        figure.suptitle('MotionTrans transfer versus epoch and effective rank')
        figure.savefig(path.parent / 'rank_vs_transfer.png', dpi=180)
        figure.savefig(path.parent / 'rank_vs_transfer.pdf')
        plt.close(figure)


if __name__ == '__main__':
    main()
