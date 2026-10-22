"""Compare matched final-epoch frozen MotionTrans decoder evaluations."""
import json
from pathlib import Path


def main():
    root = Path('processing/runs')
    sam_dir = root / 'joint_all_sam_holdout_motiontrans_fresh_8k_50ep/evaluations/motiontrans_final'
    reference_dir = root / 'joint_all_holdout_motiontrans_fresh_8k_50ep/evaluations/motiontrans'
    a = json.loads((sam_dir / 'results.json').read_text())
    b = json.loads((reference_dir / 'results.json').read_text())
    for key in ('checkpoint_epoch', 'manifest_sha256', 'source', 'head', 'seed',
                'fit_clips', 'tune_clips', 'train_clips', 'target_shape'):
        if a[key] != b[key]:
            raise ValueError(f'Unmatched protocol: {key}')
    if a['checkpoint_epoch'] != 49 or a['source'] != 'ftp_motiontrans':
        raise ValueError('Expected final epoch-50 MotionTrans evaluations')
    if a['validation']['sample_count'] != b['validation']['sample_count'] or a['validation']['validation_clips'] != b['validation']['validation_clips']:
        raise ValueError('Evaluation sample/clip counts differ')
    metrics = {}
    for name in ('deterministic_mae', 'ensemble_mean_mae', 'deterministic_mse',
                 'ensemble_mean_mse', 'best_of_samples_clip_mae'):
        x, y = a['validation'][name], b['validation'][name]
        metrics[name] = {'non_sam': y, 'sam': x, 'sam_relative_change_percent': 100 * (x / y - 1)}
    for name in ('crps', 'mean_predictive_std', 'std_error_pearson'):
        x, y = a['validation']['regions']['all'][name], b['validation']['regions']['all'][name]
        metrics[name] = {'non_sam': y, 'sam': x}
    intervals = {level: {'non_sam': b['validation']['regions']['all']['intervals'][level],
                         'sam': a['validation']['regions']['all']['intervals'][level]}
                 for level in ('0.5', '0.8', '0.9', '0.95')}
    result = {'protocol': 'Final epoch-50, frozen encoder, original 32-512-6 decoder, identical episode partitions and seeds; 64 latent draws.',
              'caveat': 'One run per optimizer; no statistical significance claim. Distribution coverage is decoder empirical calibration, not weight uncertainty.',
              'sam_results': str(sam_dir / 'results.json'),
              'non_sam_results': str(reference_dir / 'results.json'),
              'sam_checkpoint_sha256': a['checkpoint_sha256'],
              'non_sam_checkpoint_sha256': b['checkpoint_sha256'],
              'manifest_sha256': a['manifest_sha256'],
              'validation_clips': a['validation']['validation_clips'],
              'samples': a['validation']['sample_count'],
              'selected_head_epochs': {'non_sam': b['selected_epochs'], 'sam': a['selected_epochs']},
              'metrics': metrics, 'intervals': intervals}
    output = sam_dir / 'non_sam_comparison.json'
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
