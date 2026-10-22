#!/usr/bin/env python3
"""Exclude calibration episodes, reuse codes, and evaluate task-only validation."""
import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import time

import torch
import numpy as np

from tactilelatent.aligned_frames import file_hash, write_json
from tactilelatent.expansion_models import load_teacher
from tactilelatent.pressure_selection import VERSION, MANIFEST, task_episodes, build_task_code_index


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('processing/data_expansion_v1'))
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    root = args.root
    folder = root / 'egopressure_task_selection_v1'
    folder.mkdir(parents=True, exist_ok=True)

    def progress(**fields):
        state = dict(updated_at=time.time(), **fields)
        write_json(folder / 'status.json', state)
        print(json.dumps(state), flush=True)

    source = root / 'egopressure_manifest.json'
    document = json.loads(source.read_text())
    rows = sorted(document['episodes'], key=lambda r: (r['split'], r['episode_id']))
    selected = task_episodes(rows)
    excluded_ids = {r['episode_id'] for r in rows} - {r['episode_id'] for r in selected}
    excluded = [r for r in rows if r['episode_id'] in excluded_ids]
    selection = dict(episodes=selected, complete=document['complete'], revision=document['revision'],
        selection_version=VERSION, acquisition_manifest_sha256=file_hash(source),
        excluded_episodes=[dict(episode_id=r['episode_id'], split=r['split'], frames=r['frames'])
                           for r in excluded],
        split_episode_counts=dict(Counter(r['split'] for r in selected)),
        frames=sum(r['frames'] for r in selected))
    write_json(root / MANIFEST, selection)
    checkpoint = root / 'teachers/egopressure_balanced_v2/best.pt'
    parent = root / 'codes/egopressure_balanced_v2/index.json'
    destination = root / 'codes/egopressure_task_selection_v1'
    index = build_task_code_index(rows, parent, checkpoint, destination, args.device, progress)
    parent_index=json.loads(parent.read_text())
    parent_records={r['episode_id']:r for r in parent_index['episodes']}
    producer_sampling=json.loads(checkpoint.with_name('sampling_index.json').read_text())['episodes']
    support=[]
    for row in excluded:
        sample=producer_sampling[row['episode_id']]
        with np.load(parent_records[row['episode_id']]['path'],allow_pickle=False) as codes:
            n=int(codes['valid'].sum())
        support.append(dict(episode_id=row['episode_id'],split=row['split'],
            sampleable_frames=len(sample['valid']),supported_cached_frames=n))
    support_audit=dict(episodes=support,total_sampleable_frames=sum(r['sampleable_frames'] for r in support),
        total_supported_cached_frames=sum(r['supported_cached_frames'] for r in support))
    write_json(folder/'calibration_support_audit.json',support_audit)
    index['historical_calibration_support']=dict(
        sampleable_frames=support_audit['total_sampleable_frames'],
        supported_cached_frames=support_audit['total_supported_cached_frames'],
        audit_sha256=file_hash(folder/'calibration_support_audit.json'))
    write_json(destination/'index.json',index)
    progress(stage='task_only_validation', validation_episodes=selection['split_episode_counts']['val'])
    # Use the existing balanced teacher's ordinary fixed validation protocol.
    spec = importlib.util.spec_from_file_location('pressure_teacher_runner', 'scripts/train_egopressure_teacher.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    torch.set_num_threads(4)
    sample_index = runner.sampling_index(selected, folder)
    model, _ = load_teacher(checkpoint, args.device)
    validation = runner.evaluate(model, selected, sample_index, args.device)
    accepted = (validation['contact_to_zero_ratio'] < .9 and
                validation['all_mae'] < validation['all_zero_baseline_mae'])
    quality = dict(accepted=accepted, teacher_sha256=file_hash(checkpoint), metrics=validation,
        selection_version=VERSION, validation_episodes=selection['split_episode_counts']['val'],
        teacher_retrained=False, historical_calibration_support=index['historical_calibration_support'],
        reason='task-only validation must beat zero globally and on contact')
    write_json(folder / 'quality.json', quality)
    if not accepted:
        progress(stage='blocked_quality', reason=quality['reason'])
        raise SystemExit(1)
    path = root / 'source_teacher_overrides.json'
    overrides = json.loads(path.read_text()) if path.exists() else {}
    if path.exists() and not (folder / 'source_teacher_overrides_before.json').exists():
        write_json(folder / 'source_teacher_overrides_before.json', overrides)
    if overrides.get('egopressure', {}).get('teacher_sha256', quality['teacher_sha256']) != quality['teacher_sha256']:
        raise ValueError('A newer teacher is active; do not replace it with a historical teacher')
    overrides['egopressure'] = dict(checkpoint=str(checkpoint), teacher_sha256=quality['teacher_sha256'],
        code_index=str(destination / 'index.json'), code_index_sha256=file_hash(destination / 'index.json'),
        quality=str(folder / 'quality.json'), episode_selection=VERSION)
    write_json(path, overrides)
    progress(stage='complete', selected_episodes=len(selected), excluded_episodes=len(excluded),
        split_episode_counts=selection['split_episode_counts'], frames=index['frames'],
        cached_code_statistics='refitted using task-training episodes only', teacher_retrained=False,
        validation=validation)


if __name__ == '__main__':
    main()
