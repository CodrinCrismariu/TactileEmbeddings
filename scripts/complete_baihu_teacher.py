#!/usr/bin/env python3
"""Finish the authorized native four-surface AE, cache codes, then publish.

This does not enable an unverified mirror mapping or start joint training.
The failed/pending historical teacher directory is preserved.
"""
import fcntl
import json
from pathlib import Path
import time

import torch

from tactilelatent.aligned_frames import file_hash, write_json
from tactilelatent.data_expansion import ROOT, marker_batch, report
from tactilelatent.expansion_models import ExpansionMarkerAutoencoder
from tactilelatent.expansion_training import source_rows, train_teacher, build_codes, publish


def main():
    folder = ROOT / 'teachers' / 'baihu_native_v2'
    folder.mkdir(parents=True, exist_ok=True)
    lock = (folder / 'run.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.set_num_threads(4)
    rows = source_rows('baihu')
    status_path = folder / 'status.json'

    def status(stage, **fields):
        record = dict(stage=stage, source='baihu', updated_at=time.time(),
                      epochs=30, seed=45, code_dim=32, surfaces=4, **fields)
        write_json(status_path, record)
        print(json.dumps(record), flush=True)
        report('teachers', stage=stage, source='baihu', epochs=30, seed=45,
               code_dim=32, surfaces=4, **fields)

    # Real native batches from each platform, including their actual masks and
    # reference geometry, must make it through a complete optimization update.
    torch.manual_seed(45)
    probe = ExpansionMarkerAutoencoder(displacement_scale=.06337994120925243).cuda()
    optimizer = torch.optim.AdamW(probe.parameters(), lr=.001)
    checks = []
    for platform in sorted({r['platform'] for r in rows}):
        row = next(r for r in rows if r['platform'] == platform)
        batch = marker_batch(row, [0, row['frames'] // 2, row['frames'] - 1])
        y, valid = batch['marker_target'].cuda(), batch['marker_valid'].cuda()
        ref = batch['marker_reference'][None].expand(len(y), -1, -1).cuda()
        surface = batch['marker_surface'][None].expand(len(y), -1).cuda()
        optimizer.zero_grad(set_to_none=True)
        prediction = probe(y, valid, ref, surface)
        loss = probe.loss(prediction, y, valid, surface)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite Baihu teacher update')
        loss.backward()
        optimizer.step()
        checks.append(dict(platform=platform, episode=row['episode_id'],
                           points=y.shape[1], loss=float(loss.detach())))
    del probe, optimizer
    torch.cuda.empty_cache()
    write_json(folder / 'preflight.json', dict(passed=True, checks=checks))
    status('training', preflight_passed=True)
    checkpoint = train_teacher('baihu', rows, folder=folder)
    best = json.loads((folder / 'best_metrics.json').read_text())
    val = best['val']
    accepted = val['mae'] < .9 * val['zero_baseline_mae']
    quality = dict(accepted=accepted, teacher_sha256=file_hash(checkpoint),
                   validation=val, tests=dict(beats_zero_by_10_percent=accepted),
                   mirrored_mapping_verified=False)
    write_json(folder / 'quality.json', quality)
    if not accepted:
        status('teacher_rejected', quality=quality)
        return  # Completed quality rejection; no cache, promotion, or new search.
    status('caching', best_epoch=best['epoch'], validation=val)
    destination = ROOT / 'codes' / 'baihu_native_v2'
    build_codes('baihu', rows, checkpoint, destination=destination)
    # Preserve the active pressure selection and every historical teacher/cache.
    path = ROOT / 'source_teacher_overrides.json'
    overrides = json.loads(path.read_text()) if path.exists() else {}
    overrides['baihu'] = dict(checkpoint=str(checkpoint), teacher_sha256=file_hash(checkpoint),
        code_index=str(destination / 'index.json'), code_index_sha256=file_hash(destination / 'index.json'),
        quality=str(folder / 'quality.json'))
    write_json(path, overrides)
    publish()
    status('complete', best_epoch=best['epoch'], validation=val,
           joint_training_started=False, mirroring_enabled=False)


if __name__ == '__main__':
    main()
