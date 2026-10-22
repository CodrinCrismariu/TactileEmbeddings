#!/usr/bin/env python3
"""Fresh teacher/cache parity, all-source validation, and held-out exclusion."""
from tactilelatent.expansion_runtime import configure
configure()

if __name__=='__main__':
    import json
    from pathlib import Path
    import numpy as np
    import torch
    from tactilelatent import train_vjepa_markers as trainer
    from tactilelatent.acquisition_training import install
    from tactilelatent.acquisition_joint import ROOT,LAYOUTS,EXCLUDED,native_rows
    from tactilelatent.acquisition_data import _targets
    from tactilelatent.aligned_frames import write_json,file_hash
    from tactilelatent.marker_data import marker_collate
    from tactilelatent.vjepa_model import VJEPABackbone
    install(trainer);torch.set_num_threads(4);torch.manual_seed(45)
    cfg=json.loads(Path('configs/joint_vjepa21_all_sources_32d_mirror.json').read_text())
    train,val,registry,mc,provenance=trainer.prepare(cfg)
    device=torch.device('cuda');teachers=trainer.MixedMarkerTeachers(registry,device)
    model=trainer.MarkerCodeVJEPA(**mc).to(device).eval()
    backbone=VJEPABackbone(cfg['backbone_repo'],cfg['backbone_checkpoint']).to(device).eval()
    assert not set(train.indices_by_source)&EXCLUDED
    assert set(train.indices_by_source)==set(val.indices_by_source)
    cache_checks=[]
    for source,spec in LAYOUTS.items():
        if spec.get('direct'):continue
        rows,_=native_rows(source)
        info=provenance['acquisitions']['sources'][source]
        index=json.loads(Path(info['index']).read_text());records={r['episode_id']:r for r in index['episodes']}
        for split in ('train','val'):
            row=next(r for r in rows if r['split']==split);native=_targets(row['targets']['path'],row['targets']['sha256'])
            with np.load(records[row['episode_id']]['path'],allow_pickle=False) as saved:
                valid=np.flatnonzero(saved['valid']);ix=valid[[0,-1]]
                cached=saved['codes'][ix].copy()
            y=torch.from_numpy(native['values'][ix].astype(np.float32)).to(device)
            mask=torch.from_numpy(native['valid'][ix]&np.isfinite(native['values'][ix])).to(device)
            with torch.inference_mode():fresh=teachers.models[source].encode(y,mask).float().cpu()
            torch.testing.assert_close(fresh,torch.from_numpy(cached),rtol=2e-5,atol=2e-5)
            cache_checks.append(dict(source=source,split=split,episode=row['episode_id'],frames=ix.tolist(),
                max_absolute_error=float((fresh-torch.from_numpy(cached)).abs().max())))
    batches=[marker_collate([val[i] for i in ids[:2]]) for source,ids in sorted(val.indices_by_source.items())]
    validation=trainer.validate(model,backbone,teachers,batches,device)
    paired=trainer.validate_mirror_pairs(model,backbone,teachers,batches,device,cfg)
    assert set(validation)==set(train.indices_by_source)
    assert set(paired)==set(cfg['mirror_augmentation']['sources'])
    assert all(np.isfinite(v['supervision_loss']) for v in validation.values())
    artifact=dict(passed=True,cache_checks=cache_checks,validation=validation,paired_validation=paired,
        note='Random task weights: integration checks, not model-quality or transfer results.',
        train_windows={s:len(ids) for s,ids in train.indices_by_source.items()},
        validation_windows={s:len(ids) for s,ids in val.indices_by_source.items()},
        heldout_excluded=True,excluded_sources=sorted(EXCLUDED),config=cfg,
        implementation={n:file_hash(Path('src/tactilelatent')/n) for n in ('acquisition_joint.py','acquisition_training.py')})
    write_json(ROOT/'verification.json',artifact)
    print(json.dumps(dict(passed=True,sources=len(validation),mirror_sources=list(paired),cache_checks=cache_checks)),flush=True)
