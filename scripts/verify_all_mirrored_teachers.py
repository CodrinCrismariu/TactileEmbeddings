#!/usr/bin/env python3
"""Validate frozen mirror decoders/codes and prepare joint data without training."""
from tactilelatent.expansion_runtime import configure
configure()

if __name__=='__main__':
    import json
    from pathlib import Path
    import torch
    from tactilelatent.aligned_frames import file_hash,write_json
    from tactilelatent.bilateral_mirror_preparation import runtime
    from tactilelatent import mirror_preparation as prep
    from tactilelatent.mirror_cache_verification import verify_codes
    from tactilelatent.mirror_joint import read_preparation
    from tactilelatent.bimanual_mirror import load_policy

    root=Path('processing/bimanual_mirror_v4');ready=json.loads((root/'ready.json').read_text())
    if not ready['complete']:raise ValueError('Wait for all adapted teachers/caches to pass')
    policy=load_policy('configs/bimanual_mirror_policy_v3.json');checks=[]
    # New architectures were trained from fresh initial weights. Preserve
    # reference-teacher metadata separately from their actual selected metrics.
    with runtime():
        for source in ('opengraph','ftp_unit_bimanual'):
            entry=ready['sources'][source];quality_path=Path(entry['teacher_quality'])
            quality=json.loads(quality_path.read_text());teacher=entry['teacher']
            for key in ('validation','completed_epochs','best_epoch'):
                if key in teacher:teacher.setdefault('reference_teacher_'+key,teacher[key])
            teacher.update(validation=quality['validation'],completed_epochs=quality['completed_epochs'],best_epoch=quality['best_epoch'])
            quality['teacher']=teacher;write_json(quality_path,quality)
            index_path=Path(entry['code_index']);index=json.loads(index_path.read_text());index['teacher']=teacher
            write_json(index_path,index);entry['code_index_sha256']=file_hash(index_path)
            rows,_=prep.source_inputs(source)
            verification=verify_codes(rows,index,policy,torch.device('cuda'))
            checks.append(dict(source=source,fresh_code_max_absolute_error=verification['max_absolute_error']))
    write_json(root/'ready.json',ready)
    cfg_path=Path('configs/joint_vjepa21_all_sources_32d_mirror_v2.json');cfg=json.loads(cfg_path.read_text())
    read_preparation(cfg)
    from tactilelatent import train_vjepa_markers as trainer
    from tactilelatent.acquisition_training import install
    install(trainer)
    train,val,registry,mc,provenance=trainer.prepare(cfg)
    teachers=trainer.MixedMarkerTeachers(registry,torch.device('cpu'))
    for source,entry in ready['sources'].items():
        if entry.get('numeric'):
            if source in teachers.models:raise ValueError('Scalar force source acquired an unnecessary AE')
            continue
        if not entry['teacher'].get('mirror_adapted'):raise ValueError('Mirror training selected an original-only teacher')
        quality=json.loads(Path(entry['teacher_quality']).read_text())
        if not quality['accepted'] or not quality['parity_achieved']:raise ValueError('Mirror teacher failed quality/parity')
        if registry['sources'][source]['sha256']!=entry['teacher']['sha256']:raise ValueError('Joint teacher override differs')
        if any(p.requires_grad for p in teachers.models[source].parameters()):raise ValueError('Teacher is not frozen')
    assert len(train.indices_by_source)==17 and 'ftp_motiontrans' not in train.indices_by_source
    assert set(train.indices_by_source)==set(val.indices_by_source)
    result=dict(passed=True,sources={s:dict(numeric=e.get('numeric',False),ready=e['ready'],
        teacher_sha256=e.get('teacher',{}).get('sha256')) for s,e in ready['sources'].items()},
        fresh_code_checks=checks,joint_sources=sorted(train.indices_by_source),joint_training_started=False,
        future_config=str(cfg_path),data_validation_only=True)
    write_json(root/'integration_verification.json',result)
    print(json.dumps(result),flush=True)
