#!/usr/bin/env python3
"""Repair the collapsed Baihu teacher; preserve old checkpoints and targets."""
import argparse
import fcntl
import json
from pathlib import Path
import time

import numpy as np
import torch

from tactilelatent.aligned_frames import file_hash, json_hash, write_json
from tactilelatent.baihu_ae_data import BaihuAEFrames, evaluate_baihu
from tactilelatent.baihu_autoencoder import BaihuMarkerAutoencoder
from tactilelatent.data_expansion import ROOT
from tactilelatent.expansion_training import source_rows, build_codes, publish
from tactilelatent.train_alignment import atomic_checkpoint


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'teachers/baihu_native_v3')
    parser.add_argument('--prepare-codes',action='store_true')
    args=parser.parse_args();folder=args.output;folder.mkdir(parents=True,exist_ok=True)
    lock=(folder/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(4);torch.manual_seed(45);rows=source_rows('baihu');started=time.monotonic()
    def status(stage,**fields):
        record=dict(fields,stage=stage,source='baihu',training_version='native_v3',seed=45,epochs=30,
                    updated_at=time.time(),elapsed_seconds=time.monotonic()-started)
        write_json(folder/'status.json',record)
        with (folder/'progress.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
    status('loading_native_frames')
    train=BaihuAEFrames(rows,'train',100000);val=BaihuAEFrames(rows,'val',10000)
    if {r['episode_id'] for r in train.records}&{r['episode_id'] for r in val.records}:
        raise ValueError('Baihu teacher episode leakage')
    metadata={r['episode_id']:json.loads(Path(r['marker_path']).with_name('complete.json').read_text()) for r in rows}
    stats=[metadata[r['episode_id']]['loss_statistics'] for r in rows if r['split']=='train']
    scale=max((sum(s['square_sum'] for s in stats)/sum(s['coordinate_count'] for s in stats))**.5,1e-6)
    model=BaihuMarkerAutoencoder(latent_dim=32,width=128,queries=8,heads=4,displacement_scale=scale)
    config=dict(source='baihu',seed=45,epochs=30,model=model.config,representation='marker_displacement',
        training_version='native_v3',batch_size=128,learning_rate=.001,min_learning_rate=1e-6,
        weight_decay=.0001,frame_stride=2,max_train_frames=100000,max_val_frames=10000,
        rows_sha256=json_hash(rows),target_snapshot_sha256=json_hash({r['episode_id']:file_hash(r['marker_path']) for r in rows}),
        manifest=str(ROOT/'manifest.json'),train_sampling=train.signature,val_sampling=val.signature,
        implementation={str(p):file_hash(p) for p in [Path(__file__),
            *[Path('src/tactilelatent')/n for n in ('baihu_ae_data.py','baihu_autoencoder.py',
                'marker_autoencoder.py','marker_model.py','expansion_models.py','data_expansion.py')]]})
    if (folder/'config.json').exists() and json.loads((folder/'config.json').read_text())!=config:
        raise ValueError('Repair provenance changed; preserve this run and use a fresh folder')
    write_json(folder/'config.json',config)
    write_json(folder/'sampling.json',dict(train=train.records,val=val.records))
    model=model.cuda();train.to('cuda');val.to('cuda')
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=30,eta_min=1e-6)
    start=0;best=float('inf')
    if (folder/'last.pt').exists():
        old=torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
        if old['config']!=config:raise ValueError('Repair resume provenance differs')
        model.load_state_dict(old['model'],strict=True);optimizer.load_state_dict(old['optimizer'])
        scheduler.load_state_dict(old['scheduler']);start=old['epoch'];best=old['best_score']
        torch.set_rng_state(old['torch_rng']);torch.cuda.set_rng_state_all(old['cuda_rng'])
        history=folder/'metrics.jsonl'
        if history.exists():history.write_text(''.join(l+'\n' for l in history.read_text().splitlines() if json.loads(l)['epoch']<=start))
    else:
        atomic_checkpoint(folder/'epoch_000.pt',dict(model_type='expansion_marker_autoencoder_v2',
            model_config=model.config,model=model.state_dict(),config=config,epoch=0))
    probe=val.batch(np.arange(min(128,len(val))))
    for epoch in range(start,30):
        began=time.monotonic();model.train();order=np.random.default_rng(45+epoch).permutation(len(train));total=0.
        for offset in range(0,len(order),128):
            ix=order[offset:offset+128];b=train.batch(ix);optimizer.zero_grad(set_to_none=True)
            prediction=model(b['target'],b['valid'],b['reference'],b['surface'])
            loss=model.loss(prediction,b['target'],b['valid'],b['surface'])
            if not torch.isfinite(loss):raise ValueError('Nonfinite Baihu repair loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            total+=loss.item()*len(ix)
        metrics=evaluate_baihu(model,val);scheduler.step();improved=metrics['mae']<best;best=min(best,metrics['mae'])
        hidden=[]
        def capture(module,args,out):hidden.append(float((out==0).float().mean()))
        handle=model.to_code[3].register_forward_hook(capture)
        with torch.no_grad():model(probe['target'],probe['valid'],probe['reference'],probe['surface'])
        handle.remove()
        record=dict(epoch=epoch+1,source='baihu',train_loss=total/len(train),val=metrics,
            selection_score=metrics['mae'],hidden_zero_fraction=hidden[0],seconds=time.monotonic()-began)
        state=dict(model_type='expansion_marker_autoencoder_v2',model_config=model.config,model=model.state_dict(),
            config=config,epoch=epoch+1,best_score=best,optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
            validation=metrics,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
        if improved:atomic_checkpoint(folder/'best.pt',state);write_json(folder/'best_metrics.json',record)
        atomic_checkpoint(folder/'last.pt',state)
        with (folder/'metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        status('training',**record)
    checkpoint=folder/'best.pt';best_record=json.loads((folder/'best_metrics.json').read_text());metrics=best_record['val']
    tests=dict(beats_zero_by_10_percent=metrics['mae']<.9*metrics['zero_baseline_mae'],
        active_beats_zero_by_10_percent=metrics['active_mae']<.9*metrics['active_zero_baseline_mae'],
        responds_to_input=metrics['zero_input_output_change_px']>1e-5,
        changes_within_episodes=metrics['within_episode_code_std']>1e-5)
    quality=dict(accepted=all(tests.values()),teacher_sha256=file_hash(checkpoint),validation=metrics,
                 tests=tests,best_epoch=best_record['epoch'],completed_epochs=30)
    write_json(folder/'quality.json',quality)
    if not quality['accepted']:status('teacher_rejected',quality=quality);return
    status('teacher_accepted',quality=quality)
    if args.prepare_codes:
        # Release training pools before all-frame frozen code extraction.
        del model,optimizer,train,val,probe;torch.cuda.empty_cache()
        destination=ROOT/'codes/baihu_native_v3';status('caching')
        build_codes('baihu',rows,checkpoint,destination=destination)
        overrides_path=ROOT/'source_teacher_overrides.json'
        overrides=json.loads(overrides_path.read_text()) if overrides_path.exists() else {}
        overrides['baihu']=dict(checkpoint=str(checkpoint),teacher_sha256=file_hash(checkpoint),
            code_index=str(destination/'index.json'),code_index_sha256=file_hash(destination/'index.json'),
            quality=str(folder/'quality.json'))
        write_json(overrides_path,overrides);publish()
    status('complete',quality=quality,codes_prepared=args.prepare_codes,joint_training_started=False)


if __name__=='__main__':main()
