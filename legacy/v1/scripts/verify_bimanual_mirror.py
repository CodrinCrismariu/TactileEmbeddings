#!/usr/bin/env python3
"""Disposable real-RGB V-JEPA/SAM updates for prepared bilateral sources.

Writes paired visual audits and diagnostics, never a joint-training checkpoint.
"""
from tactilelatent.expansion_runtime import configure
configure()

import argparse
import copy
import json
from pathlib import Path
import time

import torch

from tactilelatent.aligned_frames import write_json
from tactilelatent import train_vjepa_markers as trainer
from tactilelatent.marker_code_model import MarkerCodeVJEPA, MixedMarkerTeachers
from tactilelatent.marker_data import marker_collate
from tactilelatent.mirror_joint import bind, augment, paired_validate
from tactilelatent.sam import SAMStep, rng_snapshot
from tactilelatent.vjepa_model import VJEPABackbone
from tactilelatent.vjepa_spatial import spatial_video_features


def plot_pair(path,original,mirrored,teachers):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    source=original['source'][0];fig,axes=plt.subplots(2,3,figsize=(12,7))
    with torch.inference_mode():
        for row,(name,item) in enumerate([('Recorded clip',original),('Mirrored clip',mirrored)]):
            available=item['view_mask'][0,-1].nonzero().flatten().tolist();slot=0 if 0 in available else available[0]
            axes[row,0].imshow(item['rgb'][0,-1,slot].permute(1,2,0));axes[row,0].set_title(name+' · current RGB')
            if 'marker_target' in item:
                n=int(item['mirror_marker_count'][0]);reference=item['marker_reference'][0,:n].cpu()
                surface=item['marker_surface'][0,:n].cpu();mask=item['marker_valid'][0,0,:n].cpu()
                target=item['marker_target'][0,0,:n].cpu()
                code=item['code_target'].to(teachers.device);stats=teachers._mirror_statistics[source]
                raw=code*torch.tensor(stats['std'],device=code.device)+torch.tensor(stats['mean'],device=code.device)
                prediction=teachers.decode_marker(source,raw,item)[0,0,:n].cpu()
                for col,(title,vectors) in enumerate([('Measured displacement',target),('Frozen AE reconstruction',prediction)],start=1):
                    x=reference[:,0]+surface*1.2;y=reference[:,1]
                    axes[row,col].scatter(x[mask],y[mask],c=surface[mask],cmap='tab10',s=3)
                    axes[row,col].quiver(x[mask],y[mask],vectors[mask,0],vectors[mask,1],
                        angles='xy',scale_units='xy',scale=20,color='tab:red',width=.004)
                    axes[row,col].invert_yaxis();axes[row,col].set_aspect('equal')
                    axes[row,col].set_title(title+'\nSame arrow gain: 1 spacing unit = 0.05 display units')
                for ax in axes[row]:ax.axis('off')
                continue
            y=item['target'][0,0].cpu();mask=item['target_mask'][0,0].cpu()
            if y.shape[-2:]==(1,1):
                axes[row,1].bar(['Left','Right'],y.flatten());axes[row,1].set_ylim(0,1)
                axes[row,1].set_title('Recorded force channels · native bounded scale')
                axes[row,2].text(.5,.5,'Direct numeric head\nNo tactile autoencoder',ha='center',va='center')
            else:
                native=torch.where(mask,y,torch.nan).numpy()
                code=item['code_target'].to(teachers.device)
                # Pair caches are normalized using the accepted teacher's
                # train-only statistics; this is an AE oracle, not a trained
                # joint prediction.
                stats=teachers._mirror_statistics[source]
                raw=code*torch.tensor(stats['std'],device=code.device)+torch.tensor(stats['mean'],device=code.device)
                pred=teachers.models[source].decode(raw.reshape(-1,raw.shape[-1])).cpu()[0]
                pred=torch.where(mask,pred,torch.nan).numpy()
                axes[row,1].imshow(np.concatenate(native,axis=1),vmin=0,vmax=1,cmap='inferno')
                axes[row,2].imshow(np.concatenate(pred,axis=1),vmin=0,vmax=1,cmap='inferno')
                axes[row,1].set_title('Measured pressure · left | right')
                axes[row,2].set_title('Frozen AE reconstruction · left | right')
            for ax in axes[row]:ax.axis('off')
    fig.suptitle(source+' · same frame IDs, native target resolution');fig.tight_layout()
    fig.savefig(path,dpi=150);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preparation',default='processing/bimanual_mirror_v1')
    p.add_argument('--base-config',default='configs/joint_vjepa21_spatial_current_32d.json')
    args=p.parse_args();root=Path(args.preparation);ready=json.loads((root/'ready.json').read_text())
    cfg=json.loads(Path(args.base_config).read_text())
    sources=[s for s,i in ready['sources'].items() if i.get('ready') and (s!='baihu' or cfg.get('data_expansion'))]
    if not sources:raise RuntimeError('No verified/prepared bilateral source is ready for a GPU update')
    cfg['mirror_augmentation']=dict(enabled=True,probability=.5,sources=sources,
        policy=ready['policy_path'],preparation=str(root))
    cfg['training_entrypoint']='scripts/train_mirrored_vjepa.py'
    family='expanded' if cfg.get('data_expansion') else 'current'
    version=root.name.removeprefix('bimanual_mirror_')
    cfg['output']=f'processing/runs/joint_vjepa21_spatial_{family}32d_mirror_{version}'
    write_json(root/'available_joint_config.json',cfg)
    torch.set_num_threads(4);torch.manual_seed(45);device=torch.device('cuda')
    if cfg.get('data_expansion'):
        from tactilelatent.expansion_joint import prepare as expanded_prepare,ExpandedTeachers
        trainer.MARKER_SOURCES=(*trainer.MARKER_SOURCES,'baihu')
        prepared=expanded_prepare(cfg,trainer.prepare,trainer)
        teacher_class=ExpandedTeachers
    else:
        prepared=trainer.prepare(cfg);teacher_class=MixedMarkerTeachers
    train,val,registry,mc,provenance=bind(cfg,prepared)
    model=MarkerCodeVJEPA(**mc).to(device).train()
    teachers=teacher_class(registry,device)
    teachers._mirror_statistics={s:mc['statistics'][s] for s in sources if s in mc['statistics']}
    backbone=VJEPABackbone(cfg['backbone_repo'],cfg['backbone_checkpoint']).to(device).eval()
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    sam=SAMStep(optimizer,cfg['sam_rho']);records=[];pairs=[];began=time.monotonic()
    folder=root/'verification';folder.mkdir(parents=True,exist_ok=True)
    for source in sources:
        original=marker_collate([train[train.indices_by_source[source][0]]])
        pair=augment(original,cfg,0,0,flags=[True])
        model.train();losses=[]
        for current in (original,pair):
            batch=trainer.supervision(current,teachers,model,device)
            with torch.no_grad(),torch.autocast('cuda',torch.bfloat16):features=spatial_video_features(backbone,current,device)
            before=rng_snapshot()
            with torch.autocast('cuda',torch.bfloat16):
                out=trainer.forward(model,features,batch);loss,_,_=trainer.task_loss(model,out,batch)
            if not torch.isfinite(loss) or out['mu'].shape!=(1,1,32):raise ValueError('Invalid mirrored current-frame training update')
            loss.backward();trainer.replay_update(model,optimizer,sam,[dict(features=features,batch=batch,rng=before)],device)
            losses.append(float(loss.detach()))
        plot_pair(folder/(source+'.png'),original,pair,teachers)
        pairs.append(marker_collate([val[val.indices_by_source[source][0]]]))
        records.append(dict(source=source,loss_original=losses[0],loss_mirrored=losses[1],
            source_frame_indices=original['sensor_frame_indices'].tolist(),sam_updates=2,
            native_target_shape=list(original['target'].shape[2:]) if 'target' in original else 'native_marker_set'))
        print(json.dumps(records[-1]),flush=True)
    if any(p.grad is not None for p in backbone.parameters()) or any(p.grad is not None for t in teachers.models.values() for p in t.parameters()):
        raise ValueError('A frozen teacher/backbone received gradients')
    diagnostics=paired_validate(model,backbone,teachers,pairs,device,cfg)
    write_json(folder/'report.json',dict(passed=True,sources=records,paired_validation=diagnostics,
        blocked={s:i for s,i in ready['sources'].items() if not i.get('ready')},
        seconds=time.monotonic()-began,production_joint_run_started=False,provenance=provenance))


if __name__=='__main__':main()
