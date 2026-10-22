"""Process-local integration hooks; historical training modules stay pinned."""
from collections import defaultdict

import numpy as np
import torch

from .acquisition_joint import LAYOUTS, AllSourceTeachers, AllSourceVJEPA, prepare


@torch.no_grad()
def validate_numeric(model,backbone,teachers,original,device):
    from .vjepa_spatial import spatial_video_features
    from .train_alignment import alignment_loss
    source=original['source'][0];target=original['target'].to(device).float();valid=original['target_mask'].to(device)
    batch=teachers.supervision(original,model)
    with torch.autocast(device.type,torch.bfloat16,enabled=device.type=='cuda'):
        features=spatial_video_features(backbone,original,device)
        output=model(features,original['view_mask'].to(device),source,sample_latent=False)
    _,loss,_=alignment_loss(output,batch,0.)
    oracle=None
    if source in model.direct_numeric:prediction=model.decode_numeric(source,output['pred'])
    else:
        teacher=teachers.models[source]
        code=model.raw_code(source,output['sensor_code']).flatten(0,1)
        truth=model.raw_code(source,batch['code_target']).flatten(0,1)
        prediction=teacher.decode(code.float()).reshape_as(target)
        oracle=teacher.decode(truth.float()).reshape_as(target)
    if not torch.isfinite(prediction).all():raise ValueError('Nonfinite acquisition prediction')
    absolute=(prediction-target).abs()
    axes=(0,1,*range(3,target.ndim))
    counts=valid.sum(axes);sums=(absolute*valid).sum(axes)
    result=dict(supervision_loss=float(loss),native_mae=float(sums.sum()/counts.sum().clamp_min(1)),
        absolute_sum=float(sums.sum()),valid_cells=int(counts.sum()),per_channel_absolute=sums.cpu().tolist(),
        per_channel_counts=counts.cpu().tolist(),target='native_numeric',frames=len(target))
    if oracle is not None:result['oracle_absolute']=float(((oracle-target).abs()*valid).sum())
    return result


def install(trainer):
    original_prepare=trainer.prepare;original_validate=trainer.validate
    trainer.prepare=lambda cfg,allow_pilot=False:prepare(cfg,original_prepare,trainer,allow_pilot)
    trainer.MixedMarkerTeachers=AllSourceTeachers;trainer.MarkerCodeVJEPA=AllSourceVJEPA
    trainer.MARKER_SOURCES=(*trainer.MARKER_SOURCES,'baihu')

    @torch.no_grad()
    def validate(model,backbone,teachers,loader,device):
        result={};totals=defaultdict(list)
        for batch in loader:
            source=batch['source'][0]
            if source in LAYOUTS or source=='egopressure':totals[source].append(validate_numeric(model,backbone,teachers,batch,device))
            else:totals[source].append(original_validate(model,backbone,teachers,[batch],device)[source])
        for source,rows in totals.items():
            if source not in LAYOUTS and source!='egopressure':
                result[source]={k:sum(r[k] for r in rows)/len(rows) if rows[0][k] is not None and isinstance(rows[0][k],(int,float))
                                else rows[0][k] for k in rows[0]}
                continue
            count=sum(r['valid_cells'] for r in rows);sums=np.sum([r['per_channel_absolute'] for r in rows],axis=0)
            counts=np.sum([r['per_channel_counts'] for r in rows],axis=0)
            r=dict(target='native_numeric',supervision_loss=float(np.mean([r['supervision_loss'] for r in rows])),
                native_mae=sum(r['absolute_sum'] for r in rows)/max(count,1),valid_cells=count,
                per_channel_mae=np.divide(sums,counts,out=np.zeros_like(sums),where=counts>0).tolist())
            if 'oracle_absolute' in rows[0]:r['frozen_ae_mae']=sum(a['oracle_absolute'] for a in rows)/max(count,1)
            if source=='egopressure':r.update(target='pressure_grid',normalized_mae=r['native_mae'])
            if source=='trex':
                for name,ids in [('force',np.arange(60)%6<3),('moment',np.arange(60)%6>=3)]:
                    r[name+'_mae']=float(sums[ids].sum()/max(counts[ids].sum(),1))
            if source=='hrdexdb_inspire_f1':
                for i,name in enumerate(['normal','tangential','direction']):
                    r[name+'_mae']=float(sums[i::3].sum()/max(counts[i::3].sum(),1))
            result[source]=r
        return result

    trainer.validate=validate
    return trainer
