"""Compare an old eight-output model and a current-frame model on identical inputs.

This measures source validation, not MotionTrans transfer. The frozen backbone
is computed once per batch; only current-frame predictions enter either score.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from tactilelatent.aligned_frames import file_hash, write_json
from tactilelatent.joint_data import SourceBatchSampler
from tactilelatent.marker_code_model import MixedMarkerTeachers,model_from_state
from tactilelatent.marker_data import marker_collate
from tactilelatent.marker_targets import MARKER_SOURCES
from tactilelatent.train_alignment import alignment_loss
from tactilelatent.train_vjepa_markers import prepare,supervision,forward
from tactilelatent.vjepa_model import VJEPABackbone
from tactilelatent.vjepa_spatial import spatial_video_features


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epoch',type=int,default=6)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--extra',action='append',default=[],metavar='NAME=RUN@EPOCH',
                        help='Additional marker-code checkpoint scored on the same clips')
    args=parser.parse_args();root=args.output;root.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);device=torch.device('cuda');started=time.monotonic()
    def report(stage,**details):
        write_json(root/'status.json',dict(stage=stage,seconds=time.monotonic()-started,**details))
    report('loading')
    runs={'old32':Path('processing/runs/joint_vjepa21_spatial_marker_codes_32d_v1'),
          'current128':Path('processing/runs/joint_vjepa21_spatial_current128d_v1')}
    epochs={name:args.epoch for name in runs}
    for extra in args.extra:
        name,location=extra.split('=',1);run,epoch=location.rsplit('@',1)
        if name in runs:raise ValueError('Duplicate model name')
        runs[name]=Path(run);epochs[name]=int(epoch)
    states={};models={};checkpoints={}
    for name,run in runs.items():
        path=run/'viewer_epochs'/f'epoch_{epochs[name]:03d}.pt'
        states[name]=torch.load(path,map_location='cpu',weights_only=False)
        assert states[name]['epoch']+1==epochs[name]
        models[name]=model_from_state(states[name]).to(device).eval().requires_grad_(False)
        checkpoints[name]=dict(path=str(path),sha256=file_hash(path),epoch=epochs[name],latent_dim=states[name]['model_config']['latent_dim'])
    cfg=states['current128']['config']
    _,val,registry,_,provenance=prepare(cfg)
    for source,teacher in registry['sources'].items():
        for state in states.values():assert state['teachers']['sources'][source]['sha256']==teacher['sha256']
    teachers=MixedMarkerTeachers(registry,device)
    backbone=VJEPABackbone(cfg['backbone_repo'],cfg['backbone_checkpoint']).to(device).eval()
    sampler=SourceBatchSampler(val,cfg['batch_size'],seed=cfg['seed'],max_windows_per_source=cfg['validation_windows_per_source'])
    loader=DataLoader(val,batch_sampler=sampler,num_workers=2,pin_memory=True,collate_fn=marker_collate)
    totals=defaultdict(lambda:defaultdict(lambda:defaultdict(float)))
    clips=defaultdict(list)
    with torch.inference_mode():
        for step,original in enumerate(loader):
            source=original['source'][0]
            if source in ('freetacman','ftp_motiontrans'):raise ValueError('Unexpected source')
            batch=supervision(original,teachers,models['current128'],device)
            with torch.autocast('cuda',torch.bfloat16):
                features=spatial_video_features(backbone,original,device)
                outputs={name:forward(model,features,batch) for name,model in models.items()}
            for name,output in outputs.items():
                if output['mu'].shape[1]==8:
                    outputs[name]={k:(v[:,-1:] if torch.is_tensor(v) else v) for k,v in output.items()}
                if outputs[name]['mu'].shape[1]!=1:raise ValueError('Expected a single evaluated current frame')
            for name,output in outputs.items():
                loss,_,_=alignment_loss(output,batch,0.)
                row=totals[source][name];row['loss']+=loss.item();row['batches']+=1
                if source in MARKER_SOURCES:
                    pred=teachers.decode_marker(source,models[name].raw_code(source,output['sensor_code']),batch)
                    target=batch['marker_target'];valid=batch['marker_valid']&torch.isfinite(target).all(-1)
                    error=(pred.float()-torch.nan_to_num(target)).abs()*batch['marker_spacing'][:,None,:,None]
                    row['absolute']+=error[valid].sum().item();row['cells']+=2*valid.sum().item()
                else:
                    pred=(teachers.decode(source,models[name].raw_code(source,output['sensor_code']))
                          if 'sensor_code' in output else output['pred'])
                    valid=batch['target_mask'];error=(pred.float()-batch['target']).abs()
                    row['absolute']+=error[valid].sum().item();row['cells']+=valid.sum().item()
            clips[source].extend(dict(episode_id=e,frame_indices=ix.tolist())
                                 for e,ix in zip(original['episode_id'],original['frame_indices']))
            if step%50==0:report('comparing',batches_done=step+1,batches_total=len(loader))
    results={}
    for source,models_metrics in totals.items():
        results[source]=dict(metric='marker_mae_px' if source in MARKER_SOURCES else 'normalized_mae',clips=len(clips[source]))
        for name,row in models_metrics.items():
            results[source][name]=dict(mae=row['absolute']/row['cells'],supervision=row['loss']/row['batches'])
        results[source]['mae_change_percent']=100*(results[source]['current128']['mae']/results[source]['old32']['mae']-1)
        results[source]['mae_change_vs_old32_percent']={name:100*(results[source][name]['mae']/results[source]['old32']['mae']-1) for name in models}
    result=dict(epoch=args.epoch,checkpoints=checkpoints,data=provenance['data'],results=results,
        mean_supervision={name:sum(row[name]['supervision'] for row in results.values())/len(results) for name in models},
        clips=clips,seconds=time.monotonic()-started,
        caveats=['Both evaluated on the current-frame validation targets and identical compressed RGB.',
                 'Training differs in width, one versus eight supervised outputs, FreeTacMan inclusion and RGB storage.',
                 'Source validation does not establish MotionTrans transfer or a causal benefit of latent width.'])
    write_json(root/'results.json',result)
    report('complete',sources=len(results))
    print(json.dumps({k:v for k,v in result.items() if k not in ('clips','checkpoints','data')}))


if __name__=='__main__':main()
