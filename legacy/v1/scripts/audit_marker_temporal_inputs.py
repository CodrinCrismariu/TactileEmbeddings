"""Temporal-input and shifted-window checks of frozen FreeTacMan prediction."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from tactilelatent.aligned_frames import scene_rgb, write_json
from tactilelatent.alignment_data import AlignedClipDataset
from tactilelatent.marker_autoencoder import load_marker_autoencoder
from tactilelatent.marker_code_model import model_from_state
from tactilelatent.marker_data import MarkerCache, MarkerTargetClips, marker_collate
from tactilelatent.vjepa_model import VJEPABackbone
from tactilelatent.vjepa_spatial import spatial_video_features
from audit_marker_dynamics import ROOT, RUN, sums, summarize


def main():
    torch.set_num_threads(4);device=torch.device('cuda')
    state=torch.load(RUN/'viewer_epochs/epoch_030.pt',map_location='cpu',weights_only=False)
    cfg=state['config'];model=model_from_state(state).to(device).eval().requires_grad_(False)
    registry=json.loads(Path(cfg['marker_teachers']).read_text())
    ae=load_marker_autoencoder(registry['sources']['freetacman']['checkpoint'],device)[0]
    backbone=VJEPABackbone(cfg['backbone_repo'],cfg['backbone_checkpoint']).to(device).eval().requires_grad_(False)
    base=AlignedClipDataset(cfg['manifest'],split='val',sources=['freetacman'],rgb_size=384,include_targets=False)
    eligible=[i for i,(e,s) in enumerate(base.windows) if s+16<base.episodes[e]['frames']]
    ids=[eligible[i] for i in np.linspace(0,len(eligible)-1,64,dtype=int)]
    data=MarkerTargetClips(base,MarkerCache(cfg['marker_targets']))
    loader=DataLoader(Subset(data,ids),batch_size=2,num_workers=2,collate_fn=marker_collate,shuffle=False)
    by_episode={r['episode_id']:r for r in base.episodes}
    records=[];overlap_absolute=overlap_count=normal_temporal2=static_temporal2=matched_dot=normal2=static2=0.
    for batch in loader:
        reference=batch['marker_reference'].to(device);surface=batch['marker_surface'].to(device)
        b,t,n=batch['marker_target'].shape[:3]
        ref=reference[:,None].expand(-1,t,-1,-1).reshape(b*t,n,2)
        sid=surface[:,None].expand(-1,t,-1).reshape(b*t,n)
        predictions={}
        shifted=[]
        for episode,frames in zip(batch['episode_id'],batch['frame_indices']):
            rgb,mask=scene_rgb(by_episode[episode],frames.numpy()+2,size=384)
            shifted.append(rgb)
        inputs={'normal':batch['rgb'],
                'repeated_middle_frame':batch['rgb'][:,4:5].expand_as(batch['rgb']),
                'reversed_input':batch['rgb'].flip(1),
                'shifted_start_by_two':torch.stack(shifted)}
        with torch.no_grad(),torch.autocast('cuda',torch.bfloat16):
            for name,rgb in inputs.items():
                altered={**batch,'rgb':rgb}
                tokens=spatial_video_features(backbone,altered,device,chunk_size=2)
                out=model(tokens,batch['view_mask'].to(device),'freetacman',sample_latent=False)
                raw=model.raw_code('freetacman',out['sensor_code'])
                with torch.autocast('cuda',enabled=False):prediction=ae.decode(raw.float().reshape(b*t,-1),ref,sid).reshape(b,t,n,2)
                predictions[name]=prediction.cpu().numpy()*batch['marker_spacing'].numpy()[:,None,:,None]
        y=batch['marker_target'].numpy()*batch['marker_spacing'].numpy()[:,None,:,None]
        valid=batch['marker_valid'].numpy()&np.isfinite(y).all(-1)
        for i in range(b):
            row=dict(episode=batch['episode_id'][i],frames=batch['frame_indices'][i].tolist(),metrics={})
            for name in ('normal','repeated_middle_frame','reversed_input'):
                p=predictions[name][i]
                if name=='reversed_input':p=p[::-1]
                row['metrics'][name]=sums(p,y[i],valid[i])
            records.append(row)
        common=np.broadcast_to(valid[:,1:,:,None],predictions['normal'][:,1:].shape)
        overlap_absolute+=float(np.abs(predictions['normal'][:,1:]-predictions['shifted_start_by_two'][:,:-1])[common].sum())
        overlap_count+=float(common.sum())
        dm=np.broadcast_to((valid[:,1:]&valid[:,:-1])[...,None],predictions['normal'][:,1:].shape)
        dn=np.diff(predictions['normal'],axis=1)[dm];ds=np.diff(predictions['repeated_middle_frame'],axis=1)[dm]
        normal_temporal2+=float((dn.astype(np.float64)**2).sum());static_temporal2+=float((ds.astype(np.float64)**2).sum())
        matched_dot+=float((dn.astype(np.float64)*ds).sum());normal2+=float((dn.astype(np.float64)**2).sum());static2+=float((ds.astype(np.float64)**2).sum())
    result=dict(clips=len(records),sampling='64 uniformly spaced eligible validation windows; no selection by prediction quality',
        metrics={name:summarize([r['metrics'][name] for r in records]) for name in records[0]['metrics']},
        same_frame_shifted_window_prediction_difference_mae_px=overlap_absolute/overlap_count,
        repeated_frame_predicted_change_rms_relative_to_normal=float(np.sqrt(static_temporal2/normal_temporal2)),
        repeated_frame_vs_normal_predicted_change_cosine=float(matched_dot/np.sqrt(normal2*static2)),
        caution='Repeated/reversed clips are out-of-distribution diagnostics, not new validation protocols; temporal-position and context effects can contribute.')
    write_json(ROOT/'temporal_input_checks.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
