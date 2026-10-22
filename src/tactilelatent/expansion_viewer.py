"""Add verified expansion data to the existing viewer without changing old IDs."""
from collections import OrderedDict
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .autoencoder_viewer import Viewer
from .data_expansion import ROOT,marker_arrays,marker_batch,pressure_targets,denormalize_pressure
from .expansion_models import load_teacher
from .expansion_render import LABELS,marker_payload
from .media import video_frames
from .pressure_selection import read_task_manifest


class ExpandedViewer(Viewer):
    def __init__(self,*args,**kwargs):
        self.expansion_rows=[];self.expansion_ids={};self.expansion_networks=OrderedDict()
        super().__init__(*args,**kwargs)

    def _scan_models(self):
        models=super()._scan_models()
        folders=[]
        for source in ('baihu','egopressure'):
            folders.append((source,ROOT/'teachers'/source))
            folders.extend((source,p) for p in sorted((ROOT/'teachers').glob(source+'_balanced_*')))
        for source,folder in folders:
            path=folder/'best.pt'
            if not path.exists() or not (folder/'best_metrics.json').exists():continue
            try:
                config=json.loads((folder/'config.json').read_text());metrics=json.loads((folder/'best_metrics.json').read_text())
            except (FileNotFoundError,json.JSONDecodeError):continue
            completed=metrics['epoch']
            if (folder/'last.pt').exists():
                history=folder/'metrics.jsonl'
                if history.exists():
                    for line in reversed(history.read_text().splitlines()):
                        try:completed=json.loads(line)['epoch'];break
                        except (ValueError,KeyError):continue
            repaired=config.get('training_version')=='balanced_v2'
            label=' · balanced v2' if repaired else ''
            rejected=ROOT/'quality'/f'{source}_rejected_teachers.json'
            if rejected.exists() and folder.name==source:label+=' · zero-collapse (legacy)'
            preferred=repaired and metrics['val'].get('contact_to_zero_ratio',1)<.9 and metrics['val']['mae']<metrics['val']['zero_baseline_mae']
            models.append(dict(path=str(path),name=f'{source} AE · {config["model"]["latent_dim"]}D'+label,
                source=source,config=config['model'],score=metrics['selection_score'],best_epoch=metrics['epoch'],
                completed_epochs=completed,validation_mae=metrics['val']['mae'],reduced_resolution=False,
                representation=config['representation'],expansion_teacher=True,preferred=preferred,
                contact_mae=metrics['val'].get('contact_mae'),background_mae=metrics['val'].get('background_mae')))
        return models

    def refresh_expansion(self):
        rows=[]
        p=ROOT/'baihu_available.json'
        if p.exists():rows.extend(json.loads(p.read_text())['episodes'])
        p=ROOT/'egopressure_manifest.json'
        if p.exists() and (ROOT/'pressure_normalization.json').exists():rows.extend(read_task_manifest(ROOT)['episodes'])
        for row in rows:
            key=(row['source'],row['episode_id'])
            if key not in self.expansion_ids:
                self.expansion_ids[key]=len(self.expansion_rows);self.expansion_rows.append(row)

    def catalog(self):
        with self.lock:
            self.refresh_expansion();result=super().catalog()
            for i,row in enumerate(self.expansion_rows):
                result['episodes'].append(dict(id=len(self.episodes)+i,name=row['episode_id'],split=row['split'],
                    source=row['source'],frames=row['frames'],cameras=sorted(row['views']),
                    marker_ready=row['source']=='baihu',sensor_videos=[],camera_labels={k:None for k in row['views']}))
            if any(r['source']=='baihu' for r in self.expansion_rows):
                result['datasets'].append(dict(id='baihu',name='Baihu-VTouch',sensor='Four optical sensors · marker displacement',
                    labels=LABELS,target_kind='optical_rgb',target_shape=[12,240,240],hand_layout=None))
            if any(r['source']=='egopressure' for r in self.expansion_rows):
                result['datasets'].append(dict(id='egopressure',name='EgoPressure',sensor='Native 224×224 hand UV pressure',
                    labels=['Active hand · native UV pressure'],target_kind='pressure_grid',target_shape=[1,224,224],hand_layout=None))
            result['data_expansion']={}
            for job in ('baihu','egopressure','feel','teachers'):
                p=ROOT/f'{job}_status.json'
                if p.exists():result['data_expansion'][job]=json.loads(p.read_text())
            repair=ROOT/'teachers/egopressure_balanced_v2/status.json'
            if repair.exists():
                state=json.loads(repair.read_text());result['data_expansion']['egopressure_repair']=state
                result['training']=dict(status=state['stage'],source='egopressure',
                    stage='64D pressure AE repair'+(f' · epoch {state["epoch"]}/{state.get("total_epochs",30)}' if 'epoch' in state else ''))
            return result

    def _episode(self,index,frame):
        if index<len(self.episodes):return super()._episode(index,frame)
        self.refresh_expansion();i=index-len(self.episodes)
        if not 0<=i<len(self.expansion_rows):raise ValueError('Unknown expansion episode')
        row=self.expansion_rows[i]
        if not 0<=frame<row['frames']:raise ValueError('Frame outside episode')
        return row

    def frame(self,episode,frame,model,prediction=None):
        if episode<len(self.episodes):return super().frame(episode,frame,model,prediction)
        with self.lock,torch.inference_mode():
            row=self._episode(episode,frame);network=None
            if model!=-1:
                if not 0<=model<len(self.models) or self.models[model]['source']!=row['source']:
                    raise ValueError('Select an autoencoder trained for this source')
                info=self.models[model];path=Path(info['path']);stamp=path.stat().st_mtime_ns
                if model not in self.expansion_networks or self.expansion_networks[model][0]!=stamp:
                    self.expansion_networks[model]=(stamp,load_teacher(path)[0])
                network=self.expansion_networks[model][1]
            if row['source']=='baihu':
                data,_=marker_arrays(row)
                if network is not None:
                    b=marker_batch(row,[frame]);prediction=network(b['marker_target'],b['marker_valid'],
                        b['marker_reference'][None],b['marker_surface'][None])[0].numpy()*b['marker_spacing'].numpy()[:,None]
                return marker_payload(row,data,frame,prediction)
            value=pressure_targets(row,[frame]);valid=np.isfinite(value)
            pred=None if network is None else network(torch.from_numpy(value),torch.from_numpy(valid))[0].numpy()
            if prediction is not None:pred=np.asarray(prediction,np.float32)
            y=value[0];mask=valid[0];error=None if pred is None else np.abs(pred-y)
            def cells(a):return [float(v) if ok and np.isfinite(v) else None for v,ok in zip(a.flat,mask.flat)]
            cfg=json.loads((ROOT/'pressure_normalization.json').read_text());scale=cfg['scale_counts']
            physical_y=denormalize_pressure(y,scale);physical_p=None if pred is None else denormalize_pressure(pred,scale)
            physical_mask=mask if pred is None else mask&np.isfinite(physical_p)
            return dict(source='egopressure',frame=frame,target_kind='pressure_grid',target_shape=[1,224,224],
                labels=[f'{row["hand_side"].title()} hand · UV pressure'],original=cells(y),
                reconstruction=None if pred is None else cells(pred),error=None if error is None else cells(error),
                mae=None if error is None or not mask.any() else float(error[mask].mean()),
                hand_mae=[None if error is None or not mask.any() else float(error[mask].mean())],
                contact_mae=None if error is None or not (mask&(y>0)).any() else float(error[mask&(y>0)].mean()),
                background_mae=None if error is None or not (mask&(y==0)).any() else float(error[mask&(y==0)].mean()),
                contact_f1=None,
                pressure_mae_counts=None if physical_p is None or not physical_mask.any() else float(np.abs(physical_p-physical_y)[physical_mask].mean()),
                pressure_normalization='p/(p+s), train-only fixed scale; reversible',scale_counts=scale,
                source_frame_id=row['source_frame_ids'][frame],hand_side=row['hand_side'])

    def video(self,episode,frame,camera):
        if episode<len(self.episodes):return super().video(episode,frame,camera)
        row=self._episode(episode,frame)
        if camera not in row['views']:raise ValueError('Unavailable camera')
        value=video_frames(row['views'][camera],[frame],size=384)[0]
        image=Image.fromarray((value.permute(1,2,0).numpy()*255).round().astype(np.uint8))
        f=BytesIO();image.save(f,'JPEG',quality=90);return f.getvalue()

    def metadata(self,episode,camera):
        if episode<len(self.episodes):return super().metadata(episode,camera)
        row=self._episode(episode,0)
        if camera not in row['views']:raise ValueError('Unavailable camera')
        return dict(video_frames=row['frames'],fps=30.,width=384,height=384)

    def optical(self,episode,frame,gain=8.):
        if episode<len(self.episodes):return super().optical(episode,frame,gain)
        return self.frame(episode,frame,-1)

    def tracking_catalog(self):
        from .marker_viewer import tracking_catalog
        result=tracking_catalog(self);self.refresh_expansion()
        result['sources'].append('baihu')
        for i,row in enumerate(self.expansion_rows):
            if row['source']=='baihu':result['episodes'].append(dict(index=len(self.episodes)+i,source='baihu',
                name=row['episode_id'],split=row['split'],frames=row['frames'],camera=sorted(row['views'])[0],pilot=False))
        return result

    def tracking_frame(self,episode,frame):
        if episode<len(self.episodes):
            from .marker_viewer import tracking_frame
            return tracking_frame(self,episode,frame)
        return self.frame(episode,frame,-1)
