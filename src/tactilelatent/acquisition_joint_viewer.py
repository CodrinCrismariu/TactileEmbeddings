"""Viewer compatibility for integrated checkpoints without changing old loaders."""
import copy
from io import BytesIO
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from .acquisition_joint import AllSourceVJEPA,AllSourceTeachers,LAYOUTS
from .acquisition_scene_viewer import add_attention_sources,joint_row,mapped_scene_rgb


def install():
    from . import joint_viewer as module
    base=module.JointViewer
    original_shapes=module.source_shapes

    def source_shapes(cfg,include_retired_images=False):
        shapes=original_shapes(cfg,include_retired_images)
        if cfg.get('acquisition_integration'):
            shapes.update({s:[spec['input_dim']] for s,spec in LAYOUTS.items()})
        return shapes

    def load(state):
        network=AllSourceVJEPA(**state['model_config']);network.load_state_dict(state['model'],strict=True)
        return network

    module.MarkerCodeVJEPA=AllSourceVJEPA;module.marker_code_from_state=load
    module.MixedMarkerTeachers=AllSourceTeachers;module.source_shapes=source_shapes

    class AllSourceJointViewer(base):
        def catalog(self):
            result=super().catalog()
            visible=[]
            for model in result['models']:
                path=Path(model['config']['output'])/'status.json' if model['config'].get('output') else None
                status=json.loads(path.read_text()) if path and path.exists() else {}
                if model['config'].get('acquisition_integration') and status.get('stage')=='stopped_at_user_request' and not status.get('completed_epochs',0):
                    continue
                visible.append(model)
            result['models']=visible
            for model in result['models']:
                if model['config'].get('acquisition_integration'):
                    model['name']+=' · 17 sources + mirroring'
            feel=next((d for d in result['datasets'] if d['id']=='feel'),None)
            if feel:
                result['datasets']=[d for d in result['datasets'] if d['id']!='feel']
                for source,label in [('feel_lab','FEEL · Lab'),('feel_kitchen','FEEL · Kitchen')]:
                    result['datasets'].append(dict(feel,id=source,name=label,target_shape=[LAYOUTS[source]['input_dim']],
                        reconstruction_available=True,direct_prediction_only=source=='feel_lab'))
            for episode in result['episodes']:
                if episode['source']=='feel':episode['source']='feel_lab' if episode['target_shape']==[5] else 'feel_kitchen'
                if episode.get('measured_only'):episode['reconstruction_available']=True
            for dataset in result['datasets']:
                if dataset.get('measured_only'):dataset['reconstruction_available']=True
            add_attention_sources(result['models'],result['episodes'])
            return result

        def _attention_run(self,episode,frame,model,epoch,views):
            row=joint_row(self.viewer._episode(episode,frame))
            if views not in ('ego','all'):raise ValueError('Choose ego or all scene cameras')
            if views=='ego' and '0' not in row['views']:raise ValueError('This episode has no ego camera')
            if model not in self.runs:self._scan()
            run=self.runs.get(model)
            if run is None:raise ValueError('Unknown joint model; refresh the model list')
            if run['config'].get('model_type') not in module.SPATIAL_TYPES:
                raise ValueError('Attention overlays require a spatial-attention V-JEPA model')
            if epoch not in {item['id'] for item in run['snapshots']}:raise ValueError('That epoch was not saved')
            if row['source'] not in run['source_shapes']:raise ValueError('This sensor has no head in the selected model')
            return row,run

        def _video_features(self,episode,indices,mode,cfg):
            if episode<len(self.viewer.episodes):return super()._video_features(episode,indices,mode,cfg)
            key=('mapped-scene-spatial-v1',module.VJEPA_ID,module.VJEPA_REVISION,module.VJEPA_SHA,
                 cfg.get('model_type'),episode,tuple(map(int,indices)),mode,384)
            if key in self.features:
                self.features.move_to_end(key)
                features,mask=self.features[key]
                return features.to(self.device),mask.to(self.device)
            row=self.viewer._episode(episode,int(indices[-1]))
            rgb,mask=mapped_scene_rgb(row,indices,mode)
            if self.video_backbone is None:
                self.video_backbone=module.VJEPABackbone(cfg['backbone_repo'],cfg['backbone_checkpoint']).to(self.device).eval()
            batch=dict(rgb=rgb[None],view_mask=mask[None])
            with torch.autocast(self.device.type,torch.bfloat16,enabled=self.device.type=='cuda'):
                features=module.spatial_video_features(self.video_backbone,batch,self.device,cfg.get('video_chunk_size',2))
            self.viewer._keep_recent(self.features,key,(features.cpu(),mask[None]),2)
            return features,mask[None].to(self.device)

        def attention_image(self,episode,frame,model,epoch,views,camera,history):
            if episode<len(self.viewer.episodes):return super().attention_image(episode,frame,model,epoch,views,camera,history)
            if epoch=='last':epoch='last.pt'
            with self.lock:
                row,run=self._attention_run(episode,frame,model,epoch,views)
                slots={'0'} if views=='ego' else set(row['views'])
                if camera not in slots:raise ValueError('This camera is not an input to the selected prediction')
                if not 0<=history<8:raise ValueError('History slot must be between zero and seven')
                indices,_=module.prediction_clip(run['config'],row,frame)
                rgb,_=mapped_scene_rgb(row,[int(indices[history])],views)
                pixels=(rgb[0,int(camera)].permute(1,2,0).numpy()*255).round().clip(0,255).astype(np.uint8)
                buffer=BytesIO();Image.fromarray(pixels).save(buffer,'JPEG',quality=92)
                return buffer.getvalue()

        def frame(self,episode,frame,model,epoch='last.pt',views='ego'):
            if episode<len(self.viewer.episodes):return super().frame(episode,frame,model,epoch,views)
            from .data_expansion import marker_batch
            row=joint_row(self.viewer._episode(episode,frame));source=row['source']
            if model in ('','-1'):
                result=self.viewer.frame(episode,frame,-1);result['source']=source;return result
            if views not in ('ego','all'):raise ValueError('Choose ego or all scene cameras')
            with self.lock,torch.inference_mode():
                self._refresh_storage()
                if model not in self.runs:self._scan()
                run=self.runs.get(model)
                if run is None or not run['config'].get('acquisition_integration'):
                    raise ValueError('Select an integrated checkpoint for this new sensor')
                if source not in run['source_shapes']:raise ValueError('This source has no trained head')
                if epoch=='last':epoch='last.pt'
                if epoch not in {s['id'] for s in run['snapshots']}:raise ValueError('Select a saved epoch')
                (network,teachers,actual_epoch,shapes),stamp=self._load(run,epoch)
                began=time.monotonic();indices,position=module.prediction_clip(run['config'],row,frame)
                features,mask=self._video_features(episode,indices,views,run['config'])
                with torch.autocast(self.device.type,torch.bfloat16,enabled=self.device.type=='cuda'):
                    out=network(features,mask.to(self.device),source,sample_latent=False)
                if 'sensor_code' in out:
                    code=network.raw_code(source,out['sensor_code'][:,position:position+1])
                    if source=='baihu':
                        batch=marker_batch(row,[frame]);batch={k:v[None].to(self.device) for k,v in batch.items()}
                        prediction=teachers.decode_marker(source,code,batch)[0,0]
                    else:prediction=teachers.decode(source,code)[0,0]
                else:prediction=network.decode_numeric(source,out['pred'])[0,position]
                prediction=prediction.float().cpu().numpy()
                if source=='baihu':
                    from .expansion_render import marker_payload
                    from .data_expansion import marker_arrays
                    data,_=marker_arrays(row);spacing=data['spacing_px']
                    result=marker_payload(row,data,frame,prediction*spacing[:,None])
                else:result=self.viewer.frame(episode,frame,-1,prediction=prediction)
                result['source']=source
                result['joint']=dict(model=run['name'],epoch=actual_epoch,snapshot=epoch,views=views,
                    clip_frames=indices.tolist(),selected_position=position,seconds=time.monotonic()-began,
                    backbone='V-JEPA 2.1 ViT-B/16',latent='mean',uncertainty='none',
                    camera_slots=['0'] if views=='ego' else sorted(row['views']))
                return result

    module.JointViewer=AllSourceJointViewer
