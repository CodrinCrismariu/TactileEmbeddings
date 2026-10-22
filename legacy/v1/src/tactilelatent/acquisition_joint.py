"""Native numeric acquisitions added to the existing twelve-source pipeline.

Only tactile codes are cached. RGB is decoded on demand through the recorded
frame maps; native targets, camera roles and episode splits are immutable.
"""
from collections import OrderedDict, defaultdict
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from .aligned_frames import file_hash, json_hash, write_json
from .acquisition_data import _targets
from .media import video_frames
from .numeric_autoencoder import load_numeric_teacher
from .marker_code_model import MarkerCodeVJEPA, MixedMarkerTeachers
from .expansion_models import load_teacher as load_expansion_teacher

ROOT=Path('processing/joint_acquisitions_v1')
LAYOUTS={
    'feel_lab':dict(acquisition='feel',environment='Lab',input_dim=5,direct=True),
    'feel_kitchen':dict(acquisition='feel',environment='Kitchen',input_dim=12,code_dim=6),
    'hrdexdb_allegro_v5':dict(acquisition='hrdexdb',embodiment='allegro_v5',input_dim=4,direct=True),
    'hrdexdb_inspire_f1':dict(acquisition='hrdexdb',embodiment='inspire_f1',input_dim=24,code_dim=12),
    'trex':dict(acquisition='trex',input_dim=60,code_dim=16),
}
EXCLUDED={'ftp_motiontrans','freetacman','tacquad','tdex'}


def native_rows(source):
    spec=LAYOUTS[source];path=Path('processing/data_queue_v2')/spec['acquisition']/'manifest.json'
    document=json.loads(path.read_text())
    if document.get('schema')!='acquisition-native-v2':raise ValueError('Expected native acquisition manifest')
    rows=[r for r in document['episodes'] if all(r.get(k)==spec[k] for k in ('environment','embodiment') if k in spec)]
    if not rows or any(r['targets']['shape'][1:]!=[spec['input_dim']] for r in rows):
        raise ValueError('Acquisition sensor layout changed')
    channels=rows[0]['targets']['channel_order']
    if any(r['targets']['channel_order']!=channels for r in rows):raise ValueError('Native channel order differs')
    if {r['split'] for r in rows}!={'train','val'}:raise ValueError('Episode-disjoint source splits required')
    return rows,dict(manifest=str(path),sha256=file_hash(path),revision=document['revision'],channel_order=channels)


def history_mapping(row,end,native=None):
    """Eight chronological RGB samples ending at this native tactile row.

    T-Rex's frame count is an episode-segment length inside a video shard;
    its upper bound includes the segment offset. HRDex uses explicit indices.
    """
    indices=np.arange(end-14,end+1,2,dtype=np.int64)
    if indices[0]<0 or indices[-1]>=row['frames']:raise ValueError('History crosses episode boundary')
    if native is None:native=_targets(row['targets']['path'],row['targets']['sha256'])
    if np.any(np.diff(native['timestamps'][indices])<=0):raise ValueError('Nonchronological tactile history')
    if 'rgb_frame_valid' in native and not native['rgb_frame_valid'][indices].all():
        raise ValueError('Missing or ambiguous RGB timestamp')
    views={}
    for slot,view in row['views'].items():
        if view['role'] not in ('ego','left_wrist','right_wrist','fixed_scene'):
            raise ValueError('Unknown camera role')
        if 'frame_indices_key' in view:
            if view.get('frame_indices_artifact')!=row['targets']['path']:raise ValueError('External camera mapping')
            mapped=native[view['frame_indices_key']][indices];limit=view.get('frames')
        else:
            offset=int(view.get('frame_offset',0));mapped=indices+offset
            limit=offset+view['frames'] if 'frames' in view else None
        if (mapped<0).any() or (limit is not None and (mapped>=limit).any()) or (np.diff(mapped)<=0).any():
            raise ValueError('Missing/nonchronological camera history')
        views[slot]=mapped
    if not views:raise ValueError('No scene camera')
    return indices,views


def load_rgb(row,end):
    indices,views=history_mapping(row,end)
    rgb=torch.zeros(8,3,3,384,384);mask=torch.zeros(8,3,dtype=torch.bool)
    for slot,mapped in views.items():
        rgb[:,int(slot)]=video_frames(row['views'][slot]['path'],mapped.tolist(),size=384)
        mask[:,int(slot)]=True
    return rgb,mask,indices


def prepare_acquisitions(device='cuda',root=ROOT):
    """Resumable target-code preparation, with training-only statistics."""
    root=Path(root);root.mkdir(parents=True,exist_ok=True);sources={}
    for source,spec in LAYOUTS.items():
        rows,manifest=native_rows(source);folder=root/'codes'/source;folder.mkdir(parents=True,exist_ok=True)
        teacher=None;info=None
        if not spec.get('direct'):
            path=Path('processing/acquisition_teachers_v1')/source/'best.pt'
            teacher,state=load_numeric_teacher(path,device);teacher.requires_grad_(False)
            if (state['source']!=spec['acquisition'] and state['source']!=source) or state['model_config']['input_dim']!=spec['input_dim']:
                raise ValueError('Numeric teacher source/layout mismatch')
            if state['provenance']['manifest_sha256']!=manifest['sha256'] or not state.get('accepted'):
                raise ValueError('Teacher has no accepted validation/provenance receipt')
            info=dict(checkpoint=str(path),sha256=file_hash(path),code_dim=spec['code_dim'],
                target_shape=[spec['input_dim']],representation='native_numeric',model_config=state['model_config'])
        signature=dict(version='joint-acquisitions-v1',source=source,spec=spec,manifest=manifest,
            teacher_sha256=info['sha256'] if info else None,normalization='train only; no episode offset')
        signature_path=folder/'signature.json'
        if signature_path.exists() and json.loads(signature_path.read_text())!=signature:
            raise ValueError('Acquisition preparation changed; use a new version')
        write_json(signature_path,signature)
        count=0;total=None;squares=None;native_count=np.zeros(spec['input_dim'],np.int64)
        native_squares=np.zeros(spec['input_dim']);records=[];window_counts=defaultdict(int);rejected=defaultdict(int)
        for number,row in enumerate(rows):
            native=_targets(row['targets']['path'],row['targets']['sha256'])
            x=native['values'].reshape(row['frames'],-1).astype(np.float32)
            mask=native['valid'].reshape(x.shape)&np.isfinite(x)
            if (not np.array_equal(native['frame_indices'],np.arange(row['frames']))
                    or not np.isfinite(native['timestamps']).all() or np.any(np.diff(native['timestamps'])<0)):
                raise ValueError('Native chronology changed')
            path=folder/(row['episode_id']+'.npz');receipt=path.with_suffix('.json')
            if path.exists() and receipt.exists():
                old=json.loads(receipt.read_text())
                if old['row_sha256']!=json_hash(row) or file_hash(path)!=old['sha256']:
                    raise ValueError('Cached numeric targets changed')
                with np.load(path,allow_pickle=False) as f:codes=f['codes'].copy();valid=f['valid'].copy();ends=f['ends'].copy()
            else:
                valid=mask.any(1);codes=np.zeros((len(x),spec.get('code_dim',0)),np.float32)
                if teacher is not None:
                    with torch.inference_mode():
                        for start in range(0,len(x),8192):
                            y=torch.from_numpy(x[start:start+8192]).to(device)
                            m=torch.from_numpy(mask[start:start+8192]).to(device)
                            codes[start:start+8192]=teacher.encode(y,m).float().cpu().numpy()
                    if not np.isfinite(codes).all():raise ValueError('Nonfinite frozen numeric code')
                ends=[]
                for end in range(14,len(x),15):
                    if not valid[end]:continue
                    try:history_mapping(row,end,native)
                    except ValueError:continue
                    ends.append(end)
                ends=np.array(ends,np.int64)
                temporary=path.with_suffix('.tmp')
                with temporary.open('wb') as stream:np.savez_compressed(stream,codes=codes,valid=valid,
                    frame_indices=np.arange(len(x),dtype=np.int64),ends=ends)
                temporary.replace(path)
                write_json(receipt,dict(row_sha256=json_hash(row),sha256=file_hash(path)))
            if codes.shape!=(len(x),spec.get('code_dim',0)) or valid.shape!=(len(x),):raise ValueError('Invalid code geometry')
            window_counts[row['split']]+=len(ends);rejected[row['split']]+=len(range(14,len(x),15))-len(ends)
            if row['split']=='train':
                safe=np.where(mask,x,0).astype(np.float64);native_count+=mask.sum(0);native_squares+=(safe*safe).sum(0)
                if teacher is not None:
                    supported=codes[valid].astype(np.float64)
                    if total is None:total=np.zeros(spec['code_dim']);squares=total.copy()
                    total+=supported.sum(0);squares+=(supported*supported).sum(0);count+=len(supported)
            records.append(dict(episode_id=row['episode_id'],split=row['split'],path=str(path),sha256=file_hash(path),
                row_sha256=json_hash(row),target_sha256=row['targets']['sha256'],frames=row['frames'],windows=len(ends)))
            if number%100==0:write_json(root/'status.json',dict(stage='caching_native_codes',source=source,
                episodes_done=number+1,episodes=len(rows)))
        if not all(window_counts[k]>0 for k in ('train','val')):raise ValueError('No supported source split')
        scale=np.sqrt(native_squares/np.maximum(native_count,1));scale=np.where(scale>1e-8,scale,1.)
        normalization=dict(scale=scale.tolist(),offset=[0.]*len(scale),count=native_count.tolist(),fit_split='train',
            method='asinh(native/train RMS)',nonnegative=[source!='trex' and not (source=='hrdexdb_inspire_f1' and i%3==2) for i in range(len(scale))])
        statistics=None
        if teacher is not None:
            mean=total/max(count,1);std=np.maximum(np.sqrt(np.maximum(squares/max(count,1)-mean*mean,0)),1e-6)
            statistics=dict(mean=mean.tolist(),std=std.tolist(),fit_split='train',frames=count)
        index=dict(complete=True,signature=signature,episodes=records,statistics=statistics,
            native_normalization=normalization,windows=dict(window_counts),excluded_windows=dict(rejected))
        write_json(folder/'index.json',index)
        sources[source]=dict(index=str(folder/'index.json'),index_sha256=file_hash(folder/'index.json'),teacher=info,
            native_normalization=normalization,statistics=statistics,windows=dict(window_counts),excluded_windows=dict(rejected))
        print(json.dumps(dict(event='source_prepared',source=source,windows=dict(window_counts),excluded_windows=dict(rejected))),flush=True)
        del teacher
    ready=dict(ready=True,version='joint-acquisitions-v1',sources=sources,heldout='ftp_motiontrans',
        excluded_sources=sorted(EXCLUDED),rgb_cache=False,native_targets_unchanged=True)
    write_json(root/'ready.json',ready);write_json(root/'status.json',dict(stage='complete',sources=list(sources)))
    return ready


class AcquisitionClips(Dataset):
    def __init__(self,base,ready,split):
        self.base=base;self.ready=ready;self.episodes=list(base.episodes);self.windows=list(base.windows)
        self.indices_by_source=defaultdict(list,{s:list(v) for s,v in base.indices_by_source.items()});self.loaded=OrderedDict();self.records={}
        for source,entry in ready['sources'].items():
            if source in EXCLUDED:raise ValueError('Excluded source in acquisition inputs')
            if file_hash(entry['index'])!=entry['index_sha256']:raise ValueError('Acquisition code index changed')
            index=json.loads(Path(entry['index']).read_text());rows,manifest=native_rows(source)
            if not index['complete'] or index['signature']['manifest']!=manifest:raise ValueError('Acquisition manifest changed')
            records={r['episode_id']:r for r in index['episodes']}
            self.records[source]=records
            for row in rows:
                if row['split']!=split:continue
                if records[row['episode_id']]['row_sha256']!=json_hash(row):raise ValueError('Episode/frame mapping changed')
                ep=len(self.episodes);self.episodes.append(dict(row,source=source))
                data=self.codes(source,row['episode_id'])
                for end in data['ends']:
                    self.indices_by_source[source].append(len(self.windows));self.windows.append((ep,int(end)-14))
        if set(self.indices_by_source)&EXCLUDED:raise ValueError('Held-out or removed source in joint data')
        self.loaded.clear()

    def codes(self,source,episode):
        key=(source,episode)
        if key not in self.loaded:
            info=self.records[source][episode]
            if file_hash(info['path'])!=info['sha256']:raise ValueError('Cached numeric code changed')
            with np.load(info['path'],allow_pickle=False) as f:self.loaded[key]={k:f[k].copy() for k in f.files}
            while len(self.loaded)>4:self.loaded.popitem(last=False)
        self.loaded.move_to_end(key);return self.loaded[key]

    def __len__(self):return len(self.windows)

    def __getitem__(self,index):
        if index<len(self.base):return self.base[index]
        ep,start=self.windows[index];row=self.episodes[ep];source=row['source'];end=start+14
        rgb,view_mask,history=load_rgb(row,end)
        target=_targets(row['targets']['path'],row['targets']['sha256']);values=target['values'][end:end+1].astype(np.float32)
        valid=target['valid'][end:end+1]&np.isfinite(values);code=self.codes(source,row['episode_id'])
        item=dict(rgb=rgb,view_mask=view_mask,source=source,episode_id=row['episode_id'],
            history_frame_indices=torch.from_numpy(history),frame_indices=torch.tensor([end]),sensor_frame_indices=torch.tensor([end]),
            target=torch.from_numpy(np.where(valid,values,0.)),target_mask=torch.from_numpy(valid))
        stats=self.ready['sources'][source]['statistics']
        if stats is not None:item.update(code_target=torch.from_numpy((code['codes'][end:end+1]-np.array(stats['mean'],np.float32))/np.array(stats['std'],np.float32)),
            code_valid=torch.from_numpy(code['valid'][end:end+1].copy()))
        return item


class AllSourceVJEPA(MarkerCodeVJEPA):
    """Same visual architecture; tiny direct heads for Lab and Allegro."""
    def __init__(self,direct_numeric=None,**kwargs):
        super().__init__(**kwargs);self.direct_numeric=direct_numeric or {};self.direct_scales=nn.ModuleDict()
        for source,info in self.direct_numeric.items():
            d=len(info['scale']);self.numeric_heads[source]=nn.Linear(kwargs.get('latent_dim',32),d)
            norm=nn.Module();norm.register_buffer('scale',torch.tensor(info['scale'],dtype=torch.float32));self.direct_scales[source]=norm

    def normalize_numeric(self,source,value):return torch.asinh(value/self.direct_scales[source].scale)

    def decode_numeric(self,source,value):return (torch.sinh(value.float().clamp(-15,15))*self.direct_scales[source].scale).clamp_min(0)


class AllSourceTeachers(MixedMarkerTeachers):
    def __init__(self,registry,device,chunk_frames=32):
        extra={s:i for s,i in registry['sources'].items() if i.get('representation')=='native_numeric' or i.get('bilateral') or s in ('baihu','egopressure')}
        super().__init__({**registry,'sources':{s:i for s,i in registry['sources'].items() if s not in extra}},device,chunk_frames)
        for source,info in extra.items():
            if file_hash(info['checkpoint'])!=info['sha256']:raise ValueError('Frozen teacher changed')
            from .bilateral_autoencoder import load_bilateral_teacher
            load=load_bilateral_teacher if info.get('bilateral') else load_numeric_teacher if info.get('representation')=='native_numeric' else load_expansion_teacher
            self.models[source]=load(info['checkpoint'],device)[0].requires_grad_(False)
            if info.get('representation')=='marker_displacement':self.marker_sources.add(source)

    @torch.no_grad()
    def supervision(self,original,model):
        source=original['source'][0]
        if source in model.direct_numeric:
            y=original['target'].to(self.device).float();mask=original['target_mask'].to(self.device)
            return dict(target=model.normalize_numeric(source,y),target_mask=mask)
        return super().supervision(original,model)


def prepare(cfg,base_prepare,trainer,allow_pilot=False):
    from .expansion_joint import prepare as expanded_prepare
    from .mirror_joint import bind
    if cfg['heldout_source']!='ftp_motiontrans':raise ValueError('MotionTrans must remain held out')
    train,val,registry,mc,provenance=expanded_prepare(cfg,base_prepare,trainer,allow_pilot)
    receipts=[]
    for binding in cfg['mirror_bindings']:
        local=copy.deepcopy(cfg);local['mirror_augmentation']={**cfg['mirror_augmentation'],**binding}
        train,val,registry,mc,provenance=bind(local,(train,val,registry,mc,provenance))
        receipts.append(copy.deepcopy(provenance['mirror_augmentation']))
    provenance['mirror_augmentation']=dict(bindings=receipts,probability=cfg['mirror_augmentation']['probability'],
        sources=cfg['mirror_augmentation']['sources'],latent_equality_constraint=False)
    path=Path(cfg['acquisition_preparation'])/'ready.json';ready=json.loads(path.read_text())
    if not ready['ready'] or set(ready['sources'])!=set(LAYOUTS):raise ValueError('Prepare all acquisitions first')
    registry=copy.deepcopy(registry);mc=copy.deepcopy(mc);mc['direct_numeric']={}
    for source,entry in ready['sources'].items():
        if entry['teacher']:
            registry['sources'][source]=entry['teacher'];mc['sensor_code_dims'][source]=entry['teacher']['code_dim']
            mc['statistics'][source]={k:entry['statistics'][k] for k in ('mean','std')}
        else:
            mc['numeric_shapes'][source]=[LAYOUTS[source]['input_dim']]
            mc['direct_numeric'][source]=entry['native_normalization']
    train=AcquisitionClips(train,ready,'train');val=AcquisitionClips(val,ready,'val')
    if len(train.indices_by_source)!=17 or set(train.indices_by_source)!=set(val.indices_by_source):
        raise ValueError('Expected all seventeen supported native sensor layouts')
    provenance['teachers']=registry
    provenance['acquisitions']=dict(ready_sha256=file_hash(path),sources=ready['sources'],
        direct_prediction_sources=list(mc['direct_numeric']),heldout='ftp_motiontrans',visual_cache=False,
        implementation={n:file_hash(Path(__file__).with_name(n)) for n in ('acquisition_joint.py','acquisition_training.py','numeric_autoencoder.py','bilateral_autoencoder.py')})
    write_json(Path(cfg['acquisition_preparation'])/'teachers.json',registry)
    write_json(Path(cfg['acquisition_preparation'])/'model_config.json',mc)
    return train,val,registry,mc,provenance
