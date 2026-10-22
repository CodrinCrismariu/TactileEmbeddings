"""Compose verified historical codes and new codes for current-frame training.

The old ten-source preparation is reused verbatim before adding two sources.
No historical code index is rewritten or assigned a new teacher identity.
"""
from collections import OrderedDict,defaultdict
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .aligned_frames import file_hash,manifest_signature
from .data_expansion import ROOT,marker_batch,pressure_targets
from .expansion_models import load_teacher
from .marker_code_model import MixedMarkerTeachers
from .marker_targets import MARKER_SOURCES as HISTORICAL_MARKERS
from .media import video_frames
from .pressure_selection import task_episodes,require_task_episodes,VERSION as PRESSURE_SELECTION


class ExpandedTeachers(MixedMarkerTeachers):
    def __init__(self,registry,device,chunk_frames=32):
        old={**registry,'sources':{s:i for s,i in registry['sources'].items() if s!='baihu'}}
        super().__init__(old,device,chunk_frames)
        info=registry['sources']['baihu']
        if file_hash(info['checkpoint'])!=info['sha256']:raise ValueError('Baihu teacher changed')
        self.models['baihu']=load_teacher(info['checkpoint'],device)[0]
        self.marker_sources.add('baihu')


class ExpandedCurrentClips(Dataset):
    def __init__(self,base,rows,indices,split):
        rows=task_episodes(rows)
        if 'egopressure' in indices:
            require_task_episodes([{**r,'source':'egopressure'} for r in indices['egopressure']['episodes']])
        self.base=base;self.episodes=list(base.episodes);self.windows=list(base.windows)
        self.indices_by_source=defaultdict(list,{s:list(v) for s,v in base.indices_by_source.items()})
        self.new_items=[];self.code_indices=indices;self.loaded=OrderedDict()
        self.records={s:{r['episode_id']:r for r in index['episodes']} for s,index in indices.items()}
        for row in rows:
            if row['split']!=split:continue
            if row['source'] in ('ftp_motiontrans','freetacman'):raise ValueError('Excluded source in new training inputs')
            ep=len(self.episodes);self.episodes.append(row)
            codes=self.codes(row)
            for start in range(0,row['frames']-14,15):
                if not codes['valid'][start+14]:continue
                self.indices_by_source[row['source']].append(len(self.windows))
                self.windows.append((ep,start));self.new_items.append((row,start))
        if set(self.indices_by_source)!=set(base.indices_by_source)|{'baihu','egopressure'}:
            raise ValueError('Expanded training needs supported clips from exactly the two added sources')

    def codes(self,row):
        key=(row['source'],row['episode_id'])
        if key not in self.loaded:
            record=self.records[row['source']][row['episode_id']];path=Path(record['path'])
            if file_hash(path)!=record['sha256']:raise ValueError('Expanded codes changed')
            with np.load(path,allow_pickle=False) as f:d={k:f[k].copy() for k in f.files}
            if not np.array_equal(d['frame_indices'],np.arange(row['frames'])):raise ValueError('Expanded code chronology differs')
            dimension=len(self.code_indices[row['source']]['statistics']['mean'])
            if (d['codes'].shape!=(row['frames'],dimension) or d['codes'].dtype!=np.float32
                    or d['valid'].shape!=(row['frames'],) or d['valid'].dtype!=np.bool_
                    or not np.isfinite(d['codes']).all()):raise ValueError('Expanded code shape/type/values differ')
            self.loaded[key]=d
            while len(self.loaded)>4:self.loaded.popitem(last=False)
        self.loaded.move_to_end(key);return self.loaded[key]

    def __len__(self):return len(self.windows)

    def __getitem__(self,index):
        if index<len(self.base):return self.base[index]
        row,start=self.new_items[index-len(self.base)]
        history=np.arange(start,start+15,2);current=history[-1:]
        rgb=torch.zeros(8,3,3,384,384);mask=torch.zeros(8,3,dtype=torch.bool)
        for slot,path in row['views'].items():
            rgb[:,int(slot)]=video_frames(path,history,size=384);mask[:,int(slot)]=True
        d=self.codes(row);stats=self.code_indices[row['source']]['statistics']
        code=(d['codes'][current]-np.asarray(stats['mean'],np.float32))/np.asarray(stats['std'],np.float32)
        item=dict(rgb=rgb,view_mask=mask,source=row['source'],episode_id=row['episode_id'],
            history_frame_indices=torch.from_numpy(history),frame_indices=torch.from_numpy(current),
            sensor_frame_indices=torch.from_numpy(current),code_target=torch.from_numpy(code),
            code_valid=torch.from_numpy(d['valid'][current]))
        if row['source']=='baihu':item.update(marker_batch(row,current))
        else:
            target=torch.from_numpy(pressure_targets(row,current));valid=torch.isfinite(target)
            item.update(target=torch.where(valid,target,0.),target_mask=valid)
        return item


def prepare(cfg,base_prepare,trainer,allow_pilot=False):
    if allow_pilot:raise ValueError('Expanded joint training requires complete production artifacts')
    ready=json.loads((ROOT/'ready.json').read_text())
    if not ready['ready'] or cfg['heldout_source']!='ftp_motiontrans' or cfg.get('prediction_mode')!='current_frame':
        raise ValueError('Expanded data not ready or wrong held-out/current-frame protocol')
    if (file_hash(ROOT/'manifest.json')!=ready['manifest_sha256']
            or file_hash(ROOT/'teachers.json')!=ready['teachers_sha256']):
        raise ValueError('Expanded publication provenance differs')
    base_cfg=json.loads(Path(cfg['base_config']).read_text())
    for key in ('latent_dim','adapter_hidden_dims','attention','prediction_mode'):base_cfg[key]=cfg[key]
    # Only the local trainer module's dispatch changes. Imported historical
    # marker extraction/cache modules and their fingerprints remain untouched.
    active=trainer.MARKER_SOURCES;trainer.MARKER_SOURCES=HISTORICAL_MARKERS
    try:train,val,registry,mc,provenance=base_prepare(base_cfg)
    finally:trainer.MARKER_SOURCES=active
    expansion=json.loads((ROOT/'teachers.json').read_text());rows=json.loads((ROOT/'manifest.json').read_text())['episodes']
    rows=[r for r in rows if r['source'] in ('baihu','egopressure')]
    require_task_episodes(rows)
    code_paths={s:Path(ready.get('code_index_paths',{}).get(s,str(ROOT/'codes'/s/'index.json'))) for s in ('baihu','egopressure')}
    indices={s:json.loads(code_paths[s].read_text()) for s in ('baihu','egopressure')}
    registry=copy.deepcopy(registry);mc=copy.deepcopy(mc)
    for source,info in expansion['sources'].items():
        if file_hash(code_paths[source])!=ready['code_index_sha256'][source]:
            raise ValueError('Expanded code index changed after publication')
        if not indices[source]['complete'] or indices[source]['teacher_sha256']!=file_hash(info['checkpoint']):
            raise ValueError('Expanded code cache/teacher mismatch')
        selected=[r for r in rows if r['source']==source]
        from .aligned_frames import json_hash
        snapshot=json_hash({r['episode_id']:file_hash(r['marker_path'] if source=='baihu' else r['target_path']) for r in selected})
        if snapshot!=indices[source]['target_snapshot_sha256']:raise ValueError('Expanded native targets changed after code caching')
        registry['sources'][source]=info;mc['sensor_code_dims'][source]=info['code_dim']
        mc['statistics'][source]={k:indices[source]['statistics'][k] for k in ('mean','std')}
    train=ExpandedCurrentClips(train,rows,indices,'train');val=ExpandedCurrentClips(val,rows,indices,'val')
    if len(train.indices_by_source)!=12 or 'ftp_motiontrans' in train.indices_by_source:raise ValueError('Expected twelve source datasets')
    provenance['base_data']=provenance['data'];provenance['data']=manifest_signature(cfg['manifest'])
    provenance['teachers']=registry
    provenance['expansion']=dict(manifest_sha256=ready['manifest_sha256'],teachers_sha256=ready['teachers_sha256'],
        code_index_sha256={s:file_hash(code_paths[s]) for s in indices},
        implementation={name:file_hash(Path(__file__).with_name(name)) for name in
            ('data_expansion.py','expansion_models.py','expansion_joint.py','pressure_selection.py')},feel='pilot only')
    provenance['expansion']['egopressure_selection']=PRESSURE_SELECTION
    return train,val,registry,mc,provenance
