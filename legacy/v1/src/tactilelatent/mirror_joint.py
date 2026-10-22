"""Opt-in native-target mirroring for current-frame frozen-code joint runs."""
from collections import OrderedDict, defaultdict
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .aligned_frames import file_hash, json_hash
from .bimanual_mirror import load_policy, require_spec, apply_batch, transform_targets, TARGET_FIELDS


def artifact_policy_hash(policy, ready, entry, source):
    """Reuse an unchanged source's artifacts under their original policy.

    A newer combined policy does not retroactively change a teacher or cache's
    provenance. Reuse requires the complete source mapping/evidence to agree.
    """
    path = entry.get('artifact_policy_path')
    if not path:
        return ready['policy_sha256']
    original = load_policy(path)
    digest = json_hash(original)
    if digest != entry['artifact_policy_sha256']:
        raise ValueError('Inherited mirror artifact policy changed')
    if original['sources'][source] != policy['sources'][source]:
        raise ValueError('Cannot inherit mirror artifacts for a changed source mapping')
    return digest


def read_preparation(cfg, active_sources=None):
    options=cfg.get('mirror_augmentation',{})
    if not options.get('enabled',False):return None
    if cfg.get('heldout_source')!='ftp_motiontrans' or cfg.get('prediction_mode')!='current_frame':
        raise ValueError('Mirroring requires the held-out MotionTrans/current-frame protocol')
    sources=options['sources']
    if len(sources)!=len(set(sources)) or not 0<=options.get('probability',.5)<=1:
        raise ValueError('Invalid mirror source list/probability')
    policy=load_policy(options['policy'])
    errors=[]
    for source in sources:
        try:require_spec(policy,source)
        except ValueError as error:errors.append(str(error))
    if errors:raise ValueError('Mirror preparation blocked:\n'+'\n'.join(errors))
    path=Path(options['preparation'])/'ready.json'
    if not path.exists():raise ValueError('Finish mirror teacher/code preparation before joint training')
    ready=json.loads(path.read_text())
    if ready['policy_sha256']!=json_hash(policy):raise ValueError('Mirror policy changed after code preparation')
    for name,digest in ready.get('implementation',{}).items():
        if file_hash(name)!=digest:raise ValueError('Mirror preparation extension changed')
    audit_path=path.with_name('policy_audit.json')
    if not audit_path.exists():raise ValueError('Mirror preparation has no geometry-evidence audit')
    audit=json.loads(audit_path.read_text())
    if audit['policy_sha256']!=ready['policy_sha256']:raise ValueError('Geometry audit belongs to another policy')
    for source in sources:
        evidence=audit['sources'][source]
        if not evidence.get('verified') or any(file_hash(p)!=digest for p,digest in evidence['evidence_sha256'].items()):
            raise ValueError(f'{source}: verified coordinate evidence changed')
        if active_sources is not None and source not in active_sources:
            errors.append(f'{source}: requested bilateral source is absent from this joint configuration')
        entry=ready['sources'].get(source,{})
        if not entry.get('ready'):errors.append(f'{source}: {entry.get("reason",entry.get("stage","not prepared"))}')
        elif not entry.get('numeric'):
            artifact_hash=artifact_policy_hash(policy,ready,entry,source)
            if file_hash(entry['teacher']['checkpoint'])!=entry['teacher']['sha256']:
                raise ValueError('Prepared mirror teacher changed')
            if file_hash(entry['code_index'])!=entry['code_index_sha256']:
                raise ValueError('Prepared mirror code index changed')
            quality=json.loads(Path(entry['teacher_quality']).read_text())
            if not quality['accepted'] or quality['teacher']['sha256']!=entry['teacher']['sha256']:
                raise ValueError('Mirror teacher has no accepted quality receipt')
            if quality['signature']['policy_sha256']!=artifact_hash:
                raise ValueError('Mirror teacher belongs to a different artifact policy')
            verification=Path(entry['code_index']).with_name('verification.json')
            if not verification.exists() or not json.loads(verification.read_text()).get('passed'):
                raise ValueError('Mirror cache has not passed fresh-encoding verification')
            if json.loads(verification.read_text())['index_sha256']!=entry['code_index_sha256']:
                raise ValueError('Mirror cache verification belongs to a different index')
    if errors:raise ValueError('Mirror preparation blocked:\n'+'\n'.join(errors))
    return policy,ready


class PairCodeCache:
    def __init__(self,path,teacher_sha256,policy_sha256):
        self.index=json.loads(Path(path).read_text());self.loaded=OrderedDict()
        sig=self.index['signature']
        if not self.index['complete'] or sig['teacher_sha256']!=teacher_sha256 or sig['policy_sha256']!=policy_sha256:
            raise ValueError('Paired sensor code provenance mismatch')
        self.rows={r['episode_id']:r for r in self.index['episodes']}

    def read(self,row,indices):
        info=self.rows[row['episode_id']]
        if info['row_sha256']!=json_hash(row):raise ValueError('Mirrored episode/frame contract changed')
        key=row['episode_id'];indices=np.asarray(indices,np.int64)
        if indices.ndim!=1 or (indices<0).any() or (indices>=row['frames']).any():
            raise ValueError('Mirrored code frame indices outside episode')
        if key not in self.loaded:
            if file_hash(info['path'])!=info['sha256']:raise ValueError('Paired code artifact changed')
            with np.load(info['path'],allow_pickle=False) as f:
                d={k:f[k].copy() for k in f.files}
            dim=len(self.index['statistics']['mean'])
            if (d['codes'].shape!=(2,row['frames'],dim) or d['codes'].dtype!=np.float32
                    or d['valid'].shape!=(2,row['frames']) or d['valid'].dtype!=np.bool_
                    or not np.isfinite(d['codes']).all()):raise ValueError('Invalid paired code arrays')
            np.testing.assert_array_equal(d['frame_indices'],np.arange(row['frames']))
            np.testing.assert_array_equal(d['valid'][0],d['valid'][1])
            self.loaded[key]=d
            while len(self.loaded)>2:self.loaded.popitem(last=False)
        self.loaded.move_to_end(key);d=self.loaded[key];stats=self.index['statistics']
        codes=(d['codes'][:,indices]-np.asarray(stats['mean'],np.float32))/np.asarray(stats['std'],np.float32)
        return torch.from_numpy(codes),torch.from_numpy(d['valid'][:,indices].copy())


class MirroredCodeClips(Dataset):
    """Preserve every existing window; attach two teacher-code variants."""
    def __init__(self,dataset,caches,sources):
        self.dataset=dataset;self.caches=caches;self.sources=set(sources)
        self.episodes=dataset.episodes;self.windows=dataset.windows
        self.indices_by_source=dataset.indices_by_source

    def __len__(self):return len(self.dataset)

    def __getitem__(self,index):
        item=dict(self.dataset[index]);source=item['source']
        if source not in self.sources:return item
        row=self.episodes[self.windows[index][0]]
        if 'marker_reference' in item:item['mirror_marker_count']=torch.tensor(len(item['marker_reference']))
        if source in self.caches:
            codes,valid=self.caches[source].read(row,item['frame_indices'].numpy())
            if not valid[0].any():
                raise ValueError('Mirrored teacher lost support for a previously valid current-frame clip')
            item.update(code_target=codes[0],code_valid=valid[0],mirror_code_target=codes[1],mirror_code_valid=valid[1])
        return item


def bind(cfg,prepared):
    train,val,registry,mc,provenance=prepared
    data=read_preparation(cfg,set(train.indices_by_source))
    if data is None:return prepared
    policy,ready=data;options=cfg['mirror_augmentation'];sources=options['sources'];caches={}
    registry=copy.deepcopy(registry);mc=copy.deepcopy(mc);provenance=copy.deepcopy(provenance)
    for source in sources:
        entry=ready['sources'][source]
        if entry.get('numeric'):continue
        if registry['sources'][source]['code_dim']!=entry['teacher']['code_dim']:
            raise ValueError('Mirroring cannot change a sensor-code dimension')
        caches[source]=PairCodeCache(entry['code_index'],entry['teacher']['sha256'],
                                    artifact_policy_hash(policy,ready,entry,source))
        # Recheck actual native target hashes, not just saved code-index claims.
        from .mirror_preparation import NativeTargets,target_snapshot
        rows=[r for r in train.episodes+val.episodes if r['source']==source]
        # Official test episodes can also be cached; include the index's full
        # row identity in provenance while validating active train/val targets.
        expected_rows={r['episode_id']:r['row_sha256'] for r in caches[source].index['episodes']}
        if any(expected_rows.get(r['episode_id'])!=json_hash(r) for r in rows):
            raise ValueError('Active dataset differs from mirror cache episode identities')
        all_rows=sorted([r for r in json.loads(Path(cfg['manifest']).read_text())['episodes'] if r['source']==source],
                        key=lambda r:(r['split'],r['episode_id']))
        signature=caches[source].index['signature']
        if json_hash(all_rows)!=signature['rows_sha256'] or target_snapshot(all_rows,NativeTargets())!=signature['target_snapshot_sha256']:
            raise ValueError('Native tactile targets changed after mirrored code caching')
        registry['sources'][source]=entry['teacher']
        mc['statistics'][source]={k:entry['statistics'][k] for k in ('mean','std')}
    provenance['mirror_augmentation']=dict(policy_sha256=ready['policy_sha256'],
        probability=options.get('probability',.5),sources={s:ready['sources'][s] for s in sources},
        implementation={name:file_hash(Path(__file__).with_name(name)) for name in
                        ('bimanual_mirror.py','mirror_joint.py','mirror_preparation.py','mirror_cache_verification.py','train_vjepa_markers.py')},
        code_verification_sha256={s:file_hash(Path(ready['sources'][s]['code_index']).with_name('verification.json'))
                                  for s in sources if not ready['sources'][s].get('numeric')},
        inference_transport='none',latent_equality_constraint=False)
    provenance['teachers']=registry
    return MirroredCodeClips(train,caches,sources),MirroredCodeClips(val,caches,sources),registry,mc,provenance


_POLICIES={}


def augment(batch,cfg,epoch,step,flags=None):
    options=cfg.get('mirror_augmentation',{})
    if not options.get('enabled',False):return batch
    path=options['policy']
    if path not in _POLICIES:_POLICIES[path]=load_policy(path)
    return apply_batch(batch,_POLICIES[path],options['sources'],options.get('probability',.5),
                       cfg['seed'],epoch,step,flags)


def _decode(model,teachers,output,batch,source):
    if 'sensor_code' not in output:return output['pred'].float()
    code=model.raw_code(source,output['sensor_code'])
    if 'marker_target' in batch:return teachers.decode_marker(source,code,batch).float()
    shape=code.shape[:-1];teacher=teachers.models[source]
    with torch.autocast(code.device.type,enabled=False):
        value=teacher.decode(code.float().reshape(-1,code.shape[-1]))
    return value.reshape(*shape,*value.shape[1:]).float()


@torch.no_grad()
def paired_validate(model,backbone,teachers,loader,device,cfg):
    """Diagnostic only: do not mix mirrored examples into primary validation."""
    from .vjepa_spatial import spatial_video_features
    from .train_vjepa_markers import forward,supervision
    options=cfg.get('mirror_augmentation',{})
    if not options.get('enabled',False):return {}
    model.eval();totals=defaultdict(lambda:defaultdict(float));policy=load_policy(options['policy'])
    for original in loader:
        source=original['source'][0]
        if source not in options['sources']:continue
        mirrored=augment(original,cfg,0,0,flags=[True]*len(original['source']))
        predictions=[]
        for name,item in [('original',original),('mirrored',mirrored)]:
            batch=supervision(item,teachers,model,device)
            with torch.autocast(device.type,torch.bfloat16,enabled=device.type=='cuda'):
                features=spatial_video_features(backbone,item,device)
                output=forward(model,features,batch)
            prediction=_decode(model,teachers,output,batch,source)
            y=batch.get('target',batch.get('marker_target'));mask=batch.get('target_mask')
            if mask is None:mask=batch['marker_valid'][...,None].expand_as(y)
            mask=mask&torch.isfinite(y)&torch.isfinite(prediction)
            totals[source][name+'_absolute']+=(prediction-y)[mask].abs().sum().item()
            totals[source][name+'_count']+=mask.sum().item()
            predictions.append(prediction)
        expected=predictions[0].clone();spec=require_spec(policy,source)
        for i in range(len(expected)):
            if spec['kind']=='markers':
                n=int(original['mirror_marker_count'][i]);value={k:original[k][i] for k in TARGET_FIELDS if k in original}
                value={k:(v[...,:n,:] if k=='marker_target' else v[...,:n] if k=='marker_valid' else v[:n]) for k,v in value.items()}
                value=to_prediction_device(value,predictions[0].device)
                value['marker_target']=predictions[0][i,...,:n,:]
                expected[i,...,:n,:]=transform_targets(value,spec)['marker_target']
            else:
                expected[i]=transform_targets(dict(target=predictions[0][i]),spec)['target']
        y=mirrored.get('target',mirrored.get('marker_target')).to(device)
        mask=mirrored.get('target_mask')
        if mask is None:mask=mirrored['marker_valid'][...,None].expand_as(y)
        mask=mask.to(device)&torch.isfinite(y)&torch.isfinite(expected)&torch.isfinite(predictions[1])
        totals[source]['equivariance_absolute']+=(predictions[1]-expected)[mask].abs().sum().item()
        totals[source]['equivariance_count']+=mask.sum().item()
    return {s:{name+'_mae':v[name+'_absolute']/max(v[name+'_count'],1)
               for name in ('original','mirrored','equivariance')} for s,v in totals.items()}


def to_prediction_device(batch,device):
    return {k:v.to(device) for k,v in batch.items()}
