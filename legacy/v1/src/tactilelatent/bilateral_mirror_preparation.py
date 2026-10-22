"""Repair the two remaining mirror teachers; never start joint training."""
from contextlib import contextmanager
import copy
import fcntl
import json
from pathlib import Path

import torch

from .aligned_frames import file_hash,json_hash,write_json
from .bilateral_autoencoder import BilateralVectorAutoencoder,BilateralMarkerAutoencoder,load_bilateral_teacher
from .bimanual_mirror import load_policy,transform_targets,verify_evidence
from . import mirror_balanced_training as balanced
from . import mirror_preparation as prep


@contextmanager
def runtime():
    previous=prep.load_model,prep.verify_codes,balanced.load_gel_model,balanced.reflect_batch
    original_load,_,gel_load,reflect=previous
    def load(info,device):
        if info.get('bilateral'):return load_bilateral_teacher(info['checkpoint'],device)
        # The original reference and fresh bilateral parent share a source but
        # must retain different architectures. Inspect only the small header.
        state=torch.load(info['checkpoint'],map_location='cpu',weights_only=False,mmap=True)
        if state.get('model_type','').startswith('bilateral_'):return load_bilateral_teacher(info['checkpoint'],device)
        return gel_load(info,device) if info.get('representation')=='marker_displacement' else original_load(info,device)
    def reflect_batch(batch,spec):
        return transform_targets(batch,spec) if spec['kind']=='taxels' else reflect(batch,spec)
    from .mirror_cache_verification import verify_codes
    prep.load_model=load;prep.verify_codes=verify_codes;balanced.load_gel_model=load;balanced.reflect_batch=reflect_batch
    try:yield
    finally:prep.load_model,prep.verify_codes,balanced.load_gel_model,balanced.reflect_batch=previous


def run(config_path='configs/mirror_adapted_teachers_v4.json'):
    cfg=json.loads(Path(config_path).read_text());root=Path(cfg['output']);root.mkdir(parents=True,exist_ok=True)
    lock=(root/'prepare.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(4);device=torch.device('cuda')
    if cfg['training_scope']!='sensor autoencoders only; joint training prohibited':raise ValueError('Unexpected training scope')
    policy=load_policy(cfg['policy']);audit=verify_evidence(policy)
    write_json(root/'policy_audit.json',dict(policy_sha256=json_hash(policy),sources=audit))
    config=dict(cfg,bilateral_model_sha256=file_hash(Path(__file__).with_name('bilateral_autoencoder.py')),
        extension_sha256=file_hash(__file__))
    results={}
    old=json.loads(Path('processing/bimanual_mirror_v3/ready.json').read_text())
    old_policy=load_policy('configs/bimanual_mirror_policy_v2.json')
    for source in ('egotouch','baihu'):
        entry=copy.deepcopy(old['sources'][source])
        if not entry['ready'] or not entry['parity_achieved']:raise ValueError('An inherited teacher lacks mirror parity')
        if policy['sources'][source]!=old_policy['sources'][source]:raise ValueError('An inherited mapping changed')
        entry.update(artifact_policy_path='configs/bimanual_mirror_policy_v2.json',artifact_policy_sha256=json_hash(old_policy))
        results[source]=entry
    results['ftp_rdp_bimanual']=dict(ready=True,numeric=True,reason='Two direct scalar targets; no decoder needed.',policy_sha256=json_hash(policy))
    for source in cfg['sources']:results[source]=dict(ready=False,stage='queued_teacher_adaptation')
    def publish():
        ready=dict(complete=all(e.get('ready') for e in results.values()),
            complete_for_verified_sources=all(e.get('ready') for e in results.values()),
            policy_path=str(Path(cfg['policy']).resolve()),policy_sha256=json_hash(policy),sources=results,
            implementation={str(Path(__file__).relative_to(Path.cwd())):file_hash(__file__),
                'src/tactilelatent/bilateral_autoencoder.py':file_hash(Path(__file__).with_name('bilateral_autoencoder.py'))},
            joint_training_started=False)
        write_json(root/'ready.json',ready);return ready
    publish()
    with runtime():
        for source in cfg['sources']:
            try:
                rows,info=prep.source_inputs(source);torch.manual_seed(cfg['seed'])
                model=(BilateralVectorAutoencoder(latent_dim=info['code_dim']) if source=='opengraph' else
                    BilateralMarkerAutoencoder(**{k:v for k,v in info['model_config'].items() if k!='architecture'}))
                folder=root/'initialization';folder.mkdir(exist_ok=True);path=folder/(source+'.pt')
                if not path.exists():
                    from .train_alignment import atomic_checkpoint
                    atomic_checkpoint(path,dict(model_type=model.model_type,model_config=model.config,model=model.state_dict(),
                        source=source,bilateral=True,epoch=-1,config=dict(contact_weight=3.,loss_threshold=.1,mae_weight=.5)))
                parent=dict(checkpoint=str(path.resolve()),sha256=file_hash(path),original_checkpoint=info['checkpoint'])
                del model
                best=balanced.train_source(source,rows,info,parent,policy,root,config,device)
                if not best['accepted'] or not best['parity_achieved']:
                    results[source]=dict(ready=False,stage='quality_not_met',quality=str(root/'teachers'/source/'quality.json'))
                    publish();continue
                teacher={**best['teacher'],'bilateral':True};best['teacher']=teacher
                write_json(root/'teachers'/source/'quality.json',best)
                balanced.progress(root,stage='caching_codes',source=source)
                index=prep.cache_codes(source,rows,teacher,policy,root,device)
                path=root/'codes'/source/'index.json'
                results[source]=dict(ready=True,numeric=False,teacher=teacher,statistics=index['statistics'],
                    code_index=str(path.resolve()),code_index_sha256=file_hash(path),
                    teacher_quality=str((root/'teachers'/source/'quality.json').resolve()),parity_achieved=True)
            except Exception as error:
                import traceback
                traceback.print_exc()
                results[source]=dict(ready=False,stage='failed',reason=f'{type(error).__name__}: {error}')
            publish()
    ready=publish();balanced.progress(root,stage='complete' if ready['complete'] else 'complete_with_explicit_blocks',
        sources=results,joint_training_started=False)
    return ready
