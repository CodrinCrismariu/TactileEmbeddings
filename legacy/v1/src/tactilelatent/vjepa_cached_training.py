"""Validated cached-video inputs and explicit migration of the stopped pair."""
import json
from pathlib import Path

import numpy as np
import torch

from .aligned_frames import file_hash
from .vjepa_cache import VJEPAFeatureCache, CachedAlignedClips
from .vjepa_model import verify_backbone_files, clip_view_mask, video_features

# Only the deployed trainer before cached inputs may migrate to this path.
LEGACY_TRAINER_SHA256 = 'ebadb6dbd275740904e6242faf86ac2394dde15db562f428e0f000976d2669e0'


def bind_feature_cache(cfg, train, val, provenance):
    root = Path(cfg['feature_cache'])
    cache = VJEPAFeatureCache(root)
    complete = json.loads((root/'complete.json').read_text())
    signature = cache.index['signature']
    backbone_info = verify_backbone_files(cfg['backbone_repo'], cfg['backbone_checkpoint'])
    expected_code = {name: file_hash(Path(__file__).with_name(name)) for name in
                     ('vjepa_model.py', 'alignment_data.py', 'vjepa_cache.py')}
    if (not complete.get('complete') or complete.get('signature') != signature
            or signature.get('data') != provenance['data'] or signature.get('backbone') != backbone_info
            or signature.get('code') != expected_code
            or signature.get('storage') != 'float32'
            or signature.get('camera_storage') != 'available_views_separate'
            or signature.get('inference') != 'eval-no-grad-bfloat16-autocast-float32-spatial-mean'
            or (signature.get('frames'), signature.get('stride'), signature.get('window_step')) != (8,2,15)
            or signature.get('teacher_registry_sha256') != file_hash(cfg['teachers'])
            or complete.get('total_clips') != cache.index['total_clips']
            or complete.get('total_camera_clips') != cache.index['total_camera_clips']):
        raise ValueError('Completed clip cache differs from training data/backbone/code contract')
    for source in provenance['code_caches']:
        actual = file_hash(Path(cfg['code_cache'])/source/'index.json')
        if signature['code_cache_indices'].get(source) != actual:
            raise ValueError('Clip cache was built with different sensor-code eligibility')
    sources = set(train.indices_by_source)
    if 'ftp_motiontrans' in sources or set(val.indices_by_source) != sources or len(sources) != 11:
        raise ValueError('Cached source training must exclude MotionTrans and retain eleven sources')
    partition_counts = {}
    for split, dataset in [('train',train), ('val',val)]:
        for source in sorted(sources):
            key = f'{source}/{split}'
            folder = root/source/split
            recorded = complete['partitions'].get(key)
            if recorded is None or recorded != json.loads((folder/'complete.json').read_text()):
                raise ValueError('Partition completion metadata differs')
            # One startup checksum pass; training reads memory-mapped arrays.
            if file_hash(folder/'features.npy') != recorded['features_sha256']:
                raise ValueError('Corrupt cached video features')
            meta, clips, values, lookup = cache._partition(source,split)
            expected = set()
            for index in dataset.indices_by_source[source]:
                ep, start = dataset.windows[index]
                row = dataset.episodes[ep]
                if row['split'] != split or row['source'] != source:
                    raise ValueError('Wrong source/split in training windows')
                expected.add((row['episode_id'],int(start)))
                if row['episode_id'] not in lookup or lookup[row['episode_id']][0]['camera_slots'] != sorted(map(int,row['views'])):
                    raise ValueError('Cached camera roles differ from chronological data')
            actual = {(meta['episodes'][int(ep)]['episode_id'],int(start)) for ep,start,_ in clips}
            if expected != actual or len(expected) != len(clips):
                raise ValueError('Cached window coverage differs from source dataset')
            partition_counts[key] = len(clips)
    info = dict(root=str(root), index_sha256=file_hash(root/'index.json'),
                complete_sha256=file_hash(root/'complete.json'), signature=signature,
                source_partition_clips=partition_counts, verification='full_source_partition_sha256_and_window_mapping')
    # Avoid inheriting opened mmap objects into workers; each opens its own readers.
    cache.partitions.clear()
    return CachedAlignedClips(train,cache), CachedAlignedClips(val,cache), backbone_info, info


def encoded_inputs(backbone, batch, device, chunk_size=2):
    if 'video_features' in batch:
        if backbone is not None or 'rgb' in batch:
            raise ValueError('Cached training must not instantiate a backbone or load RGB')
        value = batch['video_features']
        if (value.dtype != torch.float32 or value.ndim != 4
                or tuple(value.shape[1:]) != (4,3,768) or not torch.isfinite(value).all()):
            raise ValueError('Invalid cached float32 video feature batch')
        clip_view_mask(batch['view_mask'])
        if len(value) != len(batch['view_mask']):
            raise ValueError('Cached clip/mask batch mismatch')
        return value.detach().to(device,non_blocking=True)
    if backbone is None:
        raise ValueError('Missing cached full-clip features; live fallback is forbidden')
    return video_features(backbone,batch,device,chunk_size)


def check_resume_contract(saved, cfg, provenance, adopt_feature_cache=False):
    """Allow precisely a live->cache input change; preserve optimizer/schedule."""
    if saved['config'] == cfg and saved['provenance'] == provenance:
        return False
    if not adopt_feature_cache or not cfg.get('feature_cache') or saved['config'].get('feature_cache'):
        raise ValueError('Resume configuration/data/code/backbone provenance differs')
    previous, current = saved['provenance'], provenance
    if (saved['config'] != {k:v for k,v in cfg.items() if k != 'feature_cache'}
            or {k:v for k,v in previous.items() if k != 'implementation'} !=
               {k:v for k,v in current.items() if k not in ('implementation','feature_cache')}
            or previous['implementation'].get('train_vjepa.py') != LEGACY_TRAINER_SHA256
            or {k:v for k,v in previous['implementation'].items() if k != 'train_vjepa.py'} !=
               {k:v for k,v in current['implementation'].items() if k not in ('train_vjepa.py','vjepa_cached_training.py')}
            or 'feature_cache' not in current):
        raise ValueError('Cache adoption cannot change models, targets, sampling or optimization settings')
    return True
