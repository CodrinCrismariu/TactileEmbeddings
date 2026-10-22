"""Resumable full-clip V-JEPA cache, spatially pooled and camera-separated.

Stored features are float32 [camera_clips, 4, 768]. They are NOT individual
frame encodings. Inference uses the same bfloat16 autocast and float32 spatial
mean as live training. No camera averaging, dropout, or touch input is applied.
"""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .aligned_frames import file_hash, json_hash, manifest_signature, scene_rgb, sensor_indices, targets, write_json
from .alignment_data import AlignedClipDataset
from .vjepa_model import VJEPABackbone, video_features, verify_backbone_files

CACHE_VERSION = 'vjepa21-full-clip-pooled-camera-float32-v1'


def partition_plan(dataset):
    """Keep dataset/split boundaries; compact available cameras without padding."""
    groups = defaultdict(list)
    for ep, start in dataset.windows:
        row = dataset.episodes[ep]
        groups[(row['source'], row['split'])].append((ep, start))
    plans = []
    for (source, split), windows in sorted(groups.items()):
        episodes, ep_lookup, records, offset = [], {}, [], 0
        for ep, start in windows:
            row = dataset.episodes[ep]
            slots = sorted(map(int, row['views']))
            if not slots or any(slot not in (0, 1, 2) for slot in slots):
                raise ValueError('Invalid scene camera slots')
            if ep not in ep_lookup:
                ep_lookup[ep] = len(episodes)
                episodes.append(dict(episode_id=row['episode_id'], manifest_episode_index=ep, camera_slots=slots))
            records.append((ep_lookup[ep], int(start), offset))
            offset += len(slots)
        clips = np.asarray(records, dtype=np.int64)
        plans.append(dict(source=source, split=split, episodes=episodes, clips=clips,
                          camera_clips=offset, clips_sha256=json_hash(records)))
    return plans


def plan_metadata(plan):
    return {**{k:v for k,v in plan.items() if k != 'clips'}, 'count':len(plan['clips']),
            'feature_shape':[plan['camera_clips'],4,768], 'dtype':'float32'}


def open_partition(folder, plan):
    """Return writable array and committed prefix; incomplete tails are retried."""
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    metadata = plan_metadata(plan)
    index_path = folder/'index.json'
    if index_path.exists():
        if json.loads(index_path.read_text()) != metadata:
            raise ValueError('Cache clip/camera mapping changed')
        if not np.array_equal(np.load(folder/'clips.npy'), plan['clips']):
            raise ValueError('Corrupt cache clip index')
    else:
        temporary=folder/'clips.tmp'
        with temporary.open('wb') as stream:
            np.save(stream,plan['clips'],allow_pickle=False)
        temporary.replace(folder/'clips.npy');write_json(index_path,metadata)
    shape=tuple(metadata['feature_shape'])
    final, partial = folder/'features.npy', folder/'features.partial.npy'
    if final.exists():
        array=np.load(final,mmap_mode='r')
        if array.shape != shape or array.dtype != np.float32:
            raise ValueError('Invalid final cache shape or precision')
        return array,len(plan['clips']),True
    if partial.exists():
        array=np.load(partial,mmap_mode='r+')
        progress=json.loads((folder/'progress.json').read_text()) if (folder/'progress.json').exists() else {'completed_clips':0}
        completed=int(progress['completed_clips'])
    else:
        array=np.lib.format.open_memmap(partial,mode='w+',dtype=np.float32,shape=shape)
        completed=0
    if array.shape!=shape or array.dtype!=np.float32 or not 0<=completed<=len(plan['clips']):
        raise ValueError('Invalid partial cache')
    expected_rows=int(plan['clips'][completed,2]) if completed<len(plan['clips']) else plan['camera_clips']
    if (folder/'progress.json').exists() and json.loads((folder/'progress.json').read_text()).get('completed_camera_clips') != expected_rows:
        raise ValueError('Cache committed row count differs')
    return array,completed,False


def commit_prefix(folder, array, plan, completed):
    """Durable data first, atomic progress second; interrupted writes are ignored."""
    folder=Path(folder)
    array.flush()
    with (folder/'features.partial.npy').open('rb') as stream:
        os.fsync(stream.fileno())
    rows=int(plan['clips'][completed,2]) if completed<len(plan['clips']) else plan['camera_clips']
    write_json(folder/'progress.json',dict(completed_clips=completed,completed_camera_clips=rows))


def finalize_partition(folder, plan):
    folder=Path(folder)
    path=folder/'features.npy'
    if not path.exists():
        (folder/'features.partial.npy').replace(path)
    array=np.load(path,mmap_mode='r')
    if array.shape!=(plan['camera_clips'],4,768) or array.dtype!=np.float32:
        raise ValueError('Wrong cache shape/precision')
    for start in range(0,len(array),1024):
        if not np.isfinite(array[start:start+1024]).all():
            raise ValueError('Nonfinite cached video features')
    complete=dict(complete=True,clips=len(plan['clips']),camera_clips=plan['camera_clips'],
                  features_sha256=file_hash(path),clips_file_sha256=file_hash(folder/'clips.npy'),bytes=path.stat().st_size)
    write_json(folder/'complete.json',complete)
    return complete


class ClipImages(Dataset):
    def __init__(self, episodes, plan, start=0):
        self.episodes,self.plan,self.start=episodes,plan,start
    def __len__(self):
        return len(self.plan['clips'])-self.start
    def __getitem__(self,index):
        local=index+self.start
        ep,start,_=self.plan['clips'][local]
        info=self.plan['episodes'][int(ep)]
        row=self.episodes[info['manifest_episode_index']]
        indices=int(start)+np.arange(8)*2
        rgb,mask=scene_rgb(row,indices,size=384)
        return dict(rgb=rgb,view_mask=mask,clip_index=local)


class VJEPAFeatureCache:
    """Strict read-only cache; incomplete partitions cannot be used for training."""
    def __init__(self, root, expected_signature=None, require_complete=True):
        self.root=Path(root)
        self.index=json.loads((self.root/'index.json').read_text())
        if self.index['version']!=CACHE_VERSION:
            raise ValueError('Unsupported feature cache')
        if expected_signature is not None and self.index['signature']!=expected_signature:
            raise ValueError('Stale feature cache provenance')
        if require_complete and not (self.root/'complete.json').exists():
            raise ValueError('V-JEPA cache is not complete')
        self.partitions={}

    def _partition(self,source,split):
        key=(source,split)
        if key not in self.partitions:
            folder=self.root/source/split
            complete=json.loads((folder/'complete.json').read_text())
            meta=json.loads((folder/'index.json').read_text())
            if not complete.get('complete'):
                raise ValueError('Unfinished partition')
            if meta['source']!=source or meta['split']!=split:
                raise ValueError('Dataset/split identity mismatch')
            if file_hash(folder/'clips.npy') != complete['clips_file_sha256']:
                raise ValueError('Corrupt clip mapping')
            if (folder/'features.npy').stat().st_size != complete['bytes']:
                raise ValueError('Truncated feature file')
            clips=np.load(folder/'clips.npy',mmap_mode='r')
            values=np.load(folder/'features.npy',mmap_mode='r')
            if list(values.shape)!=meta['feature_shape'] or values.dtype!=np.float32 or len(clips)!=meta['count']:
                raise ValueError('Wrong cached feature shape')
            lookup={}
            for i,ep in enumerate(meta['episodes']):
                selected=np.flatnonzero(clips[:,0]==i)
                if ep['episode_id'] in lookup:
                    raise ValueError('Duplicate episode identifier within partition')
                lookup[ep['episode_id']]=(ep, selected, clips[selected,1])
            self.partitions[key]=(meta,clips,values,lookup)
        return self.partitions[key]

    def read(self,source,split,episode_id,frame_indices):
        indices=np.asarray(frame_indices,dtype=np.int64)
        if indices.shape!=(8,) or not np.array_equal(indices,indices[0]+np.arange(8)*2):
            raise ValueError('Cache requires exact ordered eight-frame stride-two clip')
        _,clips,values,lookup=self._partition(source,split)
        if episode_id not in lookup:
            raise KeyError('Episode absent from this source/split cache')
        ep,selected,starts=lookup[episode_id]
        position=int(np.searchsorted(starts,indices[0]))
        if position>=len(starts) or starts[position]!=indices[0]:
            raise KeyError('Full clip is not cached; do not reuse individual-frame features')
        record=clips[selected[position]]
        slots=ep['camera_slots'];offset=int(record[2])
        packed=np.array(values[offset:offset+len(slots)],copy=True)
        if not np.isfinite(packed).all():
            raise ValueError('Nonfinite cached feature')
        result=torch.zeros(4,3,768,dtype=torch.float32)
        result[:,slots]=torch.from_numpy(packed).permute(1,0,2)
        mask=torch.zeros(8,3,dtype=torch.bool);mask[:,slots]=True
        return result,mask


class CachedAlignedClips(Dataset):
    """An existing aligned split with video features replacing RGB decoding.

    It retains the original sampling/window index and tactile-code/target paths.
    Pass only source-training datasets here when fitting the joint model.
    """
    def __init__(self, aligned, feature_cache):
        self.aligned,self.feature_cache=aligned,feature_cache
        self.episodes,self.windows,self.indices_by_source=aligned.episodes,aligned.windows,aligned.indices_by_source
        if aligned.frames!=8 or aligned.stride!=2 or aligned.rgb_size!=384:
            raise ValueError('Dataset clip/preprocessing differs from feature cache')
    def __len__(self):
        return len(self.aligned)
    def __getitem__(self,index):
        episode,start=self.windows[index];row=self.episodes[episode]
        indices=start+np.arange(8)*2
        encoded,mask=self.feature_cache.read(row['source'],row['split'],row['episode_id'],indices)
        item=dict(video_features=encoded,view_mask=mask,source=row['source'],episode_id=row['episode_id'],
                  frame_indices=torch.from_numpy(indices),sensor_frame_indices=torch.from_numpy(sensor_indices(row,indices)))
        if row['source'] in self.aligned.caches:
            item['code_target'],item['code_valid']=self.aligned.caches[row['source']].read(row['episode_id'],indices)
        if self.aligned.include_targets or (self.aligned.caches and row['source'] not in self.aligned.caches):
            value=torch.from_numpy(targets(row,indices));valid=torch.isfinite(value)
            item.update(target=torch.where(valid,value,0.),target_mask=valid)
        return item


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/joint_vjepa21_pair_32d_v1.json')
    parser.add_argument('--output',default='processing/vjepa21_clip_cache_v1')
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--commit-clips',type=int,default=512)
    parser.add_argument('--max-clips',type=int,help='Bounded verification only; never marks a partial cache complete')
    args=parser.parse_args()
    if min(args.batch_size,args.commit_clips)<1 or args.workers<0:
        raise ValueError('Invalid cache batch/worker settings')
    cfg=json.loads(Path(args.config).read_text())
    torch.set_num_threads(4)
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    # Prevent two builders from writing the same memory map.
    import fcntl
    lock=(root/'build.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    write_json(root/'status.json',dict(stage='indexing',pid=os.getpid()))
    data_signature=manifest_signature(cfg['manifest'])
    audit=json.loads(Path(cfg['audit']).read_text())
    if audit.get('signature')!=data_signature or not audit.get('passed') or not audit.get('VLA_complete_pixels'):
        raise ValueError('Native chronological audit failed or changed')
    registry=json.loads(Path(cfg['teachers']).read_text())
    dataset=AlignedClipDataset(cfg['manifest'],split=None,code_cache=cfg['code_cache'],registry=registry,
        include_targets=False,data_signature=data_signature,rgb_size=384)
    plans=partition_plan(dataset)
    if {p['source'] for p in plans}!={r['source'] for r in dataset.episodes} or len({p['source'] for p in plans})!=12:
        raise ValueError('Expected all twelve datasets, including separately cached MotionTrans')
    backbone_info=verify_backbone_files(cfg['backbone_repo'],cfg['backbone_checkpoint'])
    signature=dict(data=data_signature,backbone=backbone_info,storage='float32',inference='eval-no-grad-bfloat16-autocast-float32-spatial-mean',
        camera_storage='available_views_separate',frames=8,stride=2,window_step=15,
        teacher_registry_sha256=file_hash(cfg['teachers']),
        code_cache_indices={s:file_hash(Path(cfg['code_cache'])/s/'index.json') for s in dataset.caches},
        code={name:file_hash(Path(__file__).with_name(name)) for name in ('vjepa_model.py','alignment_data.py','vjepa_cache.py')})
    index=dict(version=CACHE_VERSION,signature=signature,partitions=[dict(source=p['source'],split=p['split'],
        clips=len(p['clips']),camera_clips=p['camera_clips'],clips_sha256=p['clips_sha256']) for p in plans],
        total_clips=sum(len(p['clips']) for p in plans),total_camera_clips=sum(p['camera_clips'] for p in plans),
        feature_bytes=sum(p['camera_clips'] for p in plans)*4*768*4)
    if (root/'index.json').exists():
        if json.loads((root/'index.json').read_text())!=index:
            raise ValueError('Existing cache uses different data/code/backbone; choose a new output')
    else:
        if shutil.disk_usage(root).free < index['feature_bytes']+5_000_000_000:
            raise RuntimeError('Insufficient free disk space for complete cache plus reserve')
        write_json(root/'index.json',index)
    device=torch.device('cuda')
    backbone=VJEPABackbone(cfg['backbone_repo'],cfg['backbone_checkpoint']).to(device).eval()
    completed_before=0
    for p in plans:
        folder=root/p['source']/p['split']
        if (folder/'complete.json').exists():completed_before+=len(p['clips'])
        elif (folder/'progress.json').exists():completed_before+=json.loads((folder/'progress.json').read_text())['completed_clips']
    done=completed_before;new_clips=0;new_camera_clips=0;began=time.monotonic()
    plans.sort(key=lambda p:(p['camera_clips'],p['source'],p['split']))
    for plan in plans:
        folder=root/plan['source']/plan['split']
        array,start,finished=open_partition(folder,plan)
        if finished:
            finalize_partition(folder,plan)
            continue
        data=ClipImages(dataset.episodes,plan,start)
        loader=DataLoader(data,batch_size=args.batch_size,num_workers=args.workers,pin_memory=True)
        committed=start;completed=start
        for original in loader:
            with torch.no_grad(),torch.autocast('cuda',torch.bfloat16):
                encoded=video_features(backbone,original,device,chunk_size=2).cpu().numpy()
            for j,number in enumerate(original['clip_index'].tolist()):
                if number!=completed:raise ValueError('Cache extraction order changed')
                ep,_,offset=plan['clips'][number]
                slots=plan['episodes'][int(ep)]['camera_slots']
                packed=encoded[j][:,slots,:].transpose(1,0,2)
                if packed.shape!=(len(slots),4,768) or not np.isfinite(packed).all():
                    raise ValueError('Invalid encoder output')
                array[int(offset):int(offset)+len(slots)]=packed
                completed+=1;done+=1;new_clips+=1;new_camera_clips+=len(slots)
            if completed-committed>=args.commit_clips or completed==len(plan['clips']):
                commit_prefix(folder,array,plan,completed);committed=completed
                elapsed=time.monotonic()-began
                camera_done=0
                for p in plans:
                    progress=root/p['source']/p['split']/'progress.json'
                    if (progress.parent/'complete.json').exists():camera_done+=p['camera_clips']
                    elif progress.exists():camera_done+=json.loads(progress.read_text())['completed_camera_clips']
                status=dict(stage='extracting',pid=os.getpid(),source=plan['source'],split=plan['split'],
                    completed_clips=done,total_clips=index['total_clips'],completed_camera_clips=camera_done,
                    total_camera_clips=index['total_camera_clips'],seconds_this_session=elapsed,
                    camera_clips_per_second=new_camera_clips/max(elapsed,1e-6),
                    estimated_seconds_remaining=(index['total_camera_clips']-camera_done)*elapsed/max(new_camera_clips,1),
                    expected_feature_bytes=index['feature_bytes'])
                write_json(root/'status.json',status);print(json.dumps(status),flush=True)
            if args.max_clips and new_clips>=args.max_clips:
                commit_prefix(folder,array,plan,completed)
                write_json(root/'status.json',dict(stage='bounded_verification_complete',completed_clips=done,total_clips=index['total_clips']))
                return
        del array
        finalize_partition(folder,plan)
    # Confirm every source/split is finished and source files remained unchanged.
    if manifest_signature(cfg['manifest'])!=data_signature:
        raise ValueError('Source files changed during extraction')
    completes={f'{p["source"]}/{p["split"]}':json.loads((root/p['source']/p['split']/'complete.json').read_text()) for p in plans}
    if sum(x['clips'] for x in completes.values())!=index['total_clips']:
        raise ValueError('Incomplete cache')
    record=dict(stage='complete',complete=True,total_clips=index['total_clips'],total_camera_clips=index['total_camera_clips'],
                feature_bytes=sum(x['bytes'] for x in completes.values()),partitions=completes,signature=signature,
                seconds_this_session=time.monotonic()-began)
    write_json(root/'complete.json',record);write_json(root/'status.json',record)
    print(json.dumps({k:v for k,v in record.items() if k not in ('partitions','signature')}),flush=True)


if __name__=='__main__':
    main()
