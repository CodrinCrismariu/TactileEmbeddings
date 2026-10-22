"""Resumable, verified, camera-only H.264 conversion and archive compaction.

Original source frame counts are retained. Tactile/force payloads are copied
unchanged when mixed archives must be repacked. Only V-JEPA camera keys move.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile
import time
import zipfile

import av
import h5py
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from tactilelatent.aligned_frames import file_hash, write_json, targets
from tactilelatent.sensor_data import ftp_group
from tactilelatent.storage import episode_key

ROOT=Path('processing/storage_v1');VIDEO=ROOT/'rgb';RECEIPTS=ROOT/'receipts'
RESERVE=80_000_000_000


def stat(path):
    s=Path(path).stat();return [s.st_size,s.st_mtime_ns]


def discover(rows):
    jobs={};maps={};drop_tar=defaultdict(set);drop_h5=defaultdict(set);drop_zip=defaultdict(set)
    for row in rows:
        if row['source']=='freetacman':continue
        ep=episode_key(row);maps[ep]=dict(source=row['source'],episode_id=row['episode_id'],views={})
        for slot,path in row['views'].items():
            if row['source']=='ftp_vla_touch':
                raw=row['raw_scene'];prefix=raw['prefix']+'/'+raw['folders'][str(slot)]+'/'
                spec=dict(kind='zip',original=raw['archive'],prefix=prefix,frames=row['frames'],rotation=raw['rotation_quarters'][str(slot)],fps=None)
                offset=0;drop_zip[raw['archive']].add(prefix)
            elif 'tar_path' in row:
                group=ftp_group(row);array=group['data/'+path]
                spec=dict(kind='zarr',original=row['tar_path'],array='data/'+path,group=row['zarr_name'],
                          archive_group=row.get('archive_group','Unit_Bimanual'),row=row,frames=len(array),fps=None)
                offset=row['target_start']
                prefix=f"{row.get('archive_group','Unit_Bimanual')}/{row['zarr_name']}/data/{path}/"
                drop_tar[row['tar_path']].add(prefix)
            elif row['source']=='opentouch':
                with h5py.File(path,'r') as f:n=len(f['data'][row['hdf5_demo']]['rgb_images_jpeg'])
                dataset='data/'+row['hdf5_demo']+'/rgb_images_jpeg'
                spec=dict(kind='h5',original=path,dataset=dataset,frames=n,fps=None)
                offset=0;drop_h5[path].add(dataset)
            else:
                with av.open(path) as f:
                    stream=f.streams.video[0];n=stream.frames;rate=float(stream.average_rate)
                    if not n or rate<=0:raise ValueError(f'Unknown source video frame contract: {path}')
                spec=dict(kind='video',original=path,frames=n,fps=rate)
                offset=row.get('video_start',0)
            identity={k:v for k,v in spec.items() if k!='row'}
            key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:24]
            spec['key']=key;spec['output']=str(VIDEO/(key+'.mp4'))
            jobs[key]=spec;maps[ep]['views'][str(slot)]=dict(job=key,frame_offset=offset)
    return jobs,maps,drop_tar,drop_h5,drop_zip


def frames(spec):
    kind=spec['kind']
    if kind=='video':
        with av.open(spec['original']) as f:
            stream=f.streams.video[0];stream.thread_type='AUTO';stream.thread_count=2
            origin=stream.start_time or 0;rate=float(stream.average_rate)
            for i,frame in enumerate(f.decode(video=0)):
                position=round((frame.pts-origin)*float(stream.time_base)*rate)
                if position!=i:raise ValueError('Source video frame timestamps are not chronological CFR')
                yield frame.reformat(width=384,height=384,format='rgb24').to_ndarray()
    elif kind=='zarr':
        array=ftp_group(spec['row'])[spec['array']]
        for start in range(0,len(array),32):
            raw=np.asarray(array[start:start+32])
            if raw.dtype!=np.uint8 or raw.ndim!=4 or raw.shape[-1]!=3:raise ValueError('Unexpected RGB array contract')
            value=torch.from_numpy(raw.copy()).permute(0,3,1,2).float()/255.
            value=F.interpolate(value,size=(384,384),mode='bilinear',align_corners=False)
            value=(value.permute(0,2,3,1).numpy()*255).round().clip(0,255).astype(np.uint8)
            yield from value
    elif kind=='h5':
        with h5py.File(spec['original'],'r') as f:
            array=f[spec['dataset']]
            for i in range(len(array)):
                with Image.open(io.BytesIO(bytes(array[i]))) as image:
                    yield np.array(image.convert('RGB').resize((384,384),Image.Resampling.BICUBIC))
    elif kind=='zip':
        with zipfile.ZipFile(spec['original']) as archive:
            for i in range(spec['frames']):
                name=spec['prefix']+f'rgb_{i}.jpg'
                with Image.open(io.BytesIO(archive.read(name))) as image:
                    value=np.asarray(image.convert('RGB').resize((384,384),Image.Resampling.BILINEAR))
                    yield np.rot90(value,spec['rotation']).copy()


def convert(spec):
    torch.set_num_threads(1)
    if shutil.disk_usage(ROOT).free<RESERVE+2_000_000_000:raise RuntimeError('80 GB working reserve reached')
    receipt=RECEIPTS/(spec['key']+'.json');path=Path(spec['output'])
    if receipt.exists():
        saved=json.loads(receipt.read_text())
        if stat(path)!=saved['stat'] or file_hash(path)!=saved['sha256']:raise ValueError('Converted output changed')
        return saved
    temporary=path.with_suffix('.partial.mp4');samples={};source_hash=hashlib.sha256();count=0
    path.parent.mkdir(parents=True,exist_ok=True);receipt.parent.mkdir(parents=True,exist_ok=True)
    with av.open(str(temporary),'w',format='mp4') as output:
        from fractions import Fraction
        rate=Fraction(spec['fps']).limit_denominator(100000) if spec['fps'] is not None else 30
        stream=output.add_stream('libx264',rate=rate)
        stream.width=384;stream.height=384;stream.pix_fmt='yuv420p'
        stream.options={'crf':'18','preset':'fast'};stream.codec_context.thread_count=2;stream.codec_context.gop_size=16
        for count,image in enumerate(frames(spec),1):
            source_hash.update(image.tobytes())
            if count-1 in {0,spec['frames']//2,spec['frames']-1}:samples[count-1]=image.copy()
            frame=av.VideoFrame.from_ndarray(image,format='rgb24')
            for packet in stream.encode(frame):output.mux(packet)
        for packet in stream.encode():output.mux(packet)
    if count!=spec['frames']:raise ValueError('Source frame count changed during conversion')
    decoded=0;errors=[]
    with av.open(str(temporary)) as f:
        for i,frame in enumerate(f.decode(video=0)):
            if (frame.width,frame.height)!=(384,384):raise ValueError('Wrong encoded dimensions')
            if i in samples:
                error=np.abs(frame.to_ndarray(format='rgb24').astype(np.float32)-samples[i]).mean();errors.append(float(error))
            decoded+=1
    if decoded!=count or max(errors)>8.:raise ValueError(f'RGB validation failed: {decoded}/{count}, MAE {errors}')
    temporary.replace(path)
    result=dict(key=spec['key'],path=str(path),sha256=file_hash(path),stat=stat(path),frames=count,
                source_rgb_sha256=source_hash.hexdigest(),fps=spec['fps'],container_index_fps=spec['fps'] or 30,
                frame_mapping='one encoded frame per source frame; no temporal resampling',sample_rgb_mae_255=errors)
    write_json(receipt,result);return result


def preserve_digest(items):
    h=hashlib.sha256()
    for name,data in items:
        h.update(name.encode());h.update(len(data).to_bytes(8,'little'));h.update(hashlib.sha256(data).digest())
    return h.hexdigest()


def compact_tar(path,index_path,prefixes):
    old=Path(path);tmp=old.with_suffix('.compact.partial.tar')
    before=hashlib.sha256();kept=0
    with tarfile.open(old,'r:') as src,tarfile.open(tmp,'w:') as dst:
        for member in src:
            if any(member.name.startswith(p) for p in prefixes):continue
            if member.isfile():
                data=src.extractfile(member).read()
                before.update(member.name.encode());before.update(hashlib.sha256(data).digest())
                dst.addfile(member,io.BytesIO(data));kept+=1
            else:dst.addfile(member)
    after=hashlib.sha256();members={}
    with tarfile.open(tmp,'r:') as src:
        for m in src:
            if m.isfile():
                data=src.extractfile(m).read();after.update(m.name.encode());after.update(hashlib.sha256(data).digest())
                members[m.name]=[m.offset_data,m.size]
    if before.digest()!=after.digest():raise ValueError('Native non-RGB tar payload changed')
    if tmp.stat().st_size>=old.stat().st_size:
        tmp.unlink();return None
    tmp.replace(old)
    previous=json.loads(Path(index_path).read_text())
    previous['members']=members;previous['signature']=[str(old.resolve()),*stat(old)]
    write_json(index_path,previous)
    return dict(kind='tar',native_payload_sha256=before.hexdigest(),retained_members=kept)


def compact_h5(path,drops):
    old=Path(path);tmp=old.with_suffix('.compact.partial.hdf5');checks=[]
    with h5py.File(old,'r') as src,h5py.File(tmp,'w') as dst:
        def copy(group,out):
            for k,v in group.attrs.items():out.attrs[k]=v
            for name,obj in group.items():
                if obj.name.lstrip('/') in drops:continue
                if isinstance(obj,h5py.Group):copy(obj,out.create_group(name))
                else:
                    group.copy(name,out,name=name)
                    # Hash every retained target/metadata dataset in bounded slices.
                    if obj.dtype.hasobject:
                        a=obj[()];b=out[name][()]
                        if repr(a)!=repr(b):raise ValueError('HDF5 metadata changed')
                    else:
                        for start in range(0,len(obj),128) if obj.shape else [0]:
                            a=obj[start:start+128] if obj.shape else obj[()]
                            b=out[name][start:start+128] if obj.shape else out[name][()]
                            if not np.array_equal(a,b,equal_nan=obj.dtype.kind in 'fc'):raise ValueError('Native HDF5 values changed')
                    checks.append(obj.name)
        copy(src,dst)
    tmp.replace(old);return dict(kind='h5',verified_retained_datasets=checks)


def compact_zip(path,prefixes):
    old=Path(path);tmp=old.with_suffix('.compact.partial.zip');hashes={}
    with zipfile.ZipFile(old) as src,zipfile.ZipFile(tmp,'w',allowZip64=True) as dst:
        for info in src.infolist():
            if any(info.filename.startswith(p) and Path(info.filename).name.startswith('rgb_') for p in prefixes):continue
            data=src.read(info);hashes[info.filename]=hashlib.sha256(data).hexdigest();dst.writestr(info,data)
    with zipfile.ZipFile(tmp) as f:
        if any(hashlib.sha256(f.read(name)).hexdigest()!=h for name,h in hashes.items()):raise ValueError('Retained ZIP data changed')
    tmp.replace(old);return dict(kind='zip',retained_members=len(hashes),retained_sha256=file_hash(old))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--sources',nargs='+');parser.add_argument('--no-compact',action='store_true')
    args=parser.parse_args();ROOT.mkdir(parents=True,exist_ok=True);VIDEO.mkdir(exist_ok=True);RECEIPTS.mkdir(exist_ok=True)
    import fcntl
    lock=(ROOT/'migration.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    payload=json.loads((ROOT/'original_manifest.json').read_text())
    rows=[r for r in payload['episodes'] if r['source']!='freetacman' and (not args.sources or r['source'] in args.sources)]
    started=time.monotonic()
    def progress(stage,**detail):
        item=dict(stage=stage,pid=__import__('os').getpid(),elapsed_seconds=time.monotonic()-started,**detail)
        write_json(ROOT/'conversion_status.json',item);print(json.dumps(item),flush=True)
    plan_path=ROOT/('jobs_'+('_'.join(sorted(args.sources)) if args.sources else 'all')+'.json')
    if plan_path.exists():
        plan=json.loads(plan_path.read_text());jobs,maps=plan['jobs'],plan['maps']
        tar_drops,h5_drops,zip_drops=plan['tar_drops'],plan['h5_drops'],plan['zip_drops']
    else:
        jobs,maps,tar_drops,h5_drops,zip_drops=discover(rows)
        write_json(plan_path,dict(jobs=jobs,maps=maps,tar_drops={k:sorted(v) for k,v in tar_drops.items()},
                                 h5_drops={k:sorted(v) for k,v in h5_drops.items()},zip_drops={k:sorted(v) for k,v in zip_drops.items()}))
    ledger=json.loads((ROOT/'registry.json').read_text());done={}
    # Publish the exact verified loader used for this migration unit. Keep the
    # original checkpoint/data fingerprints separately in legacy_signatures.
    for filename in list(ledger['implementation'])+['src/tactilelatent/joint_data.py']:
        ledger['implementation'][filename]=file_hash(filename)
    # A completed receipt remains authoritative when the source array no longer exists.
    progress('converting',jobs=len(jobs),episodes=len(rows),workers=args.workers)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(convert,spec):key for key,spec in jobs.items()}
        for future in as_completed(futures):
            key=futures[future];done[key]=future.result()
            for ep,mapping in maps.items():
                if ep in ledger['episodes'] or not all(v['job'] in done for v in mapping['views'].values()):continue
                entry={**mapping,'views':{slot:{**done[v['job']],'frame_offset':v['frame_offset']} for slot,v in mapping['views'].items()}}
                ledger['episodes'][ep]=entry
            if len(done)%10==0 or len(done)==len(jobs):
                write_json(ROOT/'registry.json',ledger)
                progress('converting',jobs_done=len(done),jobs_total=len(jobs),output_bytes=sum(x['stat'][0] for x in done.values()))
    write_json(ROOT/'registry.json',ledger)
    if not args.no_compact:
        # Only discard raw payload after all its consumers have verified outputs.
        for path in sorted({s['original'] for s in jobs.values() if s['kind']=='video'}):
            original=Path(path)
            if not original.exists():continue
            original.unlink()
            ledger['replacements'][str(original.resolve())]=dict(deleted=True,kind='scene_video',verified_outputs=[s['output'] for s in jobs.values() if s['original']==path])
            write_json(ROOT/'registry.json',ledger)
        for path,prefixes in tar_drops.items():
            if str(Path(path).resolve()) in ledger['replacements']:continue
            index=next(r['tar_index'] for r in rows if r.get('tar_path')==path)
            progress('compacting_tar',path=path)
            receipt=compact_tar(path,index,prefixes)
            if receipt:
                for p in (path,index):ledger['replacements'][str(Path(p).resolve())]=dict(**receipt,new_stat=stat(p),deleted=False)
                write_json(ROOT/'registry.json',ledger)
        for path,drops in h5_drops.items():
            if str(Path(path).resolve()) in ledger['replacements']:continue
            progress('compacting_hdf5',path=path);receipt=compact_h5(path,drops)
            ledger['replacements'][str(Path(path).resolve())]=dict(**receipt,new_stat=stat(path),deleted=False);write_json(ROOT/'registry.json',ledger)
        for path,prefixes in zip_drops.items():
            if str(Path(path).resolve()) in ledger['replacements']:continue
            progress('compacting_zip',path=path);receipt=compact_zip(path,prefixes)
            ledger['replacements'][str(Path(path).resolve())]=dict(**receipt,new_stat=stat(path),deleted=False);write_json(ROOT/'registry.json',ledger)
    progress('complete',jobs=len(jobs),episodes=len(rows),output_bytes=sum(x['stat'][0] for x in done.values()))


if __name__=='__main__':main()
