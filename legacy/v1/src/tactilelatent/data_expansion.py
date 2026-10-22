"""Versioned acquisition and native targets for Baihu, EgoPressure and FEEL.

This module never changes historical dataset manifests or marker caches.
Only scene RGB is transcoded. Deletion requires an episode-local receipt.
"""
from contextlib import ExitStack
from fractions import Fraction
from functools import lru_cache
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tarfile
import time

import av
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from .aligned_frames import file_hash, json_hash, write_json
from .marker_targets import MarkerConfig, MarkerTracker
from .expansion_tracker import BaihuMarkerTracker

ROOT = Path('processing/data_expansion_v1')
BAIHU = Path('datasets/yxma__tactile-video-pretrain-nc/videos/baihu_vtouch')
EGO_REPO = 'eth-siplab/EgoPressure'
EGO_REV = 'cea37de99be44078f0e4b9796fd6f614d9110dc2'
FEEL_REPO = 'edessa/feel'
FEEL_REV = '0f8179649b50afcfd684c87784c67787871db361'
SURFACES = ('tactile_left_l', 'tactile_left_r', 'tactile_right_l', 'tactile_right_r')
CAMERAS = {'0': 'cam_head', '1': 'cam_hand_left', '2': 'cam_hand_right'}
RESERVE = 100 * 2**30


def setup():
    ROOT.mkdir(parents=True, exist_ok=True)


def report(job, **details):
    setup()
    row = dict(job=job, updated_at=time.time(), pid=os.getpid(),
               free_bytes=shutil.disk_usage(ROOT).free, **details)
    write_json(ROOT / f'{job}_status.json', row)
    with (ROOT / f'{job}_progress.jsonl').open('a') as f:
        f.write(json.dumps(row, allow_nan=False) + '\n')
    print(json.dumps(row, allow_nan=False), flush=True)
    return row


def reserve(additional=0):
    setup()
    if shutil.disk_usage(ROOT).free < RESERVE + additional:
        raise RuntimeError('100 GiB working reserve reached; acquisition paused')


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def atomic_npz(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.partial.npz')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **arrays)
        f.flush(); os.fsync(f.fileno())
    tmp.replace(path)


def download(repo, revision, filename, destination, expected_bytes=None):
    """Hub handles range-resume; local_dir avoids a duplicate full HF cache."""
    from huggingface_hub import hf_hub_download, get_hf_file_metadata, hf_hub_url
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=True)
    receipt = destination / '.receipts' / (json_hash(filename) + '.json')
    path = destination / filename
    if receipt.exists() and path.exists():
        saved = json.loads(receipt.read_text())
        if (saved['revision'] != revision or saved['sha256'] != file_hash(path)):
            raise ValueError(f'Download changed: {filename}')
        return path
    meta = get_hf_file_metadata(hf_hub_url(repo, filename, repo_type='dataset', revision=revision))
    reserve(meta.size or expected_bytes or 0)
    last = None
    for attempt in range(5):
        try:
            result = Path(hf_hub_download(repo, filename, repo_type='dataset', revision=revision,
                                         local_dir=destination))
            if meta.size is not None and result.stat().st_size != meta.size:
                raise ValueError('Downloaded size differs from pinned release')
            digest = file_hash(result)
            if meta.etag and len(meta.etag) == 64 and digest != meta.etag:
                raise ValueError('Downloaded SHA256 differs from Hub LFS checksum')
            receipt.parent.mkdir(parents=True, exist_ok=True)
            write_json(receipt, dict(repo=repo, revision=revision, filename=filename,
                bytes=result.stat().st_size, sha256=digest, upstream_etag=meta.etag))
            return result
        except Exception as e:
            last = e
            report('download', stage='retry', file=filename, attempt=attempt+1, error=str(e))
            time.sleep(min(2**attempt, 20))
    raise last


def video_info(path):
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        return dict(frames=s.frames, width=s.width, height=s.height, fps=float(s.average_rate))


def video_images(path):
    with av.open(str(path)) as c:
        s = c.streams.video[0]; s.thread_count = 1
        origin = s.start_time or 0; rate = float(s.average_rate)
        for i, frame in enumerate(c.decode(video=0)):
            if frame.pts is None or round((frame.pts-origin)*float(s.time_base)*rate) != i:
                raise ValueError(f'Non-chronological CFR video: {path}')
            yield frame.to_ndarray(format='rgb24')


class RGBFidelityError(ValueError):
    """Valid frame mapping, but compressed scene RGB exceeds the error limit."""
    def __init__(self, errors, indices):
        self.errors=list(errors)
        self.sample_indices=list(indices)
        super().__init__(f'Encoded RGB fidelity failed: samples={self.sample_indices}, errors={self.errors}, limit=8.0')


def encode_rgb(images, path, expected_frames, fps=30, *, pixel_format='yuv420p'):
    """Validate every encoded frame; sample fidelity before publishing."""
    if pixel_format not in ('yuv420p','yuv444p'):
        raise ValueError('Unsupported verified RGB pixel format')
    path = Path(path); receipt = path.with_suffix('.receipt.json')
    if receipt.exists():
        info = json.loads(receipt.read_text())
        if (not path.exists() or file_hash(path) != info['sha256']
                or info['frames'] != expected_frames):
            raise ValueError(f'Changed converted RGB: {path}')
        return info
    reserve(max(expected_frames*12000, 64*2**20))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.partial.mp4'); samples = {}; count = 0
    wanted = {0, expected_frames//2, expected_frames-1}
    with av.open(str(tmp), 'w', format='mp4') as out:
        stream = out.add_stream('libx264', rate=Fraction(fps).limit_denominator(100000))
        stream.width = stream.height = 384; stream.pix_fmt = pixel_format
        stream.options = {'crf': '18', 'preset': 'fast'}
        stream.codec_context.thread_count = 1; stream.codec_context.gop_size = 16
        for i, value in enumerate(images):
            image = np.asarray(Image.fromarray(value).resize((384,384), Image.Resampling.BILINEAR))
            if i in wanted: samples[i] = image.copy()
            frame = av.VideoFrame.from_ndarray(image, format='rgb24'); frame.pts = i
            for packet in stream.encode(frame): out.mux(packet)
            count += 1
        for packet in stream.encode(): out.mux(packet)
    if count != expected_frames: raise ValueError(f'RGB frame count {count} != {expected_frames}')
    errors = []; sampled_indices = []; decoded = 0
    with av.open(str(tmp)) as c:
        s = c.streams.video[0]; origin = s.start_time or 0
        for i, frame in enumerate(c.decode(video=0)):
            if (frame.width,frame.height) != (384,384) or round((frame.pts-origin)*float(s.time_base)*fps) != i:
                raise ValueError('Encoded dimensions/timestamps changed')
            if i in samples:
                errors.append(float(np.abs(frame.to_ndarray(format='rgb24').astype(np.float32)-samples[i]).mean()))
                sampled_indices.append(i)
            decoded += 1
    if decoded != count or len(errors) != len(wanted):
        raise ValueError(f'Encoded RGB validation failed: {decoded}/{count}, errors={errors}')
    if max(errors) > 8.:
        raise RGBFidelityError(errors,sampled_indices)
    tmp.replace(path)
    info = dict(path=str(path), frames=count, fps=fps, width=384, height=384,
                codec='h264', crf=18, gop=16, pixel_format=pixel_format,
                sha256=file_hash(path), bytes=path.stat().st_size,
                sample_rgb_mae_255=errors, sample_frame_indices=sampled_indices,
                max_allowed_sample_rgb_mae_255=8., frame_mapping='one encoded frame per input frame')
    write_json(receipt, info)
    return info


def discover_baihu():
    rows = []; identities = set()
    for meta in sorted(BAIHU.glob('*/*/*/metadata.json')):
        d = json.loads(meta.read_text()); folder = meta.parent
        task, platform, episode = folder.relative_to(BAIHU).parts
        identity = d['resource_info']['id']
        if identity in identities: raise ValueError('Duplicate Baihu identity')
        identities.add(identity)
        if not d['time_align_info']['aligned'] or float(d['time_align_info']['fps']) != 30:
            raise ValueError('Baihu is not synchronized at 30 fps')
        n = int(d['time_align_info']['frame_count'])
        split = 'val' if int(json_hash(identity)[:8],16) % 10 == 0 else 'train'
        views = {slot:str(ROOT/'rgb'/'baihu'/identity/(name+'.mp4'))
                 for slot,name in CAMERAS.items()
                 if name in ('cam_hand_left','cam_hand_right') or 'head' in d['data_info']['camera_info']}
        row = dict(source='baihu', episode_id='/'.join((task,platform,episode)), split=split,
            frames=n, views=views, target_kind='marker_displacement', target_shape=[12,240,240],
            target_scale=1., native_sensor_shape=[240,240], surfaces=4, task=task, platform=platform,
            resource_id=identity, original_dir=str(folder), metadata_path=str(meta),
            marker_path=str(ROOT/'markers'/identity/'tracks.npz'))
        rows.append(row)
    if not rows: raise FileNotFoundError('No local Baihu episodes')
    setup(); write_json(ROOT/'baihu_inventory.json',dict(episodes=rows, count=len(rows),
        frames=sum(r['frames'] for r in rows), source='existing local Baihu subset',
        surface_order=list(SURFACES), excluded_sources=['freetacman','ftp_motiontrans']))
    return rows


def marker_arrays(row):
    path = Path(row['marker_path']); info = json.loads(path.with_name('complete.json').read_text())
    if info['row_sha256'] != json_hash(row) or file_hash(path) != info['tracks_sha256']:
        raise ValueError('Baihu marker provenance changed')
    with np.load(path, allow_pickle=False) as f: data = {k:f[k].copy() for k in f.files}
    n,k,_ = data['displacement_px'].shape
    if n != row['frames'] or data['valid'].shape != (n,k) or set(data['surface_id']) != set(range(4)):
        raise ValueError('Baihu marker frame/surface contract changed')
    if not np.array_equal(data['frame_indices'],np.arange(n)):
        raise ValueError('Baihu marker chronology changed')
    if not np.isfinite(data['displacement_px'][data['valid']]).all():
        raise ValueError('Nonfinite valid marker')
    return data,info


def marker_batch(row, indices):
    import torch
    d,_ = marker_arrays(row); ix = np.asarray(indices,dtype=np.int64)
    if ix.ndim != 1 or not len(ix) or ix.min()<0 or ix.max()>=row['frames']:
        raise ValueError('Marker frame outside episode')
    return dict(marker_target=torch.from_numpy(d['displacement_px'][ix]/d['spacing_px'][None,:,None]),
        marker_valid=torch.from_numpy(d['valid'][ix]),
        marker_reference=torch.from_numpy(d['reference_uv'].astype(np.float32)),
        marker_surface=torch.from_numpy(d['surface_id']), marker_spacing=torch.from_numpy(d['spacing_px']))


def extract_baihu(row):
    path = Path(row['marker_path']); complete = path.with_name('complete.json')
    if complete.exists(): return marker_arrays(row)[1]
    reserve(64*2**20)
    folder = Path(row['original_dir']); n = row['frames']
    paths = [folder/(name+'.mp4') for name in SURFACES]
    for p in paths:
        if video_info(p) != dict(frames=n,width=240,height=240,fps=30.):
            raise ValueError(f'Native Baihu tactile contract changed: {p}')
    # Scan each stream independently for its first stable reference; preceding
    # frames remain missing. No reference image is persisted.
    trackers=[]; references=[]
    for p in paths:
        tracker=None
        for i,image in enumerate(video_images(p)):
            try: tracker=BaihuMarkerTracker(image); references.append(i);break
            except ValueError:
                if i>=min(n-1,119): raise
        trackers.append(tracker)
    reference=np.concatenate([t.reference for t in trackers]); k=len(reference)
    surface=np.concatenate([np.full(len(t.reference),s,np.int64) for s,t in enumerate(trackers)])
    spacing=np.concatenate([np.full(len(t.reference),t.spacing,np.float32) for t in trackers])
    displacement=np.full((n,k,2),np.nan,np.float32);valid=np.zeros((n,k),bool);quality=np.zeros((n,k),np.float32)
    offsets=np.cumsum([0]+[len(t.reference) for t in trackers])
    for s,(p,t) in enumerate(zip(paths,trackers)):
        count=0; sl=slice(offsets[s],offsets[s+1])
        for i,image in enumerate(video_images(p)):
            count+=1
            if i<references[s]:continue
            r=t.update(image);displacement[i,sl]=r['displacement_px'];valid[i,sl]=r['valid'];quality[i,sl]=r['tracking_quality']
        if count != n: raise ValueError('Baihu tactile stream ended early')
    fractions=[float(valid[:,surface==s].mean()) for s in range(4)]
    # Flag weak tracking; never clip displacement or silently certify it.
    if min(fractions)<.90:
        raise ValueError(f'Baihu tracking coverage below 90%: {fractions}; retain native videos')
    atomic_npz(path, reference_px=reference,reference_uv=reference/239.,surface_id=surface,
        spacing_px=spacing,displacement_px=displacement,valid=valid,tracking_quality=quality,
        frame_indices=np.arange(n,dtype=np.int64),sensor_frame_indices=np.arange(n,dtype=np.int64))
    normalized=displacement/spacing[None,:,None]
    info=dict(version='baihu-marker-displacement-v1',row_sha256=json_hash(row),frames=n,markers=k,
        surfaces=4,surface_order=list(SURFACES),surface_valid_fraction=fractions,valid_fraction=float(valid.mean()),
        reference_frames=references,reference_is_verified_unloaded=False,
        sensors=[t.metadata() for t in trackers],tracks_sha256=file_hash(path),
        extractor_sha256=file_hash(__file__),tracker_sha256=file_hash(Path(__file__).with_name('expansion_tracker.py')),
        association='temporal-centroid-v1',
        source_files=[dict(path=str(p),bytes=p.stat().st_size,sha256=file_hash(p)) for p in paths],
        loss_statistics=dict(square_sum=float(np.nansum(normalized.astype(np.float64)**2)),coordinate_count=int(valid.sum())*2))
    write_json(complete,info);marker_arrays(row)
    return info


def safe_delete(paths, allowed_root, ledger, reason, expected=None):
    """Write intent before unlink; append completion for restart-safe accounting."""
    ledger=Path(ledger);ledger.parent.mkdir(parents=True,exist_ok=True)
    allowed_root=Path(allowed_root).resolve(); reclaimed=0
    for value in paths:
        p=Path(value)
        if p.is_symlink() or not p.resolve().is_relative_to(allowed_root):
            raise ValueError('Deletion outside explicitly allowed dataset root')
        if not p.exists():continue
        if not p.is_file():raise ValueError('Only individually inventoried files may be deleted')
        record=dict(path=str(p),bytes=p.stat().st_size,sha256=file_hash(p),reason=reason,time=time.time())
        if expected is not None and expected.get(str(p)) != record['sha256']:
            raise ValueError(f'Original changed after verification; retain {p}')
        with ledger.open('a') as f:
            f.write(json.dumps(dict(record,event='delete_intent'))+'\n');f.flush();os.fsync(f.fileno())
        p.unlink();reclaimed+=record['bytes']
        with ledger.open('a') as f:
            f.write(json.dumps(dict(record,event='deleted'))+'\n');f.flush();os.fsync(f.fileno())
    return reclaimed


@lru_cache(maxsize=8)
def _pressure_file(path,stamp):
    pf=pq.ParquetFile(path)
    groups=[pf.read_row_group(i,columns=['frame']).column('frame').to_pylist() for i in range(pf.num_row_groups)]
    return pf,groups


@lru_cache(maxsize=4)
def _pressure_group(path,stamp,group,columns):
    pf,_=_pressure_file(path,stamp)
    return pf.read_row_group(group,columns=list(columns))


def pressure_rows(path, frame_ids, columns):
    """Bounded row-group reads joined on explicit release frame identifiers."""
    p=Path(path);s=p.stat();stamp=(s.st_size,s.st_mtime_ns)
    wanted=set(map(int,frame_ids));found={};pf,groups=_pressure_file(str(p),stamp)
    for group,ids in enumerate(groups):
        if not wanted.intersection(ids):continue
        table=_pressure_group(str(p),stamp,group,tuple(columns))
        for position,frame in enumerate(ids):
            frame=int(frame)
            if frame in wanted:
                if frame in found:raise ValueError('Duplicate explicit pressure frame ID')
                row={'frame':frame}
                for name in columns:
                    cell=table.column(name)[position]
                    row[name]=(None if not cell.is_valid else cell.values.to_numpy(zero_copy_only=False)
                               if hasattr(cell,'values') else cell.as_py())
                found[frame]=row
    return [found.get(int(i)) for i in frame_ids]


def pressure_counts(row, indices):
    ids=np.asarray(row['source_frame_ids'],dtype=np.int64)[np.asarray(indices,dtype=np.int64)]
    records=pressure_rows(row['target_path'],ids,['pressure_map','pressure_map_range'])
    values=np.full((len(ids),1,224,224),np.nan,np.float32)
    for i,r in enumerate(records):
        if r is None or r['pressure_map'] is None or r['pressure_map_range'] is None:continue
        scale=np.asarray(r['pressure_map_range'],np.float32)
        if scale.shape!=(2,) or not np.isfinite(scale).all() or scale[1]<0:continue
        p=np.asarray(r['pressure_map'],np.float32).reshape(224,224)*scale[1]
        # Negative interpolation artifacts are missing, never clipped to zero.
        values[i,0]=np.where(np.isfinite(p)&(p>=0),p,np.nan)
    return values


def normalize_pressure(value, scale):
    if not np.isfinite(scale) or scale<=0:raise ValueError('Invalid training pressure scale')
    return value/(value+scale)


def denormalize_pressure(value, scale):
    value=np.asarray(value)
    return np.divide(scale*value,1-value,out=np.full_like(value,np.nan,dtype=np.float32),where=(value>=0)&(value<1))


def pressure_targets(row, indices):
    cfg=json.loads(Path(row['pressure_normalization']).read_text())
    return normalize_pressure(pressure_counts(row,indices),cfg['scale_counts'])


def acquire_egopressure():
    inventory=json.loads(Path('processing/research/egopressure/release_inventory.json').read_text())
    if inventory['revision']!=EGO_REV:raise ValueError('EgoPressure release inventory differs')
    selected=[f for f in inventory['selected_files'] if f['path'].endswith('/cam-d.color.parquet')]
    dest=Path('datasets/eth-siplab__EgoPressure');rows=[];began=time.monotonic();downloaded=0
    for number,entry in enumerate(selected):
        rgb=entry['path'];sequence=rgb.rsplit('/',1)[0];participant=sequence.split('/')[1];name=sequence.split('/')[-1]
        out=ROOT/'rgb'/'egopressure'/participant/(name+'.mp4')
        report('egopressure',stage='acquiring_sequence',sequence=name,completed_episodes=number,total_episodes=len(selected),
            elapsed_seconds=time.monotonic()-began,bytes_acquired_this_run=downloaded)
        paths={}
        for modality in ('pressure.parquet','annotation.parquet'):
            paths[modality]=download(EGO_REPO,EGO_REV,sequence+'/'+modality,dest)
        config=download(EGO_REPO,EGO_REV,f'configs/{participant}/{name}.json',dest)
        pf=None
        if (out.with_suffix('.receipt.json').exists()
                and 'source_frame_ids' in json.loads(out.with_suffix('.receipt.json').read_text())):
            info=json.loads(out.with_suffix('.receipt.json').read_text())
            if file_hash(out)!=info['sha256']:raise ValueError('Converted EgoPressure RGB changed')
            ids=info['source_frame_ids'];n=len(ids)
        else:
            p=download(EGO_REPO,EGO_REV,rgb,dest,entry['bytes']);downloaded+=entry['bytes'];pf=pq.ParquetFile(p)
            table=pf.read(columns=['frame']);ids=list(map(int,table.column('frame').to_pylist()))
            if ids!=sorted(set(ids)):raise ValueError('RGB frame IDs not strictly chronological')
            def images():
                for batch in pf.iter_batches(batch_size=32,columns=['image']):
                    for cell in batch.column(0).to_pylist():
                        with Image.open(io.BytesIO(cell['bytes'])) as im:yield np.array(im.convert('RGB'))
            n=len(ids);info=encode_rgb(images(),out,n)
            info.update(source_frame_ids=ids,source_rgb_sha256=file_hash(p),source_revision=EGO_REV)
            write_json(out.with_suffix('.receipt.json'),info)
            safe_delete([p],dest,ROOT/'egopressure_deletions.jsonl','verified 384 RGB replacement')
        meta=json.loads(config.read_text());side=meta.get('participant',{}).get('hand_side',name.rsplit('_',1)[-1])
        row=dict(source='egopressure',episode_id=f'{participant}/{name}',participant=participant,hand_side=side,
            split='train' if int(participant.split('_')[1])<=18 else 'val',frames=n,source_frame_ids=ids,
            views={'0':str(out)},target_path=str(paths['pressure.parquet']),annotation_path=str(paths['annotation.parquet']),
            config_path=str(config),target_kind='pressure_grid',target_shape=[1,224,224],target_scale=1.,
            pressure_normalization=str(ROOT/'pressure_normalization.json'))
        rows.append(row);write_json(ROOT/'egopressure_manifest.json',dict(episodes=rows,complete=len(rows)==len(selected),revision=EGO_REV))
    # A deterministic, bounded sample of training episodes/frames estimates the
    # scale; validation participants never contribute. No pressure is clipped.
    positive=[]
    for row in rows:
        if row['split']!='train':continue
        p=pressure_counts(row,np.unique(np.linspace(0,row['frames']-1,8,dtype=int)))
        x=p[np.isfinite(p)&(p>0)]
        if len(x):positive.append(x[::max(1,len(x)//512)])
    if not positive:raise ValueError('No positive training pressure measurements')
    scale=float(np.quantile(np.concatenate(positive),.99))
    write_json(ROOT/'pressure_normalization.json',dict(scale_counts=scale,method='p/(p+s); s=train-positive-p99',
        normalization_split='train',participants=[f'p_{i:03}' for i in range(1,19)],revision=EGO_REV,
        units='Sensel counts',counts_per_newton=1736.,cell_pitch_m=.00125))
    report('egopressure',stage='complete',episodes=len(rows),frames=sum(r['frames'] for r in rows),elapsed_seconds=time.monotonic()-began)
    return rows


def acquire_feel():
    import csv
    dest=Path('datasets/edessa__feel');prefix='Kitchen/P12/'
    report('feel',stage='downloading_pilot',participant='Kitchen/P12')
    csvs=[download(FEEL_REPO,FEEL_REV,prefix+f'force_{side}_aligned.csv',dest) for side in ('left','right')]
    import_rows=[]
    for p in csvs:
        with p.open() as f:import_rows.append(list(csv.DictReader(f)))
    n=len(import_rows[0])
    if len(import_rows[1])!=n:raise ValueError('FEEL hand CSV lengths differ')
    out=ROOT/'rgb'/'feel'/'Kitchen_P12.mp4';frame_ids=np.arange(n)
    labels=np.full((n,12),np.nan,np.float32)
    for hand,rows in enumerate(import_rows):
        for i,r in enumerate(rows):
            if int(r['frame_idx'])!=i:raise ValueError('FEEL frame_idx is not chronological')
            for sensor in range(6):
                v=r[f'sensor{sensor}']
                labels[i,hand*6+sensor]=float(v) if v else np.nan
    if not out.with_suffix('.receipt.json').exists():
        archive=download(FEEL_REPO,FEEL_REV,prefix+'rgb.tar',dest)
        # Read headers first; natural numeric frame order, not lexical JPEG order.
        with tarfile.open(archive,'r:') as f:
            members=[m for m in f if m.isfile() and m.name.lower().endswith(('.jpg','.jpeg','.png'))]
        import re
        def index(m):
            digits=re.findall(r'\d+',Path(m.name).stem)
            if not digits:raise ValueError('FEEL frame name lacks explicit index')
            return int(digits[-1])
        members.sort(key=index);indices=[index(m) for m in members]
        if len(members)!=n or indices not in (list(range(n)),list(range(1,n+1))):
            raise ValueError('FEEL archive frames do not match aligned CSV IDs; retain archive for investigation')
        def images():
            with tarfile.open(archive,'r:') as f:
                for m in members:
                    with Image.open(f.extractfile(m)) as im:yield np.array(im.convert('RGB'))
        encode_rgb(images(),out,n)
        safe_delete([archive],dest,ROOT/'feel_deletions.jsonl','verified 384 RGB replacement')
    atomic_npz(ROOT/'feel_pilot_targets.npz',values=labels,valid=np.isfinite(labels),frame_indices=frame_ids)
    write_json(ROOT/'feel_pilot.json',dict(stage='pilot_only',revision=FEEL_REV,rgb=str(out),frames=n,
        targets=str(ROOT/'feel_pilot_targets.npz'),channel_order=[f'{s}_sensor{i}' for s in ('left','right') for i in range(6)],
        valid_fraction=np.isfinite(labels).mean(0).tolist(),units='released processed sensor values; not calibrated force',
        missing='NaN, never zero',excluded_from_joint=True))
    report('feel',stage='complete',frames=n,pilot_only=True,valid_fraction=np.isfinite(labels).mean(0).tolist())
    audit_feel()


def audit_feel():
    """Descriptive signal checks; block trends do not establish sensor drift."""
    meta=json.loads((ROOT/'feel_pilot.json').read_text())
    with np.load(meta['targets']) as f:values=f['values'];valid=f['valid'];ids=f['frame_indices']
    if not np.array_equal(ids,np.arange(meta['frames'])) or values.shape!=(meta['frames'],12):
        raise ValueError('FEEL pilot frame/channel contract differs')
    channels=[]
    for channel,name in enumerate(meta['channel_order']):
        x=values[:,channel];mask=valid[:,channel]&np.isfinite(x);a=x[mask]
        gaps=np.flatnonzero(np.diff(np.r_[False,~mask,False]))
        consecutive=mask[1:]&mask[:-1];delta=np.diff(x)[consecutive]
        blocks=[]
        for block in np.array_split(np.arange(len(x)),10):
            b=x[block][mask[block]]
            blocks.append(dict(first_frame=int(block[0]),last_frame=int(block[-1]),
                median=float(np.median(b)) if len(b) else None,valid_fraction=float(mask[block].mean())))
        channels.append(dict(channel=name,valid_fraction=float(mask.mean()),
            zero_fraction=float((a==0).mean()) if len(a) else None,
            quantiles=np.quantile(a,[0,.01,.5,.99,1]).tolist() if len(a) else None,
            standard_deviation=float(a.std()) if len(a) else None,
            adjacent_change_median=float(np.median(np.abs(delta))) if len(delta) else None,
            adjacent_change_p99=float(np.quantile(np.abs(delta),.99)) if len(delta) else None,
            longest_missing_run_frames=int((gaps[1::2]-gaps[::2]).max(initial=0)),time_blocks=blocks))
    write_json(ROOT/'feel_quality.json',dict(revision=meta['revision'],frames=meta['frames'],
        channels=channels,frame_alignment='explicit contiguous frame_idx matches every encoded RGB frame',
        drift_interpretation='block medians are descriptive only; activity changes confound sensor drift',
        physical_force_calibration_verified=False,excluded_from_joint=True))
    return channels
