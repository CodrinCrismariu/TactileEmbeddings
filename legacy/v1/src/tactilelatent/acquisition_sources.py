"""Source processors for the approved second acquisition queue.

All manifests describe original target rows and explicit RGB frame mappings.
Only complete, checked episodes enter manifests; failures retain staging data.
"""
import csv
import io
import json
import math
from pathlib import Path
import re
import tarfile
import time
import zipfile

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from .aligned_frames import file_hash, json_hash, write_json
from .data_expansion import RGBFidelityError, atomic_npz, encode_rgb, video_images, video_info


def split_for(group):
    return 'val' if int(json_hash(str(group))[:8],16) % 5 == 0 else 'train'


def nearest_indices(frame_times, sensor_times, tolerance=None, *, allow_equal_queries=False):
    """Nearest timestamp, no interpolation; observations outside support missing."""
    t = np.asarray(frame_times, np.float64).reshape(-1)
    s = np.asarray(sensor_times, np.float64).reshape(-1)
    if not len(s) or not np.isfinite(s).all() or not np.isfinite(t).all():
        raise ValueError('Empty/nonfinite timestamps')
    bad_query = np.diff(t)<0 if allow_equal_queries else np.diff(t)<=0
    if np.any(np.diff(s)<0) or np.any(bad_query):
        raise ValueError('Timestamps are not chronological')
    if tolerance is None:
        steps = np.diff(s); steps = steps[steps>0]
        if not len(steps):
            raise ValueError('Cannot establish sensor cadence')
        tolerance = 2.5*float(np.median(steps))
    right = np.searchsorted(s,t).clip(0,len(s)-1)
    left = (right-1).clip(0,len(s)-1)
    ix = np.where(np.abs(s[left]-t)<=np.abs(s[right]-t),left,right)
    error = s[ix]-t
    valid = (np.abs(error)<=tolerance) & (t>=s[0]) & (t<=s[-1])
    return ix.astype(np.int64),valid,error,float(tolerance)


def target_artifact(job, identity, values, timestamps, channels, *, allow_equal_timestamps=False, **arrays):
    """Lossless numeric targets: finite zero is valid; signed values unchanged."""
    values = np.asarray(values)
    if values.ndim<2 or not len(values) or not np.issubdtype(values.dtype,np.number):
        raise ValueError('Expected nonempty numeric native target array')
    n = len(values); times = np.asarray(timestamps,np.float64).reshape(-1)
    bad_order = np.diff(times)<0 if allow_equal_timestamps else np.diff(times)<=0
    if len(times)!=n or not np.isfinite(times).all() or np.any(bad_order):
        raise ValueError('Target time mapping is not strictly chronological')
    valid = np.isfinite(values)
    supplied = arrays.pop('valid',None)
    if supplied is not None:
        valid &= np.broadcast_to(supplied, values.shape)
    if len(channels)!=int(np.prod(values.shape[1:])):
        raise ValueError('Channel ordering does not describe native target shape')
    path = job.root/'targets'/(identity+'.npz')
    receipt = path.with_suffix('.json')
    if receipt.exists() and path.exists():
        saved=json.loads(receipt.read_text())
        if saved['sha256']!=file_hash(path) or saved['shape']!=list(values.shape) or saved['channel_order']!=channels:
            raise ValueError('Existing target artifact changed')
        return saved
    job.check_space(values.nbytes*2+8*2**20)
    atomic_npz(path, values=values, valid=valid, timestamps=times,
               frame_indices=np.arange(n,dtype=np.int64), **arrays)
    with np.load(path,allow_pickle=False) as check:
        if not np.array_equal(values,check['values'],equal_nan=True) or not np.array_equal(valid,check['valid']):
            raise ValueError('Native targets did not round-trip')
    info = dict(path=str(path), sha256=file_hash(path), frames=n, shape=list(values.shape),
        dtype=str(values.dtype), channel_order=channels,
        valid_fraction=valid.reshape(n,-1).mean(0).tolist(),
        measured_zero_count=int(((values==0)&valid).sum()), negative_count=int(((values<0)&valid).sum()))
    write_json(receipt,info)
    return info


def publish(job, rows, failures=None, **metadata):
    identities=[r['episode_id'] for r in rows]
    if len(set(identities))!=len(rows):
        raise ValueError('Duplicate episode identity')
    if job.id in ('freetacman','ftp_motiontrans'):
        raise ValueError('Excluded source')
    artifact=dict(schema='acquisition-native-v2', source=job.id,
        revision=job.spec.get('revision'), specification_sha256=json_hash(job.spec),
        episodes=rows, frames=sum(r['frames'] for r in rows), failures=failures or [],
        target_policy='native values and shapes; finite measured zeros valid; missing masked',
        visual_preprocessing=dict(size=384,codec='h264',crf=18),
        held_out_excluded=['ftp_motiontrans'], excluded_sources=['freetacman'],
        joint_training_started=False, **metadata)
    write_json(job.root/'manifest.json',artifact)
    job.report(accepted_episodes=len(rows), accepted_frames=artifact['frames'], failures=failures or [])
    return artifact


def convert_video(job, raw, output, expected=None, fps=None):
    info=video_info(raw)
    n=expected or info['frames']
    if not n:
        n=sum(1 for _ in video_images(raw))
    if expected and info['frames'] not in (0,expected):
        raise ValueError('Native RGB frame count differs from metadata')
    job.check_space(max(n*24000,64*2**20))
    job.report(stage='compressing_rgb', file=str(raw), video_frames=n,input_frames_done=0,
               pixel_format='yuv420p',fidelity_retry=None)
    started=time.monotonic()
    def tracked_images():
        tick=time.monotonic()
        for i,image in enumerate(video_images(raw)):
            if time.monotonic()-tick>10:
                job.check_space()
                job.report(stage='compressing_rgb',input_frames_done=i,video_frames=n,
                    current_conversion_elapsed_seconds=time.monotonic()-started)
                tick=time.monotonic()
            yield image
    try:
        result=encode_rgb(tracked_images(),output,n,fps or info['fps'])
    except RGBFidelityError as error:
        # Saturated fine detail can exceed the RGB gate after 4:2:0 chroma
        # subsampling. Preserve all frames, CRF18, size and the original gate;
        # retry with full-resolution chroma, rather than accepting worse RGB.
        reason=dict(original_pixel_format='yuv420p',retry_pixel_format='yuv444p',
                    sample_frame_indices=error.sample_indices,sample_rgb_mae_255=error.errors)
        job.report(stage='compressing_rgb',file=str(raw),pixel_format='yuv444p',
                   input_frames_done=0,fidelity_retry=reason)
        result=encode_rgb(tracked_images(),output,n,fps or info['fps'],pixel_format='yuv444p')
        result['fidelity_retry']=reason
        from .aligned_frames import write_json
        write_json(Path(output).with_suffix('.receipt.json'),result)
    elapsed=time.monotonic()-started
    job.report(last_conversion_seconds=elapsed,last_conversion_frames_per_second=n/max(elapsed,.001))
    job.retire_rgb(raw,output,result)
    return result


def trex_video_mapping(meta, camera, fps):
    prefix='videos/observation.images.'+camera
    chunk=int(meta[prefix+'/chunk_index']); shard=int(meta[prefix+'/file_index'])
    start=float(meta[prefix+'/from_timestamp']); end=float(meta[prefix+'/to_timestamp'])
    offset=round(start*fps); n=int(meta['length'])
    if abs(offset/fps-start)>1e-5 or abs((end-start)*fps-n)>1e-3:
        raise ValueError('T-Rex episode/video interval is not frame-exact')
    return f'{prefix}/chunk-{chunk:03d}/file-{shard:03d}.mp4',offset


def process_trex(job):
    inventory=job.inventory()
    for item in inventory['files']:
        p=item['path']
        if p in ('README.md','LICENSE','episodes_preview.parquet') or p.startswith('meta/'):
            job.download(p)
    info=json.loads((job.raw/'meta/info.json').read_text()); fps=float(info['fps'])
    episode_paths=sorted((job.raw/'meta/episodes').glob('**/*.parquet'))
    names=pq.ParquetFile(episode_paths[0]).schema_arrow.names
    cols=[c for c in names if c in ('episode_index','length','dataset_from_index','dataset_to_index','tasks')
          or any(c.startswith('videos/observation.images.'+cam+'/') for cam in job.spec['camera_keys'])]
    meta=[r for p in episode_paths for r in pq.read_table(p,columns=cols).to_pylist()]
    meta.sort(key=lambda r:r['episode_index'])
    if len({r['episode_index'] for r in meta})!=len(meta):
        raise ValueError('T-Rex metadata duplicates episodes')
    by_id={int(r['episode_index']):r for r in meta}
    channels=[f'{side}_{finger}_{axis}' for side in ('left','right')
        for finger in ('thumb','index','middle','ring','pinky') for axis in ('Fx','Fy','Fz','Mx','My','Mz')]
    # Episode boundaries may cross Parquet shards. Accumulate until the episode
    # changes; never pair rows based solely on shard offsets.
    pending=[]; previous=None; targets={}; global_next=0
    def flush():
        nonlocal pending
        if not pending:return
        values=np.concatenate([x[0] for x in pending]); times=np.concatenate([x[1] for x in pending])
        frames=np.concatenate([x[2] for x in pending]); indices=np.concatenate([x[3] for x in pending])
        record=by_id[previous]; n=int(record['length'])
        if not np.array_equal(frames,np.arange(n)) or not np.array_equal(indices,np.arange(record['dataset_from_index'],record['dataset_to_index'])):
            raise ValueError('T-Rex numeric episode mapping differs from released metadata')
        if not np.allclose(times,np.arange(n)/fps,atol=1e-3,rtol=0):
            raise ValueError('T-Rex timestamps and frame indices disagree')
        targets[previous]=target_artifact(job,f'episode_{previous:06d}',values,times,channels,
            global_indices=indices, upstream_imputation_mask_available=np.array(False))
        pending=[]
    numeric=[r for r in inventory['files'] if r['path'].startswith('data/') and r['path'].endswith('.parquet')]
    job.report(stage='native_targets',total_units=len(numeric),completed_units=0)
    for k,item in enumerate(numeric):
        path=job.download(item['path'])
        table=pq.read_table(path,columns=[job.spec['target'],'timestamp','frame_index','episode_index','index'])
        values=table[job.spec['target']].combine_chunks().flatten().to_numpy().reshape(-1,60)
        times=table['timestamp'].to_numpy(); frames=table['frame_index'].to_numpy()
        ep=table['episode_index'].to_numpy(); indices=table['index'].to_numpy()
        if not np.array_equal(indices,np.arange(global_next,global_next+len(indices))):
            raise ValueError('T-Rex data shards are not globally chronological')
        global_next+=len(indices)
        boundaries=np.r_[0,np.flatnonzero(np.diff(ep))+1,len(ep)]
        for lo,hi in zip(boundaries[:-1],boundaries[1:]):
            identity=int(ep[lo])
            if previous is not None and identity!=previous:flush()
            previous=identity
            pending.append((values[lo:hi],times[lo:hi],frames[lo:hi],indices[lo:hi]))
        job.report(stage='native_targets', completed_units=k+1,total_units=len(numeric))
    flush()
    if global_next!=int(info['total_frames']) or len(targets)!=len(meta):
        raise ValueError('T-Rex numeric release is incomplete')
    requirements={}
    for record in meta:
        for camera in job.spec['camera_keys']:
            p,offset=trex_video_mapping(record,camera,fps)
            requirements.setdefault(p,[]).append((offset,int(record['length'])))
    videos={}; rows=[]
    def available_rows():
        result=[]
        for r in meta:
            views={}; okay=True
            for slot,camera in enumerate(job.spec['camera_keys']):
                p,offset=trex_video_mapping(r,camera,fps)
                if p not in videos:okay=False;break
                views[str(slot)]=dict(path=videos[p]['path'],role=('ego','left_wrist','right_wrist')[slot],
                    frame_offset=offset,frames=int(r['length']),sha256=videos[p]['sha256'])
            if okay:
                episode=int(r['episode_index'])
                result.append(dict(episode_id=f'episode_{episode:06d}',frames=int(r['length']),
                    split=split_for('trex:'+str(episode)),views=views,targets=targets[episode],
                    target_kind='numeric',target_units=job.spec['target_units'],tasks=r.get('tasks',[])))
        return result
    job.report(total_units=len(requirements),completed_units=0)
    # Dict insertion order follows episodes and interleaves cameras. Processing
    # all head shards before all wrists would delay the first usable episode.
    for k,p in enumerate(requirements):
        out=job.root/'rgb'/Path(p).relative_to('videos')
        receipt=out.with_suffix('.receipt.json')
        if receipt.exists():
            result=json.loads(receipt.read_text())
            if file_hash(out)!=result['sha256']:raise ValueError('Changed T-Rex RGB replacement')
        else:
            raw=job.download(p,staging=True)
            result=convert_video(job,raw,out,fps=fps)
        for offset,n in requirements[p]:
            if offset<0 or offset+n>result['frames']:
                raise ValueError('T-Rex episode crosses RGB shard boundary')
        if hasattr(job, 'recover_rgb_cleanup'):
            job.recover_rgb_cleanup(p, out, result, expected_frames=result['frames'])
        videos[p]=result
        job.report(stage='rgb_and_alignment',completed_units=k+1,total_units=len(requirements))
        # Publish only synchronized complete three-view episodes, including
        # during long acquisition. No unconverted paths appear in the manifest.
        if k<10 or (k+1)%10==0 or k+1==len(requirements):
            rows=available_rows()
            publish(job,rows,upstream_imputation='release includes median taring and previous-frame blackout fills; no per-frame imputation mask supplied',
                physical_force_calibration_verified=False,complete_release=(k+1==len(requirements)))
    write_json(job.root/'quality.json',dict(episodes=len(rows),frames=global_next,channels=channels,
        finite_is_not_an_upstream_observed_mask=True,all_native_values_preserved=True,
        negative_values_preserved=True,all_three_camera_intervals_verified=True))


def parse_feel_csvs(paths):
    tables=[]; channels=[]; ids=None
    for side,path in paths:
        with Path(path).open(encoding='utf-8-sig') as f:
            reader=csv.DictReader(f)
            names=sorted((c for c in reader.fieldnames if re.fullmatch(r'sensor\d+',c)),
                         key=lambda c:int(c[6:]))
            if not names:raise ValueError('No final processed FEEL sensor channels')
            rows=list(reader)
        frame_ids=np.array([int(r['frame_idx']) for r in rows],np.int64)
        if not np.array_equal(frame_ids,np.arange(len(rows))):
            raise ValueError('FEEL aligned frame IDs contain gaps/duplicates')
        if ids is not None and not np.array_equal(ids,frame_ids):
            raise ValueError('FEEL left/right frame IDs differ')
        ids=frame_ids
        values=np.array([[float(r[c]) if r[c].strip() else np.nan for c in names] for r in rows],np.float32)
        tables.append(values); channels.extend([side+'_'+c for c in names])
    return np.concatenate(tables,axis=1), ids, channels


def feel_tar_members(archive, n):
    with tarfile.open(archive,'r:') as f:
        members=[m for m in f if m.isfile() and m.name.lower().endswith(('.jpg','.jpeg','.png'))]
    def index(m):
        digits=re.findall(r'\d+',Path(m.name).stem)
        if not digits:raise ValueError('FEEL frame lacks index')
        return int(digits[-1])
    members.sort(key=index); indices=[index(m) for m in members]
    if indices not in (list(range(n)),list(range(1,n+1))):
        raise ValueError('FEEL JPEG frame indices do not match aligned CSV')
    return members,indices[0]


def process_feel(job):
    files=[r['path'] for r in job.inventory()['files']]
    participants=sorted({str(Path(p).parent) for p in files if p.endswith('/rgb.tar')})
    rows=[]; failures=[]; quality=[]
    for k,prefix in enumerate(participants):
        identity=prefix.replace('/','_'); out=job.root/'rgb'/(identity+'.mp4')
        candidates=[p for p in files if str(Path(p).parent)==prefix and p.endswith('.csv') and '.before_' not in p]
        selected=[p for p in candidates if re.search(r'force_(left|right)_aligned\.csv$',p) or Path(p).name=='force_aligned_.csv']
        if not selected:
            failures.append(dict(episode=prefix,reason='No released final aligned sensor CSV; RGB not downloaded',
                                 category='unavailable_upstream_annotations',retryable=False))
            continue
        try:
            paths=[]
            for p in sorted(selected):
                side='left' if 'force_left' in p else 'right' if 'force_right' in p else 'released_hand'
                paths.append((side,job.download(p)))
            values,ids,channels=parse_feel_csvs(paths); n=len(values)
            target=target_artifact(job,identity,values,ids/30.,channels)
            original_base=None
            if prefix=='Kitchen/P12' and job.spec.get('reuse_p12'):
                old=Path('processing/data_expansion_v1/rgb/feel/Kitchen_P12.mp4')
                if old.with_suffix('.receipt.json').exists():
                    result=json.loads(old.with_suffix('.receipt.json').read_text())
                    if result['frames']!=n or file_hash(old)!=result['sha256']:
                        raise ValueError('Existing P12 RGB receipt differs')
                    result=dict(result,path=str(old)); original_base=0
                else:result=None
            else:result=None
            if result is None and out.with_suffix('.receipt.json').exists():
                result=json.loads(out.with_suffix('.receipt.json').read_text())
                if result['frames']!=n or file_hash(out)!=result['sha256']:raise ValueError('Changed FEEL RGB')
            if result is not None and (job.staging/(prefix+'/rgb.tar')).exists():
                # The receipt can survive a stop between conversion and unlink.
                # Recover only after native CSV chronology and RGB hash checks.
                mapping_path=out.with_suffix('.mapping.json')
                if mapping_path.exists():
                    mapping=json.loads(mapping_path.read_text())
                    if (mapping['frames']!=n or not mapping['chronological'] or
                            mapping['original_image_index_base'] not in (0,1) or
                            mapping['aligned_csv_index_base']!=0):
                        raise ValueError('FEEL saved chronology differs; retain archive')
                else:
                    _,base=feel_tar_members(job.staging/(prefix+'/rgb.tar'),n)
                    write_json(mapping_path,dict(original_image_index_base=base,
                        aligned_csv_index_base=0,frames=n,chronological=True))
                job.recover_rgb_cleanup(prefix+'/rgb.tar',out,result,expected_frames=n)
            if result is None:
                archive=job.download(prefix+'/rgb.tar',staging=True)
                members,original_base=feel_tar_members(archive,n)
                def images():
                    tick=time.monotonic()
                    with tarfile.open(archive,'r:') as f:
                        for i,member in enumerate(members):
                            if time.monotonic()-tick>10:
                                job.check_space()
                                job.report(stage='compressing_rgb',file=prefix,
                                           input_frames_done=i,video_frames=n)
                                tick=time.monotonic()
                            with Image.open(f.extractfile(member)) as im:yield np.array(im.convert('RGB'))
                job.check_space(max(n*24000,64*2**20))
                job.report(stage='compressing_rgb',file=prefix,video_frames=n,input_frames_done=0)
                try:
                    result=encode_rgb(images(),out,n)
                except RGBFidelityError as error:
                    reason=dict(pixel_format='yuv420p',sample_rgb_mae_255=error.errors,
                                sample_frame_indices=error.sample_indices)
                    job.report(stage='compressing_rgb',file=prefix,pixel_format='yuv444p',
                               input_frames_done=0,fidelity_retry=reason)
                    result=encode_rgb(images(),out,n,pixel_format='yuv444p')
                    result['fidelity_retry']=reason
                    write_json(out.with_suffix('.receipt.json'),result)
                write_json(out.with_suffix('.mapping.json'),dict(original_image_index_base=original_base,
                    aligned_csv_index_base=0,frames=n,chronological=True))
                job.retire_rgb(archive,out,result)
            view=dict(path=result['path'],role='ego',frame_offset=0,frames=n,sha256=result['sha256'])
            # Same participant in different environments stays in one split.
            participant=int(re.search(r'P(\d+)',prefix).group(1))
            rows.append(dict(episode_id=identity,frames=n,split=split_for('feel_participant:'+str(participant)),
                views={'0':view},targets=target,target_kind='numeric',environment=prefix.split('/')[0],
                participant=participant,target_units='released processed contact-estimation values, not calibrated SI force'))
            changes=np.diff(values,axis=0); paired=np.isfinite(values[:-1])&np.isfinite(values[1:])
            quality.append(dict(episode=identity,frames=n,channels=channels,valid_fraction=target['valid_fraction'],
                adjacent_change_mean=[float(np.abs(changes[:,i][paired[:,i]]).mean()) if paired[:,i].any() else None for i in range(values.shape[1])]))
        except Exception as error:
            from .acquisition_queue import StoragePause, retryable
            if isinstance(error,StoragePause):raise
            from PIL import UnidentifiedImageError
            category=('unavailable_upstream_annotations' if isinstance(error,UnicodeDecodeError) else
                      'unreadable_upstream_images' if isinstance(error,UnidentifiedImageError) else 'unresolved_quality')
            failures.append(dict(episode=prefix,reason=f'{type(error).__name__}: {error}',
                                 retryable=retryable(error),category=category))
        job.report(stage='processing_participants',completed_units=k+1,total_units=len(participants),episode=prefix)
        publish(job,rows,failures,complete_release=False)
    write_json(job.root/'quality.json',dict(episodes=quality,excluded=failures,
        force_calibration_verified=False,drift='descriptive signal changes do not establish sensor drift'))
    publish(job,rows,failures,complete_release=not failures)
    if failures:
        from .acquisition_queue import QualityBlock, CompletedWithExclusions
        if any(f.get('retryable') for f in failures):
            raise RuntimeError('FEEL network failures remain; retry on the next queue pass')
        if all(f.get('category') in ('unavailable_upstream_annotations','unreadable_upstream_images') for f in failures):
            raise CompletedWithExclusions(f'{len(failures)} unavailable pinned recordings excluded; valid participants ready')
        raise QualityBlock(f'{len(failures)} FEEL participants excluded; see quality.json; valid participants are ready')


# Imported lazily so numeric processing never imports GPU training or the
# source-specific object/archive readers unless that source is active.
def process_tdex(job):
    from .acquisition_pilots import tdex
    tdex(job)


def process_hrdexdb(job):
    from .acquisition_pilots import hrdexdb
    hrdexdb(job)


def process_tacquad(job):
    from .acquisition_archives import tacquad
    tacquad(job)


PROCESSORS={'trex':process_trex,'feel':process_feel,'tdex':process_tdex,
            'hrdexdb':process_hrdexdb,'tacquad':process_tacquad}
