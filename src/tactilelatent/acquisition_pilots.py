"""T-Dex and HRDexDB pilots: native arrays and explicit timestamp pairing."""
import io
import json
from pathlib import Path
import pickle
import time
import zipfile

import h5py
import numpy as np
import pyarrow.parquet as pq

from .aligned_frames import file_hash, json_hash, write_json
from .acquisition_archives import RemoteRangeFile, safe_member
from .acquisition_sources import (convert_video, nearest_indices, publish, split_for,
                                  target_artifact)
from .acquisition_quality import (UnavailableTactile, timestamp_ambiguity,
                                   supervised_history_ends)
from .data_expansion import video_info


class PrimitiveNumpyUnpickler(pickle.Unpickler):
    """Only the NumPy primitives required by the released metadata arrays."""
    def find_class(self, module, name):
        allowed={('numpy','ndarray'),('numpy','dtype'),
                 ('numpy.core.multiarray','_reconstruct'),('numpy._core.multiarray','_reconstruct'),
                 ('numpy.core.multiarray','scalar'),('numpy._core.multiarray','scalar')}
        if (module,name) not in allowed:
            raise pickle.UnpicklingError(f'Forbidden metadata global: {module}.{name}')
        return super().find_class(module,name)


def primitive_pickle(path):
    if Path(path).stat().st_size>64*2**20:
        raise ValueError('Metadata object exceeds size bound')
    with Path(path).open('rb') as handle:return PrimitiveNumpyUnpickler(handle).load()


def native_numpy(path):
    """Numeric .npy without pickle, restricted reader for official dict arrays."""
    path=Path(path)
    try:return np.load(path,allow_pickle=False)
    except ValueError as error:
        if 'Object arrays' not in str(error):raise
    if path.stat().st_size>64*2**20:raise ValueError('Object metadata exceeds size bound')
    with path.open('rb') as f:
        version=np.lib.format.read_magic(f)
        if version==(1,0):shape,_,dtype=np.lib.format.read_array_header_1_0(f)
        elif version==(2,0):shape,_,dtype=np.lib.format.read_array_header_2_0(f)
        else:raise ValueError('Unsupported object NPY header')
        if not dtype.hasobject:raise ValueError('Unexpected native NPY type')
        value=PrimitiveNumpyUnpickler(f).load()
    if not isinstance(value,np.ndarray) or value.shape!=shape or value.dtype!=dtype:
        raise ValueError('Object NPY payload differs from header')
    return value


def zip_member(job,archive,member,staging=False):
    """Download just one selected member, verify CRC and persist SHA receipt."""
    safe_member(member.filename)
    destination=(job.staging if staging else job.raw)/member.filename
    receipt=job.root/'member_receipts'/(json_hash(member.filename)+'.json')
    if receipt.exists() and destination.exists():
        d=json.loads(receipt.read_text())
        if d['crc32']!=member.CRC or d['bytes']!=member.file_size or file_hash(destination)!=d['sha256']:
            raise ValueError('Selected archive member changed')
        return destination
    destination.parent.mkdir(parents=True,exist_ok=True)
    part=destination.with_name(destination.name+'.partial')
    job.check_space(member.file_size,staging=staging,force=True)
    job.report(stage='selective_archive_download',file=member.filename,file_bytes_total=member.file_size,file_bytes_done=0)
    # Resume at member granularity, CRC is checked by ZipFile at EOF. Large depth
    # members are never selected. Incomplete RGB members are overwritten safely.
    tick=time.monotonic();count=0
    with archive.open(member) as source,part.open('wb') as out:
        while True:
            data=source.read(4*2**20)
            if not data:break
            job.check_space(len(data),staging=staging)
            out.write(data);count+=len(data)
            if time.monotonic()-tick>10:
                job.report(file_bytes_done=count);tick=time.monotonic()
    if count!=member.file_size:raise ValueError('Selected member is incomplete')
    part.replace(destination)
    write_json(receipt,dict(member=member.filename,bytes=count,crc32=member.CRC,
        sha256=file_hash(destination),archive_id=job.spec['archive_id'],
        archive_index_sha256=job.spec['archive_index_sha256']))
    return destination


def tdex(job):
    rows=[];failures=[];quality=[]
    with RemoteRangeFile('https://drive.usercontent.google.com/download',job.spec['archive_bytes'],
        {'id':job.spec['archive_id'],'export':'download','confirm':'t'}) as remote, zipfile.ZipFile(remote) as archive:
        members=archive.infolist()
        index=[dict(name=m.filename,bytes=m.file_size,compressed_bytes=m.compress_size,
                    crc32=m.CRC,offset=m.header_offset) for m in members]
        if json_hash(index)!=job.spec['archive_index_sha256']:
            raise ValueError('T-Dex Drive archive differs from the pinned member index')
        write_json(job.root/'archive_inventory.json',dict(archive_id=job.spec['archive_id'],
            bytes=job.spec['archive_bytes'],index_sha256=json_hash(index),members=index))
        available={m.filename:m for m in members}
        episodes=sorted({str(Path(m.filename).parent) for m in members if m.filename.endswith('/touch_sensor_values.h5')},
                        key=lambda x:json_hash('tdex:'+x))[:job.spec['max_episodes']]
        write_json(job.root/'selection.json',dict(episodes=episodes,scope='play-data pilot',
            seed=45,depth_excluded=True,selection='deterministic hash order at episode level'))
        for k,episode in enumerate(episodes):
            try:
                identity=Path(episode).name
                sensor=zip_member(job,archive,available[episode+'/touch_sensor_values.h5'])
                metadata=zip_member(job,archive,available[episode+'/cam_0_rgb_video.metadata'])
                with h5py.File(sensor,'r') as h:
                    native=h['sensor_values'][()];sensor_time=h['timestamps'][()].astype(np.float64)
                if native.shape[1:]!=(15,16,3) or len(sensor_time)!=len(native):
                    raise ValueError('T-Dex native XELA schema differs from [15,16,3]')
                d=primitive_pickle(metadata)
                times=np.asarray(d['timestamps'],np.float64).reshape(-1)/1000.
                camera_valid,conflicting_times,timing=timestamp_ambiguity(times,sensor_time,native)
                ix,observed,error,tolerance=nearest_indices(times,sensor_time,allow_equal_queries=True)
                observed &= camera_valid & ~np.isin(sensor_time[ix],conflicting_times)
                histories=supervised_history_ends(times,observed)
                if not len(histories):raise ValueError('T-Dex has no complete supported chronological history')
                values=native[ix]
                channels=[f'released_sensor{s}_taxel{t}_axis{a}' for s in range(15) for t in range(16) for a in range(3)]
                target=target_artifact(job,identity,values,times,channels,
                    valid=observed[:,None,None,None],sensor_indices=ix,sensor_timestamps=sensor_time[ix],
                    alignment_error_seconds=error,rgb_frame_valid=camera_valid,
                    allow_equal_timestamps=True)
                output=job.root/'rgb'/(identity+'.mp4');receipt=output.with_suffix('.receipt.json')
                if receipt.exists():
                    video=json.loads(receipt.read_text())
                    if file_hash(output)!=video['sha256'] or video['frames'] not in (len(times),len(times)+1):raise ValueError('Changed T-Dex RGB')
                    job.recover_rgb_cleanup(episode+'/cam_0_rgb_video.avi',output,video,
                                            expected_frames=video['frames'])
                else:
                    raw=zip_member(job,archive,available[episode+'/cam_0_rgb_video.avi'],staging=True)
                    native_frames=video_info(raw)['frames']
                    # The official preprocessor addresses video[i] using
                    # metadata.timestamps[i]. The sampled AVI has one further
                    # unannotated trailing frame. Preserve that frame in RGB,
                    # never invent a tactile target or trim the whole video.
                    if native_frames not in (len(times),len(times)+1):
                        raise ValueError('T-Dex RGB/metadata frame difference is not the audited zero/one-frame tail')
                    video=convert_video(job,raw,output,expected=native_frames)
                rows.append(dict(episode_id=identity,frames=len(times),split=split_for('tdex:'+identity),
                    views={'0':dict(path=str(output),role='fixed_scene',frame_offset=0,frames=video['frames'],sha256=video['sha256'])},
                    targets=target,target_kind='dense_numeric',native_target_shape=[15,16,3],
                    target_units='released native XELA values; axes and physical calibration not redefined'))
                quality.append(dict(episode=identity,alignment_coverage=float(observed.mean()),
                    max_allowed_time_error_seconds=tolerance,native_shape=list(native.shape),
                    native_timestamps_monotonic=True,scene_camera_role='fixed_scene',
                    usable_complete_histories=len(histories),timing_audit=timing,
                    partial_support_policy='missing prefixes and ambiguous capture clocks masked; no timestamp edits',
                    frame_mapping='official video[i] = metadata.timestamps[i]',
                    unannotated_trailing_rgb_frames=video['frames']-len(times)))
            except Exception as error:
                from .acquisition_queue import StoragePause, retryable
                if isinstance(error,StoragePause):raise
                failures.append(dict(episode=episode,error=f'{type(error).__name__}: {error}',retryable=retryable(error)))
            job.report(stage='processing_pilot',episode=episode,completed_units=k+1,total_units=len(episodes),
                       archive_bytes_received=remote.bytes_received)
            publish(job,rows,failures,complete_pilot=False)
    write_json(job.root/'quality.json',dict(episodes=quality,failures=failures,
        source_implementation_revision=job.spec['implementation_revision'],
        timestamp_units='official metadata milliseconds converted to seconds; native sensor seconds retained'))
    publish(job,rows,failures,complete_pilot=not failures)
    if failures:
        from .acquisition_queue import QualityBlock
        if any(f.get('retryable') for f in failures):raise RuntimeError('T-Dex network failures remain; retry on the next pass')
        raise QualityBlock(f'{len(failures)} T-Dex pilot episodes excluded; see quality.json')


def select_hrdex(episodes,spec):
    selected=[]
    for embodiment in spec['embodiments']:
        groups={}
        for row in episodes:
            if row['embodiment']==embodiment and row['has_tactile'] and row['has_videos'] and row['has_raw']:
                groups.setdefault(row['object_id'],[]).append(row)
        for group in groups.values():group.sort(key=lambda r:json_hash(r['episode_id']))
        keys=sorted(groups,key=lambda x:json_hash('hrdex_object:'+x))
        count=0
        while count<spec['max_per_embodiment']:
            progress=False
            for key in keys:
                if groups[key]:
                    selected.append(groups[key].pop(0));count+=1;progress=True
                if count>=spec['max_per_embodiment']:break
            if not progress:break
    return selected


def hrdex_native(job,row):
    prefix=row['raw_path'];emb=row['embodiment']
    if emb=='inspire_f1':
        path=job.download(prefix+'/hand/right_tactile.npy')
        clock=job.download(prefix+'/hand/right_tactile_time.npy')
        objects=native_numpy(path)
        if not len(objects) or not isinstance(objects[0],dict):raise ValueError('Unexpected Inspire F1 tactile dictionaries')
        channels=list(objects[0])
        if not all(isinstance(x,dict) and list(x)==channels for x in objects):
            raise ValueError('Inspire F1 channel identities changed within episode')
        values=np.array([[x[c] for c in channels] for x in objects],np.float64)
    else:
        path=job.download(prefix+'/hand/tactile.npy')
        values=native_numpy(path)
        if not values.size:
            # Preserve the alternate force stream for audit, but do not silently
            # substitute motor force feedback for a measured tactile target.
            if emb=='inspire_dftp':job.download(prefix+'/hand/force.npy')
            raise UnavailableTactile('Published tactile.npy is empty; separate force.npy is not a tactile-array replacement')
        clock=job.download(prefix+'/hand/time.npy')
        if values.ndim!=2:raise ValueError('Unexpected HRDex native tactile array')
        channels=[f'released_tactile_channel{i}' for i in range(values.shape[1])]
    times=np.asarray(native_numpy(clock),np.float64).reshape(-1)
    if len(times)!=len(values):raise ValueError('HRDex tactile timestamps differ in length')
    return values,times,channels


def hrdexdb(job):
    for item in job.inventory()['files']:job.download(item['path'])
    episodes=pq.read_table(job.raw/'metadata/episodes.parquet').to_pylist()
    cameras=pq.read_table(job.raw/'metadata/cameras.parquet').to_pylist()
    selected=select_hrdex(episodes,job.spec)
    write_json(job.root/'selection.json',dict(episodes=selected,
        split_policy='object-disjoint hash; all embodiments/paired humans of one object share a split',
        scene_camera_role='fixed_scene',camera_id='22641005'))
    by_episode={}
    for camera in cameras:
        if camera['camera_id']=='22641005':by_episode[camera['episode_id']]=camera
    rows=[];failures=[];quality=[]
    for k,row in enumerate(selected):
        episode=row['episode_id'];identity=episode.replace('/','_')
        try:
            native,sensor_time,channels=hrdex_native(job,row)
            time_path=job.download(row['raw_path']+'/timestamps/timestamp.npy')
            id_path=job.download(row['raw_path']+'/timestamps/frame_id.npy')
            times=np.asarray(native_numpy(time_path),np.float64).reshape(-1)
            ids=np.asarray(native_numpy(id_path),np.int64).reshape(-1)
            camera=by_episode[episode]
            if len(ids)!=len(times):raise ValueError('HRDex frame-ID and timestamp rows differ')
            output=job.root/'rgb'/(identity+'.mp4');receipt=output.with_suffix('.receipt.json')
            if receipt.exists():
                video=json.loads(receipt.read_text())
                if file_hash(output)!=video['sha256']:raise ValueError('Changed HRDex RGB')
            else:
                raw=job.download(camera['video_path'],staging=True)
                # Catalog properties are measured from a canonical camera, not
                # necessarily this stream. The actual selected stream defines
                # its frame count; explicit IDs identify annotated frames.
                video=convert_video(job,raw,output,fps=float(camera['fps']))
            n=len(times);video_frames=int(video['frames'])
            mapped=ids-1;camera_valid=(mapped>=0)&(mapped<video_frames)
            mapped=np.where(camera_valid,mapped,-1)
            # Official HRDex loader uses video index = frame_id - 1. Callback
            # timestamps can arrive out of ID order, so preserve the actual map
            # rather than quietly treating array row as video frame number.
            ix,observed,error,tolerance=nearest_indices(times,sensor_time)
            histories=supervised_history_ends(times,observed&camera_valid,mapped)
            if not len(histories):raise ValueError('HRDex has no supported chronological camera/tactile history')
            target=target_artifact(job,identity,native[ix],times,channels,
                valid=observed[:,None],sensor_indices=ix,sensor_timestamps=sensor_time[ix],
                source_frame_ids=ids,rgb_frame_indices=mapped,alignment_error_seconds=error)
            job.recover_rgb_cleanup(camera['video_path'],output,video,expected_frames=video_frames)
            rows.append(dict(episode_id=identity,frames=n,split=split_for('hrdex_object:'+row['object_id']),
                views={'0':dict(path=str(output),role='fixed_scene',camera_id=camera['camera_id'],
                    frame_indices_artifact=target['path'],frame_indices_key='rgb_frame_indices',frames=video_frames,sha256=video['sha256'])},
                targets=target,target_kind='numeric',target_units='released native tactile values; physical calibration unverified',
                embodiment=row['embodiment'],object_id=row['object_id'],success=row['success'],paired_episode_id=row['paired_episode_id']))
            quality.append(dict(episode=episode,native_shape=list(native.shape),alignment_coverage=float(observed.mean()),
                max_allowed_time_error_seconds=tolerance,
                camera_frame_id_coverage=float(camera_valid.mean()),native_video_frames=video_frames,
                timestamp_rows=n,catalog_canonical_frames=int(camera['num_frames']),
                usable_complete_histories=len(histories),
                partial_support_policy='explicit frame_id - 1; unavailable camera rows remain -1 and cannot enter clips',
                nonmonotonic_frame_id_transitions=int((np.diff(ids)<=0).sum()),
                camera_mapping='official frame_id - 1, not raw timeline row',
                clip_policy='reject histories with non-increasing mapped camera frame IDs'))
        except Exception as error:
            from .acquisition_queue import StoragePause, retryable
            if isinstance(error,StoragePause):raise
            failures.append(dict(episode=episode,embodiment=row['embodiment'],error=f'{type(error).__name__}: {error}',
                retryable=retryable(error),category='unavailable_upstream_tactile' if isinstance(error,UnavailableTactile) else 'unresolved_quality'))
        job.report(stage='processing_pilot',episode=episode,completed_units=k+1,total_units=len(selected))
        publish(job,rows,failures,complete_pilot=False)
    write_json(job.root/'quality.json',dict(episodes=quality,failures=failures,physical_calibration_verified=False))
    publish(job,rows,failures,complete_pilot=not failures)
    if failures:
        from .acquisition_queue import QualityBlock, CompletedWithExclusions
        if any(f.get('retryable') for f in failures):raise RuntimeError('HRDex network failures remain; retry on the next pass')
        if all(f.get('category')=='unavailable_upstream_tactile' for f in failures):
            raise CompletedWithExclusions(f'{len(failures)} pinned recordings contain no tactile samples; verified episodes ready')
        raise QualityBlock(f'{len(failures)} HRDex episodes excluded; see quality.json; verified episodes remain available')
