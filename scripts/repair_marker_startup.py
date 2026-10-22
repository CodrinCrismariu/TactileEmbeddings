"""Repair an untrackable startup without shifting or inventing tactile frames."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from tactilelatent.aligned_frames import target_images,sensor_indices,file_hash,json_hash,write_json
from tactilelatent.marker_targets import MarkerTracker,MarkerConfig,VERSION,detect_markers
from tactilelatent.marker_data import episode_key


def repair(manifest,root,episode):
    row=next(r for r in json.loads(Path(manifest).read_text())['episodes'] if r['source']=='freetacman' and r['episode_id']==episode)
    folder=Path(root)/row['source']/row['split']/episode_key(row)
    if (folder/'complete.json').exists():return
    config=MarkerConfig(min_markers=100);trackers=[None]*2;reference_frames=[None]*2
    # Isolated corruption can resemble dots. Require five consecutive stable
    # frames, then use the FIRST frame of that window as the reference.
    images=target_images(row,np.arange(row['frames']))
    for s in range(2):
        counts=[len(detect_markers(image,config)[0]) for image in images[s]]
        for frame in range(row['frames']-4):
            if counts[frame]<100 or any(abs(counts[f]-counts[frame])>.1*counts[frame] for f in range(frame,frame+5)):continue
            tracker=MarkerTracker(images[s][frame],config)
            if any(tracker.update(images[s][f])['valid'].mean()<.9 for f in range(frame,frame+5)):continue
            trackers[s]=MarkerTracker(images[s][frame],config);reference_frames[s]=frame;break
    if any(t is None for t in trackers):raise ValueError('A tactile surface never becomes trackable')
    reference=np.concatenate([t.reference for t in trackers]);surface=np.concatenate([np.full(len(t.reference),i,np.int64) for i,t in enumerate(trackers)])
    spacing=np.concatenate([np.full(len(t.reference),t.spacing,np.float32) for t in trackers])
    shape=np.concatenate([np.tile(np.array(t.shape[:2][::-1]),(len(t.reference),1)) for t in trackers])
    n,k=row['frames'],len(reference);delta=np.full((n,k,2),np.nan,np.float32);valid=np.zeros((n,k),bool);quality=np.zeros((n,k),np.float32)
    for start in range(0,n,16):
        for s,tracker in enumerate(trackers):
            selected=surface==s
            for j,image in enumerate(images[s][start:min(start+16,n)]):
                if start+j<reference_frames[s]:continue
                value=tracker.update(image);delta[start+j,selected]=value['displacement_px'];valid[start+j,selected]=value['valid'];quality[start+j,selected]=value['tracking_quality']
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/'tracks.tmp').open('wb') as stream:
        np.savez_compressed(stream,reference_px=reference,reference_uv=reference/(shape-1),surface_id=surface,
            spacing_px=spacing,displacement_px=delta,valid=valid,tracking_quality=quality,
            frame_indices=np.arange(n),sensor_frame_indices=sensor_indices(row,np.arange(n)))
    (folder/'tracks.tmp').replace(folder/'tracks.npz');magnitude=np.linalg.norm(delta,axis=-1)
    info=dict(version=VERSION,source=row['source'],split=row['split'],episode_id=row['episode_id'],row_sha256=json_hash(row),
        extractor_sha256=file_hash('src/tactilelatent/marker_targets.py'),loader_sha256=file_hash('src/tactilelatent/marker_data.py'),
        frames=n,markers=k,sensors=[t.metadata() for t in trackers],surfaces=2,valid_fraction=float(valid.mean()),
        min_frame_valid_fraction=float(valid.mean(1).min()),surface_valid_fraction=[float(valid[:,surface==s].mean()) for s in range(2)],
        displacement_p50_px=float(np.nanmedian(magnitude)),displacement_p95_px=float(np.nanpercentile(magnitude,95)),
        tracks_sha256=file_hash(folder/'tracks.npz'),target='signed_marker_displacement',reference_frame=None,
        reference_frames=reference_frames,reference_is_verified_unloaded=False,
        repair=dict(reason='Corrupted white/black startup frames on the first surface',script_sha256=file_hash(__file__),
                    pre_reference_observations='invalid; chronological frame axis unchanged'),
        loss_statistics=dict(square_sum=float(np.nansum((delta/spacing[None,:,None]).astype(np.float64)**2)),coordinate_count=int(valid.sum())*2))
    for s,metadata in enumerate(info['sensors']):
        metadata.update(reference_kind='first_stable_trackable_frame_after_corrupt_startup' if reference_frames[s] else 'episode_frame_zero_not_verified_unloaded',reference_frame=reference_frames[s])
    write_json(folder/'complete.json',info);print(json.dumps(dict(episode=episode,reference_frames=reference_frames,valid_fraction=info['valid_fraction'])),flush=True)


def reconcile(root):
    root=Path(root);index=json.loads((root/'index.json').read_text());retained=[]
    rows=json.loads(Path('processing/joint_alignment/manifest.json').read_text())['episodes']
    for rejected in index['rejected']:
        row=next(r for r in rows if r['source']==rejected['source'] and r['episode_id']==rejected['episode_id'])
        folder=root/row['source']/row['split']/episode_key(row)
        if not (folder/'complete.json').exists():retained.append(rejected);continue
        info=json.loads((folder/'complete.json').read_text())
        if not info.get('repair') or info['row_sha256']!=json_hash(row) or file_hash(folder/'tracks.npz')!=info['tracks_sha256']:
            raise ValueError('Repair provenance failed')
        index['episodes'].append(dict(source=row['source'],split=row['split'],episode_id=row['episode_id'],path=str(folder)))
    index['rejected']=retained;index['all_episodes_complete']=not retained
    for source in index['statistics']:
        selected=[json.loads((Path(r['path'])/'complete.json').read_text()) for r in index['episodes'] if r['source']==source and r['split']=='train']
        count=sum(r['loss_statistics']['coordinate_count'] for r in selected);square=sum(r['loss_statistics']['square_sum'] for r in selected)
        index['statistics'][source]=dict(train_coordinate_count=count,rms_spacing_units=float(np.sqrt(square/max(count,1))),valid_fraction=float(np.mean([r['valid_fraction'] for r in selected])))
    index['startup_repairs_reconciled']=True;write_json(root/'index.json',index)
    complete=json.loads((root/'complete.json').read_text());complete.update(rejected=len(retained),all_episodes_complete=not retained,statistics=index['statistics'],startup_repairs_reconciled=True);write_json(root/'complete.json',complete)
    status=json.loads((root/'status.json').read_text());status.update(stage='complete' if not retained else 'complete_with_rejected_episodes',rejected=len(retained),startup_repairs_reconciled=True);write_json(root/'status.json',status)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',default='processing/marker_displacement_v2/targets');parser.add_argument('--wait-and-reconcile',action='store_true');args=parser.parse_args()
    repair('processing/joint_alignment/manifest.json',args.root,'Slot/Slot_164')
    if args.wait_and_reconcile:
        while not (Path(args.root)/'complete.json').exists():time.sleep(10)
        reconcile(args.root)
