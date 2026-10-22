"""Native marker geometry and numeric displacement errors for viewer/audits."""
import base64
from functools import lru_cache
from io import BytesIO
import json
from pathlib import Path
import threading

import numpy as np
from PIL import Image,ImageDraw

from .marker_data import MarkerCache,episode_key
from .sensor_data import SENSORS

READ_LOCK=threading.RLock()


@lru_cache(maxsize=4)
def canonical_rows(manifest):
    return {(r['source'],r['episode_id']):r for r in json.loads(Path(manifest).read_text())['episodes']}


@lru_cache(maxsize=2)
def cache(root):
    return MarkerCache(root)


def inputs(cfg,row,frame):
    canonical=canonical_rows(cfg['manifest'])[(row['source'],row['episode_id'])]
    if canonical['frames']!=row['frames']:
        raise ValueError('Viewer and marker episode lengths differ')
    with READ_LOCK:
        reader=cache(cfg['marker_targets']);batch=reader.read(canonical,[frame])
        reference=reader.loaded[episode_key(canonical)]['reference_px']
    return canonical,batch,reference


def png(value):
    buffer=BytesIO();Image.fromarray(value).save(buffer,'PNG')
    return 'data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode()


def overlay(image,reference,delta,valid,color,gain=4):
    result=Image.fromarray(image.copy());draw=ImageDraw.Draw(result)
    for p,d,ok in zip(reference,delta,valid):
        x,y=map(float,p)
        if not ok:
            draw.line((x-3,y-3,x+3,y+3),fill=(180,180,180),width=1)
            draw.line((x-3,y+3,x+3,y-3),fill=(180,180,180),width=1)
            continue
        end=p+gain*d
        draw.ellipse((x-2,y-2,x+2,y+2),outline=(255,255,255),width=1)
        draw.line((x,y,float(end[0]),float(end[1])),fill=color,width=2)
        draw.ellipse((end[0]-1,end[1]-1,end[0]+1,end[1]+1),fill=color)
    return np.asarray(result)


def payload(cfg,row,frame,prediction=None):
    canonical,batch,reference=inputs(cfg,row,frame)
    spacing=batch['marker_spacing'].numpy();surface=batch['marker_surface'].numpy()
    target=batch['marker_target'][0].numpy()*spacing[:,None]
    valid=batch['marker_valid'][0].numpy() & np.isfinite(target).all(-1)
    if prediction is not None:
        prediction=np.asarray(prediction,dtype=np.float32)*spacing[:,None]
        if prediction.shape!=target.shape or not np.isfinite(prediction).all():
            raise ValueError('Invalid marker prediction')
    error=np.abs(prediction-target) if prediction is not None else None
    info=json.loads((Path(cfg['marker_targets'])/canonical['source']/canonical['split']/episode_key(canonical)/'complete.json').read_text())
    reference_frames=info.get('reference_frames',[0]*len(info['sensors']))
    panels=[];hand_mae=[]
    if len(info['sensors']) != len(SENSORS[row['source']][2]):
        raise ValueError('Marker surface labels do not match the retained sensor identities')
    for sid,sensor in enumerate(info['sensors']):
        selected=surface==sid;ok=valid[selected];p=reference[selected]
        h,w=sensor['shape'][:2]
        background=np.full((h,w,3),(20,25,31),dtype=np.uint8)
        measured=overlay(background,p,target[selected],ok,(30,235,255),gain=1)
        predicted=None if prediction is None else overlay(background,p,prediction[selected],np.ones(selected.sum(),bool),(255,175,30),gain=1)
        heat=Image.new('I',(w,h));draw=ImageDraw.Draw(heat)
        if prediction is not None:
            endpoint=np.linalg.norm(prediction[selected]-np.nan_to_num(target[selected]),axis=-1)/spacing[selected]
            for xy,e,seen in zip(p,endpoint,ok):
                if seen:draw.ellipse((xy[0]-4,xy[1]-4,xy[0]+4,xy[1]+4),fill=int(np.clip(e,0,1)*65535))
        mae=float(error[selected][ok].mean()) if error is not None and ok.any() else None
        hand_mae.append(mae)
        panels.append(dict(label=SENSORS[row['source']][2][sid],original=png(measured),
            reconstruction=png(predicted) if predicted is not None else None,
            error_u16=base64.b64encode(np.asarray(heat,dtype='<u2').tobytes()).decode() if prediction is not None else None,mae=mae))
    def finite_vectors(values):
        return [[float(v) for v in vector] if ok else None for vector,ok in zip(values,valid)]
    return dict(frame=frame,source=row['source'],target_kind='optical_rgb',target_shape=row['target_shape'],
        labels=SENSORS[row['source']][2],original=None,reconstruction=None,error=None,optical_panels=panels,
        mae=float(error[valid].mean()) if error is not None and valid.any() else None,hand_mae=hand_mae,
        contact_mae=None,contact_f1=None,marker_displacement=dict(units='native_pixels',axes=['x_right','y_down'],
            reference_frame=reference_frames[0] if len(set(reference_frames))==1 else None,
            reference_frames=reference_frames,reference_is_verified_unloaded=False,arrow_gain=1,
            background='sensor grid from retained reference geometry',
            tracked_fraction=float(valid.mean()),surface=surface.tolist(),reference_px=reference.tolist(),
            measured_px=finite_vectors(target),predicted_px=prediction.tolist() if prediction is not None else None,
            error_scale='endpoint error divided by marker spacing; gray/missing observations are not zero targets'))


TARGET_ROOTS=('processing/marker_displacement_v2/targets','processing/marker_displacement_v2/pilot')


def ready_root(row):
    for root in TARGET_ROOTS:
        if (Path(root)/row['source']/row['split']/episode_key(row)/'complete.json').exists():return root
    raise ValueError('Marker tracks for this episode are still being extracted')


def tracking_catalog(viewer):
    from .marker_targets import MARKER_SOURCES
    from .aligned_frames import MANIFEST
    rows=canonical_rows(MANIFEST);episodes=[]
    for i,row in enumerate(viewer.episodes):
        if row['source'] not in MARKER_SOURCES:continue
        canonical=rows[(row['source'],row['episode_id'])]
        try:root=ready_root(canonical)
        except ValueError:continue
        episodes.append(dict(index=i,source=row['source'],name=row['episode_id'],split=row['split'],
            frames=row['frames'],camera=sorted(row['views'])[0],pilot='pilot' in Path(root).parts))
    status_path=Path(TARGET_ROOTS[0])/'status.json'
    return dict(episodes=episodes,status=json.loads(status_path.read_text()) if status_path.exists() else {},
                sources=list(MARKER_SOURCES))


def tracking_frame(viewer,episode,frame):
    from .aligned_frames import MANIFEST
    row=viewer._episode(episode,frame)
    canonical=canonical_rows(MANIFEST)[(row['source'],row['episode_id'])]
    return payload(dict(manifest=MANIFEST,marker_targets=ready_root(canonical)),row,frame)
