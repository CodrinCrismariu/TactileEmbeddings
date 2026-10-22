"""Sensor-grid rendering from retained geometry, never from tactile images."""
import base64
from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw

LABELS=['Left hand · left sensor','Left hand · right sensor',
        'Right hand · left sensor','Right hand · right sensor']


def png(image):
    f=BytesIO();image.save(f,'PNG')
    return 'data:image/png;base64,'+base64.b64encode(f.getvalue()).decode()


def marker_payload(row, data, frame, prediction=None):
    if not 0<=frame<row['frames']:raise ValueError('Frame outside episode')
    ref=data['reference_px'];surface=data['surface_id'];valid=data['valid'][frame]
    target=data['displacement_px'][frame];spacing=data['spacing_px'];panels=[]
    if prediction is not None:
        prediction=np.asarray(prediction,dtype=np.float32)
        if prediction.shape!=target.shape or not np.isfinite(prediction).all():raise ValueError('Invalid marker prediction')
    error=None if prediction is None else np.abs(prediction-target)
    for sid,label in enumerate(LABELS):
        selected=surface==sid;h,w=row['native_sensor_shape']
        def panel(delta, observed, color):
            image=Image.new('RGB',(w,h),(20,25,31));draw=ImageDraw.Draw(image)
            for p,d,ok in zip(ref[selected],delta[selected],observed[selected]):
                x,y=map(float,p);draw.ellipse((x-2,y-2,x+2,y+2),outline=(130,140,150))
                if not ok:
                    draw.line((x-2,y-2,x+2,y+2),fill=(90,90,90));continue
                end=p+d
                draw.line((x,y,float(end[0]),float(end[1])),fill=color,width=2)
                draw.ellipse((end[0]-2,end[1]-2,end[0]+2,end[1]+2),fill=color)
            return png(image)
        mask=selected&valid
        heat=np.zeros((h,w),dtype='<u2')
        if error is not None:
            draw=ImageDraw.Draw(Image.fromarray(heat))
            image=Image.new('I',(w,h));draw=ImageDraw.Draw(image)
            for p,e,ok in zip(ref[selected],np.linalg.norm(prediction[selected]-np.nan_to_num(target[selected]),axis=-1)/spacing[selected],valid[selected]):
                if ok:draw.ellipse((p[0]-4,p[1]-4,p[0]+4,p[1]+4),fill=int(np.clip(e,0,1)*65535))
            heat=np.asarray(image,dtype='<u2')
        panels.append(dict(label=label,original=panel(target,valid,(20,220,255)),
            reconstruction=None if prediction is None else panel(prediction,np.ones_like(valid),(255,180,40)),
            error_u16=base64.b64encode(heat.tobytes()).decode() if prediction is not None else None,
            mae=float(error[mask].mean()) if error is not None and mask.any() else None))
    return dict(source='baihu',frame=frame,target_kind='optical_rgb',target_shape=[12,240,240],labels=LABELS,
        optical_panels=panels,original=None,reconstruction=None,error=None,
        mae=float(error[valid].mean()) if error is not None and valid.any() else None,
        hand_mae=[p['mae'] for p in panels],contact_mae=None,contact_f1=None,
        marker_displacement=dict(units='native_pixels',arrow_gain=1,tracked_fraction=float(valid.mean()),
            surface=surface.tolist(),reference_px=ref.tolist(),
            measured_px=[p.tolist() if ok else None for p,ok in zip(target,valid)],
            predicted_px=None if prediction is None else prediction.tolist(),
            background='sensor grid; original tactile imagery was retired after verification'))
