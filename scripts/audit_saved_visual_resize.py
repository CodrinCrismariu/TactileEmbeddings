"""Read-only sample audit of 384-pixel visual inputs across the saved manifest."""
import io
import json
from pathlib import Path
import statistics
import time
from collections import defaultdict

import av
import h5py
import numpy as np
from PIL import Image
import torch

from tactilelatent.aligned_frames import scene_rgb,raw_scene_images
from tactilelatent.sensor_data import ftp_group


def main():
    torch.set_num_threads(1)
    root=Path('processing/research/visual_resize');root.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(Path('processing/joint_alignment/manifest.json').read_text())
    sources=defaultdict(list)
    for row in manifest['episodes']:sources[row['source']].append(row)
    result=dict(manifest='processing/joint_alignment/manifest.json',sources={},
        scope='All source/evaluation episodes and every available camera in the active canonical manifest; excludes unused duplicate releases and pretraining archives.',
        method='Three frame-weighted sequence quantiles per source, eight chronological frames at stride two per clip. One CPU thread, warm local reads, median of three loader passes. JPEG quality 95; PNG compression level 3. Estimates exclude per-file/container metadata.',
        pixel_caveat='JPEG adds recompression. PNG preserves the rounded uint8 V-JEPA input; FTP float interpolation has sub-uint8 fractions before rounding.')
    for source,rows in sources.items():
        cumulative=np.cumsum([row['frames'] for row in rows]);chosen=sorted(set(int(np.searchsorted(cumulative,cumulative[-1]*q)) for q in (.2,.5,.8)))
        blobs=defaultdict(list);timings=[];cached_times=[];resolutions=[];resize_times=[]
        for e in chosen:
            row=rows[e];start=min(max(0,row['frames']//2-7),row['frames']-15)
            indices=np.arange(start,start+15,2)
            batch,mask=scene_rgb(row,indices,size=384)
            def original():return scene_rgb(row,indices,size=384)
            measured=[]
            for _ in range(3):
                began=time.perf_counter();original();measured.append(time.perf_counter()-began)
            timings.append(statistics.median(measured))
            local=[]
            for slot,path in row['views'].items():
                s=int(slot)
                if source=='ftp_vla_touch':
                    native=raw_scene_images(row,[int(indices[0])],slot)[0];h,w=native.shape[:2]
                    image=Image.fromarray(native)
                    began=time.perf_counter()
                    for _ in range(10):image.resize((384,384),Image.Resampling.BILINEAR)
                    resize_times.append((time.perf_counter()-began)/10)
                elif 'tar_path' in row:
                    array=ftp_group(row)['data/'+path];h,w=array.shape[1:3]
                    native=torch.ones(8,3,h,w)
                    began=time.perf_counter()
                    for _ in range(10):torch.nn.functional.interpolate(native,size=(384,384),mode='bilinear',align_corners=False)
                    resize_times.append((time.perf_counter()-began)/80)
                elif source=='opentouch':
                    with h5py.File(path,'r') as f:
                        encoded=bytes(f['data'][row['hdf5_demo']]['rgb_images_jpeg'][int(indices[0])])
                    with Image.open(io.BytesIO(encoded)) as im:image=im.convert('RGB');w,h=image.size
                    began=time.perf_counter()
                    for _ in range(10):image.resize((384,384),Image.Resampling.BICUBIC)
                    resize_times.append((time.perf_counter()-began)/10)
                else:
                    with av.open(path) as container:
                        st=container.streams.video[0];w,h=st.width,st.height;frame=next(container.decode(video=0))
                        frame.reformat(width=384,height=384,format='rgb24')
                        began=time.perf_counter()
                        for _ in range(10):frame.reformat(width=384,height=384,format='rgb24').to_ndarray()
                        resize_times.append((time.perf_counter()-began)/10)
                resolutions.append([w,h])
                for image_tensor in batch[:,s]:
                    image=Image.fromarray((image_tensor.permute(1,2,0).numpy()*255).round().clip(0,255).astype(np.uint8))
                    for name,fmt,options in [('jpeg95','JPEG',dict(quality=95)),('png','PNG',dict(compress_level=3))]:
                        stream=io.BytesIO();image.save(stream,format=fmt,**options);value=stream.getvalue()
                        blobs[name].append(value)
                        if name=='jpeg95':local.append((s,value))
            def cached():
                rgb=torch.zeros(8,3,3,384,384)
                counters=defaultdict(int)
                for slot,encoded in local:
                    with Image.open(io.BytesIO(encoded)) as image:
                        value=torch.from_numpy(np.asarray(image.convert('RGB')).copy()).permute(2,0,1).float()/255.
                    rgb[counters[slot],slot]=value;counters[slot]+=1
                return rgb
            cached();times=[]
            for _ in range(3):
                began=time.perf_counter();cached();times.append(time.perf_counter()-began)
            cached_times.append(statistics.median(times))
        frames=sum(row['frames']*len(row['views']) for row in rows)
        item=dict(episodes=len(rows),camera_frames=frames,sampled_episodes=[rows[e]['episode_id'] for e in chosen],
            native_visual_resolutions=sorted({tuple(x) for x in resolutions}),sampled_camera_frames=len(blobs['jpeg95']),
            jpeg95_mean_frame_bytes=statistics.mean(map(len,blobs['jpeg95'])),
            png_mean_frame_bytes=statistics.mean(map(len,blobs['png'])),
            jpeg95_estimated_bytes=frames*statistics.mean(map(len,blobs['jpeg95'])),
            png_estimated_bytes=frames*statistics.mean(map(len,blobs['png'])),
            original_clip_loader_ms=statistics.mean(timings)*1000,
            cached_jpeg95_clip_loader_ms=statistics.mean(cached_times)*1000,
            resize_or_video_color_reformat_ms_per_camera_frame=statistics.mean(resize_times)*1000)
        result['sources'][source]=item
        (root/'sample_benchmark.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(dict(source=source,frames=frames,original_clip_ms=item['original_clip_loader_ms'],cached_clip_ms=item['cached_jpeg95_clip_loader_ms'],jpeg95_GB=item['jpeg95_estimated_bytes']/1e9,png_GB=item['png_estimated_bytes']/1e9,resolutions=item['native_visual_resolutions'])),flush=True)
    result['total_camera_frames']=sum(i['camera_frames'] for i in result['sources'].values())
    result['total_jpeg95_estimated_bytes']=sum(i['jpeg95_estimated_bytes'] for i in result['sources'].values())
    result['total_png_estimated_bytes']=sum(i['png_estimated_bytes'] for i in result['sources'].values())
    result['complete']=True
    (root/'sample_benchmark.json').write_text(json.dumps(result,indent=2))
    print('COMPLETE',json.dumps({k:v for k,v in result.items() if k.startswith('total')}),flush=True)


if __name__=='__main__':main()
