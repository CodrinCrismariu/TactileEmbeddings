"""Measure RGB reconstruction error and true lossless-video sizes on real clips."""
import io
import json
from pathlib import Path
from collections import defaultdict
import time

import av
import numpy as np
import torch

from tactilelatent.aligned_frames import scene_rgb


def round_trip(images,codec,pixel_format,options):
    stream_bytes=io.BytesIO();began=time.perf_counter()
    with av.open(stream_bytes,'w',format='matroska') as container:
        stream=container.add_stream(codec,rate=30);stream.width=384;stream.height=384;stream.pix_fmt=pixel_format
        stream.options=options;stream.codec_context.thread_count=2;stream.codec_context.gop_size=16
        for image in images:
            for packet in stream.encode(av.VideoFrame.from_ndarray(image,format='rgb24')):container.mux(packet)
        for packet in stream.encode():container.mux(packet)
    encoding_seconds=time.perf_counter()-began;encoded=stream_bytes.getvalue();absolute=squared=count=maximum=0
    with av.open(io.BytesIO(encoded)) as container:
        for i,frame in enumerate(container.decode(video=0)):
            difference=frame.to_ndarray(format='rgb24').astype(np.float32)-images[i].astype(np.float32)
            absolute+=float(np.abs(difference).sum(dtype=np.float64));squared+=float(np.square(difference).sum(dtype=np.float64))
            count+=difference.size;maximum=max(maximum,float(np.abs(difference).max()))
    if count!=images.size:raise ValueError('Round-trip frame count changed')
    return dict(bytes=len(encoded),encoding_seconds=encoding_seconds,absolute=absolute,squared=squared,count=count,maximum=maximum)


def main():
    torch.set_num_threads(1);root=Path('processing/research/visual_resize')
    rows=json.loads(Path('processing/joint_alignment/manifest.json').read_text())['episodes'];groups=defaultdict(list)
    for row in rows:groups[row['source']].append(row)
    variants=dict(h264_crf18=('libx264','yuv420p',{'crf':'18','preset':'fast'}),
                  h264_crf12=('libx264','yuv420p',{'crf':'12','preset':'fast'}),
                  h264_rgb_lossless=('libx264rgb','rgb24',{'crf':'0','preset':'fast'}),
                  ffv1_rgb_lossless=('ffv1','bgr0',{'level':'3','coder':'1','context':'1'}))
    result=dict(sources={},variants={name:dict(codec=v[0],pixel_format=v[1],options=v[2],gop=16) for name,v in variants.items()},
        comparison='Decoded video RGB versus the existing 384x384 model input rounded to uint8, not against full-resolution originals.',
        method='Three frame-weighted episode quantiles per source; up to 64 consecutive frames per view. Two codec threads. All 12 source/evaluation datasets, no bulk conversion. Lossless codecs must return every RGB byte exactly.')
    for source,episodes in groups.items():
        cumulative=np.cumsum([r['frames'] for r in episodes]);chosen=sorted(set(int(np.searchsorted(cumulative,cumulative[-1]*q)) for q in (.2,.5,.8)))
        totals={name:defaultdict(float) for name in variants};samples=0
        for e in chosen:
            row=episodes[e];n=min(64,row['frames']);start=max(0,row['frames']//2-n//2)
            batch,_=scene_rgb(row,np.arange(start,start+n),size=384)
            for slot in row['views']:
                images=(batch[:,int(slot)].permute(0,2,3,1).numpy()*255).round().clip(0,255).astype(np.uint8)
                samples+=len(images)
                for name,settings in variants.items():
                    output=round_trip(images,*settings)
                    if name.endswith('lossless') and output['maximum']!=0:raise ValueError('Lossless codec altered RGB pixels')
                    total=totals[name]
                    for k in ('bytes','encoding_seconds','absolute','squared','count'):total[k]+=output[k]
                    total['maximum']=max(total['maximum'],output['maximum'])
            del batch,images
        frames=sum(r['frames']*len(r['views']) for r in episodes);out={}
        for name,t in totals.items():
            mse=t['squared']/t['count'];out[name]=dict(estimated_dataset_bytes=frames*t['bytes']/samples,
                sample_bytes=int(t['bytes']),sample_frames=samples,mae_255=t['absolute']/t['count'],
                rmse_255=float(np.sqrt(mse)),psnr_db=float(10*np.log10(255**2/mse)) if mse else None,
                maximum_error_255=t['maximum'],encoding_seconds=t['encoding_seconds'],squared=t['squared'],count=int(t['count']))
        result['sources'][source]=out
        (root/'quality_lossless_benchmark.json').write_text(json.dumps(result,indent=2))
        print(source,json.dumps({name:dict(GB=v['estimated_dataset_bytes']/1e9,mae=v['mae_255'],psnr=v['psnr_db']) for name,v in out.items()}),flush=True)
    result['total_estimated_bytes']={name:sum(s[name]['estimated_dataset_bytes'] for s in result['sources'].values()) for name in variants}
    result['complete']=True
    (root/'quality_lossless_benchmark.json').write_text(json.dumps(result,indent=2));print('COMPLETE',json.dumps(result['total_estimated_bytes']),flush=True)


if __name__=='__main__':main()
