"""Small in-memory H.264 resize/storage pilot; does not rewrite source datasets."""
import io
import json
from pathlib import Path
import statistics
import time
from collections import defaultdict

import av
import numpy as np
import torch

from tactilelatent.aligned_frames import scene_rgb


def main():
    torch.set_num_threads(1)
    root=Path('processing/research/visual_resize')
    rows=json.loads(Path('processing/joint_alignment/manifest.json').read_text())['episodes'];groups=defaultdict(list)
    for row in rows:groups[row['source']].append(row)
    report=dict(sources={},codec='H.264 libx264 CRF18 preset fast, GOP16, yuv420p',
        method='Up to 64 consecutive frames per episode, three frame-weighted episode quantiles per source. RGB inputs are exactly the current loader outputs rounded to uint8. Approximate size extrapolation only, before full-sequence conversion. Cached decode benchmark uses an in-memory MP4, excludes disk IO. Two encoding threads.')
    for source,episodes in groups.items():
        cumulative=np.cumsum([r['frames'] for r in episodes]);chosen=sorted(set(int(np.searchsorted(cumulative,cumulative[-1]*q)) for q in (.2,.5,.8)))
        byte_counts=[];frame_counts=[];decode_times=[];encoding_times=[]
        for e in chosen:
            row=episodes[e];n=min(64,row['frames']);start=max(0,row['frames']//2-n//2);indices=np.arange(start,start+n)
            batch,mask=scene_rgb(row,indices,size=384)
            for slot in row['views']:
                images=(batch[:,int(slot)].permute(0,2,3,1).numpy()*255).round().clip(0,255).astype(np.uint8)
                out=io.BytesIO();began=time.perf_counter()
                with av.open(out,'w',format='mp4') as container:
                    stream=container.add_stream('libx264',rate=30);stream.width=384;stream.height=384;stream.pix_fmt='yuv420p'
                    stream.options={'crf':'18','preset':'fast'};stream.codec_context.thread_count=2;stream.codec_context.gop_size=16
                    for image in images:
                        frame=av.VideoFrame.from_ndarray(image,format='rgb24')
                        for packet in stream.encode(frame):container.mux(packet)
                    for packet in stream.encode():container.mux(packet)
                encoding_times.append(time.perf_counter()-began);encoded=out.getvalue();byte_counts.append(len(encoded));frame_counts.append(n)
                wanted=np.arange(max(0,n//2-7),max(0,n//2-7)+15,2)
                def decode():
                    found={}
                    with av.open(io.BytesIO(encoded)) as container:
                        stream=container.streams.video[0];stream.thread_type='AUTO';stream.thread_count=2
                        container.seek(int(wanted[0]/30/float(stream.time_base)),stream=stream,backward=True)
                        for frame in container.decode(video=0):
                            position=round(frame.pts*float(stream.time_base)*30)
                            if position in wanted:
                                a=frame.to_ndarray(format='rgb24');found[position]=torch.from_numpy(a.copy()).permute(2,0,1).float()/255.
                            if position>=wanted[-1]:break
                    assert len(found)==8
                    return torch.stack([found[int(i)] for i in wanted])
                decode();times=[]
                for _ in range(3):
                    began=time.perf_counter();decode();times.append(time.perf_counter()-began)
                decode_times.append(statistics.median(times))
            del batch,images
        frames=sum(r['frames']*len(r['views']) for r in episodes)
        item=dict(camera_frames=frames,sample_camera_frames=sum(frame_counts),sample_mp4_bytes=sum(byte_counts),
            estimated_bytes=frames*sum(byte_counts)/sum(frame_counts),
            cached_video_decode_ms_per_camera_clip=statistics.mean(decode_times)*1000,
            encode_ms_per_camera_frame=sum(encoding_times)/sum(frame_counts)*1000)
        report['sources'][source]=item
        (root/'video_benchmark.json').write_text(json.dumps(report,indent=2))
        print(source,json.dumps(item),flush=True)
    report['estimated_total_bytes']=sum(v['estimated_bytes'] for v in report['sources'].values());report['complete']=True
    (root/'video_benchmark.json').write_text(json.dumps(report,indent=2));print('COMPLETE',report['estimated_total_bytes']/1e9,flush=True)


if __name__=='__main__':main()
