"""Small paired source-validation check of compression with a frozen checkpoint."""
import io
import json
from pathlib import Path
import time

import av
import numpy as np
import torch

from tactilelatent.aligned_frames import write_json,file_hash
from tactilelatent.marker_code_model import model_from_state,MixedMarkerTeachers
from tactilelatent.marker_data import marker_collate
from tactilelatent.marker_targets import MARKER_SOURCES
from tactilelatent.train_vjepa_markers import prepare,validate
from tactilelatent.vjepa_model import VJEPABackbone


def compress(batch,crf):
    rgb=batch['rgb'].clone();b,t,v=rgb.shape[:3]
    for example in range(b):
        for camera in range(v):
            if not batch['view_mask'][example,:,camera].any():continue
            images=(rgb[example,:,camera].permute(0,2,3,1).numpy()*255).round().clip(0,255).astype(np.uint8)
            out=io.BytesIO()
            with av.open(out,'w',format='matroska') as container:
                stream=container.add_stream('libx264',rate=30);stream.width=384;stream.height=384;stream.pix_fmt='yuv420p'
                stream.options={'crf':str(crf),'preset':'fast'};stream.codec_context.thread_count=2;stream.codec_context.gop_size=16
                for image in images:
                    for packet in stream.encode(av.VideoFrame.from_ndarray(image,format='rgb24')):container.mux(packet)
                for packet in stream.encode():container.mux(packet)
            with av.open(io.BytesIO(out.getvalue())) as container:
                decoded=[torch.from_numpy(frame.to_ndarray(format='rgb24').copy()).permute(2,0,1).float()/255. for frame in container.decode(video=0)]
            if len(decoded)!=8:raise ValueError('Compression changed frame count')
            rgb[example,:,camera]=torch.stack(decoded)
    return {**batch,'rgb':rgb}


def main():
    torch.set_num_threads(2);device=torch.device('cuda');root=Path('processing/research/visual_resize')
    checkpoint=Path('processing/runs/joint_vjepa21_spatial_marker_codes_32d_v1/epoch_009.pt')
    cfg=json.loads(Path('configs/joint_vjepa21_spatial_marker_codes_32d.json').read_text())
    _,val,registry,_,_=prepare(cfg)
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=model_from_state(state).to(device).eval().requires_grad_(False)
    backbone=VJEPABackbone(cfg['backbone_repo'],cfg['backbone_checkpoint']).to(device).eval()
    teachers=MixedMarkerTeachers(registry,device)
    result=dict(checkpoint=str(checkpoint),checkpoint_sha256=file_hash(checkpoint),epoch=state['epoch']+1,
        clips_per_source=4,sources={},heldout_excluded='ftp_motiontrans',
        limitation='Small source-validation pilot at one frozen checkpoint, without retraining. Compression applies to each eight-frame strided clip rather than a full episode. This does not measure MotionTrans transfer or establish a general accuracy guarantee.')
    for source,ids in sorted(val.indices_by_source.items()):
        if source=='ftp_motiontrans':raise ValueError('Held-out source must not be used')
        selected=[ids[min(len(ids)-1,int(len(ids)*q))] for q in (.1,.35,.65,.9)]
        items=[val[i] for i in selected];batches=[marker_collate(items[i:i+2]) for i in range(0,4,2)]
        metrics={};began=time.monotonic()
        for label,crf in [('original',None),('crf18',18),('crf12',12)]:
            data=batches if crf is None else [compress(b,crf) for b in batches]
            with torch.inference_mode():metrics[label]=validate(model,backbone,teachers,data,device)[source]
        key='marker_mae_px' if source in MARKER_SOURCES else 'normalized_mae'
        baseline=metrics['original'][key]
        result['sources'][source]=dict(metric=key,results=metrics,
            change_percent={k:100*(v[key]/baseline-1) for k,v in metrics.items() if k!='original'},
            episode_ids=[item['episode_id'] for item in items],seconds=time.monotonic()-began)
        write_json(root/'prediction_compression_check.json',result)
        print(source,json.dumps(dict(original=baseline,crf18=metrics['crf18'][key],crf12=metrics['crf12'][key],change=result['sources'][source]['change_percent'])),flush=True)
        del items,batches,data
    result['complete']=True;write_json(root/'prediction_compression_check.json',result);print('COMPLETE',flush=True)


if __name__=='__main__':main()
