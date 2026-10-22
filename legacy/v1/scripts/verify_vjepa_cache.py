"""Compare live full-video features and predictions with on-disk cache reads."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import DataLoader

from tactilelatent.aligned_frames import manifest_signature,write_json
from tactilelatent.alignment_data import AlignedClipDataset
from tactilelatent.vjepa_model import VJEPABackbone,video_features,model_from_state
from tactilelatent.vjepa_cache import (CACHE_VERSION,partition_plan,open_partition,commit_prefix,
    finalize_partition,ClipImages,VJEPAFeatureCache)

torch.set_num_threads(4)
cfg=json.load(open('configs/joint_vjepa21_pair_32d_v1.json'))
registry=json.load(open(cfg['teachers']))
d=AlignedClipDataset(cfg['manifest'],split=None,code_cache=cfg['code_cache'],registry=registry,include_targets=False,rgb_size=384,data_signature=manifest_signature(cfg['manifest']))
selected=[]
for source in ['egotouch','ftp_vla_touch','ftp_motiontrans']:
    choices=[w for w in d.windows if d.episodes[w[0]]['source']==source and d.episodes[w[0]]['split']=='train']
    selected.extend(choices[:2])
plans=partition_plan(SimpleNamespace(episodes=d.episodes,windows=selected))
root=Path('processing/vjepa2_1/cache_verification');root.mkdir(parents=True,exist_ok=True)
write_json(root/'index.json',dict(version=CACHE_VERSION,signature={'verification_only':True}))
backbone=VJEPABackbone().cuda().eval()
models={}
for name in ('deterministic','stochastic'):
    state=torch.load(Path('processing/runs/joint_vjepa21_pair_32d_v1')/name/'epoch_000.pt',map_location='cpu',weights_only=False)
    models[name]=model_from_state(state).cuda().eval()
reports=[]
for plan in plans:
    original=next(iter(DataLoader(ClipImages(d.episodes,plan),batch_size=2)))
    with torch.no_grad(),torch.autocast('cuda',torch.bfloat16):
        live=video_features(backbone,original,torch.device('cuda'))
    folder=root/plan['source']/plan['split'];array,start,finished=open_partition(folder,plan)
    if not finished:
        for j,(ep,_,offset) in enumerate(plan['clips']):
            slots=plan['episodes'][int(ep)]['camera_slots']
            array[offset:offset+len(slots)]=live[j,:,slots].permute(1,0,2).cpu().numpy()
        commit_prefix(folder,array,plan,len(plan['clips']));del array;finalize_partition(folder,plan)
    reader=VJEPAFeatureCache(root,{'verification_only':True},require_complete=False)
    for j,(ep,start,_) in enumerate(plan['clips']):
        info=plan['episodes'][int(ep)]
        cached,mask=reader.read(plan['source'],plan['split'],info['episode_id'],start+np.arange(8)*2)
        assert torch.equal(cached,live[j].cpu())
        errors={}
        with torch.no_grad():
            for name,model in models.items():
                live_mu,live_lv=model.visual.encode(live[j:j+1],original['view_mask'][j:j+1].cuda())
                cache_mu,cache_lv=model.visual.encode(cached[None].cuda(),mask[None].cuda())
                assert torch.equal(live_mu,cache_mu) and torch.equal(live_lv,cache_lv)
                errors[name]=float((live_mu-cache_mu).abs().max())
        reports.append(dict(source=plan['source'],episode=info['episode_id'],frame_indices=(start+np.arange(8)*2).tolist(),
                            exact_float32_features=True,latent_max_absolute_error=errors,camera_slots=info['camera_slots']))
record=dict(passed=True,clips=reports,cache_dtype='float32',encoder_precision='bfloat16 autocast matching training')
write_json('processing/vjepa2_1/cache_verification.json',record)
print(json.dumps(record),flush=True)
