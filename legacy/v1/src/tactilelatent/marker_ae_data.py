"""Frozen per-source snapshots of completed marker targets; no optical pixels."""
from collections import OrderedDict
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .aligned_frames import file_hash,json_hash,manifest_signature,write_json
from .marker_data import episode_key
from .marker_targets import VERSION,MARKER_SOURCES


def snapshot(manifest,root,source):
    if source not in MARKER_SOURCES:raise ValueError('Only the four marker sources have marker AEs')
    rows=[r for r in json.loads(Path(manifest).read_text())['episodes'] if r['source']==source]
    missing=[r['episode_id'] for r in rows if not (Path(root)/source/r['split']/episode_key(r)/'complete.json').exists()]
    if missing:raise FileNotFoundError(f'{source}: {len(missing)} incomplete/untrackable episodes; first={missing[0]}')
    records=[];missing=[]
    for row in rows:
        folder=Path(root)/source/row['split']/episode_key(row)
        if not (folder/'complete.json').exists():missing.append(row['episode_id']);continue
        info=json.loads((folder/'complete.json').read_text())
        if (info['row_sha256']!=json_hash(row) or info['version']!=VERSION
                or info['extractor_sha256']!=file_hash(Path(__file__).with_name('marker_targets.py'))
                or info['loader_sha256']!=file_hash(Path(__file__).with_name('marker_data.py'))
                or info['tracks_sha256']!=file_hash(folder/'tracks.npz')):
            raise ValueError(f'Invalid marker episode: {source}/{row["episode_id"]}')
        records.append(dict(row=row,path=str(folder),info=info,complete_sha256=file_hash(folder/'complete.json')))
    if missing:raise FileNotFoundError(f'{source}: {len(missing)} incomplete/untrackable episodes; first={missing[0]}')
    train=[r['info']['loss_statistics'] for r in records if r['row']['split']=='train']
    scale=float(np.sqrt(sum(x['square_sum'] for x in train)/max(sum(x['coordinate_count'] for x in train),1)))
    if not np.isfinite(scale) or scale<=0:raise ValueError('Source has no measurable training displacement')
    return dict(source=source,data=manifest_signature(manifest),root=str(root),episodes=records,
                displacement_scale=scale,source_snapshot_sha256=json_hash(records),
                representation='signed_marker_displacement',normalization_split='train')


class MarkerAEFrames(Dataset):
    """Pack selected vectors once into RAM; retain every marker at native precision.

    Only the frame sampling budget is capped, matching the earlier AE search.
    Static reference geometry is stored once per episode, rather than per frame.
    """
    def __init__(self,pinned,split,stride=2,max_frames=100000,seed=45):
        records=[r for r in pinned['episodes'] if r['row']['split']==split]
        if not records:raise ValueError('Missing AE split')
        candidates=[(e,f) for e,r in enumerate(records) for f in range(0,r['row']['frames'],stride)]
        if max_frames and len(candidates)>max_frames:
            rng=np.random.default_rng(seed);selected=sorted(rng.choice(len(candidates),max_frames,replace=False));candidates=[candidates[i] for i in selected]
        maximum=max(r['info']['markers'] for r in records);n=len(candidates)
        self.target=torch.zeros(n,maximum,2);self.valid=torch.zeros(n,maximum,dtype=torch.bool)
        self.reference=torch.zeros(len(records),maximum,2);self.surface=torch.zeros(len(records),maximum,dtype=torch.long)
        self.spacing=torch.ones(len(records),maximum);self.episode=torch.empty(n,dtype=torch.long)
        self.frames=torch.empty(n,dtype=torch.long);self.episode_ids=[r['row']['episode_id'] for r in records]
        grouped={}
        for i,(e,f) in enumerate(candidates):grouped.setdefault(e,[]).append((i,f))
        for e,items in grouped.items():
            record=records[e]
            with np.load(Path(record['path'])/'tracks.npz',allow_pickle=False) as archive:
                k=len(archive['reference_px']);ix=np.array([f for _,f in items]);ids=torch.tensor([i for i,_ in items])
                vectors=archive['displacement_px'][ix]/archive['spacing_px'][None,:,None]
                valid=archive['valid'][ix] & np.isfinite(vectors).all(-1)
                self.target[ids,:k]=torch.from_numpy(np.nan_to_num(vectors));self.valid[ids,:k]=torch.from_numpy(valid)
                self.reference[e,:k]=torch.from_numpy(archive['reference_uv'].astype(np.float32))
                self.surface[e,:k]=torch.from_numpy(archive['surface_id']);self.spacing[e,:k]=torch.from_numpy(archive['spacing_px'])
                self.episode[ids]=e;self.frames[ids]=torch.from_numpy(ix)
        supported=self.valid.any(-1)
        self.unsupported_frames=int((~supported).sum());self.indices=supported.nonzero().flatten()
        self.shape=(len(self.indices),maximum,2)
        self.signature=dict(split=split,stride=stride,max_frames=max_frames,seed=seed,
            sampled_frames=n,supported_frames=len(self.indices),unsupported_frames=self.unsupported_frames,
            source_snapshot_sha256=pinned['source_snapshot_sha256'])

    def __len__(self):return len(self.indices)

    def __getitem__(self,index):
        i=self.indices[index];e=self.episode[i]
        return dict(marker_target=self.target[i],marker_valid=self.valid[i],marker_reference=self.reference[e],
                    marker_surface=self.surface[e],marker_spacing=self.spacing[e],episode=e,frame=self.frames[i])
