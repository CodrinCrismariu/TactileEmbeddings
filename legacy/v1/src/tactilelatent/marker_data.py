"""Versioned marker targets with native chronological frame indexing."""
from collections import OrderedDict
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .aligned_frames import file_hash, json_hash, sensor_indices, target_images, write_json
from .marker_targets import MARKER_SOURCES, VERSION, MarkerConfig, MarkerTracker


def episode_key(row):
    return json_hash([row['source'], row['split'], row['episode_id']])[:24]


def extract_episode(row, folder, config=MarkerConfig(), chunk_frames=16):
    """Track every original frame in order; never reset at training clip edges."""
    if row['source'] not in MARKER_SOURCES or chunk_frames < 1:
        raise ValueError('Invalid marker source or chunk size')
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    reference_images = target_images(row, [0])
    trackers = [MarkerTracker(images[0], config) for images in reference_images]
    reference = np.concatenate([t.reference for t in trackers])
    surface = np.concatenate([np.full(len(t.reference), i, np.int64) for i,t in enumerate(trackers)])
    spacing = np.concatenate([np.full(len(t.reference), t.spacing, np.float32) for t in trackers])
    shape = np.concatenate([np.tile(np.array(t.shape[:2][::-1]), (len(t.reference),1)) for t in trackers])
    n = row['frames']; k = len(reference)
    displacement = np.full((n,k,2), np.nan, np.float32)
    valid = np.zeros((n,k), bool); quality = np.zeros((n,k), np.float32)
    frame_indices = np.arange(n, dtype=np.int64)
    for start in range(0,n,chunk_frames):
        end = min(start+chunk_frames,n)
        images = target_images(row, frame_indices[start:end])
        if len(images) != len(trackers):
            raise ValueError('Sensor ordering/count changed')
        for j in range(end-start):
            results = [t.update(images[s][j]) for s,t in enumerate(trackers)]
            displacement[start+j] = np.concatenate([r['displacement_px'] for r in results])
            valid[start+j] = np.concatenate([r['valid'] for r in results])
            quality[start+j] = np.concatenate([r['tracking_quality'] for r in results])
    arrays = dict(reference_px=reference, reference_uv=reference/(shape-1), surface_id=surface,
                  spacing_px=spacing, displacement_px=displacement, valid=valid,
                  tracking_quality=quality, frame_indices=frame_indices,
                  sensor_frame_indices=sensor_indices(row,frame_indices))
    temporary = folder/'tracks.tmp'
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(folder/'tracks.npz')
    magnitude = np.linalg.norm(displacement,axis=-1)
    result = dict(version=VERSION, source=row['source'], split=row['split'], episode_id=row['episode_id'],
                  row_sha256=json_hash(row), extractor_sha256=file_hash(Path(__file__).with_name('marker_targets.py')),
                  loader_sha256=file_hash(Path(__file__)), frames=n, markers=k,
                  sensors=[t.metadata() for t in trackers], valid_fraction=float(valid.mean()),
                  min_frame_valid_fraction=float(valid.mean(1).min()),
                  displacement_p50_px=float(np.nanmedian(magnitude)),
                  displacement_p95_px=float(np.nanpercentile(magnitude,95)),
                  tracks_sha256=file_hash(folder/'tracks.npz'), target='signed_marker_displacement',
                  reference_frame=0, reference_is_verified_unloaded=False)
    normalized = displacement/spacing[None,:,None]
    result['surfaces'] = len(trackers)
    result['surface_valid_fraction'] = [float(valid[:,surface==s].mean()) for s in range(len(trackers))]
    result['loss_statistics'] = dict(square_sum=float(np.nansum(normalized.astype(np.float64)**2)),
                                   coordinate_count=int(valid.sum())*2)
    write_json(folder/'complete.json',result)
    return result


class MarkerCache:
    def __init__(self, root, max_loaded_episodes=8):
        self.root, self.maximum = Path(root), max_loaded_episodes
        self.loaded = OrderedDict()

    def read(self, row, frame_indices):
        key = episode_key(row)
        if key not in self.loaded:
            folder = self.root/row['source']/row['split']/key
            info = json.loads((folder/'complete.json').read_text())
            if (info['version'] != VERSION or info['row_sha256'] != json_hash(row)
                    or info['tracks_sha256'] != file_hash(folder/'tracks.npz')
                    or info['extractor_sha256'] != file_hash(Path(__file__).with_name('marker_targets.py'))
                    or info['loader_sha256'] != file_hash(Path(__file__))):
                raise ValueError('Stale/corrupt marker targets or changed native frame contract')
            with np.load(folder/'tracks.npz',allow_pickle=False) as archive:
                data={k:archive[k].copy() for k in archive.files}
            self.loaded[key]=data
            while len(self.loaded)>self.maximum:
                self.loaded.popitem(last=False)
        self.loaded.move_to_end(key)
        data=self.loaded[key];ix=np.asarray(frame_indices,dtype=np.int64)
        if ix.ndim != 1 or not len(ix) or (ix<0).any() or (ix>=len(data['frame_indices'])).any() or (np.diff(ix)<=0).any():
            raise ValueError('Invalid chronological marker frame indices')
        if not np.array_equal(data['sensor_frame_indices'][ix],sensor_indices(row,ix)):
            raise ValueError('Marker/scene chronological pairing changed')
        return {'marker_target':torch.from_numpy(data['displacement_px'][ix]/data['spacing_px'][None,:,None]),
                'marker_valid':torch.from_numpy(data['valid'][ix]),
                'marker_reference':torch.from_numpy(data['reference_uv'].astype(np.float32)),
                'marker_surface':torch.from_numpy(data['surface_id']),
                'marker_spacing':torch.from_numpy(data['spacing_px'])}


class MarkerTargetClips(Dataset):
    """Replace AE supervision for just four sources, using existing clip inputs.

    Wrapping a cached visual dataset preserves its exact windows and masks.
    Any RGB/AE targets loaded by the wrapped dataset are removed for these four
    sources, so a new training objective cannot accidentally use both targets.
    """
    def __init__(self, dataset, marker_cache):
        self.dataset,self.cache=dataset,marker_cache
        self.episodes,self.windows,self.indices_by_source=dataset.episodes,dataset.windows,dataset.indices_by_source

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,index):
        row=self.episodes[self.windows[index][0]]
        if row['source'] in MARKER_SOURCES:
            # Read only scene clips. Never decode full tactile RGB or call an AE
            # for replaced sources during joint learning.
            from .aligned_frames import scene_rgb
            episode,start=self.windows[index]
            indices=np.arange(start,start+self.dataset.span,self.dataset.stride)
            rgb,mask=scene_rgb(row,indices,size=self.dataset.rgb_size)
            item=dict(rgb=rgb,view_mask=mask,source=row['source'],episode_id=row['episode_id'],
                      frame_indices=torch.from_numpy(indices),
                      sensor_frame_indices=torch.from_numpy(sensor_indices(row,indices)))
            item.update(self.cache.read(row,item['frame_indices'].numpy()))
            return item
        return dict(self.dataset[index])


def marker_collate(items):
    """Pad only the marker axis; padding is always excluded from supervision."""
    from torch.utils.data import default_collate
    if not items or len({x['source'] for x in items}) != 1:
        raise ValueError('Expected a nonempty single-source batch')
    if 'marker_target' not in items[0]:
        return default_collate(items)
    maximum=max(len(x['marker_reference']) for x in items)
    padded=[]
    for item in items:
        item=dict(item)
        for key,axis in [('marker_target',1),('marker_valid',1),('marker_reference',0),('marker_surface',0),('marker_spacing',0)]:
            value=item[key];shape=list(value.shape);shape[axis]=maximum
            out=torch.zeros(shape,dtype=value.dtype)
            if key=='marker_spacing':out.fill_(1.)
            selection=[slice(None)]*value.ndim;selection[axis]=slice(0,value.shape[axis])
            out[tuple(selection)]=value;item[key]=out
        padded.append(item)
    return default_collate(padded)
