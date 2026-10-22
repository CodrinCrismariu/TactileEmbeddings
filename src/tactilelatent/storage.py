"""Verified physical-storage migration underneath immutable episode identities.

Only scene RGB is transcoded. Historical checkpoint provenance is accepted
through an explicit original-file/replacement ledger, never an unchecked skip.
"""
import hashlib
import json
from functools import lru_cache
from pathlib import Path

REGISTRY=Path('processing/storage_v1/registry.json')


@lru_cache(maxsize=2)
def _read(stamp):
    return json.loads(REGISTRY.read_text())


def registry():
    if not REGISTRY.exists():return {}
    s=REGISTRY.stat()
    return _read((s.st_mtime_ns,s.st_size))


def episode_key(row):
    return hashlib.sha256((row['source']+'\0'+row['episode_id']).encode()).hexdigest()[:24]


def retired(source):
    return source in registry().get('retired_sources',[])


def original_stat(path):
    """Only explicitly retired/replaced files may be absent from old manifests."""
    r=registry();key=str(Path(path).resolve())
    if key in r.get('replacements',{}) or any(Path(key).is_relative_to(Path(p)) for p in r.get('retired_roots',[])):
        return r.get('original_files',{}).get(key)
    return None


def scene_entry(row):
    return registry().get('episodes',{}).get(episode_key(row))


def converted_scene(row,indices,size):
    entry=scene_entry(row)
    if entry is None:return None
    import torch
    from .media import video_frames
    rgb=torch.zeros(len(indices),3,3,size,size);mask=torch.zeros(len(indices),3,dtype=torch.bool)
    for slot,info in entry['views'].items():
        p=Path(info['path']);s=p.stat()
        if [s.st_size,s.st_mtime_ns]!=info['stat']:raise ValueError('Converted RGB changed outside migration ledger')
        rgb[:,int(slot)]=video_frames(p,indices+info.get('frame_offset',0),size=size);mask[:,int(slot)]=True
    return rgb,mask


def compatible_data(manifest,expected):
    """Explicitly authorize old models on migrated visual inputs; targets unchanged."""
    r=registry()
    if expected not in r.get('legacy_signatures',[]):return False
    p=Path(manifest)
    if hashlib.sha256(p.read_bytes()).hexdigest()!=expected['manifest_sha256']:return False
    # The conversion workers update physical receipts; approved loader code has
    # its own immutable publication so a worker cannot overwrite a newer pin.
    pin=REGISTRY.with_name('implementation.json')
    implementation=json.loads(pin.read_text())['files'] if pin.exists() else r.get('implementation',{})
    for filename,digest in implementation.items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest()!=digest:return False
    if not implementation:return False
    replacements=r.get('replacements',{})
    for name,old in r.get('original_files',{}).items():
        path=Path(name)
        if name in replacements:
            receipt=replacements[name]
            if receipt.get('deleted'):
                if path.exists():return False
            elif not path.exists() or [path.stat().st_size,path.stat().st_mtime_ns]!=receipt['new_stat']:return False
        elif any(path.is_relative_to(Path(root)) for root in r.get('retired_roots',[])):
            continue
        elif not path.exists() or [path.stat().st_size,path.stat().st_mtime_ns]!=old:
            return False
    return True
