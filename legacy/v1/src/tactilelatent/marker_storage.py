"""Explicit retirement of optical pixels after verified marker replacement.

The immutable episode manifests and cached marker codes keep their identities.
Only sources named in the committed, local retirement receipt use this contract.
"""
import json
from functools import lru_cache
from pathlib import Path

ROOT = Path('processing/marker_storage_v1')
RECEIPT = ROOT / 'retirement.json'
SOURCES = ('ftp_unit', 'ftp_unit_bimanual', 'ftp_vla_touch')


@lru_cache(maxsize=2)
def _read(stamp):
    return json.loads(RECEIPT.read_text())


def retirement():
    if not RECEIPT.exists():
        return {}
    stat = RECEIPT.stat()
    return _read((str(RECEIPT.resolve()), stat.st_mtime_ns, stat.st_size))


def marker_only(source):
    receipt = retirement()
    return bool(receipt.get('active') and source in receipt.get('sources', {}))


def retained_shape(row):
    if not marker_only(row['source']):
        return None
    shape = retirement()['sources'][row['source']]['target_shape']
    if list(row['target_shape']) != shape:
        raise ValueError('Retained marker geometry differs from the episode shape')
    return list(shape)
