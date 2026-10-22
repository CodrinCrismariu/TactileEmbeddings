"""Frozen sensor references. No episode-dependent fitting or contact thresholding.

Offsets are sensor coordinates, not an SI calibration. Empirical unloaded
envelopes describe the reviewed observations; they are not confidence intervals
or a contact classifier. Raw arrays and validity masks must be retained.
"""
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np


DEFAULT_REGISTRY = Path('processing/zero_reference_v1/references.json')
SCHEMA = 'fixed-sensor-zero-v1'


def content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def save_registry(path, payload):
    payload = dict(payload)
    payload.pop('content_sha256', None)
    payload['content_sha256'] = content_hash(payload)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


@lru_cache(maxsize=4)
def _load(path, stamp):
    result = json.loads(Path(path).read_text())
    digest = result.pop('content_sha256')
    if result.get('schema') != SCHEMA or content_hash(result) != digest:
        raise ValueError('Invalid fixed-reference schema or content hash')
    if result.get('fit_split') != 'train' or result.get('episode_refitting') is not False:
        raise ValueError('References must be frozen and fitted on training data only')
    result['content_sha256'] = digest
    return result


class FixedReferences:
    def __init__(self, path=DEFAULT_REGISTRY):
        self.path = Path(path)

    def catalog(self):
        if not self.path.exists():
            return None
        stat = self.path.stat()
        return _load(str(self.path), (stat.st_mtime_ns, stat.st_size))

    def numeric(self, source, shape, channels, revision):
        """Return a matching reference, never a guess for a different release."""
        registry = self.catalog()
        if registry is None:
            return None
        matches = [e for k, e in registry['numeric'].items()
            if (e.get('source', k) == source and list(shape) == e['target_shape']
                and list(channels) == e['channel_order'] and revision == e['revision'])]
        if not matches:
            return None
        if len(matches) != 1:
            raise ValueError('Ambiguous fixed reference for this sensor layout')
        entry = matches[0]
        for key in ('offset', 'display_limits'):
            array = np.asarray(entry[key], dtype=float)
            if array.shape != (len(channels),) or not np.isfinite(array).all():
                raise ValueError('Malformed fixed numeric reference')
        if np.any(np.asarray(entry['display_limits']) <= 0):
            raise ValueError('Display limits must be positive')
        return dict(entry, registry_sha256=registry['content_sha256'])

    def transform(self, source, values, valid, channels, revision):
        """One observation or a batch, native shape preserved, missing stays NaN.

        Deliberately takes no episode identity. Direction channels have an offset
        of zero and remain in the native convention. No rectification/deadband.
        """
        registry = self.catalog()
        if registry is None:
            raise ValueError('No fixed reference for this source')
        values, valid = np.asarray(values), np.asarray(valid, dtype=bool)
        matches = [e for k, e in registry['numeric'].items()
            if e.get('source', k) == source and list(channels) == e['channel_order']]
        if len(matches) != 1:
            raise ValueError('No unique reference for this channel order')
        shape = matches[0]['target_shape']
        entry = self.numeric(source, shape, channels, revision)
        ndim = len(shape)
        if (entry is None or list(values.shape[-ndim:]) != shape or values.shape != valid.shape):
            raise ValueError('Reference release, channel order, target shape or mask mismatch')
        offset = np.asarray(entry['offset'], dtype=float).reshape(shape)
        mask = valid & np.isfinite(values)
        return np.where(mask, values.astype(float) - offset, np.nan), mask.copy()

    def optical(self, sensor, image, revision):
        registry = self.catalog()
        if registry is None:
            return None
        image = np.asarray(image)
        key = sensor + ':' + 'x'.join(map(str, image.shape)) + ':' + str(image.dtype)
        entry = registry.get('optical', {}).get(key)
        if entry is None or entry['revision'] != revision:
            return None
        path = self.path.parent / entry['file']
        template = _template(str(path), entry['sha256'])
        if template.shape != image.shape:
            raise ValueError('Optical zero reference shape mismatch')
        # Float subtraction prevents uint8 wraparound, preserves signed change.
        return dict(entry, residual=image.astype(np.float32) - template,
                    registry_sha256=registry['content_sha256'])


@lru_cache(maxsize=8)
def _template(path, digest):
    payload = Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError('Optical reference checksum mismatch')
    with np.load(path, allow_pickle=False) as data:
        return data['reference'].copy()
