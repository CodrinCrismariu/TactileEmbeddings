"""Physical left/right augmentation; sensor codes are never transformed.

Policies are explicit about each sensor's local coordinates. Unverified
mounting conventions are a hard error, not an invitation to guess a flip.
"""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .aligned_frames import file_hash, json_hash

SOURCES = ('egotouch', 'opengraph', 'ftp_unit_bimanual', 'ftp_rdp_bimanual', 'baihu')
TARGET_FIELDS = ('target', 'target_mask', 'marker_target', 'marker_valid',
                 'marker_reference', 'marker_surface', 'marker_spacing')


def load_policy(path):
    policy = json.loads(Path(path).read_text())
    if policy.get('schema_version') != 1 or set(policy['sources']) != set(SOURCES):
        raise ValueError('Expected the complete versioned five-source mirror policy')
    for source, spec in policy['sources'].items():
        if spec.get('status') == 'verified':
            validate_spec(spec)
            if not spec.get('evidence'):
                raise ValueError(f'Missing mirror-coordinate evidence: {source}')
    return policy


def require_spec(policy, source):
    if source not in SOURCES:
        raise ValueError(f'Mirroring is restricted to bilateral sources: {source}')
    spec = policy['sources'][source]
    if spec.get('status') != 'verified':
        raise ValueError(f'{source}: mirror mapping blocked: {spec.get("reason", "unverified geometry")}')
    validate_spec(spec)
    return spec


def validate_spec(spec):
    kind = spec['kind']
    if kind not in ('grid', 'taxels', 'scalar', 'markers'):
        raise ValueError('Unknown tactile mirror representation')
    perm = np.asarray(spec['surface_permutation'] if kind == 'markers' else spec['hand_permutation'])
    if perm.ndim != 1 or sorted(perm.tolist()) != list(range(len(perm))) or not np.array_equal(perm[perm], np.arange(len(perm))):
        raise ValueError('Mirror identities must form an involutive permutation')
    if kind == 'markers':
        transforms = np.asarray(spec['surface_transforms'], np.float64)
        if transforms.shape != (len(perm), 2, 3) or not np.isfinite(transforms).all():
            raise ValueError('Every marker surface requires a finite coordinate transform')
        for sid, dst in enumerate(perm):
            a, b = transforms[sid, :, :2], transforms[sid, :, 2]
            reverse, offset = transforms[dst, :, :2], transforms[dst, :, 2]
            # Current native marker contract supports x/y reflections. An axis
            # swap needs native image aspect ratios and must not be guessed.
            if not np.allclose(a, np.diag(np.diag(a))) or not np.allclose(a.T @ a, np.eye(2)):
                raise ValueError('Marker mirrors support verified local-axis reflections only')
            if not np.allclose(reverse @ a, np.eye(2)) or not np.allclose(reverse @ b + offset, 0):
                raise ValueError('Marker coordinate transform is not self-inverting')
    if kind == 'taxels':
        channels = np.asarray(spec['channel_permutation'], np.int64)
        if channels.ndim != 2 or len(channels) != len(perm):
            raise ValueError('Taxel mirroring requires an explicit per-hand anatomical map')
        for hand, dst in enumerate(perm):
            p = channels[hand]
            if sorted(p.tolist()) != list(range(len(p))) or not np.array_equal(channels[dst][p], np.arange(len(p))):
                raise ValueError('Taxel correspondence must preserve all channels and invert exactly')


def verify_evidence(policy):
    """Recheck local evidence and pin its bytes in preparation provenance."""
    report = {}
    for source, spec in policy['sources'].items():
        if spec.get('status') != 'verified':
            report[source] = dict(verified=False, reason=spec.get('reason'))
            continue
        require_spec(policy, source)
        files = {p: file_hash(p) for p in spec['evidence']}
        if source == 'egotouch':
            import ast
            tree = ast.parse(Path(spec['evidence'][0]).read_text())
            masks = {}
            for node in tree.body:
                if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                    if node.targets[0].id in ('_NAN_L', '_NAN_R'):
                        masks[node.targets[0].id] = np.asarray(ast.literal_eval(node.value.args[0]), bool)
            if set(masks) != {'_NAN_L', '_NAN_R'} or not np.array_equal(masks['_NAN_L'][:, ::-1], masks['_NAN_R']):
                raise ValueError('EgoTouch official left/right masks do not match the mirror policy')
        elif source == 'ftp_rdp_bimanual':
            from .joint_all_config import FTP_SPECS
            keys = FTP_SPECS[source]['target_keys']
            if keys != ['left_tactile_data_gripperforce_flexivgripper', 'right_tactile_data_gripperforce_flexivgripper']:
                raise ValueError('RDP bilateral scalar-channel order changed')
        report[source] = dict(verified=True, evidence_sha256=files, mapping_sha256=json_hash(spec))
    return report


def transform_targets(item, spec):
    """Transform one unpadded tactile sample (time dimension is optional).

    Marker point identity/order is retained; surface ownership and coordinates
    transform together. NaNs and measured zero remain distinct.
    """
    validate_spec(spec)
    if 'code_target' in item:
        raise ValueError('Transform native targets before encoding; never mirror latent coordinates')
    out = dict(item)
    kind = spec['kind']
    if kind != 'markers':
        order = torch.tensor(spec['hand_permutation'], dtype=torch.long, device=item['target'].device)
        for key in ('target', 'target_mask'):
            if key not in item:
                continue
            value = item[key]
            if value.shape[-3] != len(order):
                raise ValueError('Unexpected bilateral target layout')
            value = value.index_select(-3, order)
            if kind == 'grid' and spec.get('flip_width'):
                value = value.flip(-1)
            elif kind == 'taxels':
                channels = torch.as_tensor(spec['channel_permutation'], device=value.device)
                if value.shape[-2] != 1 or value.shape[-1] != channels.shape[1]:
                    raise ValueError('Taxel geometry differs from the verified channel map')
                value = torch.stack([value[..., hand, :, :].index_select(-1, channels[hand])
                                     for hand in range(len(order))], dim=-3)
            out[key] = value
        return out
    surface = item['marker_surface'].long()
    if surface.ndim != 1 or (surface < 0).any() or (surface >= len(spec['surface_permutation'])).any():
        raise ValueError('Unexpected marker surface identities')
    transforms = torch.as_tensor(spec['surface_transforms'], device=surface.device,
                                 dtype=item['marker_reference'].dtype)[surface]
    matrix, offset = transforms[..., :2], transforms[..., 2]
    out['marker_reference'] = torch.einsum('nij,nj->ni', matrix, item['marker_reference']) + offset
    out['marker_target'] = torch.einsum('nij,...nj->...ni', matrix, item['marker_target'])
    out['marker_surface'] = torch.as_tensor(spec['surface_permutation'], device=surface.device)[surface]
    return out


def mirror_rgb(rgb, view_mask):
    if rgb.shape[-4] != 3 or view_mask.shape[-1] != 3:
        raise ValueError('Expected scene/left-wrist/right-wrist camera slots')
    order = torch.tensor([0, 2, 1], device=rgb.device)
    return rgb.index_select(-4, order).flip(-1), view_mask.index_select(-1, order.to(view_mask.device))


def sample_flags(batch, probability, seed, epoch, step):
    if not 0 <= probability <= 1:
        raise ValueError('Mirror probability must be in [0,1]')
    # No worker-local RNG: epoch-boundary resume and SAM replay cannot redraw a
    # different augmentation. IDs also distinguish individual clips in a batch.
    values = []
    for i, episode in enumerate(batch['episode_id']):
        frames = batch['frame_indices'][i].tolist()
        key = json.dumps([seed, epoch, step, i, episode, frames], separators=(',', ':')).encode()
        draw = int.from_bytes(hashlib.sha256(key).digest()[:8], 'big') / 2**64
        values.append(draw < probability)
    return torch.tensor(values, dtype=torch.bool)


def apply_batch(batch, policy, sources, probability=.5, seed=45, epoch=0, step=0, flags=None):
    """Select native/mirrored codes and transform a collated RGB/target batch."""
    source = batch['source'][0]
    if len(set(batch['source'])) != 1:
        raise ValueError('Mirror augmentation requires homogeneous source batches')
    if source not in sources:
        return batch
    spec = require_spec(policy, source)
    n = len(batch['source'])
    flags = sample_flags(batch, probability, seed, epoch, step) if flags is None else torch.as_tensor(flags, dtype=torch.bool).cpu()
    if flags.shape != (n,):
        raise ValueError('One mirror choice is required per clip')
    out = dict(batch)
    out['mirror_applied'] = flags.clone()
    fields = ('rgb', 'view_mask', *TARGET_FIELDS, 'code_target', 'code_valid')
    if flags.any():
        for key in fields:
            if key in out:
                out[key] = out[key].clone()
        if 'code_target' in batch and not {'mirror_code_target', 'mirror_code_valid'} <= set(batch):
            raise ValueError('A mirrored clip requires freshly encoded, pinned mirrored sensor codes')
        for i in flags.nonzero().flatten().tolist():
            out['rgb'][i], out['view_mask'][i] = mirror_rgb(batch['rgb'][i], batch['view_mask'][i])
            sensor = {k: batch[k][i] for k in TARGET_FIELDS if k in batch}
            if spec['kind'] == 'markers':
                if 'mirror_marker_count' not in batch:
                    raise ValueError('Marker padding needs the original point count')
                count = int(batch['mirror_marker_count'][i])
                if not 0 < count <= len(sensor['marker_reference']):
                    raise ValueError('Invalid unpadded marker count')
                for key in sensor:
                    sensor[key] = sensor[key][..., :count, :] if key == 'marker_target' else (
                        sensor[key][:count] if key not in ('marker_valid',) else sensor[key][..., :count])
            transformed = transform_targets(sensor, spec)
            for key, value in transformed.items():
                if spec['kind'] == 'markers':
                    if key == 'marker_target': out[key][i, ..., :count, :] = value
                    elif key == 'marker_valid': out[key][i, ..., :count] = value
                    else: out[key][i, :count] = value
                else:
                    out[key][i] = value
            if 'code_target' in batch:
                out['code_target'][i] = batch['mirror_code_target'][i]
                out['code_valid'][i] = batch['mirror_code_valid'][i]
    # Alternate variants are not part of the SAM replay payload.
    out.pop('mirror_code_target', None)
    out.pop('mirror_code_valid', None)
    return out
