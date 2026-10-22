"""Explicitly unverified coordinate hypotheses for visual mapping review.

These are NOT training policies. A round trip or a good AE reconstruction
cannot turn one of these hypotheses into physical mounting evidence.
"""
import base64

import numpy as np
from PIL import Image, ImageDraw

from .bimanual_mirror import validate_spec
from .marker_viewer import overlay, png

FINGERS = ('index', 'middle', 'pinky', 'ring', 'thumb')
REVIEW_REASONS = {
    'opengraph': 'Named finger identities are known. The official SDK leaves the across-finger row direction unmeasured; the older dataset thumb convention is not certified.',
    'ftp_unit_bimanual': 'Hand ownership is known; corresponding fingertip mounts and local tactile-camera reflection axes have no verified calibration.',
    'baihu': 'Four stream identities are known; jaw correspondence and local image axes remain unverified separately for D-WHEEL, PIKA, and QINGLOONG.',
}
FLIPS = {
    'none': [[1, 0, 0], [0, 1, 0]],
    'x': [[-1, 0, 1], [0, 1, 0]],
    'y': [[1, 0, 0], [0, -1, 1]],
    'xy': [[-1, 0, 1], [0, -1, 1]],
}


def candidates(source):
    """A bounded review family, not an exhaustive mounting calibration."""
    result = []
    if source == 'opengraph':
        # User-requested display-coordinate reflection: reflect each 4x4
        # finger grid independently. Reversing the flattened hand would also
        # reverse finger identities, which is incorrect.
        horizontal=np.arange(80).reshape(5,4,4)[:,:,::-1].reshape(-1).tolist()
        result.append(dict(id='horizontal_all_fingers',
            label='Swap hands + reflect every finger grid left ↔ right',
            spec=dict(kind='taxels',hand_permutation=[1,0],
                channel_permutation=[horizontal,horizontal])))
        # Keep the release's named finger order. SDK wire/canonical finger
        # permutations do not apply to these already labeled channels.
        for rows in (False, True):
            for thumb_rows in (False, True):
                for thumb_columns in (False, True):
                    mapping = np.arange(80).reshape(5, 4, 4)
                    mapping = np.stack([g[::-1 if (thumb_rows if i == 4 else rows) else 1,
                                                  ::-1 if i == 4 and thumb_columns else 1]
                                        for i, g in enumerate(mapping)]).reshape(-1).tolist()
                    key = f'rows{int(rows)}_thumbrows{int(thumb_rows)}_thumbcols{int(thumb_columns)}'
                    result.append(dict(id=key,
                        label=f'Long-finger rows {"reverse" if rows else "keep"}; thumb rows '
                              f'{"reverse" if thumb_rows else "keep"}, columns '
                              f'{"reverse" if thumb_columns else "keep"}',
                        spec=dict(kind='taxels', hand_permutation=[1, 0],
                                  channel_permutation=[mapping, mapping])))
    elif source == 'ftp_unit_bimanual':
        for axis, transform in FLIPS.items():
            result.append(dict(id=axis, label=f'Swap hands; local axes: {axis}',
                spec=dict(kind='markers', surface_permutation=[1, 0],
                          surface_transforms=[transform, transform])))
    elif source == 'baihu':
        # The two hand-exchanging surface permutations must remain separate
        # hypotheses. Neither the names nor sensor-grid symmetry decides them.
        for ownership, permutation in [('keep_jaw', [2, 3, 0, 1]), ('swap_jaw', [3, 2, 1, 0])]:
            for a, first in FLIPS.items():
                for b, second in FLIPS.items():
                    transforms = [None] * 4
                    transforms[0] = transforms[permutation[0]] = first
                    transforms[1] = transforms[permutation[1]] = second
                    result.append(dict(id=f'{ownership}_{a}_{b}',
                        label=f'{ownership.replace("_", " ")}; pair 1 axes {a}; pair 2 axes {b}',
                        spec=dict(kind='markers', surface_permutation=permutation,
                                  surface_transforms=transforms)))
    for entry in result:
        validate_spec(entry['spec'])
    return result


def candidate_spec(source, identifier):
    for entry in candidates(source):
        if identifier == entry['id']:
            return entry['spec']
    raise ValueError('Unknown coordinate-review hypothesis; this is not a verified training mapping')


def marker_payload(row, batch, labels, frame, prediction=None):
    """Render transformed reference geometry and signed vectors at native size."""
    h, w = row.get('native_sensor_shape', row['target_shape'][-2:])
    reference = batch['marker_reference'].cpu().numpy() * np.array([w - 1, h - 1])
    spacing = batch['marker_spacing'].cpu().numpy()
    surface = batch['marker_surface'].cpu().numpy()
    target = batch['marker_target'][0].cpu().numpy() * spacing[:, None]
    valid = batch['marker_valid'][0].cpu().numpy() & np.isfinite(target).all(-1)
    if prediction is not None:
        prediction = np.asarray(prediction, np.float32) * spacing[:, None]
        if prediction.shape != target.shape or not np.isfinite(prediction).all():
            raise ValueError('Invalid transformed marker reconstruction')
    error = None if prediction is None else np.abs(prediction - target)
    panels = []
    for sid, label in enumerate(labels):
        selected = surface == sid
        background = np.full((h, w, 3), (20, 25, 31), np.uint8)
        heat = Image.new('I', (w, h))
        if prediction is not None:
            draw = ImageDraw.Draw(heat)
            endpoint = np.linalg.norm(prediction - np.nan_to_num(target), axis=-1) / spacing
            for p, e in zip(reference[selected & valid], endpoint[selected & valid]):
                draw.ellipse((p[0]-4, p[1]-4, p[0]+4, p[1]+4),
                             fill=int(np.clip(e, 0, 1)*65535))
        supported = selected & valid
        panels.append(dict(label=label,
            original=png(overlay(background, reference[selected], target[selected],
                                 valid[selected], (30, 235, 255), gain=1)),
            reconstruction=None if prediction is None else png(overlay(background, reference[selected],
                prediction[selected], np.ones(selected.sum(), bool), (255, 175, 30), gain=1)),
            error_u16=None if prediction is None else base64.b64encode(np.asarray(heat, dtype='<u2').tobytes()).decode(),
            mae=float(error[supported].mean()) if error is not None and supported.any() else None))
    return dict(frame=frame, source=row['source'], target_kind='optical_rgb',
        target_shape=list(row['target_shape']), labels=labels, optical_panels=panels,
        original=None, reconstruction=None, error=None,
        mae=float(error[valid].mean()) if error is not None and valid.any() else None,
        hand_mae=[p['mae'] for p in panels], contact_mae=None, contact_f1=None,
        marker_displacement=dict(units='native_pixels', arrow_gain=1, reference_frames=[0]*len(labels),
            reference_is_verified_unloaded=False, tracked_fraction=float(valid.mean()),
            surface=surface.tolist(), reference_px=reference.tolist(),
            measured_px=[p.tolist() if ok else None for p, ok in zip(target, valid)],
            predicted_px=None if prediction is None else prediction.tolist(),
            background='retained sensor geometry; no tactile image access'))
