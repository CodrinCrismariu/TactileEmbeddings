#!/usr/bin/env python3
"""Build source-wide references without rewriting any acquisition targets.

Run: .venv/bin/python scripts/build_zero_references.py
The reviewed frame list is explicit: first frames are NOT assumed unloaded.
Changing this list creates a new calibration study, not an automatic refit.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from tactilelatent.aligned_frames import file_hash
from tactilelatent.tacquad_viewer import TacQuadBrowser
from tactilelatent.zero_reference import SCHEMA, save_registry


NOTES = {
    'trex': ('Released nominal zero; upstream episode taring',
        'The release subtracts each episode\'s first-15-clean-frame median. The saved offsets are not supplied. '
        'Use the released numeric zero unchanged; no additional episode tare is performed. '
        'A true common pre-tare physical reference cannot be recovered from these targets.'),
    'feel': ('Released contact-estimation zero; upstream drift correction',
        'Processed sensor columns use upstream piecewise zero correction and are contact-estimation values, '
        'not calibrated force. Preserve numeric zero; a physical global offset for the raw piezoresistors '
        'has not been established. Missing channels remain missing.'),
    'tdex': ('Estimated common unloaded reference; limited calibration sample',
        'One 720-component offset fitted to three visually screened, open-hand recordings. '
        'No episode refitting. Physical gains and any upstream acquisition tare are unverified. '
        'The reviewed sample is too small to certify all episodes as calibrated.'),
    'hrdexdb_allegro_v5': ('Common nominal reference; unloaded offsets vary',
        'One median offset across visually screened training snapshots. Unloaded readings vary between recordings, '
        'so this is not a reliable exact-contact zero. The empirical envelope includes offset drift and sensor effects.'),
    'hrdexdb_inspire_f1': ('Common nominal reference; unloaded offsets vary',
        'One median force offset across visually screened training snapshots. Unloaded force readings vary substantially '
        'between recordings. Direction coordinates (including -1) are not forces and are preserved, without zero subtraction.'),
}


def family_ids(source, count):
    if source == 'trex':
        return (np.arange(count) % 6 >= 3).astype(int)
    if source in ('tdex', 'hrdexdb_inspire_f1'):
        return np.arange(count) % 3
    return np.zeros(count, dtype=int)


def load_values(row):
    target = row['targets']
    if file_hash(target['path']) != target['sha256']:
        raise ValueError('Native target checksum changed: ' + row['episode_id'])
    with np.load(target['path'], allow_pickle=False) as data:
        return data['values'].copy(), data['valid'].copy()


def numeric_references(root, evidence_root):
    candidates = json.loads((evidence_root / 'candidate_window_statistics.json').read_text())
    references, selections, audit = {}, {}, {}
    active_policy = Path('configs/data_queue_active.json')
    removed = json.loads(active_policy.read_text()).get('removed_sources', {}) if active_policy.exists() else {}
    for acquisition in ('feel', 'tdex', 'hrdexdb', 'trex'):
        if acquisition in removed:
            continue
        path = root / acquisition / 'manifest.json'
        manifest = json.loads(path.read_text())
        grouped = defaultdict(list)
        for row in manifest['episodes']:
            source = 'hrdexdb_' + row['embodiment'] if acquisition == 'hrdexdb' else acquisition
            if acquisition == 'feel':
                source = 'feel_' + row['environment'].lower()
            grouped[source].append(row)
        for reference_key, rows in grouped.items():
            source = 'feel' if acquisition == 'feel' else reference_key
            if source not in NOTES:
                continue
            shape = rows[0]['targets']['shape'][1:]
            channels = rows[0]['targets']['channel_order']
            count = len(channels)
            selected = []
            if acquisition == 'hrdexdb':
                # These exact snapshots were reviewed in the saved montage pages.
                by_name = {x['episode']: x for x in candidates[rows[0]['embodiment']]}
                selected = [dict(episode=r['episode_id'], split=r['split'],
                    frames=[by_name[r['episode_id']]['indices'][0]],
                    evidence=r['embodiment'] + '_candidates_' + str(i // 12) + '.png')
                    for i, r in enumerate(rows)]
            elif acquisition == 'tdex':
                by_name = {x['episode']: x for x in candidates['tdex']}
                selected = [dict(episode=name, split='train',
                    frames=[by_name[name]['indices'][j] for j in (0, 7, 14)],
                    evidence='tdex_selected_windows.png')
                    for name in ('demonstration_6', 'demonstration_8', 'demonstration_10')]
            by_name = {r['episode_id']: r for r in rows}
            samples = {'train': [], 'val': []}
            for selection in selected:
                row = by_name[selection['episode']]
                if row['split'] != selection['split']:
                    raise ValueError('Calibration split changed')
                values, valid = load_values(row)
                x = values[selection['frames']].reshape(len(selection['frames']), count).astype(float)
                mask = valid[selection['frames']].reshape(x.shape) & np.isfinite(x)
                if not mask.all():
                    raise ValueError('Selected snapshot contains missing values')
                samples[row['split']].append(x)
                selection.update(target_sha256=row['targets']['sha256'],
                    rgb_sha256=row['views']['0']['sha256'])
            offset = np.zeros(count, dtype=float)
            envelope = None
            subtractable = np.ones(count, dtype=bool)
            if source == 'hrdexdb_inspire_f1':
                subtractable[2::3] = False
            if samples['train']:
                # Equal episode weighting: frames -> episode median -> source median.
                offset = np.median([np.median(x, axis=0) for x in samples['train']], axis=0)
                offset[~subtractable] = 0
                x = np.concatenate(samples['train']) - offset
                envelope = [[float(lo), float(hi)] if ok else None for lo, hi, ok in
                    zip(x.min(axis=0), x.max(axis=0), subtractable)]
            # Fixed training-only display limits; no range changes when switching episodes.
            maximum = np.zeros(count)
            training_frames = 0
            inventory = []
            for row in rows:
                if row['targets']['shape'][1:] != shape or row['targets']['channel_order'] != channels:
                    raise ValueError('Mixed target layouts inside one sensor source')
                inventory.append(dict(episode=row['episode_id'], split=row['split'],
                    target_sha256=row['targets']['sha256']))
                if row['split'] != 'train':
                    if file_hash(row['targets']['path']) != row['targets']['sha256']:
                        raise ValueError('Native validation target checksum changed: ' + row['episode_id'])
                    continue
                values, valid = load_values(row)
                values = values.reshape(len(values), count).astype(float)
                valid = valid.reshape(values.shape) & np.isfinite(values)
                maximum = np.maximum(maximum, np.where(valid, np.abs(values - offset), 0).max(axis=0))
                training_frames += len(values)
            families = family_ids(source, count)
            limits = [float(maximum[families == group].max()) or 1.0 for group in families]
            validation = None
            if samples['val'] and envelope is not None:
                check = np.concatenate(samples['val']) - offset
                low = np.array([e[0] if e else -np.inf for e in envelope])
                high = np.array([e[1] if e else np.inf for e in envelope])
                validation = dict(episodes=len(samples['val']),
                    component_fraction_outside_training_envelope=float(
                        ((check[:, subtractable] < low[subtractable]) |
                         (check[:, subtractable] > high[subtractable])).mean()),
                    max_abs_force_residual=float(np.abs(check[:, subtractable]).max()))
            references[reference_key] = dict(source=source, revision=manifest.get('revision'), target_shape=shape,
                channel_order=channels, offset=offset.tolist(), display_limits=limits,
                subtractable=subtractable.tolist(), unloaded_envelope=envelope,
                status=NOTES[source][0], explanation=NOTES[source][1],
                physical_zero_verified=False, deadband=False, episode_refitting=False,
                units=rows[0].get('target_units'),
                reference_fit_episodes=[s['episode'] for s in selected if s['split'] == 'train'],
                reference_fit_observations=sum(len(s['frames']) for s in selected if s['split'] == 'train'),
                display_fit_episodes=sum(r['split'] == 'train' for r in rows),
                display_fit_frames=training_frames, inventory_sha256=hashlib.sha256(
                    json.dumps(inventory, sort_keys=True).encode()).hexdigest(),
                manifest_sha256=file_hash(path), validation=validation)
            selections[reference_key] = selected
            audit[reference_key] = dict(episodes=len(rows), native_targets_unchanged=True,
                offset_min=float(offset[subtractable].min()), offset_max=float(offset[subtractable].max()),
                largest_unloaded_abs_residual=(max(max(abs(x) for x in e) for e in envelope if e)
                    if envelope is not None else None), validation=validation)
            print(reference_key, audit[reference_key], flush=True)
    return references, selections, audit


def optical_references(root, output):
    """One neutral template per optical sensor/native shape, not per object.

    Pre-contact images are annotation-based candidates, not certified force-free
    calibration. The contact's OWN interval is used; never another stream's time.
    """
    policy = Path('configs/data_queue_active.json')
    if policy.exists() and 'tacquad' in json.loads(policy.read_text()).get('removed_sources', {}):
        return {}, {}, dict(status='removed_by_user', diagnostics={}, skipped=[])
    if not (root / 'tacquad/manifest.json').exists():
        return {}, {}, dict(status='unavailable_or_removed', diagnostics={}, skipped=[])
    browser = TacQuadBrowser(root / 'tacquad/manifest.json')
    contacts = browser.catalog()['contacts']
    revision = browser.manifest['revision']
    buckets = defaultdict(list)
    skipped = []
    for i, row in enumerate(contacts):
        streams = {s['id']: s for s in browser.streams(i)['streams']}
        # Object-disjoint across subsets. The split is fixed before looking at images.
        split = 'val' if int(hashlib.sha256(('45:' + row['folder']).encode()).hexdigest(), 16) % 5 == 0 else 'train'
        for sensor, annotation in [('gelsight', 'gelsight_mini'), ('digit', 'digit'), ('duragel', 'duragel')]:
            interval = row['intervals'].get(annotation)
            if not row['interval_valid'].get(annotation) or not interval or interval[0] < 10:
                skipped.append(dict(contact=i, sensor=sensor, reason='No verified annotation margin of at least 10 frames'))
                continue
            stream = streams.get(sensor)
            if not stream or stream['first'] != '0.png':
                skipped.append(dict(contact=i, sensor=sensor, reason='Unverified native filename origin'))
                continue
            name = browser._member(i, sensor, 0)
            image = np.asarray(browser.reader.image(name)).copy()
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
                skipped.append(dict(contact=i, sensor=sensor, reason='Unexpected native optical format'))
                continue
            middle = (interval[0] + interval[1]) // 2
            filename = str(middle) + '.png'
            if filename not in stream['frame_names']:
                skipped.append(dict(contact=i, sensor=sensor, reason='Contact annotation cannot address native frame'))
                continue
            loaded_name = browser._member(i, sensor, stream['frame_names'].index(filename))
            loaded = np.asarray(browser.reader.image(loaded_name)).copy()
            if loaded.shape != image.shape:
                raise ValueError('Native optical shape changed inside one recording')
            key = sensor + ':' + 'x'.join(map(str, image.shape)) + ':' + str(image.dtype)
            buckets[key].append(dict(image=image, contact_image=loaded, object=row['folder'], split=split,
                member=name, contact_member=loaded_name, contact=i,
                sha256=hashlib.sha256(browser.reader.read(name)).hexdigest()))
        if i % 30 == 0:
            print('TacQuad candidates', i, '/', len(contacts), flush=True)
    references, audit, selections = {}, {}, {}
    for key, observations in buckets.items():
        train = [x for x in observations if x['split'] == 'train']
        if len({x['object'] for x in train}) < 5:
            skipped.append(dict(sensor=key, reason='Fewer than five training objects'))
            continue
        # Equal object weighting even when an object appears in multiple subsets.
        object_images = defaultdict(list)
        for x in train:
            object_images[x['object']].append(x['image'])
        template = np.median([np.median(v, axis=0) for v in object_images.values()], axis=0).astype(np.float32)
        filename = 'neutral_' + key.replace(':', '_') + '.npz'
        np.savez_compressed(output / filename, reference=template)
        diagnostic = {}
        for split in ('train', 'val'):
            rows = [x for x in observations if x['split'] == split]
            if not rows:
                continue
            pre = np.array([np.abs(x['image'].astype(float) - template).mean() for x in rows])
            contact = np.array([np.abs(x['contact_image'].astype(float) - template).mean() for x in rows])
            diagnostic[split] = dict(observations=len(rows), objects=len({x['object'] for x in rows}),
                precontact_mae_median=float(np.median(pre)), precontact_mae_max=float(pre.max()),
                contact_mae_median=float(np.median(contact)),
                fraction_contact_larger_than_own_precontact=float((contact > pre).mean()))
        passed = diagnostic['train']['fraction_contact_larger_than_own_precontact'] >= .9
        references[key] = dict(sensor=key.split(':')[0], revision=revision,
            target_shape=list(template.shape), native_dtype='uint8', file=filename,
            sha256=file_hash(output / filename),
            status=('Estimated fixed optical background; not calibrated pressure' if passed else
                    'Fixed optical background fails contact separation; do not use as no-contact calibration'),
            training_contact_separation_check_passed=passed,
            physical_zero_verified=False, episode_refitting=False,
            fit_objects=sorted(object_images), diagnostics=diagnostic,
            display_difference_max=64,
            explanation='RGB minus one frozen neutral texture, at native resolution. '
                'Annotation-based pre-contact candidates may include lighting drift. '
                'A nonzero residual alone does not establish contact; no threshold is applied.')
        selections[key] = [{k: v for k, v in x.items() if k not in ('image', 'contact_image')} for x in observations]
        audit[key] = diagnostic
        print(key, diagnostic, flush=True)
    if browser.reader:
        browser.reader.close()
    return references, selections, dict(diagnostics=audit, skipped=skipped,
        tac3d='Force/torque/displacement nominal zero retained; physical offset not independently verified. '
              'Absolute pos arrays are geometry, not no-contact measurements. '
              'No own-stream unloaded annotations: do not borrow optical timing or zero a position array.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('processing/data_queue_v2'))
    parser.add_argument('--output', type=Path, default=Path('processing/zero_reference_v1'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    numeric, selections, numeric_audit = numeric_references(args.root, args.output)
    optical, optical_selections, optical_audit = optical_references(args.root, args.output)
    policy = Path('configs/data_queue_active.json')
    removed = json.loads(policy.read_text()).get('removed_sources', {}) if policy.exists() else {}
    save_registry(args.output / 'references.json', dict(schema=SCHEMA, fit_split='train',
        episode_refitting=False, excluded_from_fit=['ftp_motiontrans', 'freetacman', *removed],
        retired_sources=removed,
        implementation={str(p): file_hash(p) for p in
            (Path(__file__), Path('src/tactilelatent/zero_reference.py'))},
        numeric=numeric, optical=optical,
        existing_sources=dict(egopressure='Physical released pressure zero already maps to normalized zero.',
            baihu='Marker displacement zero is the saved reference geometry; it is not certified unloaded. '
                  'Do not refit or change already-trained teacher coordinates.')))
    for name, payload in [('reviewed_numeric_frames.json', selections),
                          ('optical_candidates.json', optical_selections),
                          ('audit.json', dict(numeric=numeric_audit, optical=optical_audit))]:
        (args.output / name).write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    print('Saved', args.output / 'references.json', flush=True)


if __name__ == '__main__':
    main()
