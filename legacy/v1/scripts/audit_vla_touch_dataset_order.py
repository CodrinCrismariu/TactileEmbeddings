#!/usr/bin/env python3
"""Recover tactile chronology by exact pixels, preserving the FTP archives.

Match complete episode multisets of SHA256 image hashes to numbered raw JPEGs.
Only unique episode matches and consecutive raw frame numbers are accepted.
Repeated identical frames are interchangeable for display, and are counted.
Output is a reversible, episode-local chronological-to-stored frame lookup.
"""
import argparse
from collections import Counter, defaultdict, deque
import hashlib
import io
import json
from pathlib import Path
import re
import time
import zipfile

import numpy as np
from PIL import Image

from tactilelatent.native_optical import native_images
from tactilelatent.sensor_data import ftp_group


def digest(image):
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


def signature(hashes):
    return tuple(sorted(Counter(hashes).items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path('processing/joint_all/manifest_sensor_roles_corrected_v2.json'))
    parser.add_argument('--raw-root', type=Path, default=Path('datasets/AllenBi21__VLA-Touch/raw_data'))
    parser.add_argument('--output', type=Path, default=Path('processing/audits/vla_touch_native_dynamics/dataset_frame_order.json'))
    args = parser.parse_args()
    started = time.monotonic()
    rows = [r for r in json.loads(args.manifest.read_text())['episodes'] if r['source'] == 'ftp_vla_touch']
    by_family = {'water_cup.zip': 'Cup_', 'wipe.zip': 'Wipe_', 'mango.zip': 'Mango'}
    pattern = re.compile(r'(^|/)episode_(\d+)/gelsight/gel_(\d+)\.jpg$')
    corrections, unmatched, scene_checks = {}, [], []
    archives = {}
    for archive_name, prefix in by_family.items():
        archive_path = args.raw_root / archive_name
        archives[archive_name] = {'path': str(archive_path), 'bytes': archive_path.stat().st_size,
                                 'sha256': hashlib.file_digest(archive_path.open('rb'), 'sha256').hexdigest()}
        candidates = defaultdict(list)
        for row in rows:
            if not row['episode_id'].startswith(prefix):
                continue
            hashes = [digest(image) for image in native_images(row, range(row['frames']))[0]]
            candidates[signature(hashes)].append((row, hashes))
        print(json.dumps({'family': prefix, 'ftp_episodes_indexed': sum(map(len, candidates.values()))}), flush=True)
        with zipfile.ZipFile(archive_path) as archive:
            episodes = defaultdict(list)
            for name in archive.namelist():
                match = pattern.search(name)
                if match:
                    episodes[int(match.group(2))].append((int(match.group(3)), name))
            for raw_episode, files in sorted(episodes.items()):
                files.sort()
                hashes = [digest(np.asarray(Image.open(io.BytesIO(archive.read(name))).convert('RGB')))
                          for _, name in files]
                matches = candidates.get(signature(hashes), [])
                consecutive = [number for number, _ in files] == list(range(len(files)))
                if len(matches) != 1 or not consecutive:
                    unmatched.append({'archive': archive_name, 'raw_episode': raw_episode,
                                      'raw_frames': len(files), 'candidate_episodes': [r['episode_id'] for r, _ in matches],
                                      'consecutive_frames': consecutive})
                    continue
                row, stored_hashes = matches[0]
                if row['episode_id'] in corrections:
                    raise ValueError('Multiple raw episodes matched one FTP episode')
                positions = defaultdict(deque)
                for index, value in enumerate(stored_hashes):
                    positions[value].append(index)
                order = [positions[value].popleft() for value in hashes]
                unique = len(set(hashes))
                corrections[row['episode_id']] = {
                    'chronological_to_stored_index': order, 'raw_archive': archive_name,
                    'target_start': row['target_start'], 'zarr_name': row['zarr_name'],
                    'tar_path': row['tar_path'],
                    'raw_episode': raw_episode, 'frames': len(files), 'unique_images': unique,
                    'positions_changed': sum(i != value for i, value in enumerate(order)),
                    'chronology_pixel_changes': sum(hashes[i] != stored_hashes[i] for i in range(len(files))),
                    'exact_pixel_multiset_match': True,
                    'duplicate_frames': len(files) - unique}
                # Verify camera chronology independently at the first/last raw indices.
                episode_root = files[0][1].split('/gelsight/')[0]
                group = ftp_group(row)
                for camera, key in row['views'].items():
                    for number in sorted({0, len(files) - 1}):
                        stored = np.asarray(group['data/' + key][row['target_start'] + number])
                        if stored.ndim == 4:
                            stored = stored[0]
                        camera_matches = []
                        comparable = False
                        for folder in ('camera1', 'camera2'):
                            name = f'{episode_root}/{folder}/rgb_{number}.jpg'
                            if name not in archive.NameToInfo:
                                continue
                            image = np.asarray(Image.open(io.BytesIO(archive.read(name))).convert('RGB'))
                            if image.shape == stored.shape:
                                comparable = True
                                if np.array_equal(image, stored):
                                    camera_matches.append(folder)
                        scene_checks.append({'episode_id': row['episode_id'], 'camera': camera,
                                             'frame': number, 'exact_raw_camera_matches': camera_matches,
                                             'comparison_possible': comparable,
                                             'note': None if comparable else 'FTP scene dimensions differ from raw camera; no resizing performed for this audit.'})
            print(json.dumps({'family': prefix, 'raw_episodes': len(episodes),
                              'matched_total': len(corrections), 'unmatched_total': len(unmatched)}), flush=True)
    missing = [r['episode_id'] for r in rows if r['episode_id'] not in corrections]
    result = {'source': 'ftp_vla_touch', 'manifest': str(args.manifest),
              'method': 'Complete native RGB SHA256 multiset match; numeric gel_N.jpg chronology; no resizing or similarity inference.',
              'episodes': corrections, 'unmatched_raw_episodes': unmatched, 'unmatched_ftp_episodes': missing,
              'archives': archives, 'scene_checks': scene_checks,
              'summary': {'ftp_episodes': len(rows), 'matched_episodes': len(corrections),
                          'reordered_episodes': sum(r['chronology_pixel_changes'] > 0 for r in corrections.values()),
                          'constant_episodes': sum(r['unique_images'] == 1 for r in corrections.values()),
                          'frames': sum(r['frames'] for r in corrections.values()),
                          'positions_changed': sum(r['positions_changed'] for r in corrections.values()),
                          'scene_checks': len(scene_checks),
                          'scene_check_failures': sum(r['comparison_possible'] and not r['exact_raw_camera_matches'] for r in scene_checks),
                          'scene_checks_skipped': sum(not r['comparison_possible'] for r in scene_checks),
                          'seconds': time.monotonic() - started}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result['summary']), flush=True)


if __name__ == '__main__':
    main()
