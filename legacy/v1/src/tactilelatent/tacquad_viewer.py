"""Native TacQuad archive browser with independent, explicitly unpaired streams."""
import json
from pathlib import Path
import re
import sqlite3
import threading
from io import BytesIO

import numpy as np
from PIL import Image

from .acquisition_archives import TacQuadReader
from .zero_reference import FixedReferences


def natural_key(value):
    return [int(v) if v.isdigit() else v for v in re.split(r'(\d+)', value)]


class TacQuadBrowser:
    def __init__(self, manifest, zero_references=None):
        self.path = Path(manifest)
        self.lock = threading.RLock()
        self.reader = None
        self.contacts = None
        self.members = {}
        self.zero_references = FixedReferences() if zero_references is None else FixedReferences(zero_references)

    def catalog(self):
        with self.lock:
            if not self.path.exists():
                return dict(available=False, contacts=[])
            if self.contacts is None:
                self.manifest = json.loads(self.path.read_text())
                self.contacts = [json.loads(line) for line in
                    Path(self.manifest['contact_intervals']).read_text().splitlines() if line.strip()]
            return dict(available=True, framewise_alignment_verified=False,
                contacts=[dict(id=i, **row) for i, row in enumerate(self.contacts)])

    def streams(self, contact):
        with self.lock:
            self.catalog()
            if self.contacts is None or not 0 <= contact < len(self.contacts):
                raise ValueError('Unknown TacQuad contact')
            row = self.contacts[contact]
            subset = {'contact_fine.csv': 'data_fine', 'contact_indoor.csv': 'data_indoor',
                      'contact_outdoor.csv': 'data_outdoor'}[Path(row['csv_member']).name]
            prefix = f'{subset}/{row["folder"]}/'
            if contact not in self.members:
                # Prefix range avoids interpreting object names as SQL LIKE patterns.
                with sqlite3.connect(f'file:{self.manifest["member_index"]}?mode=ro', uri=True) as database:
                    names = [r[0] for r in database.execute(
                        'SELECT name FROM members WHERE name >= ? AND name < ?', (prefix, prefix+'\uffff'))]
                groups = {}
                for name in names:
                    relative = name[len(prefix):]
                    if '/' not in relative:
                        continue
                    stream, filename = relative.split('/', 1)
                    suffix = Path(filename).suffix.lower()
                    if suffix not in ('.png', '.jpg', '.jpeg', '.npy'):
                        continue
                    # Different Tac3D payload kinds are distinct streams; never
                    # alternate force arrays and marker positions on one timeline.
                    if suffix == '.npy':
                        kind = re.sub(r'[_-]?\d+$', '', Path(filename).stem)
                        stream += '/' + kind
                    groups.setdefault(stream, []).append(name)
                self.members[contact] = {key: sorted(value, key=natural_key) for key, value in groups.items()}
            groups = self.members[contact]
            return dict(contact=contact, description=row['text'], alignment_level=row['alignment_level'],
                intervals=row['intervals'], interval_valid=row['interval_valid'],
                missing_reasons=row.get('interval_missing_reasons'), framewise_alignment_verified=False,
                streams=[dict(id=key, frames=len(names), kind='array' if names[0].endswith('.npy') else 'image',
                    first=Path(names[0]).name, last=Path(names[-1]).name,
                    frame_names=[Path(name).name for name in names]) for key, names in groups.items()])

    def _member(self, contact, stream, index):
        self.streams(contact)
        names = self.members[contact].get(stream, [])
        if not 0 <= index < len(names):
            raise ValueError('Unknown native stream or frame; no neighbouring frame is substituted')
        if self.reader is None:
            self.reader = TacQuadReader(self.path)
        return names[index]

    def member(self, contact, stream, index):
        with self.lock:
            name = self._member(contact, stream, index)
            mime = 'image/png' if name.endswith('.png') else ('image/jpeg' if name.endswith(('.jpg', '.jpeg'))
                    else 'application/octet-stream')
            return self.reader.read(name), mime

    def array(self, contact, stream, index):
        with self.lock:
            name = self._member(contact, stream, index)
            if not name.endswith('.npy'):
                raise ValueError('Select a released numeric array stream')
            data = self.reader.array(name)
            if not np.issubdtype(data.dtype, np.number) or data.size > 100_000:
                raise ValueError('Array is not a bounded numeric payload; download the native member instead')
            return dict(member=name, shape=list(data.shape), dtype=str(data.dtype),
                values=[v.item() if np.isfinite(v) else None for v in data.flat],
                normalization='none; native values')

    def reference(self, contact, stream, index, *, render=False):
        """Fixed optical difference only; native downloads stay byte-identical."""
        with self.lock:
            if stream not in ('gelsight', 'digit', 'duragel'):
                raise ValueError('No verified optical reference for this stream')
            name = self._member(contact, stream, index)
            image = np.asarray(self.reader.image(name))
            result = self.zero_references.optical(stream, image, self.manifest['revision'])
            if result is None:
                raise ValueError('No fixed reference matches this release and native image format')
            residual = result.pop('residual')
            if not render:
                return dict(result, member=name, mean_absolute_difference=float(np.abs(residual).mean()),
                    max_absolute_difference=float(np.abs(residual).max()),
                    normalization='signed native RGB minus one fixed sensor template; no episode refitting')
            # Display magnitude on a fixed 0..64 RGB-count scale, never per-frame min/max.
            magnitude = np.abs(residual).mean(axis=-1) / result['display_difference_max']
            magnitude = np.clip(magnitude, 0, 1)
            rgb = np.stack([magnitude, np.sqrt(magnitude) * .65, np.zeros_like(magnitude)], axis=-1)
            buffer = BytesIO()
            Image.fromarray(np.rint(rgb * 255).astype(np.uint8)).save(buffer, 'PNG')
            return buffer.getvalue(), 'image/png'
