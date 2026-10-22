"""Exact EgoPressure UV-to-surface metadata; the MANO UV files are user supplied.

The dataset already provides posed vertices. No MANO pickle, model evaluation,
approximate hand geometry, or inferred texel-to-finger assignment is needed.
"""
from functools import lru_cache
import json
import os
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pyarrow.parquet as pq

from .aligned_frames import file_hash


SETUP_URL = 'https://github.com/eth-siplab/EgoPressure/blob/master/mano_models/README.md'


def model_directories():
    configured = os.environ.get('EGOPRESSURE_MANO_PATH')
    return ([Path(configured).expanduser()] if configured else []) + [
        Path('mano_models'), Path('datasets/eth-siplab__EgoPressure/mano_models'),
        Path('processing/upstream/EgoPressure/mano_models')]


@lru_cache(maxsize=4)
def _read_uv(path, stamp, size):
    """Keep separate vertex and UV indices so chart seams join on the surface."""
    uv, faces, uv_faces = [], [], []
    for line in Path(path).read_text().splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == 'vt':
            uv.append([float(fields[1]), float(fields[2])])
        elif fields[0] == 'f':
            if len(fields) != 4:
                raise ValueError('MANO UV mesh must have triangular faces')
            corners = [token.split('/') for token in fields[1:]]
            if any(len(c) < 2 or not c[1] for c in corners):
                raise ValueError('MANO mesh lacks the exact UV indices')
            faces.append([int(c[0])-1 for c in corners])
            uv_faces.append([int(c[1])-1 for c in corners])
    coordinates = np.asarray(uv, np.float64)
    face_array = np.asarray(faces, np.int64)
    uv_array = np.asarray(uv_faces, np.int64)
    if (coordinates.ndim != 2 or coordinates.shape[1] != 2 or not len(face_array) or
            not np.isfinite(coordinates).all() or coordinates.min() < 0 or coordinates.max() > 1 or
            face_array.min() < 0 or face_array.max() >= 778 or uv_array.min() < 0 or
            uv_array.max() >= len(coordinates)):
        raise ValueError('MANO UV mesh has invalid coordinates or indices')
    return dict(vertex_count=778, faces=faces, uv_faces=uv_faces, uv=uv,
                sha256=file_hash(path), uv_origin='bottom', image_origin='top',
                pixel_mapping='column=u*(W-1); row=(1-v)*(H-1)',
                setup_url=SETUP_URL)


def surface_assets():
    sides = {}
    for side in ('left', 'right'):
        filename = f'MANO_UV_{side}.obj'
        path = next((p/filename for p in model_directories() if (p/filename).is_file()), None)
        if path is None:
            sides[side] = dict(available=False, required_file=filename)
            continue
        try:
            stat = path.stat()
            mesh = _read_uv(str(path.resolve()), stat.st_mtime_ns, stat.st_size)
            sides[side] = dict(available=True, mesh=mesh, filename=filename)
        except (ValueError, OSError, IndexError) as error:
            sides[side] = dict(available=False, required_file=filename, error=str(error))
    return dict(sides=sides, setup_url=SETUP_URL,
                directory='mano_models/', dataset_vertices_used=True,
                needs_mano_pickle=False, native_texture_shape=[224,224])


@lru_cache(maxsize=2)
def _annotation_rows(path, stamp, size):
    table = pq.read_table(path, columns=['frame', 'has_annotation', 'hand_side',
                                        'vertices', 'joint_position'])
    ids = table['frame'].to_numpy()
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate native annotation frame identifiers')
    # Keep Arrow's compact float buffers instead of millions of Python floats.
    return table, {int(value):i for i,value in enumerate(ids)}


def surface_frame(viewer, episode, frame):
    with viewer.lock:
        row = viewer._episode(episode, frame)
        if row['source'] != 'egopressure':
            raise ValueError('Hand surface is available only for EgoPressure')
        source_id = int(row['source_frame_ids'][frame])
        result = dict(frame=frame, source_frame_id=source_id, hand_side=row['hand_side'],
                      pose_source='recorded annotation; identical geometry for all pressure comparisons')
        path = Path(row['annotation_path'])
        stat = path.stat()
        table, indices = _annotation_rows(str(path), stat.st_mtime_ns, stat.st_size)
        index = indices.get(source_id)
        annotation = None if index is None else table.slice(index,1).to_pylist()[0]
        if not annotation or not annotation['has_annotation']:
            return dict(result, available=False, reason='No recorded hand pose at this native frame; use UV atlas')
        if annotation['hand_side'] != row['hand_side']:
            raise ValueError('Recorded hand side disagrees with the selected episode')
        vertices = np.asarray(annotation['vertices'], np.float64)
        joints = np.asarray(annotation['joint_position'], np.float64)
        if (vertices.size != 778*3 or joints.size != 21*3 or
                not np.isfinite(vertices).all() or not np.isfinite(joints).all()):
            raise ValueError('Invalid recorded MANO geometry')
        return dict(result, available=True, vertices=vertices.reshape(778,3).tolist(),
                    joints=joints.reshape(21,3).tolist())


def make_pressure_handler(base, viewer):
    class Handler(base):
        def do_GET(self):
            url = urlparse(self.path)
            route = '/api/'+url.path.split('/api/',1)[1] if '/api/' in url.path else url.path
            if not (route.endswith('/pressure-surface-viewer.js') or
                    route in ('/api/egopressure/surface-assets','/api/egopressure/surface')):
                return super().do_GET()
            try:
                if route.endswith('/pressure-surface-viewer.js'):
                    content = Path(__file__).with_name('pressure_surface_viewer.js').read_bytes()
                    mime = 'text/javascript; charset=utf-8'
                else:
                    query = parse_qs(url.query)
                    value = (surface_assets() if route.endswith('surface-assets') else
                             surface_frame(viewer, int(query.get('episode',['0'])[0]),
                                           int(query.get('frame',['0'])[0])))
                    content = json.dumps(value, allow_nan=False).encode()
                    mime = 'application/json'
                self.send_response(200)
            except (BrokenPipeError, ConnectionResetError):
                return
            except Exception as error:
                content = json.dumps(dict(error=str(error))).encode()
                mime = 'application/json'
                self.send_response(400)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(content)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            try:
                self.wfile.write(content)
            except (BrokenPipeError, ConnectionResetError):
                pass
    return Handler
