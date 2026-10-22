"""Browse accepted acquisitions without changing historical loaders or targets.

New sensor sources are measurement-only until an actual teacher is trained.
IDs live in a separate namespace so growing acquisition manifests cannot move
historical episodes. Camera mappings use the acquisition's explicit indices.
"""
from functools import lru_cache
from io import BytesIO
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch
from PIL import Image

from .acquisition_data import _targets
from .expansion_viewer import ExpandedViewer
from .media import video_frames
from .zero_reference import FixedReferences
from .feel_hand_layout import feel_hand_layout
from .aligned_frames import file_hash
from .numeric_autoencoder import load_numeric_teacher


ROOT = Path('processing/data_queue_v2')
FIRST_ID = 1_000_000
NAMES = {'trex': 'T-Rex', 'feel': 'FEEL',
         'hrdexdb_allegro_v5': 'HRDexDB · Allegro V5',
         'hrdexdb_inspire_f1': 'HRDexDB · Inspire F1'}
ROLES = {'ego': 'Ego', 'left_wrist': 'Left wrist', 'right_wrist': 'Right wrist',
         'fixed_scene': 'Fixed scene camera'}


def source_id(source, row):
    return f'hrdexdb_{row["embodiment"]}' if source == 'hrdexdb' else source


def groups_for(source, channels):
    """Group released indices; do not invent taxel geometry or calibration."""
    if source == 'tdex':
        return [dict(label=f'Released XELA sensor {s}', indices=list(range(s*48, (s+1)*48)))
                for s in range(15)]
    if source == 'trex':
        return [dict(label=f'{hand.title()} {finger}', indices=list(range(i*6, (i+1)*6)))
                for i, (hand, finger) in enumerate((h, f) for h in ('left', 'right')
                    for f in ('thumb', 'index', 'middle', 'ring', 'pinky'))]
    if source == 'feel':
        # Ownership must come from explicit episode metadata, not channel count.
        return [dict(label='Hand identity unverified · released sensor indices',
                     indices=list(range(channels)), hand=None)]
    if source == 'hrdexdb_inspire_f1':
        labels = ('Little', 'Ring', 'Middle', 'Index', 'Thumb', 'Palm right', 'Palm middle', 'Palm left')
        return [dict(label=label, indices=list(range(i*3, (i+1)*3))) for i, label in enumerate(labels)]
    return [dict(label='Released tactile channels · anatomical mapping unverified',
                 indices=list(range(channels)))]


@lru_cache(maxsize=4)
def display_scales(path, digest, shared_source=None):
    """Display-only ranges; shared modes cover every episode frame."""
    data = _targets(path, digest)
    count = len(data['values'])
    indices = (np.arange(count) if shared_source is not None else
               np.unique(np.linspace(0, count-1, min(count, 2048)).astype(int)))
    values = data['values'][indices].reshape(len(indices), -1).astype(float)
    valid = data['valid'][indices].reshape(values.shape) & np.isfinite(values)
    maximum = np.max(np.where(valid, np.abs(values), 0), axis=0)
    if shared_source is not None:
        if shared_source == 'trex':
            if len(maximum) != 60:
                raise ValueError('Shared wrench display expects ten six-component fingertips')
            families = (np.arange(60) % 6 >= 3).astype(int)
        elif shared_source == 'hrdexdb_allegro_v5':
            if len(maximum) != 4:
                raise ValueError('Allegro display expects four released tactile channels')
            families = np.zeros(4, dtype=int)
        elif shared_source == 'hrdexdb_inspire_f1':
            if len(maximum) != 24:
                raise ValueError('Inspire F1 display expects eight three-component sensors')
            families = np.arange(24) % 3
        elif shared_source == 'feel':
            if len(maximum) not in (5, 12):
                raise ValueError('FEEL display expects the released Lab or Kitchen layout')
            families = np.zeros(len(maximum), dtype=int)
        else:
            raise ValueError('Unverified shared display channel convention')
        # Share only comparable quantities. In particular, force/moment and
        # normal/tangential/direction have independent ranges. Zero-only
        # channels must not inflate the common range through their fallback.
        maximum = np.asarray([maximum[families == family].max() for family in families])
    return np.where(maximum > 0, maximum, 1).tolist()


def mapped_frame(row, data, frame, camera):
    if camera not in row['views']:
        raise ValueError('Camera is unavailable for this episode')
    view = row['views'][camera]
    if 'rgb_frame_valid' in data and not bool(data['rgb_frame_valid'][frame]):
        raise ValueError('RGB timestamp is missing or ambiguous at this observation')
    if 'frame_indices_key' in view:
        if view.get('frame_indices_artifact') != row['targets']['path']:
            raise ValueError('Unexpected external RGB frame mapping')
        index = int(data[view['frame_indices_key']][frame])
        limit = view.get('frames')
    else:
        offset = int(view.get('frame_offset', 0))
        index = frame + offset
        # In offset-based manifests, frames is the episode segment length,
        # not the total length of the multi-episode video shard.
        limit = offset + view['frames'] if 'frames' in view else None
    if index < 0 or (limit is not None and index >= limit):
        raise ValueError('No recorded RGB frame maps to this tactile observation')
    return index


class AcquisitionViewer(ExpandedViewer):
    def __init__(self, *args, acquisition_root=None, zero_references=None, acquisition_teachers=None, **kwargs):
        self.acquisition_root = Path(acquisition_root) if acquisition_root is not None else ROOT
        self.acquisition_rows = {}
        self.acquisition_keys = {}
        self.acquisition_stamps = {}
        self.acquisition_manifests = {}
        self.acquisition_manifest_hashes = {}
        self.acquisition_teacher_root = Path(acquisition_teachers or 'processing/acquisition_teachers_v1')
        self.acquisition_networks = {}
        self.zero_references = FixedReferences() if zero_references is None else FixedReferences(zero_references)
        self.feel_display = json.loads(Path('configs/feel_display_v1.json').read_text())
        super().__init__(*args, **kwargs)

    def _scan_models(self):
        models=super()._scan_models()
        for folder in sorted(self.acquisition_teacher_root.glob('*')):
            if not (folder/'best.pt').exists() or not (folder/'best_metrics.json').exists():continue
            try:
                config=json.loads((folder/'config.json').read_text())
                metrics=json.loads((folder/'best_metrics.json').read_text())
                state=json.loads((folder/'status.json').read_text()) if (folder/'status.json').exists() else {}
                quality=json.loads((folder/'quality.json').read_text()) if (folder/'quality.json').exists() else {}
            except (OSError,ValueError):continue
            source=config['source'];spec=config['spec']
            if source not in NAMES or source=='hrdexdb_allegro_v5' or source=='feel' and spec['input_dim']==5:continue
            accepted=quality.get('accepted',False)
            models.append(dict(path=str(folder/'best.pt'),name=f'{folder.name} AE · {spec["latent_dim"]}D'+
                (' · baseline check pending' if not accepted else ''),source=source,config=config['model'],
                score=metrics['selection_score'],best_epoch=metrics['epoch'],completed_epochs=state.get('epoch',metrics['epoch']),
                validation_mae=metrics['val']['mae'],normalized_mae=metrics['val']['score'],
                per_quantity_mae=metrics['val']['per_quantity_mae'],target_shape=[spec['input_dim']],
                reduced_resolution=False,native_numeric_teacher=True,representation='native_numeric',preferred=accepted))
        return models

    def refresh_acquisitions(self):
        for source in ('feel', 'hrdexdb', 'trex'):
            path = self.acquisition_root / source / 'manifest.json'
            if not path.exists():
                continue
            stamp = (path.stat().st_mtime_ns, path.stat().st_size)
            if self.acquisition_stamps.get(source) == stamp:
                continue
            manifest = json.loads(path.read_text())
            if manifest.get('schema') != 'acquisition-native-v2':
                continue
            for native in manifest['episodes']:
                source_name = source_id(source, native)
                if source_name not in NAMES:
                    continue
                key = (source_name, native['episode_id'])
                if key not in self.acquisition_keys:
                    self.acquisition_keys[key] = FIRST_ID + len(self.acquisition_keys)
                enriched = dict(native, source=source_name, acquisition_source=source)
                if source == 'feel':
                    enriched['sensor_layout'] = feel_hand_layout(native['targets']['channel_order'],
                        environment=native.get('environment'), revision=manifest.get('revision'))
                self.acquisition_rows[self.acquisition_keys[key]] = enriched
            self.acquisition_manifests[source] = manifest
            self.acquisition_manifest_hashes[source] = file_hash(path)
            self.acquisition_stamps[source] = stamp

    def catalog(self):
        with self.lock:
            result = super().catalog()
            self.refresh_acquisitions()
            represented = set()
            for index, row in self.acquisition_rows.items():
                source = row['source']
                direct=source=='hrdexdb_allegro_v5' or source=='feel' and row['targets']['shape'][1:]==[5]
                reconstruction_available=not direct and any(m['source']==source and m.get('native_numeric_teacher')
                    and m['target_shape']==row['targets']['shape'][1:] for m in result['models'])
                result['episodes'].append(dict(id=index, name=row['episode_id'], source=source,
                    split=row['split'], frames=row['frames'], cameras=sorted(row['views']),
                    camera_labels={k: ROLES.get(v['role'], v['role']) for k, v in row['views'].items()},
                    sensor_videos=[], measured_only=True,target_shape=row['targets']['shape'][1:],
                    direct_prediction_only=direct,reconstruction_available=reconstruction_available))
                if 'sensor_layout' in row:
                    result['episodes'][-1]['sensor_layout'] = row['sensor_layout']
                if source in represented:
                    continue
                represented.add(source)
                target = row['targets']
                manifest = self.acquisition_manifests[row['acquisition_source']]
                result['datasets'].append(dict(id=source, name=NAMES[source],
                    sensor=row.get('target_units', 'Native released sensor readings'),
                    target_kind='native_numeric', target_shape=target['shape'][1:],
                    labels=row.get('sensor_layout', {}).get('labels', target['channel_order']),
                    hand_layout=None, measured_only=True, episode_dependent_layout=source == 'feel',
                    reconstruction_available=any(m['source']==source and m.get('native_numeric_teacher') for m in result['models']),
                    direct_prediction_only=source=='hrdexdb_allegro_v5',
                    measurements_available=True, groups=row.get('sensor_layout', {}).get('groups',
                        groups_for(source, len(target['channel_order']))),
                    acquisition_revision=manifest.get('revision'),
                    display_policy=(self.feel_display if source == 'feel' and
                        manifest.get('revision') == self.feel_display['revision'] else None),
                    acquisition_complete=manifest.get('complete_release', source != 'trex')))
            result['acquisition_status'] = {}
            for source in ('feel', 'hrdexdb', 'trex'):
                path = self.acquisition_root / source / 'status.json'
                if path.exists():
                    result['acquisition_status'][source] = json.loads(path.read_text())
            result['tacquad_available'] = False  # Removed at the user's request; old tabs may still inspect this flag.
            state=self.acquisition_teacher_root/'status.json'
            result['numeric_teacher_training']=json.loads(state.read_text()) if state.exists() else None
            return result

    def _reconstruct_numeric(self,row,values,valid,model):
        if row['source']=='hrdexdb_allegro_v5' or row['source']=='feel' and values.shape[-1]==5:
            raise ValueError('This sensor layout uses direct targets without an autoencoder')
        if not 0<=model<len(self.models):raise ValueError('No trained native reconstruction exists for this model')
        info=self.models[model]
        if (not info.get('native_numeric_teacher') or info['source']!=row['source'] or
                info['target_shape']!=list(values.shape[-1:])):
            raise ValueError('Select an autoencoder trained for this native sensor layout')
        path=Path(info['path']);stamp=(path.stat().st_mtime_ns,path.stat().st_size)
        cached=self.acquisition_networks.get(model)
        if cached is None or cached[0]!=stamp:
            network,state=load_numeric_teacher(path)
            if state['provenance']['manifest_sha256']!=self.acquisition_manifest_hashes[row['acquisition_source']]:
                raise ValueError('Numeric teacher acquisition provenance changed')
            cached=(stamp,network,state);self.acquisition_networks[model]=cached
        with torch.inference_mode():
            prediction=cached[1](torch.from_numpy(np.array(values,dtype=np.float32,copy=True)),
                torch.from_numpy(np.array(valid,dtype=bool,copy=True))).numpy()
        if not np.isfinite(prediction).all():raise ValueError('Nonfinite numeric reconstruction')
        return prediction,dict(model_id=model,name=info['name'],epoch=cached[2]['epoch'],
            code_dim=info['config']['latent_dim'],accepted=bool(cached[2].get('accepted')))

    def _episode(self, index, frame):
        if index < FIRST_ID:
            return super()._episode(index, frame)
        with self.lock:
            if index not in self.acquisition_rows:
                self.refresh_acquisitions()
            if index not in self.acquisition_rows:
                raise ValueError('Unknown acquisition episode; refresh the catalog')
            row = self.acquisition_rows[index]
            if not 0 <= frame < row['frames']:
                raise ValueError('Frame outside episode')
            return row

    def frame(self, episode, frame, model, prediction=None):
        if episode < FIRST_ID:
            return super().frame(episode, frame, model, prediction)
        if prediction is not None and model!=-1:
            raise ValueError('Choose either a visual prediction or a sensor reconstruction teacher')
        with self.lock:
            row = self._episode(episode, frame)
            target = row['targets']
            data = _targets(target['path'], target['sha256'])
            values = data['values'][frame].reshape(-1)
            valid = data['valid'][frame].reshape(-1) & np.isfinite(values)
            original = [v.item() if ok else None for v, ok in zip(values, valid)]
            reconstructed=None;errors=None;teacher=None;mae=None
            if prediction is not None:
                predicted=np.asarray(prediction,dtype=np.float32)
                if predicted.shape!=values.shape or not np.isfinite(predicted).all():raise ValueError('Invalid native visual prediction')
                error=np.abs(predicted-values)
                reconstructed=[float(v) if ok else None for v,ok in zip(predicted,valid)]
                errors=[float(v) if ok else None for v,ok in zip(error,valid)]
                mae=float(error[valid].mean()) if valid.any() else None
                teacher=dict(kind='visual',model_id=None,mae=mae)
            elif model!=-1:
                predicted,teacher=self._reconstruct_numeric(row,values[None],valid[None],model)
                error=np.abs(predicted[0]-values)
                reconstructed=[float(v) if ok else None for v,ok in zip(predicted[0],valid)]
                errors=[float(v) if ok else None for v,ok in zip(error,valid)]
                mae=float(error[valid].mean()) if valid.any() else None
                teacher['mae']=mae
            times = data['timestamps']
            mapping = {}
            for camera in row['views']:
                try:
                    mapping[camera] = mapped_frame(row, data, frame, camera)
                except ValueError:
                    mapping[camera] = None
            shared = {}
            if row['source'] in ('trex', 'hrdexdb_allegro_v5', 'hrdexdb_inspire_f1', 'feel'):
                limits = display_scales(target['path'], target['sha256'], row['source'])
                shared['shared_display_scales'] = limits
                if row['source'] == 'trex':
                    shared['shared_wrench_scales'] = limits  # Existing open viewer tabs.
            reference = self.zero_references.numeric(row['source'], data['values'].shape[1:],
                target['channel_order'], self.acquisition_manifests[row['acquisition_source']].get('revision'))
            return dict(source=row['source'], episode=episode, frame=frame,
                target_kind='native_numeric', target_shape=list(data['values'].shape[1:]),
                labels=row.get('sensor_layout', {}).get('labels', target['channel_order']),
                original=original, reconstruction=reconstructed, error=errors,
                mae=mae, hand_mae=[mae,None], contact_mae=None, contact_f1=None,
                acquisition=dict(native_dtype=str(values.dtype), units=row.get('target_units'),
                    direct_prediction_only=row['source']=='hrdexdb_allegro_v5' or row['source']=='feel' and len(values)==5,
                    display_scales=display_scales(target['path'], target['sha256']),
                    zero_reference=reference,
                    reconstruction=teacher,
                    display_policy=(self.feel_display if row['source'] == 'feel' and
                        self.acquisition_manifests[row['acquisition_source']].get('revision') == self.feel_display['revision'] else None),
                    sensor_layout=row.get('sensor_layout'), native_channel_order=target['channel_order'],
                    **shared,
                    valid_count=int(valid.sum()), channels=len(values),
                    timestamp_seconds=float(times[frame]), elapsed_seconds=float(times[frame]-times[0]),
                    source_frame_id=int(data.get('source_frame_ids', data['frame_indices'])[frame]),
                    camera_frame_indices=mapping,
                    alignment_error_seconds=(float(data['alignment_error_seconds'][frame])
                        if 'alignment_error_seconds' in data and np.isfinite(data['alignment_error_seconds'][frame]) else None),
                    groups=row.get('sensor_layout', {}).get('groups', groups_for(row['source'], len(values))),
                    measured_only=True, normalization='none; native signed readings'))

    def video(self, episode, frame, camera):
        if episode < FIRST_ID:
            return super().video(episode, frame, camera)
        with self.lock:
            row = self._episode(episode, frame)
            data = _targets(row['targets']['path'], row['targets']['sha256'])
            mapped = mapped_frame(row, data, frame, camera)
            rgb = video_frames(Path(row['views'][camera]['path']), [mapped], size=384)[0]
            image = Image.fromarray((rgb.permute(1, 2, 0).numpy()*255).round().astype(np.uint8))
            output = BytesIO()
            image.save(output, 'JPEG', quality=90)
            return output.getvalue()

    def metadata(self, episode, camera):
        if episode < FIRST_ID:
            return super().metadata(episode, camera)
        with self.lock:
            row = self._episode(episode, 0)
            if camera not in row['views']:
                raise ValueError('Camera is unavailable for this episode')
            times = _targets(row['targets']['path'], row['targets']['sha256'])['timestamps']
            delta = np.diff(times[:4096])
            delta = delta[np.isfinite(delta) & (delta > 0)]
            return dict(video_frames=row['frames'], fps=float(1/np.median(delta)) if len(delta) else None,
                        width=384, height=384, camera_role=row['views'][camera]['role'])

    def acquisition_series(self, episode, frame, channel, radius=150, model=-1):
        with self.lock:
            if episode < FIRST_ID:
                raise ValueError('Select an acquired numeric sensor episode')
            row = self._episode(episode, frame)
            data = _targets(row['targets']['path'], row['targets']['sha256'])
            values = data['values'].reshape(row['frames'], -1)
            valid = data['valid'].reshape(values.shape) & np.isfinite(values)
            if not 0 <= channel < values.shape[1] or not 1 <= radius <= 1000:
                raise ValueError('Invalid channel or history radius')
            start, stop = max(0, frame-radius), min(row['frames'], frame+radius+1)
            x, mask = values[start:stop, channel], valid[start:stop, channel]
            reconstructed=None
            if model!=-1:
                output,_=self._reconstruct_numeric(row,values[start:stop],valid[start:stop],model)
                reconstructed=[float(v) if ok else None for v,ok in zip(output[:,channel],mask)]
            return dict(channel=channel, label=row.get('sensor_layout', {}).get('labels',
                            row['targets']['channel_order'])[channel],
                native_label=row['targets']['channel_order'][channel],
                frame_indices=list(range(start, stop)),
                seconds=(data['timestamps'][start:stop]-data['timestamps'][0]).tolist(),
                values=[v.item() if ok and np.isfinite(v) else None for v, ok in zip(x, mask)],
                reconstruction=reconstructed,
                selected_frame=frame, native_values=True)


def make_acquisition_handler(base, viewer):
    class Handler(base):
        def do_GET(self):
            url = urlparse(self.path)
            route = '/api/' + url.path.split('/api/', 1)[1] if '/api/' in url.path else url.path
            query = parse_qs(url.query)
            relevant = (route.endswith('/acquisition-viewer.js') or route.rstrip('/').endswith('/tacquad')
                        or route.startswith('/api/acquisition/') or route.startswith('/api/tacquad/'))
            if not relevant:
                return super().do_GET()
            try:
                status = 200
                if route.endswith('/acquisition-viewer.js'):
                    content = Path(__file__).with_name('acquisition_viewer.js').read_bytes()
                    mime = 'text/javascript; charset=utf-8'
                elif route.rstrip('/').endswith('/tacquad') or route.startswith('/api/tacquad/'):
                    status = 410
                    content = json.dumps(dict(error='TacQuad was removed from this project.')).encode()
                    mime = 'application/json'
                elif route == '/api/acquisition/series':
                    result = viewer.acquisition_series(int(query['episode'][0]), int(query['frame'][0]),
                        int(query['channel'][0]), int(query.get('radius', ['150'])[0]),
                        int(query.get('model',['-1'])[0]))
                    content, mime = json.dumps(result, allow_nan=False).encode(), 'application/json'
                else:
                    self.send_error(404)
                    return
                self.send_response(status)
            except (BrokenPipeError, ConnectionResetError):
                return
            except Exception as error:
                content, mime = json.dumps(dict(error=str(error))).encode(), 'application/json'
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
