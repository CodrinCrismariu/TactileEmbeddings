"""Paired native/mirrored sensor audits with fresh frozen-AE inference."""
from io import BytesIO
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image, ImageOps
import torch

from .aligned_frames import file_hash, json_hash
from .autoencoder_viewer import make_handler
from .bimanual_mirror import load_policy, require_spec, transform_targets, verify_evidence
from .acquisition_viewer import AcquisitionViewer
from .sensor_autoencoder import make_autoencoder
from .mirror_mapping_review import candidates, candidate_spec, marker_payload, REVIEW_REASONS


CAMERA_SWAP = {'0': '0', '1': '2', '2': '1'}


class MirroredViewer(AcquisitionViewer):
    def __init__(self, *args, mirror_policy='configs/bimanual_mirror_policy_v3.json',
                 mirror_preparation='processing/bimanual_mirror_v4', **kwargs):
        self.mirror_policy_path = Path(mirror_policy)
        self.mirror_policy = load_policy(mirror_policy)
        self.mirror_evidence = verify_evidence(self.mirror_policy)
        self.mirror_root = Path(mirror_preparation)
        self.mirror_history = [(self.mirror_root, self.mirror_policy)]
        if self.mirror_policy['version'] != 'bilateral-native-coordinates-v1':
            previous = Path('processing/bimanual_mirror_v3')
            if self.mirror_root.resolve()!=previous.resolve():
                self.mirror_history.append((previous,load_policy('configs/bimanual_mirror_policy_v2.json')))
            previous = Path('processing/bimanual_mirror_v2')
            if self.mirror_root.resolve()!=previous.resolve():
                self.mirror_history.append((previous,load_policy('configs/bimanual_mirror_policy_v2.json')))
            old_policy = Path('configs/bimanual_mirror_policy_v1.json')
            if old_policy.exists():
                self.mirror_history.append((Path('processing/bimanual_mirror_v1'), load_policy(old_policy)))
        self.mirror_checkpoint_hashes = {}
        super().__init__(*args, **kwargs)

    def _scan_models(self):
        models = super()._scan_models()
        # A new production Baihu teacher is kept separate from the historical
        # unfinished folder, and becomes inspectable as soon as an epoch saves.
        from .expansion_viewer import ROOT as expansion_root
        for version in ('v2', 'v3'):
            folder = expansion_root / 'teachers' / f'baihu_native_{version}'
            if (folder / 'best.pt').exists() and (folder / 'best_metrics.json').exists():
                config = json.loads((folder / 'config.json').read_text())
                metrics = json.loads((folder / 'best_metrics.json').read_text())
                history_path = folder / 'metrics.jsonl'
                history = history_path.read_text().splitlines() if history_path.exists() else []
                completed = metrics['epoch']
                for line in reversed(history):
                    try:
                        completed = json.loads(line)['epoch']; break
                    except (ValueError, KeyError):
                        continue
                quality_path = folder / 'quality.json'
                quality = json.loads(quality_path.read_text()) if quality_path.exists() else None
                quality_note = (' · native quality failed' if quality and not quality['accepted'] else
                                ' · in training' if quality is None else '')
                models.append(dict(path=str(folder / 'best.pt'), name=f'baihu AE · 32D · native {version}'+(' · normalized code projection' if version=='v3' else '')+quality_note,
                    source='baihu', config=dict(config['model'], architecture='marker_set'), score=metrics['selection_score'],
                    best_epoch=metrics['epoch'], completed_epochs=completed,
                    validation_mae=metrics['val']['mae'], reduced_resolution=False,
                    representation='marker_displacement', expansion_teacher=True,
                    preferred=bool(quality and quality['accepted'])))
        for root, artifact_policy in self.mirror_history:
            models = self._mirrored_teachers(models, root, artifact_policy)
        return models

    def _mirrored_teachers(self, models, root, artifact_policy):
        for source, spec in self.mirror_policy['sources'].items():
            # Only complete verified coordinate maps may enter this audit.
            if (spec['status'] != 'verified' or spec['kind'] not in ('grid', 'markers','taxels')
                    or artifact_policy['sources'][source] != spec):
                continue
            p = root / 'teachers' / source / 'quality.json'
            if not p.exists():
                continue
            quality = json.loads(p.read_text())
            teacher = quality.get('teacher')
            if not quality.get('accepted') or not teacher or not quality.get('adapted'):
                continue
            if quality['signature']['policy_sha256'] != json_hash(artifact_policy):
                continue
            path = Path(teacher['checkpoint'])
            stamp = (path.stat().st_mtime_ns, path.stat().st_size)
            if self.mirror_checkpoint_hashes.get(str(path)) != stamp:
                if file_hash(path) != teacher['sha256']:
                    raise ValueError('Accepted mirrored teacher checksum changed')
                self.mirror_checkpoint_hashes[str(path)] = stamp
            if any(Path(m['path']).resolve() == path.resolve() for m in models):
                continue
            metrics = quality['validation']
            continued='decision' in quality
            completed=quality.get('completed_epochs',quality['signature']['epochs'])
            models.append(dict(path=str(path), name=f'{source} AE · mirror-adapted teacher'+(' · shared hands' if teacher.get('bilateral') else ' · balanced continuation' if continued else ''),
                source=source, config=teacher['model_config'],
                score=(metrics['original']['mae'] + metrics['mirrored']['mae']) / 2,
                best_epoch=quality.get('best_epoch'), completed_epochs=completed,
                validation_mae=metrics['original']['mae'],
                mirrored_validation_mae=metrics['mirrored']['mae'],
                reduced_resolution=False, mirror_adapted=True,
                preferred=bool(quality.get('parity_achieved')),
                mirror_parity_achieved=quality.get('parity_achieved'),
                mirror_parity_ratios=quality.get('decision',{}).get('ratios'),
                **(dict(representation='marker_displacement', expansion_teacher=source=='baihu')
                   if spec['kind']=='markers' else {}),
                **(dict(manifest='processing/storage_v1/manifest.json',
                        marker_targets='processing/marker_displacement_storage_v1/targets')
                   if source=='ftp_unit_bimanual' else {}),
                training_note=f"{completed} mirror fine-tuning epochs; "
                              f"parent {Path(teacher['original_checkpoint']).parent.name}"))
        return models

    def _training_ready(self, source):
        path = self.mirror_root / 'ready.json'
        if not path.exists():
            return False
        ready = json.loads(path.read_text())
        return (ready.get('policy_sha256') == json_hash(self.mirror_policy)
                and bool(ready.get('sources', {}).get(source, {}).get('ready')))

    def catalog(self):
        result = super().catalog()
        status=self.mirror_root/'status.json'
        if status.exists():result['mirror_training']=json.loads(status.read_text())
        result['mirror_sources'] = {}
        for source, spec in self.mirror_policy['sources'].items():
            enabled = spec['status'] == 'verified'
            review = candidates(source) if not enabled else []
            result['mirror_sources'][source] = dict(enabled=enabled,
                review_available=bool(review),
                review_candidates=[dict(id=c['id'], label=c['label']) for c in review],
                training_enabled=enabled and self._training_ready(source), status=spec['status'],
                verification_basis=spec.get('verification_basis', 'release_geometry'),
                description=('Swap hands and horizontally reflect both native pressure grids'
                             if spec['kind'] == 'grid' else
                             ('Swap hands and jaw sides; horizontally reflect marker references and signed displacement vectors'
                              if source=='baihu' else
                              'Swap hands; horizontally reflect marker references and signed displacement vectors')
                             if spec['kind'] == 'markers' else
                             'Swap hands; reflect each named 4×4 finger grid left ↔ right'
                             if spec['kind']=='taxels' else 'Swap left/right scalar force values; preserve sign and magnitude')
                            if enabled else None,
                reason=None if enabled else REVIEW_REASONS.get(source, spec.get('reason')))
        for episode in result['episodes']:
            availability = result['mirror_sources'].get(episode['source'], {})
            if availability.get('enabled') or availability.get('review_available'):
                episode['mirror_cameras'] = sorted(CAMERA_SWAP[k] for k in episode['cameras'])
        result['mirror_policy'] = self.mirror_policy['version']
        return result

    def _spec(self, source, candidate=None):
        if candidate:
            if self.mirror_policy['sources'][source]['status'] == 'verified':
                raise ValueError('Verified sources do not accept speculative coordinate overrides')
            return candidate_spec(source, candidate)
        return require_spec(self.mirror_policy, source)

    def _marker_frame(self, row, frame, model, spec):
        from .mirror_preparation import NativeTargets, predict
        from .sensor_data import SENSORS
        from .expansion_render import LABELS
        if not hasattr(self, '_mirror_targets'):
            self._mirror_targets = NativeTargets()
        if row['source'] == 'ftp_unit_bimanual':
            from .marker_viewer import canonical_rows
            row = canonical_rows('processing/storage_v1/manifest.json')[(row['source'], row['episode_id'])]
        batch = transform_targets(self._mirror_targets.read(row, [frame]), spec)
        prediction = None
        if model != -1:
            if not 0 <= model < len(self.models):
                raise ValueError('Unknown marker teacher')
            info = self.models[model]
            if info['source'] != row['source'] or info.get('representation') != 'marker_displacement':
                raise ValueError('Select a marker-displacement autoencoder for this sensor')
            path = Path(info['path']); stamp = path.stat().st_mtime_ns
            if model not in self.model_cache or self.model_cache[model][0] != stamp:
                if row['source'] == 'baihu':
                    from .expansion_models import load_teacher
                    network = load_teacher(path)[0]
                else:
                    from .marker_autoencoder import load_marker_autoencoder
                    network = load_marker_autoencoder(path)[0]
                self._keep_recent(self.model_cache, model, (stamp, network), 2)
            prediction = predict(self.model_cache[model][1], batch)[0].numpy()
        labels = LABELS if row['source'] == 'baihu' else SENSORS[row['source']][2]
        return marker_payload(row, batch, labels, frame, prediction)

    def mirror_frame(self, episode, frame, model, candidate=None):
        with self.lock, torch.inference_mode():
            row = self._episode(episode, frame)
            spec = self._spec(row['source'], candidate)
            metadata = dict(policy=self.mirror_policy['version'], kind=spec['kind'],
                verified=candidate is None,
                hardware_geometry_verified=spec.get('hardware_geometry_verified'),
                verification_basis=spec.get('verification_basis', 'release_geometry') if candidate is None else 'unverified_hypothesis',
                training_enabled=candidate is None and self._training_ready(row['source']),
                candidate=candidate, platform=row.get('platform'),
                recorded_camera_by_mirrored_camera={CAMERA_SWAP[k]: k for k in row.get('views', {})})
            if spec['kind'] == 'markers':
                result = self._marker_frame(row, frame, model, spec)
                result['mirror'] = dict(**metadata, surface_permutation=spec['surface_permutation'],
                    surface_transforms=spec['surface_transforms'],
                    reconstruction='fresh frozen AE encoding of transformed vectors and reference geometry')
                return result
            # Read native values, then transform measurements BEFORE encoding.
            native = super().frame(episode, frame, -1)
            original = np.asarray([np.nan if x is None else x for x in native['original']],
                                  dtype=np.float32).reshape(native['target_shape'])
            value = torch.from_numpy(original)
            b = transform_targets(dict(target=value, target_mask=torch.isfinite(value)), spec)
            y, valid = b['target'].numpy(), b['target_mask'].numpy()
            prediction = np.full_like(y, np.nan)
            if model != -1:
                if not 0 <= model < len(self.models) or self.models[model]['source'] != row['source']:
                    raise ValueError('Select an autoencoder trained for this sensor dataset')
                info = self.models[model]
                metadata['teacher_mirror_adapted']=bool(info.get('mirror_adapted'))
                if row['source']=='opengraph' and not metadata['teacher_mirror_adapted']:
                    metadata['reconstruction_note']='This teacher was trained on original targets; it has not been fine-tuned for this mirrored mapping.'
                if info.get('reduced_resolution') or info.get('representation') == 'marker_displacement':
                    raise ValueError('A native pressure-grid teacher is required')
                stamp = Path(info['path']).stat().st_mtime_ns
                if model not in self.model_cache or self.model_cache[model][0] != stamp:
                    saved = torch.load(info['path'], map_location='cpu', weights_only=False)
                    network = make_autoencoder(saved['model_config'])
                    network.load_state_dict(saved['model'], strict=True)
                    self._keep_recent(self.model_cache, model,
                                      (stamp, network.eval().requires_grad_(False)), 2)
                network = self.model_cache[model][1]
                clean = torch.where(b['target_mask'], b['target'], 0).unsqueeze(0)
                prediction = network(clean, b['target_mask'].unsqueeze(0))[0].numpy()
            error = np.abs(prediction - y)
            pressure = spec['kind'] == 'grid'
            contact, predicted_contact = valid & (y > .05), valid & (prediction > .05)
            denominator = int(contact.sum() + predicted_contact.sum())
            measured = model != -1

            def cells(a):
                return [float(x) if ok and np.isfinite(x) else None
                        for x, ok in zip(a.flat, valid.flat)]

            result = dict(frame=frame, source=row['source'], target_kind=native['target_kind'],
                stored_sensor_frame=native['stored_sensor_frame'], target_shape=list(y.shape),
                labels=native['labels'], original=cells(y), reconstruction=cells(prediction),
                error=cells(error), mae=float(error[valid].mean()) if measured and valid.any() else None,
                hand_mae=[float(error[i][valid[i]].mean()) if measured and valid[i].any() else None
                          for i in range(len(y))],
                contact_mae=float(error[contact].mean()) if pressure and measured and contact.any() else None,
                contact_f1=2 * int((contact & predicted_contact).sum()) / denominator
                           if pressure and measured and denominator else None,
                mirror=dict(**metadata,
                    hand_permutation=spec['hand_permutation'], flip_width=spec.get('flip_width', False),
                    reconstruction='fresh frozen AE encoding of transformed native measurements'
                                   if measured else 'direct numeric measurements; no autoencoder',
                    ))
            if spec['kind'] == 'scalar':
                scale = row['target_transform_scale']

                def raw(a):
                    return [float(scale * np.arctanh(2 * x - 1))
                            if ok and np.isfinite(x) and 0 < x < 1 else None
                            for x, ok in zip(a.flat, valid.flat)]

                result.update(raw_original=raw(y), raw_reconstruction=raw(prediction))
            return result

    def mirror_video(self, episode, frame, camera, candidate=None):
        row = self._episode(episode, frame)
        self._spec(row['source'], candidate)
        recorded = CAMERA_SWAP.get(camera)
        if recorded not in row['views']:
            raise ValueError('This mirrored camera is unavailable; its recorded counterpart is missing')
        image = ImageOps.mirror(Image.open(BytesIO(super().video(episode, frame, recorded))))
        output = BytesIO()
        image.save(output, 'PNG')  # Preserve the decoded preview pixels exactly.
        return output.getvalue()


def make_mirror_handler(viewer, joint=None):
    from .pressure_surface import make_pressure_handler
    from .acquisition_viewer import make_acquisition_handler
    base = make_acquisition_handler(make_pressure_handler(make_handler(viewer, joint), viewer), viewer)

    class Handler(base):
        def do_GET(self):
            url = urlparse(self.path)
            route = '/api/' + url.path.split('/api/', 1)[1] if '/api/' in url.path else url.path
            if route not in ('/api/mirror/frame', '/api/mirror/video'):
                return super().do_GET()
            query = parse_qs(url.query)
            try:
                episode = int(query.get('episode', ['0'])[0])
                frame = int(query.get('frame', ['0'])[0])
                candidate = query.get('candidate', [None])[0]
                if route.endswith('/video'):
                    content = viewer.mirror_video(episode, frame, query.get('camera', ['0'])[0], candidate)
                    mime = 'image/png'
                else:
                    result = viewer.mirror_frame(episode, frame, int(query.get('model', ['-1'])[0]), candidate)
                    content = json.dumps(result, allow_nan=False).encode()
                    mime = 'application/json'
                self.send_response(200)
            except (BrokenPipeError, ConnectionResetError):
                return
            except Exception as exc:
                content, mime = json.dumps(dict(error=str(exc))).encode(), 'application/json'
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
