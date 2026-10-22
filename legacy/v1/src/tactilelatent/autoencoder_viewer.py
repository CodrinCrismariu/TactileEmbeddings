"""Local browser viewer for native sensor outputs and AE reconstructions.

Run scripts/view_egotouch_autoencoder.py. Sensor AEs use CPU; the visual-model
comparison page loads DINO on demand and uses CUDA when available.
"""

import argparse
import base64
import math
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
from pathlib import Path
import threading
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image
import torch

from .joint_all_config import ALL_TARGETS
from .sensor_autoencoder import make_autoencoder
from .sensor_data import SENSORS, load_targets, scene_frames, scene_metadata
from .native_optical import native_images, native_shape, native_targets
from .opentouch_layout import hand_layout, mapped_mask
from .aligned_frames import (MANIFEST as ALIGNED_MANIFEST, raw_scene_images,
                             sensor_indices, validate_frame_order)


class Viewer:
    def __init__(self, manifest, checkpoint_roots, cache="processing/sensor_ae_cache_v1",
                 training="processing/runs/sensor_autoencoders_native_v2",
                 frame_order="processing/audits/vla_touch_native_dynamics/dataset_frame_order.json"):
        from .storage import retired
        self.episodes = [r for r in json.loads(Path(manifest).read_text())["episodes"]
                         if r["source"] in SENSORS and not retired(r['source'])]
        self.checkpoint_roots, self.cache_root, self.training_root = checkpoint_roots, Path(cache), Path(training)
        self.models = self._scan_models()
        primary = Path(checkpoint_roots[0]).resolve()
        self.models.sort(key=lambda m: (not Path(m["path"]).resolve().is_relative_to(primary),
                                       m["score"] if m["score"] is not None else float("inf")))
        self.pressure_cache, self.model_cache, self.video_metadata = OrderedDict(), OrderedDict(), {}
        self.disk_cache = {}
        self.optical_reference_cache = OrderedDict()
        self.tactile_frame_orders = {}
        for row in self.episodes:
            if row.get("sensor_frame_order") is not None:
                if sorted(row["sensor_frame_order"]) != list(range(row["frames"])):
                    raise ValueError("Invalid embedded tactile frame-order mapping")
                self.tactile_frame_orders[row["episode_id"]] = row["sensor_frame_order"]
        if frame_order is not None and Path(frame_order).exists():
            audit = json.loads(Path(frame_order).read_text())
            if audit.get("source") != "ftp_vla_touch":
                raise ValueError("Incorrect tactile frame-order source")
            for row in self.episodes:
                if row["source"] != "ftp_vla_touch":
                    continue
                record = audit["episodes"].get(row["episode_id"])
                if record is None:
                    continue
                try:
                    order = validate_frame_order(row, record)
                except ValueError as error:
                    raise ValueError("Frame-order lookup does not match the current episode") from error
                if row["episode_id"] in self.tactile_frame_orders and self.tactile_frame_orders[row["episode_id"]] != order:
                    raise ValueError("Embedded and external frame-order mappings differ")
                self.tactile_frame_orders[row["episode_id"]] = order
        self.lock = threading.RLock()

    def _sensor_frame(self, row, frame):
        order = self.tactile_frame_orders.get(row["episode_id"]) if row["source"] == "ftp_vla_touch" else None
        return int(sensor_indices(row, [frame], order)[0]) if order is not None else frame

    def _scan_models(self):
        models = []
        for root in self.checkpoint_roots:
            for checkpoint in sorted(Path(root).rglob("best.pt")):
                config_path = checkpoint.parent / "config.json"
                if not config_path.exists() or any("smoke" in part.lower() for part in
                        (Path(root).name, *checkpoint.relative_to(root).parts)):
                    continue
                try:
                    cfg = json.loads(config_path.read_text())
                    metrics_path = checkpoint.parent / "best_metrics.json"
                    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
                except (FileNotFoundError, json.JSONDecodeError):
                    continue  # A checkpoint may be arriving while training writes it.
                model = cfg.get("model", cfg)
                from .storage import retired
                if retired(cfg.get('source','egotouch')):
                    continue
                if model.get('representation')=='marker_displacement':
                    if not (checkpoint.parent/'source_snapshot.json').exists():continue
                    source=cfg['source']
                    state=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
                    if state.get('model_type')!='marker_autoencoder_v1':continue
                    history=checkpoint.parent/'metrics.jsonl'
                    completed=metrics.get('epoch')
                    if history.exists():
                        for line in reversed(history.read_text().splitlines()):
                            try:completed=json.loads(line)['epoch'];break
                            except (ValueError,KeyError):continue
                    partial=any(part.startswith('marker_autoencoders_freetacman_available') for part in checkpoint.parts)
                    models.append(dict(path=str(checkpoint),name=f"Marker AE · {checkpoint.parent.name}"+(' · processed snapshot' if partial else ''),source=source,
                        config=state['model_config'],score=metrics.get('selection_score'),best_epoch=metrics.get('epoch'),
                        completed_epochs=completed,validation_mae=metrics.get('val',{}).get('mae_px'),
                        reduced_resolution=False,representation='marker_displacement',marker_targets=cfg['marker_targets'],manifest=cfg['manifest']))
                    continue
                if model.get("architecture") not in ("conv", "mlp"):
                    continue
                source = cfg.get("source", "egotouch")
                from .marker_storage import marker_only
                if marker_only(source):
                    continue
                from .storage import retired
                if retired(source):
                    continue
                if source not in SENSORS or ALL_TARGETS[source][0] in ("force_tanh", "torque_tanh"):
                    continue
                if ALL_TARGETS[source][0] == "optical_rgb" and not cfg.get("native_optical", False):
                    continue
                score = metrics.get("selection_score", metrics.get("val", {}).get("mae"))
                completed = metrics.get("epoch")
                history_path = checkpoint.parent / "metrics.jsonl"
                if history_path.exists():
                    for line in reversed(history_path.read_text().splitlines()):
                        try:
                            completed = json.loads(line)["epoch"]
                            break
                        except (json.JSONDecodeError, KeyError):
                            continue
                models.append({"path": str(checkpoint), "name": checkpoint.parent.name,
                                    "source": cfg.get("source", "egotouch"),
                                    "config": model, "score": score,
                                    "best_epoch": metrics.get("epoch"),
                                    "completed_epochs": completed,
                                    "validation_mae": metrics.get("val", {}).get("mae"),
                                    "reduced_resolution": cfg.get("source") in ALL_TARGETS and ALL_TARGETS[cfg["source"]][0] == "optical_rgb" and not cfg.get("native_optical", False)})
        return models

    def catalog(self):
        with self.lock:
            # Preserve IDs when new checkpoints appear; an active frame request
            # must not silently switch architecture after a catalog refresh.
            ids = {m["path"]: i for i, m in enumerate(self.models)}
            available = self._scan_models()
            visible = {m["path"] for m in available}
            for model in available:
                if model["path"] in ids: self.models[ids[model["path"]]] = model
                else:
                    ids[model["path"]] = len(self.models); self.models.append(model)
        training_path = self.training_root / "status.json"
        training = json.loads(training_path.read_text()) if training_path.exists() else {"status": "not_started"}
        marker_status=Path('processing/runs/marker_autoencoders_v1/status.json')
        if marker_status.exists():
            marker_training=json.loads(marker_status.read_text())
            training=dict(status=marker_training['stage'],source=marker_training.get('source'),
                stage='marker AE'+(f" · {marker_training['code_dim']}D" if 'code_dim' in marker_training else '')+
                      (f" · epoch {marker_training['epoch']}" if 'epoch' in marker_training else ''))
        from .marker_viewer import ready_root
        from .marker_storage import marker_only
        def marker_ready(row):
            if row['source'] not in ('freetacman','ftp_unit','ftp_unit_bimanual','ftp_vla_touch'):return None
            try:ready_root(row);return True
            except ValueError:return False
        return {"episodes": [{"id": i, "name": r["episode_id"], "split": r["split"],
                              "source": r["source"],
                              "marker_ready": marker_ready(r),
                              "frames": r["frames"], "cameras": sorted(r["views"]),
                              "camera_labels": {k: ("Ego view · " + Path(v).name if r["source"] == "freetacman" else None) for k,v in r["views"].items()},
                              "tactile_chronology": ("Tactile order corrected from raw numbered frames" if r["source"] == "ftp_vla_touch" and r["episode_id"] in self.tactile_frame_orders else None),
                              "sensor_videos": [Path(p).name for p in r.get("tactile_videos", [])]}
                             for i, r in enumerate(self.episodes)],
                "datasets": [{"id": source, "name": SENSORS[source][0], "sensor": SENSORS[source][1],
                              "marker_only": marker_only(source),
                              "hand_layout": hand_layout() if source == "opentouch" else None,
                              "labels": SENSORS[source][2], "target_kind": ALL_TARGETS[source][0],
                              "target_shape": (native_shape(next(r for r in self.episodes if r["source"] == source))
                                               if ALL_TARGETS[source][0] == "optical_rgb" else ALL_TARGETS[source][1])}
                             for source in SENSORS if any(r["source"] == source for r in self.episodes)],
                "training": training,
                "models": [{"id": i, **{k: v for k, v in m.items() if k != "path"}}
                           for i, m in enumerate(self.models) if m["path"] in visible]}

    def _episode(self, index, frame):
        if not 0 <= index < len(self.episodes):
            raise ValueError("Unknown episode")
        row = self.episodes[index]
        if not 0 <= frame < row["frames"]:
            raise ValueError("Frame outside episode")
        return row

    @staticmethod
    def _keep_recent(cache, key, value, limit):
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > limit:
            cache.popitem(last=False)

    def frame(self, episode, frame, model, prediction=None):
        row = self._episode(episode, frame)
        if model != -1 and not 0 <= model < len(self.models):
            raise ValueError("Unknown architecture")
        if model != -1 and self.models[model]["source"] != row["source"]:
            raise ValueError("Select a checkpoint trained for this sensor dataset")
        if model != -1 and self.models[model].get("reduced_resolution"):
            raise ValueError("This checkpoint used reduced-resolution targets and is superseded")
        if model!=-1 and self.models[model].get('representation')=='marker_displacement':
            from .marker_autoencoder import load_marker_autoencoder
            from .marker_viewer import inputs as marker_inputs,payload as marker_payload
            with self.lock,torch.inference_mode():
                info=self.models[model];stamp=Path(info['path']).stat().st_mtime_ns
                if model not in self.model_cache or self.model_cache[model][0]!=stamp:
                    ae,_=load_marker_autoencoder(info['path'])
                    self._keep_recent(self.model_cache,model,(stamp,ae),2)
                ae=self.model_cache[model][1];cfg=dict(manifest=info['manifest'],marker_targets=info['marker_targets'])
                _,b,_=marker_inputs(cfg,row,frame)
                reconstructed=ae(b['marker_target'],b['marker_valid'],b['marker_reference'][None],b['marker_surface'][None])[0].numpy()
                return marker_payload(cfg,row,frame,reconstructed)
        from .marker_storage import marker_only
        if marker_only(row['source']):
            if model != -1 or prediction is not None:
                raise ValueError('Select a marker model; original tactile images were retired')
            from .marker_viewer import tracking_frame
            return tracking_frame(self,episode,frame)
        # Serialize cache changes and inference; browser requests can overlap.
        with self.lock:
            if row["source"] == "egotouch" and episode not in self.pressure_cache:
                with np.load(row["target_path"]) as data:
                    target = np.stack([data[f"{hand}_pressure_grid"] for hand in ("left", "right")], axis=1)
                if target.shape != (row["frames"], 2, 21, 21):
                    raise ValueError("Pressure/video manifest frame counts differ")
                self._keep_recent(self.pressure_cache, episode, target.astype(np.float32), 2)
            if row["source"] == "egotouch":
                self.pressure_cache.move_to_end(episode)
                original = self.pressure_cache[episode][frame]
            else:
                original = (native_targets(row, [self._sensor_frame(row, frame)])[0] if ALL_TARGETS[row["source"]][0] == "optical_rgb"
                            else self._cached_target(row, frame))
            valid = np.isfinite(original)
            if not valid.any():
                raise ValueError("Frame has no valid pressure cells")
            target = torch.from_numpy(np.where(valid, original, 0.)).unsqueeze(0)
            has_prediction = model != -1 or prediction is not None
            prediction = np.full_like(original, np.nan) if prediction is None else np.asarray(prediction, dtype=np.float32)
            if prediction.shape != original.shape:
                raise ValueError("Prediction and native sensor target shapes differ; resizing is not permitted")
            if model != -1:
                stamp = Path(self.models[model]["path"]).stat().st_mtime_ns
                if model not in self.model_cache or self.model_cache[model][0] != stamp:
                    saved = torch.load(self.models[model]["path"], map_location="cpu", weights_only=False)
                    network = make_autoencoder(saved["model_config"])
                    network.load_state_dict(saved["model"])
                    network.eval().requires_grad_(False)
                    self._keep_recent(self.model_cache, model, (stamp, network), 2)
                self.model_cache.move_to_end(model)
                with torch.inference_mode():
                    prediction = self.model_cache[model][1](target, torch.from_numpy(valid).unsqueeze(0))[0].numpy()
            # Mask only display and viewer metrics, after native AE inference.
            # Existing checkpoints were trained on all finite electrical channels.
            if row["source"] == "opentouch":
                valid = valid & mapped_mask()
                if not valid.any():
                    raise ValueError("Frame has no valid mapped OpenTouch channels")
            error = np.abs(prediction - np.where(valid, original, 0.))
            kind = ALL_TARGETS[row["source"]][0]
            pressure = kind in ("pressure_grid", "taxel_vector") and row["source"] != "opentouch"
            contact = valid & (original > .05)
            predicted_contact = valid & (prediction > .05)
            tp = int((contact & predicted_contact).sum())
            denom = int(contact.sum() + predicted_contact.sum())
            def cells(array):
                return [float(value) if ok and np.isfinite(value) else None for value, ok in zip(array.flat, valid.flat)]
            hand_mae = [float(error[i][valid[i]].mean()) if valid[i].any() and has_prediction else None for i in range(len(original))]
            result = {"frame": frame, "source": row["source"], "target_kind": kind,
                      "stored_sensor_frame": self._sensor_frame(row, frame),
                      "target_shape": list(original.shape), "labels": SENSORS[row["source"]][2],
                      "original": cells(original) if kind != "optical_rgb" else None,
                      "reconstruction": cells(prediction) if kind != "optical_rgb" else None,
                      "error": cells(error) if kind != "optical_rgb" else None,
                      "mae": float(error[valid].mean()) if has_prediction else None,
                      "hand_mae": hand_mae,
                      "contact_mae": float(error[contact].mean()) if pressure and contact.any() and has_prediction else None,
                      "contact_f1": 2 * tp / denom if pressure and denom and has_prediction else None}
            if row["source"] == "opentouch":
                result.update(hand_layout_id=hand_layout()["id"],
                              mapped_channels=169, excluded_channels=87,
                              metric_channels=int(valid.sum()))
            if kind in ("force_tanh", "torque_tanh"):
                scale = row["target_transform_scale"]
                def raw_values(array):
                    return [float(scale * np.arctanh(2 * x - 1)) if ok and np.isfinite(x) and 0 < x < 1 else None
                            for x, ok in zip(array.flat, valid.flat)]
                result.update(raw_original=raw_values(original), raw_reconstruction=raw_values(prediction))
            if kind == "optical_rgb":
                panels = []
                for i, label in enumerate(SENSORS[row["source"]][2]):
                    sensor = slice(3 * i, 3 * i + 3)
                    def png_rgb(value):
                        buffer = BytesIO()
                        image = np.clip(np.nan_to_num(value).transpose(1, 2, 0) * 255, 0, 255).round().astype(np.uint8)
                        Image.fromarray(image).save(buffer, "PNG")
                        return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
                    panels.append({"label": label, "original": png_rgb(original[sensor]),
                        "reconstruction": png_rgb(prediction[sensor]) if has_prediction else None,
                        "error_u16": base64.b64encode(np.clip(np.nan_to_num(error[sensor].mean(axis=0)) * 65535, 0, 65535).round().astype('<u2').tobytes()).decode() if has_prediction else None,
                        "mae": float(error[sensor].mean()) if has_prediction else None})
                result.update(original=None, reconstruction=None, error=None, optical_panels=panels)
            return result

    def _cached_target(self, row, frame):
        key = (row["source"], row["split"])
        index_path = self.cache_root / row["source"] / "index.json"
        if key not in self.disk_cache and index_path.exists():
            index = json.loads(index_path.read_text())
            selected = index["splits"].get(row["split"])
            if selected:
                self.disk_cache[key] = (index["signature"]["stride"],
                    {r["episode_id"]: selected["offsets"][i] for i, r in enumerate(selected["episodes"])},
                    np.load(index_path.parent / f"{row['split']}.npy", mmap_mode="r"))
        if key in self.disk_cache:
            stride, offsets, data = self.disk_cache[key]
            if frame % stride == 0 and row["episode_id"] in offsets:
                return np.asarray(data[offsets[row["episode_id"]] + frame // stride]).copy()
        return load_targets(row, [frame])[0]

    def video(self, episode, frame, camera):
        row = self._episode(episode, frame)
        if camera not in row["views"]:
            raise ValueError("Camera is unavailable for this episode")
        buffer = BytesIO()
        from .storage import scene_entry
        if "raw_scene" in row and scene_entry(row) is None:
            image = Image.fromarray(raw_scene_images(row, [frame], camera)[0])
        else:
            rgb = scene_frames(row, [frame], camera, size=480)[0]
            image = Image.fromarray((rgb.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))
        image.save(buffer, "JPEG", quality=85)
        return buffer.getvalue()

    def _native_optical(self, row, frame):
        """Decode native pixels, applying verified raw chronology when available."""
        return [image[0] for image in native_images(row, [self._sensor_frame(row, frame)])]

    def optical(self, episode, frame, gain=8.):
        row = self._episode(episode, frame)
        from .marker_storage import marker_only
        if marker_only(row['source']):
            from .marker_viewer import tracking_frame
            result=tracking_frame(self,episode,frame)
            return dict(frame=frame,stored_sensor_frame=self._sensor_frame(row,frame),
                reference_frame=0,stored_reference_frame=self._sensor_frame(row,0),gain=1,
                marker_displacement=result['marker_displacement'],
                panels=[dict(label=p['label'],width=row['target_shape'][2],height=row['target_shape'][1],
                    original=p['original'],change=p['original'],change_mae=None,
                    identical_to_reference=False) for p in result['optical_panels']])
        if ALL_TARGETS[row["source"]][0] != "optical_rgb":
            raise ValueError("Source-resolution inspection requires an optical sensor")
        if not math.isfinite(gain) or not 1 <= gain <= 32:
            raise ValueError("Change gain must be between 1 and 32")
        with self.lock:
            if episode not in self.optical_reference_cache:
                self._keep_recent(self.optical_reference_cache, episode,
                                  self._native_optical(row, 0), 2)
            self.optical_reference_cache.move_to_end(episode)
            reference = self.optical_reference_cache[episode]
            original = reference if frame == 0 else self._native_optical(row, frame)
            def png(array):
                buffer = BytesIO()
                Image.fromarray(array).save(buffer, "PNG")
                return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
            panels = []
            for label, image, first in zip(SENSORS[row["source"]][2], original, reference):
                difference = np.abs(image.astype(np.float32) - first.astype(np.float32))
                panels.append({"label": label, "width": image.shape[1], "height": image.shape[0],
                               "original": png(image),
                               "change": png(np.clip(difference * gain, 0, 255).astype(np.uint8)),
                               "change_mae": float(difference.mean() / 255.),
                               "identical_to_reference": bool(np.array_equal(image, first))})
            return {"frame": frame, "stored_sensor_frame": self._sensor_frame(row, frame),
                    "reference_frame": 0, "stored_reference_frame": self._sensor_frame(row, 0),
                    "gain": gain, "panels": panels}

    def metadata(self, episode, camera):
        row = self._episode(episode, 0)
        if camera not in row["views"]:
            raise ValueError("Camera is unavailable for this episode")
        key = (episode, camera)
        if key not in self.video_metadata:
            self.video_metadata[key] = scene_metadata(row, camera)
        return self.video_metadata[key]


def make_handler(viewer, joint=None):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            url = urlparse(self.path)
            query = parse_qs(url.query)
            # Some browser IDE proxies strip their prefix; others preserve it.
            route = url.path
            if "/api/" in route:
                route = "/api/" + route.split("/api/", 1)[1]
            try:
                if route.endswith('/attention-viewer.js'):
                    content=Path(__file__).with_name('attention_viewer.js').read_bytes();mime='text/javascript; charset=utf-8'
                elif route.rstrip('/').endswith('/markers'):
                    content=Path(__file__).with_name('marker_viewer.html').read_bytes()
                    mime='text/html; charset=utf-8'
                elif route in ('/api/marker/catalog','/api/marker/frame'):
                    from .marker_viewer import tracking_catalog,tracking_frame
                    catalog_fn=viewer.tracking_catalog if hasattr(viewer,'tracking_catalog') else lambda:tracking_catalog(viewer)
                    frame_fn=viewer.tracking_frame if hasattr(viewer,'tracking_frame') else lambda e,f:tracking_frame(viewer,e,f)
                    result=(catalog_fn() if route.endswith('/catalog') else
                            frame_fn(int(query.get('episode',['0'])[0]),int(query.get('frame',['0'])[0])))
                    content,mime=json.dumps(result,allow_nan=False).encode(),'application/json'
                elif route.endswith("/") or route.endswith("/joint") or route == f"/proxy/{self.server.server_port}":
                    content = Path(__file__).with_name("autoencoder_viewer.html").read_bytes()
                    mime = "text/html; charset=utf-8"
                elif route == "/api/catalog":
                    content, mime = json.dumps(viewer.catalog()).encode(), "application/json"
                elif route == "/api/joint/catalog" and joint is not None:
                    content, mime = json.dumps(joint.catalog()).encode(), "application/json"
                elif route in ('/api/joint/attention','/api/joint/attention-image') and joint is not None:
                    args=(int(query.get('episode',['0'])[0]),int(query.get('frame',['0'])[0]),
                          query.get('model',[''])[0],query.get('epoch',['last'])[0],query.get('views',['ego'])[0])
                    if route.endswith('attention-image'):
                        content=joint.attention_image(*args,query.get('camera',['0'])[0],int(query.get('history',['7'])[0]));mime='image/jpeg'
                    else:
                        expected=query.get('expected_epoch',[None])[0]
                        result=joint.attention(*args,expected_epoch=int(expected) if expected is not None else None)
                        content=json.dumps(result,allow_nan=False).encode();mime='application/json'
                elif route == "/api/joint/frame" and joint is not None:
                    result = joint.frame(int(query.get("episode", ["0"])[0]),
                        int(query.get("frame", ["0"])[0]), query.get("model", [""])[0],
                        query.get("epoch", ["last"])[0], query.get("views", ["ego"])[0])
                    content, mime = json.dumps(result, allow_nan=False).encode(), "application/json"
                elif route in ("/api/frame", "/api/video", "/api/metadata", "/api/optical"):
                    episode = int(query.get("episode", ["0"])[0])
                    frame = int(query.get("frame", ["0"])[0])
                    camera = query.get("camera", ["0"])[0]
                    if route == "/api/video":
                        content, mime = viewer.video(episode, frame, camera), "image/jpeg"
                    else:
                        if route == "/api/frame":
                            result = viewer.frame(episode, frame, int(query.get("model", ["0"])[0]))
                        elif route == "/api/optical":
                            result = viewer.optical(episode, frame, float(query.get("gain", ["8"])[0]))
                        else:
                            result = viewer.metadata(episode, camera)
                        content, mime = json.dumps(result, allow_nan=False).encode(), "application/json"
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", mime)
                self.send_header("Content-Length", str(len(content)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(content)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as exc:
                content = json.dumps({"error": str(exc)}).encode()
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)

        def log_message(self, format, *args):
            if args and str(args[1]) != "200":
                super().log_message(format, *args)
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=ALIGNED_MANIFEST if Path(ALIGNED_MANIFEST).exists()
                        else "processing/joint_all/manifest_sensor_roles_corrected_v2.json")
    parser.add_argument("--checkpoint-roots", nargs="+", default=[
        "processing/runs/marker_autoencoders_freetacman_available_v1/trials",
        "processing/runs/marker_autoencoders_freetacman_available_32d_v1/trials",
        "processing/runs/marker_autoencoders_v1/trials",
        "processing/runs/egotouch_ae_search_v4/trials", "processing/runs/egotouch_ae_sensor_code_grid_v1",
        "processing/runs/sensor_autoencoders_v1/trials", "processing/runs/sensor_autoencoders_native_v2/trials",
        *[str(path) for architecture in ("conv", "mlp")
          for path in sorted(Path("processing/runs").glob(f"egotouch_ae_{architecture}_*"))
          if path.is_dir() and "smoke" not in path.name.lower()]])
    parser.add_argument("--cache", default="processing/sensor_ae_cache_v1")
    parser.add_argument("--training", default="processing/runs/sensor_autoencoders_native_v2")
    parser.add_argument("--frame-order", default="processing/audits/vla_touch_native_dynamics/dataset_frame_order.json")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--joint-roots", nargs="+", default=["processing/runs", "processing/joint_alignment"])
    parser.add_argument("--joint-device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    torch.set_num_threads(2)
    from .mirror_viewer import MirroredViewer, make_mirror_handler
    viewer = MirroredViewer(args.manifest, args.checkpoint_roots, args.cache, args.training, args.frame_order)
    from .joint_viewer import JointViewer
    joint = JointViewer(viewer, args.joint_roots, args.joint_device)
    server = ThreadingHTTPServer((args.host, args.port), make_mirror_handler(viewer, joint))
    print(f"Sensor AE viewer: http://{args.host}:{args.port} ({len(viewer.models)} models, {len(viewer.episodes)} episodes)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
