"""On-demand scene-to-touch inference for the existing browser viewer."""
import hashlib
import json
import pickle
from pathlib import Path
import threading
import time

import numpy as np
import torch

from .aligned_frames import manifest_signature, scene_rgb
from .alignment_model import MODEL_TYPE, FrozenSensorDecoders, JointSensorAlignment, model_from_state
from .joint_all_config import ALL_TARGETS
from .joint_model import JointTactileVAE
from .model import DinoBackbone, MODEL_ID, REVISION
from .vjepa_model import (MODEL_TYPE as VJEPA_TYPE, BACKBONE_ID as VJEPA_ID, REVISION as VJEPA_REVISION,
                          CHECKPOINT_SHA256 as VJEPA_SHA, JointVJEPAAlignment,
                          VJEPABackbone, video_features, model_from_state as vjepa_from_state)
from .vjepa_spatial import (MODEL_TYPE as SPATIAL_VJEPA_TYPE, JointSpatialVJEPA,
                            spatial_video_features, model_from_state as spatial_vjepa_from_state)
from .marker_model import (MODEL_TYPE as MARKER_VJEPA_TYPE, MarkerSpatialVJEPA,
                           model_from_state as marker_vjepa_from_state)
from .marker_code_model import (MODEL_TYPE as MARKER_CODE_TYPE,MarkerCodeVJEPA,MixedMarkerTeachers,
                                model_from_state as marker_code_from_state)
from .marker_targets import MARKER_SOURCES
from .prepare_joint import TARGETS
from .train import MODEL_CONFIG_KEYS
from .frozen_probe import head as probe_head
from .viewer_transfer import validate_normalization, load_transfer_head
from .storage import compatible_data


VJEPA_TYPES = (VJEPA_TYPE, SPATIAL_VJEPA_TYPE, MARKER_VJEPA_TYPE, MARKER_CODE_TYPE)
SPATIAL_TYPES = (SPATIAL_VJEPA_TYPE, MARKER_VJEPA_TYPE, MARKER_CODE_TYPE)
ALIGNMENT_TYPES = (MODEL_TYPE, *VJEPA_TYPES)


def check_backbone(state):
    if state.get("model_type") in VJEPA_TYPES:
        if (state.get("backbone") != VJEPA_ID or state.get("backbone_revision") != VJEPA_REVISION
                or state["provenance"]["backbone"]["checkpoint_sha256"] != VJEPA_SHA):
            raise ValueError("Unsupported V-JEPA backbone")
        if state.get('model_type') in (MARKER_VJEPA_TYPE,MARKER_CODE_TYPE):
            from .aligned_frames import file_hash
            if file_hash(Path(state['config']['marker_targets'])/'index.json')!=state['provenance']['marker_targets']['index_sha256']:
                raise ValueError('Marker targets differ from the checkpoint provenance')
    elif state.get("backbone") != MODEL_ID or state.get("backbone_revision") != REVISION:
        raise ValueError("Unsupported DINO backbone")


def path_id(path):
    return hashlib.sha256(str(Path(path).resolve()).encode()).hexdigest()[:20]


def clip_indices(frame, length, frames=8, stride=2):
    """Keep the exact selected frame at position 3; repeat boundaries if needed."""
    if length < 1 or not 0 <= frame < length:
        raise ValueError("Frame outside episode")
    position = (frames - 1) // 2
    return np.clip(frame + (np.arange(frames) - position) * stride, 0, length - 1), position


def prediction_clip(config,row,frame):
    if config.get('prediction_mode')=='current_frame':
        from .current_frame import causal_clip_indices
        return causal_clip_indices(frame,row['frames'])
    return clip_indices(frame,row['frames'])


def checkpoint_stamp(path):
    info = Path(path).stat()
    return (info.st_ino, info.st_mtime_ns, info.st_size)


def read_checkpoint(path):
    """Atomic replacements can arrive during selection; never cache stale weights."""
    for _ in range(3):
        stamp = checkpoint_stamp(path)
        state = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        if checkpoint_stamp(path) == stamp:
            return state, stamp
    raise ValueError("Checkpoint is updating; refresh and try again")


def source_shapes(config, include_retired_images=False):
    if config.get("model_type") in ALIGNMENT_TYPES:
        registry = json.loads(Path(config["teachers"]).read_text())
        shapes = {source: info["target_shape"] for source, info in registry["sources"].items()}
        shapes.update({source: ALL_TARGETS[source][1] for source in ("ftp_rdp", "ftp_rdp_bimanual")})
    else:
        targets = ALL_TARGETS if config.get("dataset_family") == "joint_all" else TARGETS
        shapes = {source: value[1] for source, value in targets.items()}
    if config.get("train_sources"):
        shapes = {source: shapes[source] for source in config["train_sources"]}
    shapes.pop(config.get("heldout_source"), None)
    from .marker_storage import marker_only
    if not include_retired_images and config.get('model_type') not in (MARKER_VJEPA_TYPE, MARKER_CODE_TYPE):
        shapes = {source: shape for source,shape in shapes.items() if not marker_only(source)}
    return shapes


class JointViewer:
    def __init__(self, viewer, roots, device="auto"):
        self.viewer, self.roots = viewer, [Path(root) for root in roots]
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available()
                                   else "cpu" if device == "auto" else device)
        self.lock = threading.RLock()
        self.runs = {}
        self.snapshot_checks = {}
        self.transfer_checks, self.file_hashes = {}, {}
        self.scan_signatures = {}
        self.compatibility_checks = {}
        from .native_optical import native_shape
        self.native_shapes = {}
        for row in viewer.episodes:
            source = row["source"]
            if source not in self.native_shapes:
                self.native_shapes[source] = (native_shape(row) if ALL_TARGETS[source][0] == "optical_rgb"
                                              else ALL_TARGETS[source][1])
        self.backbone = None  # Loaded only when this page first requests a prediction.
        self.video_backbone = None
        from collections import OrderedDict
        self.models, self.features, self.predictions = OrderedDict(), OrderedDict(), OrderedDict()
        self.attentions=OrderedDict()

    def _inspect_snapshot(self, path):
        """Validate checkpoint structure without allocating legacy model weights."""
        stamp = checkpoint_stamp(path)
        key = str(path.resolve())
        if key in self.snapshot_checks and self.snapshot_checks[key][0] == stamp:
            return self.snapshot_checks[key][1]
        result = None
        try:
            state, stamp = read_checkpoint(path)
            check_backbone(state)
            cfg = state["config"]
            if state.get("model_type") in VJEPA_TYPES and self._hash(cfg["backbone_checkpoint"]) != VJEPA_SHA:
                raise ValueError("Stale V-JEPA backbone weights")
            if state.get("model_type") in ALIGNMENT_TYPES:
                manifest = cfg["manifest"]
                if manifest not in self.scan_signatures:
                    self.scan_signatures[manifest] = manifest_signature(manifest)
                compatibility_key = (manifest, json.dumps(state['provenance']['data'], sort_keys=True))
                if compatibility_key not in self.compatibility_checks:
                    self.compatibility_checks[compatibility_key] = (
                        self.scan_signatures[manifest] == state['provenance']['data']
                        or compatible_data(manifest, state['provenance']['data']))
                if not self.compatibility_checks[compatibility_key]:
                    raise ValueError("Stale alignment data")
                # Inspect structure without copying every epoch's weight tensors.
                constructor = {VJEPA_TYPE: JointVJEPAAlignment, SPATIAL_VJEPA_TYPE: JointSpatialVJEPA,
                               MARKER_VJEPA_TYPE: MarkerSpatialVJEPA,MARKER_CODE_TYPE:MarkerCodeVJEPA}.get(state.get('model_type'), JointSensorAlignment)
                network = constructor(**state["model_config"])
                expected = network.state_dict()
                if set(expected) != set(state["model"]) or any(
                        value.shape != state["model"][name].shape for name, value in expected.items()):
                    raise ValueError("Checkpoint architecture differs")
                shapes = {source: state["teachers"]["sources"][source]["target_shape"]
                          for source in network.adapters}
                shapes.update(network.numeric_shapes)
                if state.get('model_type')==MARKER_VJEPA_TYPE:
                    shapes.update(cfg['marker_target_shapes'])
                for source in network.adapters:
                    info = state["teachers"]["sources"][source]
                    if self._hash(info["checkpoint"]) != info["sha256"]:
                        raise ValueError("Stale decoder")
            elif state.get("model_type") is None:
                shapes = source_shapes(cfg, include_retired_images=True)
                targets = {source: (ALL_TARGETS[source][0], shape, ALL_TARGETS[source][2])
                           for source, shape in shapes.items()}
                with torch.device("meta"):
                    network = JointTactileVAE(input_dim=768, targets=targets,
                        **{name: cfg[name] for name in MODEL_CONFIG_KEYS})
                expected = network.state_dict()
                if set(expected) != set(state["model"]) or any(
                        value.shape != state["model"][name].shape for name, value in expected.items()):
                    raise ValueError("Checkpoint architecture differs")
            else:
                raise ValueError("Unsupported model type")
            result = {"epoch": int(state["epoch"]) + 1, "source_shapes": shapes}
        except (OSError, ValueError, KeyError, RuntimeError, EOFError, TypeError, pickle.UnpicklingError):
            pass
        self.snapshot_checks[key] = (stamp, result)
        return result

    def _hash(self, path):
        key, stamp = str(Path(path).resolve()), checkpoint_stamp(path)
        if key not in self.file_hashes or self.file_hashes[key][0] != stamp:
            from .aligned_frames import file_hash
            self.file_hashes[key] = (stamp, file_hash(path))
        return self.file_hashes[key][1]

    def _transfer_run(self, run, cfg):
        """Expose only evaluated decoders paired with their exact frozen encoder."""
        pairs, snapshots = {}, []
        for path in sorted((run / "evaluations").glob("motiontrans*/decoder.pt")):
            try:
                results_path = path.with_name("results.json")
                results = json.loads(results_path.read_text())
                if results.get("source") != "ftp_motiontrans" or not results.get("frozen_encoder"):
                    continue
                saved, head_stamp = read_checkpoint(path)
                encoder_path = Path(saved["config"]["checkpoint"])
                encoder_stamp = checkpoint_stamp(encoder_path)
                stamp = (*head_stamp, *encoder_stamp, *checkpoint_stamp(results_path))
                key = str(path.resolve())
                if key not in self.transfer_checks or self.transfer_checks[key][0] != stamp:
                    checked = self._inspect_snapshot(encoder_path)
                    if checked is None or saved["checkpoint_sha256"] != self._hash(encoder_path):
                        raise ValueError("Transfer decoder does not match its frozen encoder")
                    state, _ = read_checkpoint(encoder_path)
                    if (state.get("model_type") not in (None, *ALIGNMENT_TYPES)
                            or state["config"].get("heldout_source") != "ftp_motiontrans"):
                        raise ValueError("Unsupported transfer encoder")
                    if (results["checkpoint_sha256"] != saved["checkpoint_sha256"]
                            or results["checkpoint_epoch"] + 1 != checked["epoch"]
                            or results["target_shape"] != ALL_TARGETS["ftp_motiontrans"][1]):
                        raise ValueError("Transfer evaluation provenance differs")
                    latent = state["config"]["latent_dim"]
                    validate_normalization(saved,latent)
                    widths = saved["config"].get("hidden_dims", [512])
                    with torch.device("meta"):
                        network = probe_head(latent, 6, hidden_dims=widths)
                    expected = network.state_dict()
                    if set(expected) != set(saved["head"]) or any(
                            value.shape != saved["head"][name].shape for name, value in expected.items()):
                        raise ValueError("Transfer decoder architecture differs")
                    self.transfer_checks[key] = (stamp, {"epoch": checked["epoch"],
                        "head_epochs": saved["selected_epochs"], "decoder_widths": [latent, *widths, 6],
                        "head_seed": saved.get('seed'),
                        "feature_normalization": saved.get('feature_normalization','none')})
                info = self.transfer_checks[key][1]
                identifier = str(path.relative_to(run))
                pairs[identifier] = {"encoder": str(encoder_path), "decoder": str(path)}
                snapshots.append({"id": identifier, **info, "stamp": list(stamp),
                    "label": f"Encoder epoch {info['epoch']} · decoder " +
                             " → ".join(map(str, info["decoder_widths"])) +
                             (f" · head seed {info['head_seed']}" if info.get('head_seed') is not None else '')})
            except (OSError, ValueError, KeyError, RuntimeError, EOFError, TypeError, pickle.UnpicklingError):
                continue
        if not snapshots:
            return None
        snapshots.sort(key=lambda item: -item["epoch"])
        identifier = path_id(run / "motiontrans_transfer_viewer")
        name=run.name + ' · MotionTrans transfer'
        if cfg.get('model_type') in VJEPA_TYPES:
            name=f"V-JEPA 2.1 B · {cfg['latent_mode']} · {cfg['latent_dim']}D · MotionTrans transfer"
            if cfg['model_type'] in (SPATIAL_VJEPA_TYPE,MARKER_VJEPA_TYPE,MARKER_CODE_TYPE):name+=' · spatial attention'
            if cfg['model_type']==MARKER_VJEPA_TYPE:name+=' · marker targets'
            if cfg['model_type']==MARKER_CODE_TYPE:name+=' · marker AE codes'
            if cfg.get('prediction_mode')=='current_frame':name+=' · current frame + 7 history'
        return {"id": identifier, "path": str(run), "name": name,
                "config": cfg, "source_shapes": {"ftp_motiontrans": ALL_TARGETS["ftp_motiontrans"][1]},
                "snapshots": snapshots, "transfer_heads": pairs,
                "completed_epochs": snapshots[0]["epoch"], "total_epochs": cfg.get("epochs"),
                "smoke": False, "kind": "Frozen encoder + fitted MotionTrans decoder"}

    def _scan(self):
        # A catalog scan shares one fresh source snapshot across saved epochs.
        self.scan_signatures = {}
        self.compatibility_checks = {}
        paths = set()
        for root in self.roots:
            if "smoke" not in root.name.lower():
                paths.update(path for path in root.rglob("config.json")
                             if not any("smoke" in part.lower() for part in path.relative_to(root).parts))
        found = {}
        for path in sorted(paths):
            try:
                cfg = json.loads(path.read_text())
                deterministic = (cfg.get("model_type") in ALIGNMENT_TYPES and
                    cfg.get("latent_objective") == "deterministic" and cfg.get("latent_mode") == "deterministic"
                    and not cfg.get("stochastic"))
                if ((cfg.get("latent_objective") != "vae" and not deterministic)
                        or ("width" not in cfg and cfg.get("model_type") not in ALIGNMENT_TYPES)):
                    continue
                if not (cfg.get("dataset_family") == "joint_all" or cfg.get("model_type") in ALIGNMENT_TYPES
                        or "joint_six" in str(cfg.get("manifest", ""))):
                    continue
                run = path.parent
                identifier = path_id(run)
                shapes = source_shapes(cfg)
                if not any(list(shape) == list(self.native_shapes.get(source, []))
                           for source, shape in shapes.items()):
                    continue
                completed, best_epoch = None, None
                history = run / "metrics.jsonl"
                if history.exists():
                    # A partially written last line must not hide a completed epoch.
                    for line in reversed(history.read_text().splitlines()):
                        try:
                            epoch = json.loads(line)["epoch"]
                            completed = epoch if cfg.get("model_type") in ALIGNMENT_TYPES else epoch + 1
                            break
                        except (ValueError, KeyError):
                            continue
                metrics = run / "best_metrics.json"
                if metrics.exists():
                    best_epoch = json.loads(metrics.read_text()).get("epoch")
                snapshots = []
                candidates = list(run.glob("epoch_*.pt")) + list((run / "viewer_epochs").glob("epoch_*.pt"))
                candidates += [run / "best.pt", run / "last.pt"]
                for checkpoint in sorted(set(candidates)):
                    if not checkpoint.exists():
                        continue
                    checked = self._inspect_snapshot(checkpoint)
                    if checked is None:
                        continue
                    key = str(checkpoint.relative_to(run))
                    number = checked["epoch"]
                    if checkpoint.stem.startswith("epoch_"):
                        label = f"Epoch {number}" + (" · weights" if "viewer_epochs" in key else "")
                    else:
                        label = ("Latest saved" if checkpoint.stem == "last" else "Best validation")
                        if number is not None:
                            label += f" · epoch {number}"
                    snapshots.append({"id": key, "epoch": number, "label": label,
                                      "stamp": list(checkpoint_stamp(checkpoint))})
                snapshots.sort(key=lambda item: (item["id"] != "last.pt", item["id"] != "best.pt",
                                                -(item["epoch"] or 0), item["id"]))
                if not snapshots:
                    continue
                name = str(Path(run.parent.name) / run.name) if run.name == "training_smoke" else run.name
                if cfg.get("model_type") in VJEPA_TYPES:
                    name = f"V-JEPA 2.1 B · {cfg['latent_mode']} · {cfg['latent_dim']}D"
                    if cfg['model_type'] in (SPATIAL_VJEPA_TYPE,MARKER_VJEPA_TYPE,MARKER_CODE_TYPE):
                        name += ' · spatial attention · live'
                    if cfg['model_type']==MARKER_VJEPA_TYPE:name+=' · marker targets'
                    if cfg['model_type']==MARKER_CODE_TYPE:name+=' · marker AE codes'
                    if cfg.get('prediction_mode')=='current_frame':name+=' · current frame + 7 history'
                found[identifier] = {"id": identifier, "path": str(run), "name": name,
                    "config": cfg, "source_shapes": shapes, "snapshots": snapshots,
                    "completed_epochs": completed, "total_epochs": cfg.get("epochs"),
                    "smoke": "smoke" in str(run),
                    "kind": "Sensor-code alignment" if cfg.get("model_type") in ALIGNMENT_TYPES else "Direct tactile decoder"}
                if cfg.get("heldout_source") == "ftp_motiontrans":
                    transfer = self._transfer_run(run, cfg)
                    if transfer is not None:
                        found[transfer["id"]] = transfer
            except (OSError, ValueError, KeyError):
                continue
        self.runs = found

    def catalog(self):
        with self.lock:
            self._scan()
            models = [{key: value for key, value in run.items() if key not in ("path", "transfer_heads")}
                      for run in self.runs.values()]
            for model in models:model['attention_supported']=model['config'].get('model_type') in SPATIAL_TYPES
            models.sort(key=lambda run: (run["smoke"], run["kind"] != "Sensor-code alignment", run["name"]))
        base = self.viewer.catalog()
        return {**base, "models": models, "training": {"status": "Saved checkpoints refresh automatically"},
                "device": str(self.device)}

    def _load(self, run, snapshot):
        pair = run.get("transfer_heads", {}).get(snapshot)
        path = Path(pair["encoder"]) if pair else Path(run["path"]) / snapshot
        stamp = (*checkpoint_stamp(path), *checkpoint_stamp(pair["decoder"])) if pair else checkpoint_stamp(path)
        key = (run["id"], snapshot)
        if key in self.models and self.models[key][0] == stamp:
            self.models.move_to_end(key)
            return self.models[key][1:], stamp
        # Release stale/older GPU models before allocating the replacement.
        if key in self.models:
            del self.models[key]
        while len(self.models) >= 2:
            self.models.popitem(last=False)
        state, encoder_stamp = read_checkpoint(path)
        if pair:
            head_state, head_stamp = read_checkpoint(pair["decoder"])
            if head_state["checkpoint_sha256"] != self._hash(path):
                raise ValueError("Transfer decoder does not match its frozen encoder")
            stamp = (*encoder_stamp, *head_stamp)
        else:
            stamp = encoder_stamp
        check_backbone(state)
        cfg = state["config"]
        if state.get("model_type") in ALIGNMENT_TYPES:
            if (manifest_signature(cfg["manifest"]) != state["provenance"]["data"]
                    and not compatible_data(cfg['manifest'], state['provenance']['data'])):
                raise ValueError("Checkpoint training-data provenance differs")
            loader = {VJEPA_TYPE: vjepa_from_state, SPATIAL_VJEPA_TYPE: spatial_vjepa_from_state,
                      MARKER_VJEPA_TYPE: marker_vjepa_from_state,MARKER_CODE_TYPE:marker_code_from_state}.get(state.get('model_type'), model_from_state)
            network = loader(state)
            decoder = (MixedMarkerTeachers(state['teachers'],self.device) if state.get('model_type')==MARKER_CODE_TYPE
                       else FrozenSensorDecoders(state["teachers"], self.device))
            shapes = {**{s: info["target_shape"] for s, info in state["teachers"]["sources"].items()},
                      **state["model_config"]["numeric_shapes"]}
            trained = set(network.adapters) | set(network.numeric_heads)
            if state.get('model_type')==MARKER_VJEPA_TYPE:
                shapes.update(cfg['marker_target_shapes']);trained.update(network.marker_heads)
            shapes = {s: shapes[s] for s in trained}
        else:
            shapes = source_shapes(cfg, include_retired_images=True)
            targets = {source: (ALL_TARGETS[source][0], shape, ALL_TARGETS[source][2])
                       for source, shape in shapes.items()}
            network = JointTactileVAE(input_dim=768, targets=targets,
                                     **{key: cfg[key] for key in MODEL_CONFIG_KEYS})
            network.load_state_dict(state["model"], strict=True)
            decoder = None
        if pair:
            decoder = load_transfer_head(head_state,cfg['latent_dim'])
            decoder.to(self.device).eval().requires_grad_(False)
            shapes = run["source_shapes"]
        network.to(self.device).eval().requires_grad_(False)
        entry = (stamp, network, decoder, int(state["epoch"]) + 1, shapes)
        self.viewer._keep_recent(self.models, key, entry, 2)
        return entry[1:], stamp

    def _features(self, episode, indices, mode):
        row = self.viewer.episodes[episode]
        unique, inverse = np.unique(indices, return_inverse=True)
        if self.backbone is None:
            self.backbone = DinoBackbone().to(self.device).eval().requires_grad_(False)
        # Retain this request's hits locally: inserting newly encoded frames
        # into the bounded LRU may evict a hit that is still needed below.
        selected = {}
        for index in unique:
            key = (episode, int(index), mode)
            if key in self.features:
                selected[int(index)] = self.features[key]
                self.features.move_to_end(key)
        missing = [int(index) for index in unique if int(index) not in selected]
        if missing:
            input_row = {**row, "views": {"0": row["views"]["0"]}} if mode == "ego" else row
            rgb, masks = scene_rgb(input_row, missing)
            # Migrated scene storage may return all physical camera slots.
            # Apply the viewer input selection explicitly before encoding.
            if mode=='ego':rgb[:,1:]=0;masks[:,1:]=False
            rgb, masks = rgb.to(self.device), masks.to(self.device)
            images = rgb.flatten(0, 1)[masks.flatten()]
            encoded = torch.cat([self.backbone(chunk) for chunk in images.split(8)])
            restored = encoded.new_zeros(len(missing) * 3, *encoded.shape[1:])
            restored[masks.flatten()] = encoded
            restored = restored.reshape(len(missing), 3, *encoded.shape[1:]).cpu()
            for i, index in enumerate(missing):
                selected[index] = (restored[i], masks[i].cpu())
                self.viewer._keep_recent(self.features, (episode, index, mode), selected[index], 64)
        values = [selected[int(index)] for index in unique]
        return (torch.stack([value[0] for value in values])[inverse].unsqueeze(0).to(self.device),
                torch.stack([value[1] for value in values])[inverse].unsqueeze(0).to(self.device))

    def _video_features(self, episode, indices, mode, cfg):
        # Video features depend on the ENTIRE ordered clip, not individual frames.
        key = (VJEPA_ID, VJEPA_REVISION, VJEPA_SHA, cfg.get('model_type'), episode, tuple(map(int, indices)), mode, 384)
        if key in self.features:
            self.features.move_to_end(key)
            values, mask = self.features[key]
            return values.to(self.device), mask.to(self.device)
        if self.video_backbone is None:
            self.video_backbone = VJEPABackbone(cfg['backbone_repo'], cfg['backbone_checkpoint']).to(self.device).eval()
        row = self.viewer.episodes[episode]
        input_row = {**row, 'views': {'0': row['views']['0']}} if mode == 'ego' else row
        rgb, mask = scene_rgb(input_row, indices, size=384)
        if mode=='ego':rgb[:,1:]=0;mask[:,1:]=False
        batch = {'rgb': rgb.unsqueeze(0), 'view_mask': mask.unsqueeze(0)}
        spatial = cfg.get('model_type') in (SPATIAL_VJEPA_TYPE,MARKER_VJEPA_TYPE,MARKER_CODE_TYPE)
        with torch.autocast(self.device.type, torch.bfloat16, enabled=self.device.type == 'cuda'):
            extract = spatial_video_features if spatial else video_features
            values = extract(self.video_backbone, batch, self.device, cfg.get('video_chunk_size', 2))
        mask = batch['view_mask']
        self.viewer._keep_recent(self.features, key, (values.cpu(), mask.cpu()), 2 if spatial else 64)
        return values, mask.to(self.device)

    def _refresh_storage(self):
        from .storage import REGISTRY
        stamp=REGISTRY.stat().st_mtime_ns if REGISTRY.exists() else None
        if getattr(self,'_storage_stamp',None)!=stamp:
            self.features.clear();self.predictions.clear();self.attentions.clear()
            self._storage_stamp=stamp

    def _attention_run(self,episode,frame,model,epoch,views):
        row=self.viewer._episode(episode,frame)
        if views not in ('ego','all'):raise ValueError('Choose ego or all scene cameras')
        if views=='ego' and '0' not in row['views']:raise ValueError('This episode has no ego camera')
        if model not in self.runs:self._scan()
        run=self.runs.get(model)
        if run is None:raise ValueError('Unknown joint model; refresh the model list')
        if run['config'].get('model_type') not in SPATIAL_TYPES:
            raise ValueError('Attention overlays require a spatial-attention V-JEPA model')
        if epoch not in {item['id'] for item in run['snapshots']}:raise ValueError('That epoch was not saved')
        if row['source'] not in run['source_shapes']:raise ValueError('This sensor has no head in the selected model')
        return row,run

    def attention(self,episode,frame,model,epoch='last.pt',views='ego',expected_epoch=None):
        from .attention_viewer import capture_cross_attention,encode_weights
        if epoch=='last':epoch='last.pt'
        with self.lock,torch.inference_mode():
            self._refresh_storage()
            row,run=self._attention_run(episode,frame,model,epoch,views)
            (network,_,actual_epoch,_),stamp=self._load(run,epoch)
            if expected_epoch is not None and actual_epoch!=expected_epoch:
                raise ValueError('Checkpoint updated; refresh the frame or select a fixed epoch for matching attention')
            key=(model,epoch,stamp,episode,frame,views)
            if key not in self.attentions:
                indices,position=prediction_clip(run['config'],row,frame)
                features,mask=self._video_features(episode,indices,views,run['config'])
                with torch.autocast(self.device.type,torch.bfloat16,enabled=self.device.type=='cuda'):
                    weights=capture_cross_attention(network.visual,features,mask,position,row['source'])
                slots=torch.where(mask[0].all(0))[0].tolist()
                result=dict(**encode_weights(weights),model=run['name'],epoch=actual_epoch,
                    target_frame=frame,clip_frames=indices.tolist(),camera_slots=[str(s) for s in slots],
                    query_index=position,default_image_slot=(7 if run['config'].get('prediction_mode')=='current_frame' else position),
                    temporal_groups=[indices[i:i+2].tolist() for i in range(0,8,2)],
                    kind='tactile-query cross-attention',normalization='Probability mass across all camera/time/patch tokens',
                    interpretation='Two input images share each temporal-token map. Tokens contain context from the whole clip. Attention weights are not causal pixel importance.')
                self.viewer._keep_recent(self.attentions,key,result,8)
            return self.attentions[key]

    def attention_image(self,episode,frame,model,epoch,views,camera,history):
        from io import BytesIO
        from PIL import Image
        if epoch=='last':epoch='last.pt'
        with self.lock:
            row,run=self._attention_run(episode,frame,model,epoch,views)
            slots={'0'} if views=='ego' else set(row['views'])
            if camera not in slots:raise ValueError('This camera is not an input to the selected prediction')
            if not 0<=history<8:raise ValueError('History slot must be between zero and seven')
            indices,_=prediction_clip(run['config'],row,frame)
            rgb,_=scene_rgb(row,[int(indices[history])],size=384)
            pixels=(rgb[0,int(camera)].permute(1,2,0).numpy()*255).round().clip(0,255).astype(np.uint8)
            buffer=BytesIO();Image.fromarray(pixels).save(buffer,'JPEG',quality=92)
            return buffer.getvalue()

    def frame(self, episode, frame, model, epoch="last.pt", views="ego"):
        row = self.viewer._episode(episode, frame)
        if model in ("", "-1"):
            return self.viewer.frame(episode, frame, -1)
        if epoch == "last":
            epoch = "last.pt"
        if views not in ("ego", "all"):
            raise ValueError("Choose ego or all scene cameras")
        if views == "ego" and "0" not in row["views"]:
            raise ValueError("This dataset has no ego camera; choose all available scene cameras")
        with self.lock, torch.inference_mode():
            self._refresh_storage()
            if model not in self.runs:
                self._scan()
            run = self.runs.get(model)
            if run is None:
                raise ValueError("Unknown joint model; refresh the model list")
            if epoch not in {item["id"] for item in run["snapshots"]}:
                raise ValueError("That epoch was not saved; refresh and select an available checkpoint")
            shape = run["source_shapes"].get(row["source"])
            if shape is None:
                raise ValueError("This sensor was held out or has no trained head in the selected model")
            from .native_optical import native_shape
            native = native_shape(row) if ALL_TARGETS[row["source"]][0] == "optical_rgb" else ALL_TARGETS[row["source"]][1]
            if list(shape) != list(native):
                raise ValueError("This model predicts reduced-resolution tactile images; native comparison is unavailable")
            began = time.monotonic()
            (network, decoder, actual_epoch, shapes), stamp = self._load(run, epoch)
            if list(shapes.get(row["source"], [])) != list(native):
                raise ValueError("Checkpoint sensor head does not match the native target")
            indices,position=prediction_clip(run['config'],row,frame)
            marker_mode=(run['config'].get('model_type') in (MARKER_VJEPA_TYPE,MARKER_CODE_TYPE)
                         and row['source'] in MARKER_SOURCES and not run.get('transfer_heads'))
            key = (model, epoch, stamp, episode, frame, views)
            if key not in self.predictions:
                if run["config"].get("model_type") in VJEPA_TYPES:
                    features, mask = self._video_features(episode, indices, views, run["config"])
                else:
                    features, mask = self._features(episode, indices, views)
                with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                    if run.get("transfer_heads"):
                        mu, _ = network.visual.encode(features, mask)
                        prediction = decoder(mu[:, position])[0].reshape(native).float().cpu().numpy()
                        output = None
                    else:
                        extra={}
                        if marker_mode:
                            from .marker_viewer import inputs as marker_inputs
                            _,marker_batch,_=marker_inputs(run['config'],row,frame)
                            extra=dict(marker_reference=marker_batch['marker_reference'][None].to(self.device),
                                       marker_surface=marker_batch['marker_surface'][None].to(self.device))
                        output = network(features, mask, row["source"], sample_latent=False,**extra)
                if output is None:
                    pass  # Held-out prediction comes from the matched transfer decoder.
                elif 'marker_displacement' in output:
                    prediction=output['marker_displacement'][0,position].float().cpu().numpy()
                elif marker_mode and run['config'].get('model_type')==MARKER_CODE_TYPE:
                    code=network.raw_code(row['source'],output['sensor_code'][:,position:position+1])
                    prediction=decoder.decode_marker(row['source'],code,dict(
                        marker_reference=extra['marker_reference'],marker_surface=extra['marker_surface']))[0,0].cpu().numpy()
                elif "sensor_code" in output:
                    code = network.raw_code(row["source"], output["sensor_code"][:, position:position + 1])
                    prediction = decoder.decode(row["source"], code)[0, 0].float().cpu().numpy()
                else:
                    prediction = output["pred"][0, position].float().cpu().numpy()
                if not np.isfinite(prediction).all():
                    raise ValueError("Checkpoint produced nonfinite predictions")
                self.viewer._keep_recent(self.predictions, key, prediction, 8)
            if marker_mode:
                from .marker_viewer import payload as marker_payload
                result=marker_payload(run['config'],row,frame,prediction=self.predictions[key])
            else:
                result = self.viewer.frame(episode, frame, -1, prediction=self.predictions[key])
            result["joint"] = {"model": run["name"], "epoch": actual_epoch, "snapshot": epoch,
                "views": views, "camera_slots": ["0"] if views == "ego" else sorted(row["views"]),
                "clip_frames": indices.tolist(), "selected_position": position,
                "seconds": time.monotonic() - began, "latent": "mean"}
            if run["config"].get("model_type") in VJEPA_TYPES:
                result["joint"].update(backbone="V-JEPA 2.1 ViT-B/16",
                    uncertainty="uncalibrated" if run["config"].get("stochastic") else "none")
            if run.get("transfer_heads"):
                snapshot = next(item for item in run["snapshots"] if item["id"] == epoch)
                result["joint"].update(frozen_encoder=True, decoder_epochs=snapshot["head_epochs"],
                                       decoder_widths=snapshot["decoder_widths"])
            return result
