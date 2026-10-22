"""Current viewer operations composed from the public source and model APIs."""

import threading
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path

import torch

from ..configuration import load_experiment
from ..dataset_management.joint import collate_samples
from ..dataset_management.registry import build_sources
from ..models import JointModel, SensorDecoder, VJEPABackbone, spatial_video_features
from .attention import AttentionCapture
from .catalog import ModelCatalog, current_checkpoint
from .rendering import rgb_url, target_payload


class ViewerState:
    def __init__(
        self,
        project_root,
        experiment=None,
        *,
        registry=None,
        device="cpu",
        backbone_factory=VJEPABackbone,
    ):
        self.root = Path(project_root).resolve()
        code_root = Path(__file__).resolve().parents[3]
        experiment = experiment or code_root / "configs/experiment.json"
        self.experiment = (
            load_experiment(experiment, project_root=self.root) if registry is None else None
        )
        self.registry = registry or build_sources(self.experiment, self.root)
        self.device = torch.device(device)
        self.backbone_factory = backbone_factory
        self.backbone = None
        self.models = ModelCatalog(self.root)
        self.networks = OrderedDict()
        self.decoders = OrderedDict()
        self.lock = threading.RLock()

    def catalog(self, refresh=False):
        sources = []
        for name, adapter in self.registry.sources.items():
            spec = adapter.spec
            sources.append(
                dict(
                    id=name,
                    kind=spec.target_kind,
                    native_shape=list(spec.native_shape),
                    splits=sorted({row["split"] for row in adapter.rows}),
                    teacher=bool(spec.teacher),
                    code_dim=spec.code_dim,
                    mirrored=bool(spec.mirror),
                    heldout=name == self.registry.heldout_source,
                )
            )
        return dict(
            sources=sources,
            models=self.models.public(refresh),
            historical_url="legacy/",
            inference_device=str(self.device),
            semantics="8 RGB frames, one current tactile target; native target dimensions",
        )

    def episodes(self, source, split, text=""):
        if source not in self.registry.sources:
            raise ValueError("Unknown source")
        rows = self.registry[source].rows
        return [
            dict(id=row["episode_id"], frames=row["frames"], split=row["split"])
            for row in rows
            if row["split"] == split and text.lower() in row["episode_id"].lower()
        ]

    def _sample(self, source, episode, frame, mirrored):
        if source not in self.registry.sources:
            raise ValueError("Unknown source")
        adapter = self.registry[source]
        matches = [row for row in adapter.rows if row["episode_id"] == episode]
        if len(matches) != 1:
            raise ValueError("Select an episode from the source catalog")
        row = matches[0]
        if not 0 <= frame < row["frames"]:
            raise ValueError("Frame outside episode")
        batch = collate_samples([adapter.read_frame(row, frame)])
        if mirrored:
            batch = self.registry.augment(batch, seed=45, epoch=0, step=0, flags=[True])
        return row, batch

    def _network(self, path):
        stamp = (str(path), path.stat().st_mtime_ns, path.stat().st_size)
        if stamp not in self.networks:
            state = torch.load(path, map_location="cpu", weights_only=False)
            if not current_checkpoint(state):
                raise ValueError("This historical architecture is available on the Historical page")
            model = JointModel(**state["model_config"])
            model.load_state_dict(state["model"], strict=True)
            model.to(self.device).eval().requires_grad_(False)
            self.networks[stamp] = (model, state)
            while len(self.networks) > 2:
                self.networks.popitem(last=False)
        self.networks.move_to_end(stamp)
        return self.networks[stamp]

    def _decoder(self, spec):
        key = spec.teacher["sha256"]
        if key not in self.decoders:
            self.decoders[key] = SensorDecoder(spec, self.device)
            while len(self.decoders) > 3:
                self.decoders.popitem(last=False)
        self.decoders.move_to_end(key)
        return self.decoders[key]

    def _checkpoint_spec(self, spec, state):
        teachers = state.get("teachers") or state.get("provenance", {}).get("teachers", {})
        sources = teachers.get("sources", teachers)
        info = sources.get(spec.source)
        if not info:
            raise ValueError(
                "Checkpoint has no pinned decoder for this source; use the Historical page"
            )
        info = dict(info)
        path = Path(info["checkpoint"])
        info["checkpoint"] = str(path if path.is_absolute() else self.root / path)
        return replace(spec, teacher=info)

    def _load_backbone(self):
        if self.backbone is None:
            config = self.experiment["model"] if self.experiment else {}
            settings = config.get("backbone", {})
            repo = Path(settings.get("repo", "processing/upstream/vjepa2"))
            checkpoint = Path(
                settings.get("checkpoint", "processing/vjepa2_1/vjepa2_1_vitb_dist_vitG_384.pt")
            )
            repo = repo if repo.is_absolute() else self.root / repo
            checkpoint = checkpoint if checkpoint.is_absolute() else self.root / checkpoint
            self.backbone = self.backbone_factory(repo, checkpoint).to(self.device).eval()
        return self.backbone

    @torch.inference_mode()
    def _reconstruct(self, spec, batch):
        if not spec.teacher:
            return batch.target.values.clone()
        decoder = self._decoder(spec)
        target = batch.target.values.to(self.device).float()
        valid = batch.target.valid.to(self.device)
        if batch.target.kind == "marker_displacement":
            reference = batch.target.geometry["reference"].to(self.device)
            surface = batch.target.geometry["surface"].to(self.device)
            code = decoder.model.encode(target[:, 0], valid[:, 0], reference, surface)
        else:
            code = decoder.model.encode(target.flatten(0, 1), valid.flatten(0, 1))
        return decoder.decode(code[:, None], batch.target.geometry)

    @torch.inference_mode()
    def _predict(self, spec, batch, model_id, epoch, attention):
        entry, path = self.models.resolve(model_id, epoch)
        if spec.source not in entry["sources"]:
            raise ValueError("Selected model was not trained for this sensor")
        model, state = self._network(path)
        backbone = self._load_backbone()
        features = spatial_video_features(backbone, batch, self.device)
        mask = batch.visual.view_mask.to(self.device)
        roles = (
            batch.visual.camera_roles.to(self.device)
            if batch.visual.camera_roles is not None
            else None
        )
        capture = AttentionCapture(model)
        if entry["kind"] == "transfer":
            from ..evaluation.transfer import load_probe, predict_probe

            head, probe = load_probe(entry["head"], self.device)
            with capture:
                latent, _ = model.encode(features, mask, roles)
            prediction = predict_probe(head, probe, latent).reshape_as(
                batch.target.values.to(self.device)
            )
        else:
            if attention:
                with capture:
                    output = model(features, mask, spec.source, camera_roles=roles)
            else:
                output = model(features, mask, spec.source, camera_roles=roles)
            if "sensor_code" in output:
                checkpoint_spec = self._checkpoint_spec(spec, state)
                raw = model.raw_code(spec.source, output["sensor_code"])
                prediction = self._decoder(checkpoint_spec).decode(raw, batch.target.geometry)
            else:
                prediction = (
                    model.decode_numeric(spec.source, output["pred"])
                    if spec.source in model.direct_numeric
                    else output["pred"].float()
                )
        return prediction, capture.payload() if attention else None, int(state["epoch"]) + 1

    def frame(
        self,
        source,
        episode,
        frame,
        *,
        page="reconstruction",
        mirrored=False,
        model=None,
        epoch=None,
        attention=False,
    ):
        with self.lock:
            row, batch = self._sample(source, episode, int(frame), mirrored)
            spec = self.registry[source].spec
            maps = selected_epoch = None
            if page in ("prediction", "joint"):
                prediction, maps, selected_epoch = self._predict(
                    spec, batch, model, epoch, attention
                )
            else:
                prediction = self._reconstruct(spec, batch)
            images = []
            for camera in range(batch.visual.rgb.shape[2]):
                if not bool(batch.visual.view_mask[0, -1, camera]):
                    continue
                item = dict(
                    slot=camera,
                    label=("Scene / ego", "Left wrist", "Right wrist")[camera],
                    image=rgb_url(batch.visual.rgb[0, -1, camera]),
                )
                if maps is not None:
                    item["history_groups"] = [
                        rgb_url(batch.visual.rgb[0, 2 * group + 1, camera]) for group in range(4)
                    ]
                images.append(item)
            return dict(
                source=source,
                episode=episode,
                frame=int(frame),
                frames=row["frames"],
                current_frame_ids=batch.frame_indices[0].tolist(),
                history_frame_ids=batch.visual.history_frame_indices[0].tolist(),
                mirrored=mirrored,
                epoch=selected_epoch,
                scenes=images,
                target=target_payload(batch.target, prediction, spec),
                attention=maps,
                decoder=(
                    "Checkpoint-specific frozen teacher"
                    if page in ("prediction", "joint") and spec.teacher
                    else "Accepted frozen teacher"
                    if spec.teacher
                    else "Direct numeric sensor · no autoencoder"
                ),
            )
