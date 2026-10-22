"""Verify lossless historical continuation and reject scientific mismatches."""

import copy
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_core_models import assert_same_tensors, config, inputs
from test_core_training import batch, update

from tactilelatent.dataset_management.artifacts import ArtifactPaths, file_hash
from tactilelatent.models import JointModel
from tactilelatent.models.joint import MODEL_TYPE
from tactilelatent.training.legacy_import import import_legacy_checkpoint
from tactilelatent.training.optim import SAMStep, make_scheduler, rng_snapshot


class Registry:
    """Use real artifact readers with small synthetic immutable receipts."""

    def __init__(self, root, teachers, caches):
        self.paths = ArtifactPaths(root, root, root / "archive")
        self.heldout_source = "holdout"
        self.specs = {name: SimpleNamespace(teacher=teacher) for name, teacher in teachers.items()}
        self.caches = caches
        self.import_checked = False

    def __getitem__(self, name):
        return SimpleNamespace(cache=SimpleNamespace(index_hash=self.caches[name]))

    def require_imports(self):
        self.import_checked = True


def write_json(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return str(path)


@pytest.fixture
def saved(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(45)
    random.seed(45)
    np.random.seed(45)
    mc = config()
    mc.update(input_dim=12, initialization="baseline_v1")
    model = JointModel(**mc)
    tc = dict(
        seed=45,
        epochs=3,
        batch_size=2,
        accumulation=1,
        workers=0,
        train_windows_per_source_per_epoch=8,
        validation_windows_per_source=4,
        learning_rate=0.0003,
        min_learning_rate=1e-6,
        weight_decay=0.05,
        warmup_epochs=1,
        optimizer_method="sam_adamw",
        sam_rho=0.05,
        wrist_dropout=0.3,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=tc["learning_rate"], weight_decay=tc["weight_decay"]
    )
    scheduler, sam = make_scheduler(optimizer, tc, 1), SAMStep(optimizer)
    # Populate every slot, including heads not used in the following probe.
    sum(parameter.square().sum() for parameter in model.parameters()).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    scheduler.step()
    sam.steps = 1
    teachers = {}
    for source, dimension in mc["sensor_code_dims"].items():
        path = tmp_path / f"teachers/{source}.pt"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(source.encode())
        teachers[source] = dict(checkpoint=str(path), sha256=file_hash(path), code_dim=dimension)
    teacher_path = write_json(tmp_path, "teachers.json", {"sources": teachers})
    manifest = write_json(tmp_path, "manifest.json", {"episodes": []})
    base_manifest = write_json(tmp_path, "base_manifest.json", {"episodes": ["base"]})
    base_data = {"manifest_sha256": file_hash(base_manifest)}
    norm = write_json(tmp_path, "normalization.json", {"data": base_data})
    marker = write_json(tmp_path, "marker/index.json", {"provenance": {"version": "marker"}})
    marker_teachers = write_json(tmp_path, "marker_teachers.json", {})
    codes = write_json(tmp_path, "codes/registry.json", {})
    expanded_teacher = write_json(tmp_path, "expansion/teachers.json", {})
    expansion = write_json(
        tmp_path, "expansion/ready.json", {"manifest_sha256": file_hash(manifest)}
    )
    native_index = write_json(tmp_path, "native/index.json", {"complete": True})
    acquisition_sources = {"ftp_unit": {"index_sha256": file_hash(native_index)}}
    acquisitions = write_json(tmp_path, "acquisition/ready.json", {"sources": acquisition_sources})
    mirror_policy = write_json(tmp_path, "policy.json", {"sources": {}})
    write_json(tmp_path, "mirror/ready.json", {"sources": {}})
    producer = tmp_path / "archive/producer.py"
    producer.parent.mkdir()
    producer.write_text('"""Immutable synthetic producer; never imported."""\n')
    old = dict(
        **tc,
        training_entrypoint="scripts/train_all_sources_vjepa.py",
        dataset_family="joint_all",
        model_type=MODEL_TYPE,
        latent_mode="deterministic",
        prediction_mode="current_frame",
        stochastic=False,
        beta_max=0,
        heldout_source="holdout",
        excluded_sources=[],
        rgb_size=384,
        manifest=manifest,
        normalization=norm,
        marker_targets=str(Path(marker).parent),
        teachers=teacher_path,
        marker_teachers=marker_teachers,
        code_cache=str(Path(codes).parent),
        data_expansion=str(Path(expanded_teacher).parent),
        mirror_augmentation={"enabled": False, "probability": 0.5, "sources": []},
    )
    recorded = dict(
        teachers={"sources": teachers},
        data={"manifest_sha256": file_hash(manifest)},
        base_data=base_data,
        normalization_sha256=file_hash(norm),
        marker_targets={"index_sha256": file_hash(marker), "provenance": {"version": "marker"}},
        marker_teachers_sha256=file_hash(marker_teachers),
        code_registry_sha256=file_hash(codes),
        code_caches={},
        expansion={
            "teachers_sha256": file_hash(expanded_teacher),
            "manifest_sha256": file_hash(manifest),
            "code_index_sha256": {},
        },
        acquisitions={"ready_sha256": file_hash(acquisitions), "sources": acquisition_sources},
        mirror_augmentation={"latent_equality_constraint": False, "bindings": []},
        backbone={"id": "test-backbone"},
        implementation={"producer.py": file_hash(producer)},
    )
    dc = dict(
        sources=list(mc["sensor_code_dims"]) + list(mc["numeric_shapes"]),
        heldout_source="holdout",
        excluded_sources=[],
        rgb_size=384,
        clip_frames=8,
        frame_stride=2,
        registry_paths={
            "manifest": manifest,
            "storage_manifest": base_manifest,
            "storage_normalization": norm,
            "marker_targets": str(Path(marker).parent),
            "teachers": teacher_path,
            "expansion_ready": expansion,
            "acquisition_ready": acquisitions,
        },
        mirror={
            "enabled": False,
            "probability": 0.5,
            "sources": [],
            "policy": mirror_policy,
            "preparation": str(tmp_path / "mirror"),
        },
    )
    experiment = {"training": tc, "datasets": dc, "model": {"backbone": {"video_chunk_size": 2}}}
    provenance = {"backbone": {"id": "test-backbone"}, "implementation": {"current.py": "new"}}
    generator = torch.Generator().manual_seed(45)
    state = dict(
        model_type=MODEL_TYPE,
        model_config=mc,
        model=copy.deepcopy(model.state_dict()),
        config=old,
        provenance=recorded,
        teachers={"sources": teachers},
        epoch=0,
        optimizer=copy.deepcopy(optimizer.state_dict()),
        scheduler=scheduler.state_dict(),
        sam=sam.state_dict(),
        rng=rng_snapshot(),
        python_rng=random.getstate(),
        numpy_rng=np.random.get_state(),
        loader_rng=generator.get_state(),
        best_score=0.5,
        validation={"ftp_unit": {"supervision_loss": 0.5}},
        elapsed_seconds=12.3,
    )
    path = tmp_path / "last.pt"
    torch.save(state, path)
    write_json(tmp_path, "preflight.json", {"passed": True, "config": old, "provenance": recorded})
    registry = Registry(tmp_path, teachers, {"ftp_unit": file_hash(native_index)})
    return SimpleNamespace(
        path=path,
        state=state,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        sam=sam,
        generator=generator,
        registry=registry,
        experiment=experiment,
        provenance=provenance,
        mc=mc,
    )


def imported(saved):
    return import_legacy_checkpoint(
        saved.path,
        model_config=saved.mc,
        experiment=saved.experiment,
        provenance=saved.provenance,
        registry=saved.registry,
        model=saved.model,
        optimizer=saved.optimizer,
        scheduler=saved.scheduler,
        sam=saved.sam,
        generator=saved.generator,
        batches_per_epoch=1,
    )


def test_import_preserves_state_bytes_and_next_complete_sam_update(saved):
    before = file_hash(saved.path)
    features, _ = inputs()
    random_expected, numpy_expected = random.random(), np.random.rand()
    loader_expected = torch.rand(3, generator=saved.generator)
    update(
        saved.model,
        batch(),
        features,
        optimizer=saved.optimizer,
        scheduler=saved.scheduler,
        sam=saved.sam,
    )
    expected_model = copy.deepcopy(saved.model.state_dict())
    expected_optimizer = copy.deepcopy(saved.optimizer.state_dict())
    expected_scheduler = saved.scheduler.state_dict()
    expected_rng = torch.get_rng_state().clone()
    state, receipt = imported(saved)
    assert state["epoch"] == 0 and state["elapsed_seconds"] == 12.3
    assert state["config"] == saved.experiment and state["provenance"] == saved.provenance
    assert receipt["original_metadata"]["provenance"] == saved.state["provenance"]
    assert receipt["old_checkpoint_rewritten"] is False
    assert receipt["optimizer_reinitialized"] is False
    assert saved.registry.import_checked
    assert file_hash(saved.path) == before
    json.dumps(receipt)
    assert random.random() == random_expected and np.random.rand() == numpy_expected
    assert torch.equal(torch.rand(3, generator=saved.generator), loader_expected)
    update(
        saved.model,
        batch(),
        features,
        optimizer=saved.optimizer,
        scheduler=saved.scheduler,
        sam=saved.sam,
    )
    assert_same_tensors(expected_model, saved.model.state_dict())
    for key in expected_optimizer["state"]:
        assert_same_tensors(
            expected_optimizer["state"][key], saved.optimizer.state_dict()["state"][key]
        )
    assert saved.scheduler.state_dict() == expected_scheduler
    assert torch.equal(expected_rng, torch.get_rng_state())
    assert saved.sam.steps == 2


@pytest.mark.parametrize(
    "missing", ["optimizer", "scheduler", "sam", "rng", "python_rng", "numpy_rng", "loader_rng"]
)
def test_import_rejects_incomplete_resume_state(saved, missing):
    saved.state.pop(missing)
    torch.save(saved.state, saved.path)
    with pytest.raises(ValueError, match="missing complete resumable"):
        imported(saved)


def test_import_rejects_architecture_and_head_order_changes(saved):
    saved.mc = {**saved.mc, "latent_dim": 16}
    with pytest.raises(ValueError, match="architecture"):
        imported(saved)


def test_import_rejects_training_schedule_changes(saved):
    saved.experiment["training"]["learning_rate"] = 1e-4
    with pytest.raises(ValueError, match="learning_rate"):
        imported(saved)


def test_import_rejects_changed_teacher(saved):
    saved.registry.specs["ftp_unit"].teacher = {
        **saved.registry.specs["ftp_unit"].teacher,
        "sha256": "0" * 64,
    }
    with pytest.raises(ValueError, match="teacher metadata changed"):
        imported(saved)


def test_import_rejects_changed_active_code_index(saved):
    saved.registry.caches["ftp_unit"] = "0" * 64
    with pytest.raises(ValueError, match="active acquisition code changed"):
        imported(saved)


def test_import_rejects_changed_archived_producer(saved):
    (saved.registry.paths.archived_producers / "producer.py").write_text(
        "raise RuntimeError('never run')\n"
    )
    with pytest.raises(ValueError, match="producer checksum"):
        imported(saved)


def test_import_rejects_changed_mirror_schedule(saved):
    saved.experiment["datasets"]["mirror"]["probability"] = 0.25
    with pytest.raises(ValueError, match="mirroring schedule"):
        imported(saved)
