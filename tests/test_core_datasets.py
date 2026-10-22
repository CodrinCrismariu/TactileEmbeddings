"""Focused contracts, artifact rejection and source-extension tests."""

from collections import OrderedDict
from dataclasses import replace

import numpy as np
import pytest
import torch

from tactilelatent.contracts import ClipSample, SensorTarget, SourceSpec, Supervision, VisualClip
from tactilelatent.dataset_management.artifacts import (
    ArtifactPaths,
    atomic_json,
    file_hash,
    json_hash,
)
from tactilelatent.dataset_management.cache import SensorCodeStore
from tactilelatent.dataset_management.joint import JointDataset, collate_samples
from tactilelatent.dataset_management.registry import SourceRegistry
from tactilelatent.dataset_management.sampling import SourceBatchSampler
from tactilelatent.datasets.sources.definitions import (
    SourceDefinition,
    register_source_definition,
    source_definition,
)
from tactilelatent.datasets.sources.mirroring import mirror_rgb, transform_targets


def sample(source="sensor", markers=0, frame=14):
    visual = VisualClip(
        torch.arange(8 * 3 * 3 * 4 * 4).reshape(8, 3, 3, 4, 4).float(),
        torch.ones(8, 3, dtype=torch.bool),
        torch.arange(0, 15, 2),
        torch.arange(8)[:, None].expand(8, 3).clone(),
        torch.arange(3),
    )
    if markers:
        geometry = {
            "reference": torch.arange(markers * 2).reshape(markers, 2).float() / 100,
            "surface": torch.arange(markers) % 2,
            "spacing": torch.full((markers,), 10.0),
            "count": markers,
            "point_mask": torch.ones(markers, dtype=torch.bool),
        }
        target = SensorTarget(
            torch.ones(1, markers, 2),
            torch.ones(1, markers, dtype=torch.bool),
            "marker_displacement",
            geometry,
            "marker_spacing",
        )
    else:
        target = SensorTarget(
            torch.tensor([[0.0, 2.0, 3.0]]), torch.tensor([[True, False, True]]), "native_numeric"
        )
    return ClipSample(
        visual,
        target,
        Supervision(torch.ones(1, 2), torch.ones(1, dtype=torch.bool), "code"),
        source,
        "episode",
        "train",
        torch.tensor([frame]),
        torch.tensor([frame]),
        {},
    )


def test_collate_preserves_native_shapes_missing_zero_and_camera_ids():
    batch = collate_samples([sample(), sample()])
    assert batch.target.values.shape == (2, 1, 3)
    assert batch.target.values[0, 0, 0] == 0 and batch.target.valid[0, 0, 0]
    assert not batch.target.valid[0, 0, 1]
    assert batch.visual.camera_roles.shape == (2, 3)
    with pytest.raises(ValueError):
        collate_samples([sample("a"), sample("b")])


def test_marker_padding_never_becomes_measured_zero():
    batch = collate_samples([sample(markers=3), sample(markers=5)])
    assert batch.target.values.shape == (2, 1, 5, 2)
    assert batch.target.geometry["count"].tolist() == [3, 5]
    assert not batch.target.valid[0, 0, 3:].any()
    assert not batch.target.geometry["point_mask"][0, 3:].any()
    assert batch.target.geometry["spacing"][0, 3:].tolist() == [1.0, 1.0]


def code_fixture(tmp_path, paired=False):
    paths = ArtifactPaths(tmp_path, metadata_root=tmp_path)
    row = {"source": "sensor", "episode_id": "ep", "split": "train", "frames": 3}
    teacher_path = tmp_path / "teacher.pt"
    teacher_path.write_bytes(b"fixed-teacher")
    teacher = {"checkpoint": str(teacher_path), "sha256": file_hash(teacher_path), "code_dim": 2}
    archive = tmp_path / "codes.npz"
    values = np.arange(12 if paired else 6, dtype=np.float32).reshape(
        (2, 3, 2) if paired else (3, 2)
    )
    valid = np.ones((2, 3) if paired else 3, dtype=bool)
    np.savez(archive, codes=values, valid=valid, frame_indices=np.arange(3))
    index = {
        "complete": True,
        "signature": {"teacher_sha256": teacher["sha256"]},
        "statistics": {"mean": [1.0, 2.0], "std": [2.0, 4.0], "fit_split": "train"},
        "episodes": [
            dict(
                episode_id="ep",
                split="train",
                frames=3,
                row_sha256=json_hash(row),
                path=str(archive),
                sha256=file_hash(archive),
            )
        ],
    }
    if paired:
        index["signature"].update(variants=["original", "mirrored"], policy_sha256="policy")
    path = tmp_path / "index.json"
    atomic_json(path, index)
    return paths, row, teacher, index, path


def test_codes_preserve_row_identity_and_training_statistics(tmp_path):
    paths, row, teacher, index, path = code_fixture(tmp_path)
    cache = SensorCodeStore(path, paths, teacher)
    codes, valid = cache.read(row, [1])
    torch.testing.assert_close(codes, torch.tensor([[0.5, 0.25]]))
    assert valid.tolist() == [True]
    changed = {**row, "frames": 4}
    with pytest.raises(ValueError, match="contract changed"):
        cache.read(changed, [1])
    with pytest.raises(ValueError):
        cache.read(row, [3])


def test_paired_codes_are_fresh_variants_and_stale_policies_rejected(tmp_path):
    paths, row, teacher, index, path = code_fixture(tmp_path, True)
    cache = SensorCodeStore(path, paths, teacher, "policy")
    original, _ = cache.read(row, [0])
    mirrored, _ = cache.read(row, [0], mirrored=True)
    assert not torch.equal(original, mirrored)
    with pytest.raises(ValueError, match="policy changed"):
        SensorCodeStore(path, paths, teacher, "other")
    (tmp_path / "teacher.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        SensorCodeStore(path, paths, teacher, "policy")


def test_explicit_import_receipt_pins_old_and_new_implementations(tmp_path):
    paths, row, teacher, index, path = code_fixture(tmp_path)
    consumer = tmp_path / "consumer.py"
    consumer.write_text("version=1\n")
    cache = SensorCodeStore(path, paths, teacher)
    old = path.read_bytes()
    sidecar = tmp_path / "new_import.json"
    cache.write_import(sidecar, [consumer])
    cache.require_import(sidecar, [consumer])
    assert path.read_bytes() == old
    consumer.write_text("version=2\n")
    with pytest.raises(ValueError, match="new consumer"):
        cache.require_import(sidecar, [consumer])
    with pytest.raises(ValueError, match="new sidecar"):
        cache.write_import(sidecar, [consumer])


def test_native_masks_and_grid_mirror_are_involutive():
    spec = {"kind": "grid", "hand_permutation": [1, 0], "flip_width": True}
    target = torch.arange(2 * 3 * 4).reshape(1, 2, 3, 4).float()
    mask = target % 2 == 0
    first = transform_targets({"target": target, "target_mask": mask}, spec)
    second = transform_targets(first, spec)
    torch.testing.assert_close(second["target"], target)
    assert torch.equal(second["target_mask"], mask)
    x = sample().visual
    y, m = mirror_rgb(x.rgb, x.view_mask)
    z, n = mirror_rgb(y, m)
    assert torch.equal(z, x.rgb) and torch.equal(n, x.view_mask)


def test_registering_source_adds_head_without_model_or_trainer_changes(tmp_path):
    definition = SourceDefinition("fake_extra", "native_numeric", native_shape=(3,))
    register_source_definition(definition, replace=True)
    assert source_definition("fake_extra") is definition
    old = SourceSpec("old", "fake", "native_numeric", (2,), supervision_kind="direct")
    new = SourceSpec("fake_extra", "fake", "native_numeric", (3,), supervision_kind="direct")

    class Adapter:
        def __init__(self, spec):
            self.spec = spec

        def eligible_ends(self, row):
            return np.arange(14, row["frames"], 15)

        def read_clip(self, row, end, include_targets=True):
            item = sample(self.spec.source, frame=end)
            values = torch.full((1, *self.spec.native_shape), 0.25)
            valid = torch.ones_like(values, dtype=torch.bool)
            return replace(
                item,
                target=SensorTarget(values, valid, "native_numeric"),
                supervision=Supervision(values, valid, "direct"),
            )

    adapters = [Adapter(old), Adapter(new)]
    rows = [
        ("old", {"episode_id": "a", "split": "train", "frames": 30}),
        ("fake_extra", {"episode_id": "b", "split": "train", "frames": 30}),
    ]
    registry = SourceRegistry(
        adapters,
        rows,
        {"sources": ["old", "fake_extra"], "heldout_source": "holdout"},
        ArtifactPaths(tmp_path),
        {"numeric_shapes": {"old": [2]}, "sensor_code_dims": {}, "statistics": {}},
        {},
    )
    assert registry.model_config()["numeric_shapes"] == {"old": [2], "fake_extra": [3]}
    dataset = JointDataset(registry)
    assert list(dataset.indices_by_source) == ["old", "fake_extra"]
    assert [
        batch.source for batch in [collate_samples([dataset[0]]), collate_samples([dataset[2]])]
    ] == ["old", "fake_extra"]
    from tactilelatent.models import JointModel
    from tactilelatent.training.objectives import forward_batch, supervision_loss

    model = JointModel(
        **registry.model_config(
            {
                "width": 32,
                "heads": 4,
                "layers": 1,
                "feedforward": 64,
                "dropout": 0.0,
                "grid_size": 2,
            }
        )
    )
    batch = collate_samples([dataset[2]])
    features = torch.randn(1, 4, 3, 4, 768)
    prediction = forward_batch(model, features, batch)
    assert prediction["pred"].shape == (1, 1, 3)
    loss = supervision_loss(prediction, batch)
    loss.backward()
    assert torch.isfinite(loss) and model.visual.bottleneck[-1].weight.grad is not None


def test_cycling_sampler_covers_before_repeating_and_resumes_exactly():
    class Data:
        indices_by_source = OrderedDict([("large", list(range(11))), ("small", [11, 12])])

    data = Data()
    sampler = SourceBatchSampler(data, 2, seed=45, max_windows_per_source=4, sampling_mode="cycle")
    seen = []
    for epoch in range(3):
        sampler.set_epoch(epoch)
        batches = list(sampler)
        seen.extend(i for b in batches for i in b if i < 11)
        other = SourceBatchSampler(
            data, 2, seed=45, max_windows_per_source=4, sampling_mode="cycle"
        )
        other.set_epoch(epoch)
        assert batches == list(other)
    assert len(set(seen)) == 11


def test_full_verification_rechecks_payload_even_when_already_loaded(tmp_path):
    from tactilelatent.dataset_management.api import verify
    from tactilelatent.dataset_management.registry import SourceAdapter

    paths, row, teacher, index, path = code_fixture(tmp_path)
    cache = SensorCodeStore(path, paths, teacher)
    spec = SourceSpec(
        "sensor",
        "synthetic",
        "native_numeric",
        (2,),
        code_dim=2,
        teacher=teacher,
        statistics=cache.statistics,
    )
    adapter = SourceAdapter(spec, paths, [row], {}, tmp_path, cache)
    registry = SourceRegistry(
        [adapter], [("sensor", row)], {"sources": ["sensor"]}, paths, {}, {"artifacts": {}}
    )
    report = verify(registry, full=True)
    assert report["sources"]["sensor"]["code_payload_verification"]["payload_files"] == 1
    assert report["verification_levels"]["rgb"] == "not decoded"
    assert not report["sources"]["sensor"]["code_payload_verification"][
        "native_target_quality_evaluated"
    ]
    np.savez(
        tmp_path / "codes.npz",
        codes=np.ones((3, 2), np.float32),
        valid=np.ones(3, bool),
        frame_indices=np.arange(3),
    )
    with pytest.raises(ValueError, match="checksum"):
        verify(registry, full=True)


def test_pinned_download_overrides_validate_without_network(tmp_path):
    from tactilelatent.dataset_management.providers import download_plan, resolve_download_plan

    values = dict(
        url="https://example.invalid/native.npz",
        sha256="a" * 64,
        destination=tmp_path / "datasets/new/native.npz",
    )
    plan = resolve_download_plan(None, tmp_path, **values)
    assert plan["provider"] == "https_file"
    assert resolve_download_plan(plan, tmp_path, **values) == plan
    with pytest.raises(ValueError, match="together"):
        resolve_download_plan(None, tmp_path, url=values["url"])
    with pytest.raises(ValueError, match="contradict"):
        resolve_download_plan(plan, tmp_path, **{**values, "sha256": "b" * 64})
    with pytest.raises(ValueError, match="Revision"):
        resolve_download_plan(plan, tmp_path, revision="abc")
    with pytest.raises(ValueError, match="stay inside"):
        download_plan({**plan, "destination": str(tmp_path / "outside.npz")}, tmp_path)
    with pytest.raises(PermissionError):
        download_plan(plan, tmp_path)


def test_arbitrary_verified_policy_does_not_require_a_fixed_source_count(tmp_path):
    from tactilelatent.datasets.sources.mirroring import load_policy

    path = tmp_path / "policy.json"
    atomic_json(
        path,
        {
            "schema_version": 1,
            "sources": {
                "new_bilateral": {
                    "status": "verified",
                    "kind": "scalar",
                    "hand_permutation": [1, 0],
                    "evidence": ["recorded-geometry.json"],
                }
            },
        },
    )
    assert list(load_policy(path)["sources"]) == ["new_bilateral"]


def test_new_source_manifest_builds_full_registry_and_objective_without_old_head_maps(tmp_path):
    from tactilelatent.dataset_management.registry import SourceAdapter, build_sources
    from tactilelatent.models import JointModel
    from tactilelatent.training.objectives import forward_batch, supervision_loss

    class AddedReader(SourceAdapter):
        def read_clip(self, row, end, include_targets=True):
            item = sample(self.spec.source, frame=end)
            value = torch.full((1, 3), 0.25)
            mask = torch.ones_like(value, dtype=torch.bool)
            return replace(
                item,
                target=SensorTarget(value, mask, "native_numeric"),
                supervision=Supervision(value, mask, "direct"),
            )

    register_source_definition(
        SourceDefinition(
            "synthetic_new", "native_numeric", native_shape=(3,), adapter_factory=AddedReader
        ),
        replace=True,
    )
    main = tmp_path / "manifest.json"
    atomic_json(
        main,
        {
            "schema_version": 2,
            "frame_contract": "native-chronological-v1",
            "episodes": [
                {
                    "source": "ftp_motiontrans",
                    "episode_id": "holdout",
                    "split": "train",
                    "frames": 30,
                    "target_shape": [6, 1, 1],
                    "views": {},
                }
            ],
        },
    )
    extra = tmp_path / "new_source.json"
    atomic_json(
        extra,
        {
            "schema_version": 2,
            "frame_contract": "native-chronological-v1",
            "episodes": [
                {
                    "source": "synthetic_new",
                    "episode_id": "episode",
                    "split": "train",
                    "frames": 30,
                    "target_shape": [3],
                    "views": {},
                }
            ],
        },
    )
    for name, payload in [
        ("model.json", {}),
        ("teachers.json", {"sources": {}}),
        ("normalization.json", {"data": {}}),
        ("expansion.json", {"ready": True, "manifest_sha256": file_hash(main)}),
        ("acquisition.json", {"ready": True, "sources": {}}),
    ]:
        atomic_json(tmp_path / name, payload)
    cfg = {
        "sources": ["synthetic_new"],
        "heldout_source": "ftp_motiontrans",
        "mirror": {"enabled": False},
        "registry_paths": {
            "manifest": str(main),
            "model_config": str(tmp_path / "model.json"),
            "teachers": str(tmp_path / "teachers.json"),
            "storage_normalization": str(tmp_path / "normalization.json"),
            "expansion_ready": str(tmp_path / "expansion.json"),
            "acquisition_ready": str(tmp_path / "acquisition.json"),
            "marker_targets": str(tmp_path / "unused"),
        },
        "prepared_sources": {
            "synthetic_new": {"manifest": str(extra), "manifest_sha256": file_hash(extra)}
        },
    }
    registry = build_sources(cfg, tmp_path)
    assert registry.require_imports()
    dataset = JointDataset(registry)
    batch = collate_samples([dataset[0]])
    mc = registry.model_config(
        {"width": 32, "heads": 4, "layers": 1, "feedforward": 64, "dropout": 0.0, "grid_size": 2}
    )
    assert mc["numeric_shapes"] == {"synthetic_new": [3]}
    model = JointModel(**mc)
    prediction = forward_batch(model, torch.randn(1, 4, 3, 4, 768), batch)
    objective = supervision_loss(prediction, batch)
    objective.backward()
    assert torch.isfinite(objective) and model.visual.bottleneck[-1].weight.grad is not None


def test_frozen_code_normalization_cannot_be_replaced_silently(tmp_path):
    paths, row, teacher, index, path = code_fixture(tmp_path)
    with pytest.raises(ValueError, match="statistics differ"):
        SensorCodeStore(path, paths, teacher, statistics={"mean": [0.0, 0.0], "std": [1.0, 1.0]})
