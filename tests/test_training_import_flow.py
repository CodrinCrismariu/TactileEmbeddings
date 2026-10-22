"""Verify one-time checkpoint import publication through the maintained engine.

The fixtures replace only external dataset/backbone/importer boundaries. The
actual engine, AdamW scheduler, file locking, checkpoint writer and status/metrics
publication execute without GPUs or production artifacts.
"""

import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tactilelatent.training import engine


@pytest.fixture
def import_flow(tmp_path, monkeypatch):
    import tactilelatent.dataset_management as management
    import tactilelatent.dataset_management.artifacts as artifacts
    import tactilelatent.models as models
    import tactilelatent.models.backbone as backbone_module
    import tactilelatent.training.legacy_import as legacy_import

    project = Path(__file__).resolve().parents[1]
    cfg = {
        name: json.loads((project / "configs" / f"{name}.json").read_text())
        for name in ("model", "datasets", "training", "transfer")
    }
    cfg["training"].update(output="continued", device="cpu", workers=0)
    cfg["provenance"] = {"fixture": "explicit import-flow configuration"}
    parent = tmp_path / "parent"
    parent.mkdir()
    source = parent / "last.pt"
    source.write_bytes(b"preserved parent bytes")
    metrics = "".join(
        json.dumps({"epoch": epoch, "selection_score": 1 / epoch}) + "\n" for epoch in range(1, 5)
    )
    (parent / "metrics.jsonl").write_text(metrics)
    observed = dict(imports=[], require_imports=0, storage_checks=0)

    class Dataset:
        indices_by_source = {"fixture": [0, 1, 2, 3]}

        def __len__(self):
            return 4

        def __getitem__(self, index):
            pytest.fail("Import-only mode must not read training examples")

    class Task(torch.nn.Module):
        model_type = "fixture-import-model"

        def __init__(self, **kwargs):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.full((2, 2), 0.125))
            self.visual = SimpleNamespace(latent_dim=32)

    class Backbone(torch.nn.Module):
        provenance = {"fixture": "frozen backbone identity"}

        def __init__(self, *args):
            super().__init__()

    def require_imports():
        observed["require_imports"] += 1

    registry = SimpleNamespace(
        provenance={"fixture": "dataset identity"},
        specs={"fixture": SimpleNamespace(teacher={})},
        model_config=lambda requested: {"fixture": "model config"},
        require_imports=require_imports,
    )

    def storage_check(root):
        assert root == tmp_path
        observed["storage_checks"] += 1

    def import_checkpoint(path, **kwargs):
        observed["imports"].append((path, kwargs))
        model, optimizer, scheduler = (
            kwargs["model"],
            kwargs["optimizer"],
            kwargs["scheduler"],
        )
        for _ in range(4):
            model.weight.sum().backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
        kwargs["sam"].steps = 4
        state = dict(
            schema_version=2,
            model_type=model.model_type,
            model_config=kwargs["model_config"],
            model=model.state_dict(),
            config=kwargs["experiment"],
            provenance=kwargs["provenance"],
            epoch=3,
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),
            sam=kwargs["sam"].state_dict(),
            rng=engine.rng_snapshot(),
            python_rng=random.getstate(),
            numpy_rng=np.random.get_state(),
            loader_rng=kwargs["generator"].get_state(),
            best_score=0.25,
            validation={"fixture": {"supervision_loss": 0.25}},
            elapsed_seconds=1234.5,
        )
        receipt = {"completed_epochs": 4, "optimizer_reinitialized": False}
        observed["state"], observed["receipt"] = state, receipt
        return state, receipt

    def unexpected(*args, **kwargs):
        pytest.fail("Import-only mode must not execute training or validation")

    monkeypatch.setattr(engine, "load_experiment", lambda *args, **kwargs: cfg)
    monkeypatch.setattr(management, "build_sources", lambda *args: registry)
    monkeypatch.setattr(management, "build_dataset", lambda *args: Dataset())
    monkeypatch.setattr(models, "JointModel", Task)
    monkeypatch.setattr(backbone_module, "VJEPABackbone", Backbone)
    monkeypatch.setattr(
        artifacts, "scientific_implementation_signature", lambda root: {"fixture": "code identity"}
    )
    monkeypatch.setattr(legacy_import, "import_legacy_checkpoint", import_checkpoint)
    monkeypatch.setattr(engine, "check_storage", storage_check)
    monkeypatch.setattr(engine, "train_epoch", unexpected)
    monkeypatch.setattr(engine, "validate", unexpected)
    monkeypatch.setattr(engine, "paired_validate", unexpected)
    monkeypatch.setattr(engine, "preflight", unexpected)
    return SimpleNamespace(
        root=tmp_path, cfg=cfg, source=source, metrics=metrics, observed=observed
    )


def test_import_only_publishes_complete_state_and_parent_history_without_training(import_flow):
    context = import_flow
    result = engine.train_experiment(
        {},
        project_root=context.root,
        import_checkpoint=context.source,
        import_only=True,
    )
    output = context.root / "continued"
    assert result == dict(output=str(output), completed_epochs=4, next_epoch=5, stage="imported")
    assert context.observed["require_imports"] == 1
    assert context.observed["storage_checks"] == 1
    assert len(context.observed["imports"]) == 1
    imported_path, arguments = context.observed["imports"][0]
    assert imported_path == context.source and arguments["batches_per_epoch"] == 2
    state = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
    anchor = torch.load(output / "imported_checkpoint.pt", map_location="cpu", weights_only=False)
    assert anchor["epoch"] == state["epoch"]
    assert anchor["schema_version"] == state["schema_version"]
    assert anchor["scheduler"] == state["scheduler"]
    assert state["epoch"] == 3 and state["elapsed_seconds"] == 1234.5
    assert state["optimizer"]["state"]
    assert state["sam"] == {"rho": 0.05, "steps": 4}
    assert state["scheduler"] == context.observed["state"]["scheduler"]
    assert torch.equal(state["loader_rng"], context.observed["state"]["loader_rng"])
    for name, value in state["model"].items():
        assert torch.equal(value, context.observed["state"]["model"][name])
        assert torch.equal(value, anchor["model"][name])
    snapshot = torch.load(output / "viewer_epochs/epoch_004.pt", weights_only=False)
    assert snapshot["inference_only"] and snapshot["epoch"] == 3
    assert "optimizer" not in snapshot
    assert not (output / "viewer_epochs/epoch_000.pt").exists()
    assert (output / "metrics.jsonl").read_text() == context.metrics
    assert (
        json.loads((output / "checkpoint_import.json").read_text()) == context.observed["receipt"]
    )
    status = json.loads((output / "status.json").read_text())
    assert status["stage"] == "imported" and status["pid"] is None
    assert status["completed_epochs"] == 4 and status["epoch"] == 5
    assert status["total_epochs"] == 30
    assert context.source.read_bytes() == b"preserved parent bytes"


@pytest.mark.parametrize("existing_name", ["last.pt", "metrics.jsonl", "unrelated.txt"])
def test_import_rejects_nonempty_destination_without_overwriting(import_flow, existing_name):
    context = import_flow
    output = context.root / "continued"
    output.mkdir()
    protected = output / existing_name
    protected.write_bytes(b"protected destination")
    with pytest.raises(FileExistsError):
        engine.train_experiment(
            {}, project_root=context.root, import_checkpoint=context.source, import_only=True
        )
    assert protected.read_bytes() == b"protected destination"
    assert not context.observed["imports"]
    assert not (output / "checkpoint_import.json").exists()


def test_import_rejects_gapped_parent_history_before_checkpoint_publication(import_flow):
    context = import_flow
    (context.source.parent / "metrics.jsonl").write_text(
        "".join(json.dumps({"epoch": epoch}) + "\n" for epoch in (1, 2, 4))
    )
    with pytest.raises(ValueError, match="Parent metrics"):
        engine.train_experiment(
            {}, project_root=context.root, import_checkpoint=context.source, import_only=True
        )
    output = context.root / "continued"
    assert not (output / "last.pt").exists()
    assert not (output / "imported_checkpoint.pt").exists()


@pytest.mark.parametrize(
    "options",
    [
        {"import_only": True},
        {"import_checkpoint": "last.pt", "resume": "run"},
        {"import_checkpoint": "last.pt", "preflight_only": True},
        {"import_checkpoint": "last.pt", "import_only": True, "preflight_only": True},
    ],
)
def test_import_modes_are_mutually_exclusive_before_loading_configuration(tmp_path, options):
    with pytest.raises(ValueError, match="Import-only|Checkpoint import"):
        engine.train_experiment("missing-experiment.json", project_root=tmp_path, **options)
