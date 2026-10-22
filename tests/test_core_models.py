"""CPU-only scientific parity checks for actual extracted implementations."""

import hashlib
import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from tactilelatent.models import JointModel, SensorDecoder
from tactilelatent.models.autoencoders import load_teacher, make_autoencoder

ROOT = Path(__file__).resolve().parents[1]


def baseline_module(name):
    """Import immutable reference code only inside tests, under another name."""
    namespace = "_model_parity_reference"
    if namespace not in sys.modules:
        module = ModuleType(namespace)
        module.__path__ = [str(ROOT / "legacy/v1/src/tactilelatent")]
        sys.modules[namespace] = module
        # This archive also preserves the first organization attempt, where
        # losses.py became an alias. Load its actual archived implementation
        # privately rather than resolving that alias into the current package.
        actual = ROOT / "legacy/v1/src/tactilelatent/training/losses.py"
        if actual.is_file():
            spec = importlib.util.spec_from_file_location(namespace + ".losses", actual)
            losses = importlib.util.module_from_spec(spec)
            sys.modules[namespace + ".losses"] = losses
            spec.loader.exec_module(losses)
    return importlib.import_module(namespace + "." + name)


def config():
    return dict(
        sensor_code_dims={"ftp_unit": 3, "another_sensor": 4},
        numeric_shapes={"force": [1, 1, 1], "direct": [4]},
        statistics={
            "ftp_unit": dict(mean=[0.1] * 3, std=[0.8] * 3),
            "another_sensor": dict(mean=[0.0] * 4, std=[1.0] * 4),
        },
        direct_numeric={"direct": {"scale": [1.0, 2.0, 3.0, 4.0]}},
        input_dim=12,
        latent_dim=32,
        adapter_hidden_dims=[16, 16],
        width=16,
        heads=4,
        layers=2,
        feedforward=32,
        dropout=0.1,
        grid_size=2,
        prediction_mode="current_frame",
        latent_mode="deterministic",
    )


def inputs():
    generator = torch.Generator().manual_seed(17)
    features = torch.randn(2, 4, 3, 4, 12, generator=generator)
    mask = torch.ones(2, 8, 3, dtype=torch.bool)
    mask[:, :, 2] = False
    return features, mask


def assert_same_tensors(a, b):
    assert list(a) == list(b)
    for key in a:
        if torch.is_tensor(a[key]):
            torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)
        else:
            assert a[key] == b[key]


def test_seed45_initialization_parameter_order_and_rng_match_archived_baseline():
    torch.set_num_threads(1)
    original = baseline_module("acquisition_joint").AllSourceVJEPA
    torch.manual_seed(45)
    reference = original(**config())
    reference_rng = torch.get_rng_state().clone()
    torch.manual_seed(45)
    extracted = JointModel(**config())
    assert_same_tensors(reference.state_dict(), extracted.state_dict())
    assert list(dict(reference.named_parameters())) == list(dict(extracted.named_parameters()))
    assert torch.equal(reference_rng, torch.get_rng_state())


def test_deterministic_forward_and_all_head_kinds_match_baseline():
    original = baseline_module("acquisition_joint").AllSourceVJEPA
    torch.manual_seed(45)
    reference = original(**config()).eval()
    torch.manual_seed(45)
    extracted = JointModel(**config()).eval()
    features, mask = inputs()
    for source in ("ftp_unit", "another_sensor", "force", "direct"):
        assert_same_tensors(reference(features, mask, source), extracted(features, mask, source))
    point = torch.tensor([[[-2.0, 0.0, 0.5, 2.0]]])
    torch.testing.assert_close(
        reference.normalize_numeric("direct", point),
        extracted.normalize_numeric("direct", point),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        reference.decode_numeric("direct", point),
        extracted.decode_numeric("direct", point),
        atol=0,
        rtol=0,
    )


def sam_update(model, features, mask):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.05)
    target = torch.full((2, 1, 3), 0.2)
    torch.manual_seed(91)
    before = torch.get_rng_state()
    prediction = model(features, mask, "ftp_unit")["sensor_code"]
    first_loss = (prediction - target).square().mean()
    first_loss.backward()
    after = torch.get_rng_state()
    parameters = [p for p in model.parameters() if p.grad is not None]
    norm = torch.linalg.vector_norm(
        torch.stack([torch.linalg.vector_norm(p.grad.float()) for p in parameters])
    )
    originals = [p.detach().clone() for p in parameters]
    with torch.no_grad():
        for p in parameters:
            p.add_(p.grad * (0.05 / (norm + 1e-12)))
    optimizer.zero_grad(set_to_none=True)
    torch.set_rng_state(before)
    replay = model(features, mask, "ftp_unit")["sensor_code"]
    second_loss = (replay - target).square().mean()
    second_loss.backward()
    with torch.no_grad():
        for p, original in zip(parameters, originals):
            p.copy_(original)
    torch.set_rng_state(after)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    return first_loss.detach(), second_loss.detach(), optimizer.state_dict(), torch.get_rng_state()


def test_one_sam_adamw_update_and_dropout_replay_match_baseline():
    original = baseline_module("acquisition_joint").AllSourceVJEPA
    torch.manual_seed(45)
    reference = original(**config())
    torch.manual_seed(45)
    extracted = JointModel(**config())
    features, mask = inputs()
    a = sam_update(reference, features, mask)
    b = sam_update(extracted, features, mask)
    assert_same_tensors(reference.state_dict(), extracted.state_dict())
    for left, right in zip(a[:2], b[:2]):
        assert torch.equal(left, right)
    assert a[2]["param_groups"] == b[2]["param_groups"]
    for key in a[2]["state"]:
        assert_same_tensors(a[2]["state"][key], b[2]["state"][key])
    assert torch.equal(a[3], b[3])


def test_source_independent_latent_masks_and_new_source_head():
    settings = config()
    settings["sensor_code_dims"]["new_eighteenth_source"] = 7
    settings["statistics"]["new_eighteenth_source"] = dict(mean=[0.0] * 7, std=[1.0] * 7)
    model = JointModel(**settings).eval()
    features, mask = inputs()
    first = model(features, mask, "ftp_unit")
    extra = model(features, mask, "new_eighteenth_source")
    assert torch.equal(first["mu"], extra["mu"])
    assert extra["sensor_code"].shape == (2, 1, 7)
    changed = features.clone()
    changed[:, :, 2] = float("nan")
    assert torch.equal(first["mu"], model.encode(changed, mask)[0])
    with pytest.raises(ValueError, match="available camera"):
        model.encode(features, torch.zeros_like(mask))
    inconsistent = mask.clone()
    inconsistent[:, 0, 0] = False
    with pytest.raises(ValueError, match="constant"):
        model.encode(features, inconsistent)


AE_CASES = [
    (None, dict(architecture="conv", latent_dim=2, channels=[4], hidden_dims=[8])),
    (
        None,
        dict(
            target_shape=[1, 8, 8], architecture="conv", latent_dim=2, channels=[4], hidden_dims=[8]
        ),
    ),
    (
        "expansion_pressure_autoencoder_v2",
        dict(
            target_shape=[1, 8, 8],
            architecture="conv",
            latent_dim=2,
            channels=[4],
            hidden_dims=[8],
            input_mean=0.1,
            input_scale=0.2,
        ),
    ),
    (
        "native-numeric-ae-v1",
        dict(
            input_dim=4,
            latent_dim=2,
            hidden_dims=[8],
            scale=[1.0] * 4,
            nonnegative=[True, False, True, False],
        ),
    ),
    (
        "marker_autoencoder_v1",
        dict(latent_dim=4, width=8, queries=2, heads=2, displacement_scale=0.1),
    ),
    (
        "expansion_marker_autoencoder_v2",
        dict(
            latent_dim=4,
            width=8,
            queries=2,
            heads=2,
            displacement_scale=0.1,
            surfaces=4,
            code_normalization="pre_gelu_layer_norm",
        ),
    ),
    (
        "bilateral_vector_autoencoder_v1",
        dict(target_shape=[2, 1, 80], latent_dim=4, hidden_dims=[8]),
    ),
    (
        "bilateral_marker_autoencoder_v1",
        dict(latent_dim=4, width=8, queries=2, heads=2, displacement_scale=0.1),
    ),
]


@pytest.mark.parametrize("kind,settings", AE_CASES)
def test_teacher_formats_strict_load_frozen_weights_and_bad_hash_rejection(
    kind, settings, tmp_path
):
    teacher = make_autoencoder(settings, kind)
    path = tmp_path / "teacher.pt"
    torch.save(dict(model_type=kind, model_config=teacher.config, model=teacher.state_dict()), path)
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    restored, state = load_teacher(path, expected_sha256=expected)
    assert_same_tensors(teacher.state_dict(), restored.state_dict())
    assert not restored.training and all(not p.requires_grad for p in restored.parameters())
    with pytest.raises(ValueError, match="hash mismatch"):
        load_teacher(path, expected_sha256="0" * 64)


def test_marker_decoder_handles_batched_queries_and_unequal_surface_counts(tmp_path):
    kind, settings = AE_CASES[-1]
    model = make_autoencoder(settings, kind)
    path = tmp_path / "teacher.pt"
    torch.save(dict(model_type=kind, model_config=model.config, model=model.state_dict()), path)
    spec = SimpleNamespace(
        target_kind="marker_displacement",
        teacher=dict(checkpoint=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()),
    )
    decoder = SensorDecoder(spec)
    geometry = dict(
        reference=torch.rand(2, 7, 2), surface=torch.tensor([[0, 0, 0, 1, 1, 1, 1]] * 2)
    )
    code = torch.randn(2, 1, 4)
    prediction = decoder.decode(code, geometry)
    assert prediction.shape == (2, 1, 7, 2)
    expected = decoder.model.decode(code[:, 0], geometry["reference"], geometry["surface"])
    torch.testing.assert_close(prediction[:, 0], expected, atol=0, rtol=0)


def test_model_runtime_has_no_archive_or_dataset_imports():
    import ast

    folder = ROOT / "src/tactilelatent/models"
    for path in folder.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert not any("legacy" in alias.name for alias in node.names), path
            elif isinstance(node, ast.ImportFrom):
                assert "legacy" not in (node.module or ""), path
                assert "dataset" not in (node.module or ""), path
