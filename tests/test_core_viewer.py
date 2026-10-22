"""Local HTTP and attention checks: no GPU, downloads, or training processes."""

import json
from contextlib import contextmanager
from http.server import ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from threading import Thread
from urllib.error import HTTPError
from urllib.parse import urljoin
from urllib.request import HTTPRedirectHandler, build_opener, urlopen

import pytest
import torch
from PIL import Image

from tactilelatent.contracts import SensorTarget, SourceSpec
from tactilelatent.models import JointModel
from tactilelatent.visualization.attention import AttentionCapture
from tactilelatent.visualization.catalog import ModelCatalog
from tactilelatent.visualization.legacy import LegacyViewer
from tactilelatent.visualization.rendering import target_payload
from tactilelatent.visualization.server import make_handler
from tactilelatent.visualization.state import ViewerState
from tactilelatent.visualization.training_dashboard import TrainingDashboard


class FakeState:
    def __init__(self):
        self.calls = []

    def catalog(self, refresh=False):
        return dict(
            sources=[dict(id="fixture", kind="pressure_grid", mirrored=True)],
            models=[],
            refresh=refresh,
        )

    def episodes(self, source, split, text=""):
        if source != "fixture":
            raise ValueError("Unknown source")
        return [dict(id="episode", split=split, frames=20)]

    def frame(self, source, episode, frame, **options):
        self.calls.append((source, episode, frame, options))
        return dict(source=source, episode=episode, frame=frame, options=options)


class FakeLegacy:
    def __init__(self):
        self.requests = []

    def request(self, tail):
        self.requests.append(tail)
        return 200, {"Content-Type": "text/html"}, ("historical " + tail).encode()


@contextmanager
def server_for(state, legacy=None, dashboard=None):
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), make_handler(state, legacy=legacy, dashboard=dashboard)
    )
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield "http://127.0.0.1:" + str(server.server_port)
    finally:
        server.shutdown()
        server.server_close()
        if dashboard:
            dashboard.close()
        thread.join(timeout=3)


def test_current_http_assets_proxy_prefix_and_frame_options():
    state = FakeState()
    with server_for(state) as base:
        assert b"TactileLatent" in urlopen(base + "/").read()
        assert b"loadCatalog" in urlopen(base + "/proxy/8080/viewer.js").read()
        assert b"color-scheme" in urlopen(base + "/viewer.css").read()
        assert json.load(urlopen(base + "/proxy/8080/api/catalog?refresh=1"))["refresh"]
        assert (
            json.load(urlopen(base + "/api/episodes?source=fixture&split=val"))[0]["id"]
            == "episode"
        )
        value = json.load(
            urlopen(
                base
                + "/api/frame?source=fixture&episode=episode&frame=3&page=prediction&model=abc&epoch=epoch_003.pt&mirror=1&attention=1"
            )
        )
        assert (
            value["frame"] == 3 and value["options"]["mirrored"] and value["options"]["attention"]
        )
        assert state.calls[-1][-1]["epoch"] == "epoch_003.pt"
        with pytest.raises(HTTPError) as error:
            urlopen(base + "/api/episodes?source=unknown&split=val")
        assert error.value.code == 400


def test_explicit_legacy_proxy_preserves_query_static_and_native_api_routes():
    worker = FakeLegacy()
    with server_for(FakeState(), worker) as base:
        for path, tail in [
            ("/legacy/?page=joint", "?page=joint"),
            ("/proxy/8080/legacy/api/catalog", "api/catalog"),
            ("/legacy/attention_viewer.js", "attention_viewer.js"),
        ]:
            assert urlopen(base + path).read() == ("historical " + tail).encode()
        assert worker.requests == ["?page=joint", "api/catalog", "attention_viewer.js"]


def test_legacy_directory_redirect_keeps_proxy_prefix_for_archived_relative_fetches():
    class NoRedirect(HTTPRedirectHandler):
        def redirect_request(self, *args):
            return None

    worker = FakeLegacy()
    archived_html = (
        Path(__file__).parents[1] / "legacy/v1/src/tactilelatent/autoencoder_viewer.html"
    )
    # Verify the real archived UI's URL convention, rather than substituting
    # absolute paths that would hide a missing directory redirect.
    assert "new URL('api/'+path,apiBase)" in archived_html.read_text()
    with server_for(FakeState(), worker) as base:
        for prefix in ("", "/proxy/8080"):
            original = base + prefix + "/legacy?page=joint"
            with pytest.raises(HTTPError) as error:
                build_opener(NoRedirect).open(original)
            assert error.value.code == 301
            assert error.value.headers["Location"] == prefix + "/legacy/?page=joint"
            opened = urlopen(original)
            assert opened.geturl() == base + prefix + "/legacy/?page=joint"
            assert (
                urlopen(urljoin(opened.geturl(), "api/catalog")).read() == b"historical api/catalog"
            )
            assert (
                urlopen(urljoin(opened.geturl(), "attention-viewer.js")).read()
                == b"historical attention-viewer.js"
            )


def test_training_routes_use_explicit_project_and_leave_sensor_routes_intact(tmp_path):
    dashboard = TrainingDashboard(tmp_path, journal=tmp_path / "journal")
    with server_for(FakeState(), dashboard=dashboard) as base:
        assert b"Training monitor" in urlopen(base + "/?page=training").read()
        assert b"ScalarChart" in urlopen(base + "/proxy/8080/training-dashboard.js").read()
        assert "runs" in json.load(urlopen(base + "/api/training/catalog"))
        assert json.load(urlopen(base + "/api/catalog"))["sources"][0]["id"] == "fixture"


def small_config():
    return dict(
        sensor_code_dims={"sensor": 3},
        numeric_shapes={},
        statistics={"sensor": dict(mean=[0.0] * 3, std=[1.0] * 3)},
        input_dim=12,
        width=16,
        heads=4,
        layers=2,
        feedforward=32,
        grid_size=2,
        latent_dim=32,
        prediction_mode="current_frame",
        dropout=0.0,
    )


def test_attention_capture_preserves_inference_and_masks_absent_cameras():
    model = JointModel(**small_config()).eval()
    features = torch.randn(1, 4, 3, 4, 12)
    mask = torch.ones(1, 8, 3, dtype=torch.bool)
    mask[:, :, 2] = False
    reference = model(features, mask, "sensor")
    with AttentionCapture(model) as capture:
        prediction = model(features, mask, "sensor")
    assert torch.equal(reference["mu"], prediction["mu"])
    assert torch.equal(reference["sensor_code"], prediction["sensor_code"])
    weights = torch.tensor(capture.payload()["weights"])
    assert weights.shape == (2, 4, 4, 3, 2, 2)
    assert torch.count_nonzero(weights[:, :, :, 2]) == 0
    torch.testing.assert_close(weights.flatten(2).sum(-1), torch.ones(2, 4))
    assert not capture.handles


def test_checkpoint_decoder_provenance_uses_old_teacher_not_current_one(tmp_path):
    state = object.__new__(ViewerState)
    state.root = tmp_path
    spec = SourceSpec(
        "sensor",
        "fixture",
        "pressure_grid",
        (1, 8, 8),
        teacher=dict(checkpoint="latest.pt", sha256="latest"),
    )
    saved = dict(
        teachers={"sources": {"sensor": dict(checkpoint="previous.pt", sha256="previous")}}
    )
    selected = state._checkpoint_spec(spec, saved)
    assert selected.teacher["sha256"] == "previous"
    assert selected.teacher["checkpoint"] == str(tmp_path / "previous.pt")
    assert spec.teacher["sha256"] == "latest"
    with pytest.raises(ValueError, match="pinned decoder"):
        state._checkpoint_spec(spec, {})


def test_catalog_excludes_smokes_initialization_and_rejects_epoch_path_traversal(tmp_path):
    root = tmp_path / "processing/runs/current"
    epochs = root / "viewer_epochs"
    epochs.mkdir(parents=True)
    (root / "config.json").write_text(json.dumps(dict(prediction_mode="current_frame")))
    settings = small_config()
    torch.save(
        dict(model_type="joint_vjepa21_spatial_marker_codes_v1", model_config=settings, epoch=0),
        epochs / "epoch_001.pt",
    )
    torch.save(
        dict(model_type="joint_vjepa21_spatial_marker_codes_v1", model_config=settings, epoch=-1),
        epochs / "epoch_000.pt",
    )
    catalog = ModelCatalog(tmp_path)
    public = catalog.public()
    assert len(public) == 1 and public[0]["epochs"] == ["epoch_001.pt"]
    assert catalog.resolve(public[0]["id"], "epoch_001.pt")[1] == epochs / "epoch_001.pt"
    with pytest.raises(ValueError, match="available checkpoint"):
        catalog.resolve(public[0]["id"], "../../etc/passwd")


def test_native_renderer_preserves_shape_missing_masks_and_marker_pixels():
    target = SensorTarget(
        torch.tensor([[[[[0.0, 0.5], [1.0, 0.0]]]]]),
        torch.tensor([[[[[True, False], [True, True]]]]]),
        "pressure_grid",
    )
    spec = SourceSpec("sensor", "fixture", "pressure_grid", (1, 2, 2))
    payload = target_payload(target, target.values.clone(), spec)
    assert payload["shape"] == [1, 2, 2] and payload["mae"] == 0.0
    import base64

    image = Image.open(BytesIO(base64.b64decode(payload["panels"][0]["original"].split(",", 1)[1])))
    assert image.size == (2, 2) and image.getpixel((1, 0)) == (90, 95, 105)
    marker = SensorTarget(
        torch.ones(1, 1, 3, 2),
        torch.tensor([[[True, False, True]]]),
        "marker_displacement",
        dict(
            reference=torch.zeros(1, 3, 2),
            surface=torch.tensor([[0, 1, 1]]),
            spacing=torch.tensor([[2.0, 3.0, 4.0]]),
            count=torch.tensor([3]),
        ),
    )
    spec = SourceSpec(
        "markers",
        "fixture",
        "marker_displacement",
        (-1, 2),
        metadata=dict(marker_surface_shape=[240, 320]),
    )
    payload = target_payload(marker, marker.values.clone(), spec)
    assert payload["values"] == [[2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]
    assert payload["valid"] == [True, False, True] and payload["mae"] == 0.0


def test_legacy_worker_is_explicit_cpu_only_and_preserves_archived_script(tmp_path, monkeypatch):
    archive = tmp_path / "legacy/v1"
    (archive / "scripts").mkdir(parents=True)
    script = archive / "scripts/view_egotouch_autoencoder.py"
    script.write_text("# immutable archived fixture\n")
    (tmp_path / "processing").mkdir()
    calls = []

    class Process:
        def __init__(self, command, **kwargs):
            calls.append((command, kwargs))
            self.code = None

        def poll(self):
            return self.code

        def terminate(self):
            self.code = 0

        def wait(self, timeout=None):
            return self.code

    @contextmanager
    def connection(*args, **kwargs):
        yield None

    monkeypatch.setattr("tactilelatent.visualization.legacy.subprocess.Popen", Process)
    monkeypatch.setattr("tactilelatent.visualization.legacy.socket.create_connection", connection)
    worker = LegacyViewer(archive, tmp_path, log_root=tmp_path / "logs")
    assert not calls
    worker.start()
    assert len(calls) == 1 and calls[0][1]["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert calls[0][1]["env"]["PYTHONPATH"] == str(archive / "src")
    assert calls[0][1]["cwd"] == archive
    assert (archive / "processing").is_symlink()
    assert script.read_text() == "# immutable archived fixture\n"
    worker.close()
