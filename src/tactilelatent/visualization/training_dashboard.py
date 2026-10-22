"""Read scalar histories and sample live status without touching training.

Inputs are recorded metrics/status files under configured run roots. Outputs
are normalized chart series and a separate viewer sampling journal; epoch
validation and live batch observations remain distinct. Source units and absent
metrics are preserved rather than inferred from another run's schema.
"""

import hashlib
import json
import math
import os
import threading
import time
from collections import defaultdict
from pathlib import Path
from urllib.parse import parse_qs, urlparse

SOURCE_NAMES = {
    "egotouch": "EgoTouch",
    "egotactile": "EgoTactile",
    "opentouch": "OpenTouch",
    "opengraph": "OpenGraph",
    "baihu": "Baihu",
    "egopressure": "EgoPressure",
    "ftp_unit": "FTP Unit",
    "ftp_unit_bimanual": "FTP Unit Bimanual",
    "ftp_vla_touch": "FTP VLA Touch",
    "ftp_vitamin": "FTP ViTaMIn",
    "ftp_rdp": "FTP RDP",
    "ftp_rdp_bimanual": "FTP RDP Bimanual",
    "ftp_motiontrans": "MotionTrans",
    "feel_lab": "FEEL Lab",
    "feel_kitchen": "FEEL Kitchen",
    "hrdexdb_allegro_v5": "HRDex Allegro",
    "hrdexdb_inspire_f1": "HRDex Inspire",
    "freetacman": "FreeTacMan (historical)",
    "__macro__": "Equal-weight source mean",
    "__all__": "Logged overall",
}
ALIASES = {
    "supervision": "supervision_loss",
    "code_mse": "supervision_loss",
    "train_loss": "loss",
    "mae_px": "marker_mae_px",
    "active_mae_px": "active_mae",
    "mae_spacing": "marker_mae_spacing",
    "score": "selection_score",
}
METRICS = {
    "supervision_loss": "Supervision loss",
    "loss": "Objective loss",
    "mae": "MAE · logged units",
    "normalized_mae": "Normalized MAE",
    "native_mae": "Native MAE",
    "marker_mae_px": "Marker MAE · pixels",
    "marker_mae_spacing": "Marker MAE · spacing units",
    "mse": "MSE",
    "kl": "KL",
    "contact_mae": "Contact MAE",
    "active_mae": "Active-motion MAE",
    "background_mae": "Background MAE",
    "high_pressure_mae": "High-pressure MAE",
    "contact_f1": "Contact F1",
    "psnr": "PSNR",
    "selection_score": "Selection score · run-specific",
    "temporal_change_mae": "Temporal-change MAE",
    "temporal_change_mae_px": "Temporal-change MAE · pixels",
    "endpoint_error_px": "Marker endpoint error · pixels",
}
PRUNED = {
    "evaluations",
    "features",
    "cache",
    "caches",
    "codes",
    "targets",
    "viewer_epochs",
    "checkpoints",
    "__pycache__",
    ".git",
    "tuning_trials",
}


def finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def read_json(path):
    try:
        value = json.loads(Path(path).read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def stamp(path):
    try:
        s = Path(path).stat()
        return s.st_mtime_ns, s.st_size
    except OSError:
        return 0, 0


def pid_active(pid, project, reported_at=None):
    """A stale PID reused by another process must not mark an old run live."""
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return False
    try:
        proc = Path("/proc") / str(pid)
        command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").lower()
        if reported_at is not None:
            fields = (proc / "stat").read_text().rsplit(")", 1)[1].split()
            boot = next(
                int(line.split()[1])
                for line in Path("/proc/stat").read_text().splitlines()
                if line.startswith("btime ")
            )
            started = boot + int(fields[19]) / os.sysconf("SC_CLK_TCK")
            if reported_at < started:
                return False
        return (proc / "cwd").resolve() == project and b"python" in command
    except (OSError, ValueError, IndexError, StopIteration):
        return False


def scalar_metrics(value):
    if finite(value):
        return [("loss", float(value), "loss")]
    if not isinstance(value, dict):
        return []
    result = {}
    # Prefer explicitly named supervision over its duplicated code_mse field.
    for key in sorted(value, key=lambda k: k in ALIASES):
        metric = ALIASES.get(key, key)
        if metric in METRICS and finite(value[key]) and metric not in result:
            result[metric] = (metric, float(value[key]), key)
    return list(result.values())


def normalize_history(rows, config, path):
    """Keep scalar names/units distinct; never invent validation objective loss."""
    source = config.get("source")
    if source == "feel":
        dim = config.get("model", {}).get("input_dim")
        source = "feel_lab" if dim == 5 else "feel_kitchen" if dim == 12 else source
    if not source:
        candidates = [
            s
            for s in SOURCE_NAMES
            if not s.startswith("__")
            and any(part == s or part.startswith(s + "_") for part in path.parts)
        ]
        source = max(candidates, key=len) if candidates else "__all__"
    epochs = [r.get("epoch") for r in rows if finite(r.get("epoch"))]
    offset = 1 if 0 in epochs else 0
    curves = defaultdict(dict)
    fields = defaultdict(set)

    def add(epoch, sensor, split, value):
        for metric, number, field in scalar_metrics(value):
            key = (sensor, split, metric)
            curves[key][epoch] = number
            fields[key].add(field)

    for row in rows:
        if not finite(row.get("epoch")):
            continue
        epoch = row["epoch"] + offset
        sensor = row.get("source", source)
        if sensor == "feel":
            sensor = source
        train = row.get("train")
        if isinstance(train, dict) and not scalar_metrics(train):
            for key, values in train.items():
                add(epoch, key, "train", values)
        elif train is not None:
            add(epoch, sensor, "train", train)
        if isinstance(row.get("train_loss_by_source"), dict):
            for key, loss in row["train_loss_by_source"].items():
                add(epoch, key, "train", loss)
        if finite(row.get("train_loss")):
            add(epoch, sensor, "train", {"loss": row["train_loss"]})
        validation = row.get("validation", row.get("val"))
        if isinstance(validation, dict) and "model" in validation:
            add(epoch, sensor, "validation", validation["model"])
        elif isinstance(validation, dict) and {"original", "mirrored"} & validation.keys():
            for variant in ("original", "mirrored"):
                add(epoch, sensor, "validation_" + variant, validation.get(variant))
        elif isinstance(validation, dict) and not scalar_metrics(validation):
            for key, values in validation.items():
                add(epoch, key, "validation", values)
        elif validation is not None:
            add(epoch, sensor, "validation", validation)
        if finite(row.get("selection_score")):
            add(epoch, "__all__", "validation", {"selection_score": row["selection_score"]})
        for key, values in row.get("mirror_validation", {}).items():
            if isinstance(values, dict):
                for variant in ("original", "mirrored"):
                    add(epoch, key, "validation_" + variant, {"mae": values.get(variant + "_mae")})

    macro = defaultdict(list)
    for (sensor, split, metric), values in curves.items():
        if sensor.startswith("__"):
            continue
        for epoch, value in values.items():
            macro[(split, metric, epoch)].append(value)
    for (split, metric, epoch), values in macro.items():
        if len(values) > 1:
            curves[("__macro__", split, metric)][epoch] = sum(values) / len(values)
    output = [
        dict(
            source=s,
            split=split,
            metric=metric,
            fields=sorted(fields[(s, split, metric)]),
            points=sorted(values.items()),
        )
        for (s, split, metric), values in sorted(curves.items())
    ]
    return output, max((e + offset for e in epochs), default=0), offset


class TrainingDashboard:
    def __init__(self, project, roots=None, journal=None):
        self.project = Path(project).resolve()
        self.roots = (
            [Path(r) for r in roots]
            if roots is not None
            else [
                self.project / "processing/runs",
                self.project / "processing/data_expansion_v1/teachers",
                self.project / "processing/acquisition_teachers_v1",
                *sorted((self.project / "processing").glob("bimanual_mirror_v*/teachers")),
            ]
        )
        self.journal = (
            Path(journal) if journal else self.project / "processing/viewer_training_history_v1"
        )
        self.lock = threading.RLock()
        self.paths = {}
        self.cache = {}
        self.last_scan = -math.inf
        self.sample_keys = {}
        self.stop = threading.Event()
        self.thread = None
        self.collector_error = None

    def discover(self, force=False):
        if not force and time.monotonic() - self.last_scan < 25:
            return
        found = {}
        for root in self.roots:
            for folder, dirs, files in os.walk(root, followlinks=False):
                dirs[:] = [d for d in dirs if d not in PRUNED and "smoke" not in d.lower()]
                parent = Path(folder)
                if any("smoke" in p.lower() for p in parent.relative_to(self.project).parts):
                    continue
                if "metrics.jsonl" not in files and not {"status.json", "config.json"} <= set(
                    files
                ):
                    continue
                path = (parent / "metrics.jsonl").resolve()
                if not path.is_relative_to(self.project):
                    continue
                identifier = hashlib.sha256(
                    str(path.relative_to(self.project)).encode()
                ).hexdigest()[:20]
                found[identifier] = path
        self.paths = found
        self.last_scan = time.monotonic()

    def load(self, identifier):
        path = self.paths[identifier]
        parent = path.parent
        signature = (stamp(path), stamp(parent / "config.json"), stamp(parent / "signature.json"))
        if identifier in self.cache and self.cache[identifier]["signature"] == signature:
            return self.cache[identifier]
        config = read_json(parent / "config.json") or read_json(parent / "signature.json")
        rows = []
        try:
            with path.open() as stream:
                for line in stream:
                    try:
                        row = json.loads(line)
                        if isinstance(row, dict):
                            rows.append(row)
                    except ValueError:
                        pass  # An appending writer may leave a partial final row.
        except FileNotFoundError:
            pass
        curves, completed, offset = normalize_history(rows, config, path)
        relative = str(parent.relative_to(self.project)).removeprefix("processing/runs/")
        name = relative.replace("/trials/", " / ").replace("/teachers/", " / ")
        is_ae = (
            "teacher" in relative
            or "autoencoder" in relative
            or "_ae_" in relative
            or config.get("model", {}).get("architecture") in ("conv", "mlp", "marker_set")
        )
        teacher = parent.name == "tactile_teacher"
        kind = "teacher" if teacher else "autoencoder" if is_ae else "joint"
        sources = sorted({r["source"] for r in curves if not r["source"].startswith("__")})
        model = config.get("model", {}) or read_json(parent / "quality.json").get(
            "teacher", {}
        ).get("model_config", {})
        entry = dict(
            signature=signature,
            curves=curves,
            config=config,
            metadata=dict(
                id=identifier,
                name=name,
                path=str(path.relative_to(self.project)),
                kind=kind,
                sources=sources,
                metrics=sorted({r["metric"] for r in curves}),
                completed_epochs=completed,
                epoch_offset=offset,
                total_epochs=config.get("epochs", config.get("training", {}).get("epochs")),
                latent_dim=config.get("latent_dim", model.get("latent_dim")),
                latent_mode=config.get("latent_mode", config.get("latent_objective")),
                seed=config.get("seed", config.get("training", {}).get("seed")),
                modified=stamp(path)[0] / 1e9,
                weights_available=any((parent / p).exists() for p in ("last.pt", "best.pt")),
                record_count=len(rows),
            ),
        )
        self.cache[identifier] = entry
        return entry

    def metadata(self, identifier):
        entry = self.load(identifier)
        result = dict(entry["metadata"])
        parent = self.paths[identifier].parent
        status_path = parent / "status.json"
        status = read_json(status_path)
        phase = status.get("stage", status.get("status", ""))
        active = phase not in (
            "complete",
            "completed",
            "stopped_at_user_request",
            "failed",
            "paused",
            "stopped",
        ) and pid_active(status.get("pid"), self.project, stamp(status_path)[0] / 1e9)
        # Epoch-based autoencoders sometimes omit their PID; label recent writes
        # "updating", rather than asserting that an unverified process is alive.
        updating = (
            not active
            and not status.get("pid")
            and phase in ("training", "validation")
            and time.time() - stamp(status_path)[0] / 1e9 < 30
        )
        finished = phase in ("complete", "completed") or (
            finite(result["total_epochs"]) and result["completed_epochs"] >= result["total_epochs"]
        )
        result["state"] = (
            "running"
            if active
            else "updating"
            if updating
            else "complete"
            if finished
            else "inactive"
        )
        allowed = (
            "stage",
            "status",
            "pid",
            "epoch",
            "completed_epochs",
            "total_epochs",
            "batch",
            "batches_per_epoch",
            "source",
            "loss",
            "epoch_seconds",
            "latent_dim",
            "prediction_mode",
        )
        result["status"] = {
            k: v
            for k, v in status.items()
            if k in allowed and (not isinstance(v, float) or math.isfinite(v))
        }
        if not result["sources"] and isinstance(status.get("source"), str):
            result["sources"] = [status["source"]]
        return result

    def catalog(self, force=False):
        with self.lock:
            self.discover(force)
            runs = []
            for identifier in self.paths:
                try:
                    meta = self.metadata(identifier)
                    if meta["record_count"] or meta["state"] in ("running", "updating"):
                        runs.append(meta)
                except (OSError, ValueError, TypeError):
                    continue
            runs.sort(
                key=lambda r: (r["state"] not in ("running", "updating"), -r["modified"], r["name"])
            )
            return dict(
                runs=runs,
                metrics=METRICS,
                source_names=SOURCE_NAMES,
                server_time=time.time(),
                refresh_seconds=5,
                live_sampling_seconds=2,
                note="Completed-epoch curves use saved logs. Live batch samples begin when this monitor starts.",
            )

    def sample_once(self):
        with self.lock:
            for meta in self.catalog()["runs"]:
                status = meta["status"]
                if (
                    meta["state"] not in ("running", "updating")
                    or status.get("stage") != "training"
                ):
                    continue
                if not all(finite(status.get(k)) for k in ("loss", "epoch", "batch")):
                    continue
                if not isinstance(status.get("source"), str):
                    continue
                key = tuple(status.get(k) for k in ("pid", "epoch", "batch", "source", "loss"))
                if self.sample_keys.get(meta["id"]) == key:
                    continue
                self.sample_keys[meta["id"]] = key
                self.journal.mkdir(parents=True, exist_ok=True)
                row = {
                    k: status[k]
                    for k in ("epoch", "batch", "source", "loss", "batches_per_epoch", "pid")
                    if k in status
                }
                row["time"] = time.time()
                with (self.journal / (meta["id"] + ".jsonl")).open("a") as stream:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")

    def history(self, identifiers):
        if not identifiers or len(identifiers) > 8:
            raise ValueError("Select between one and eight runs")
        with self.lock:
            self.discover()
            if any(i not in self.paths for i in identifiers):
                raise ValueError("Unknown run; refresh the model list")
            runs = []
            for identifier in dict.fromkeys(identifiers):
                entry = self.load(identifier)
                samples = []
                path = self.journal / (identifier + ".jsonl")
                if path.exists():
                    from collections import deque

                    with path.open() as stream:
                        for line in deque(stream, maxlen=15000):
                            try:
                                row = json.loads(line)
                                if (
                                    isinstance(row, dict)
                                    and finite(row.get("loss"))
                                    and finite(row.get("time"))
                                ):
                                    samples.append(row)
                            except ValueError:
                                pass
                runs.append(
                    dict(**self.metadata(identifier), curves=entry["curves"], live_samples=samples)
                )
            return dict(runs=runs, server_time=time.time(), collector_error=self.collector_error)

    def start(self):
        if self.thread is not None:
            return

        def collect():
            while not self.stop.is_set():
                try:
                    self.sample_once()
                    self.collector_error = None
                except Exception as error:
                    self.collector_error = str(error)
                self.stop.wait(2)

        self.thread = threading.Thread(target=collect, daemon=True, name="training-status-monitor")
        self.thread.start()

    def close(self):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=3)


def make_training_handler(base, dashboard=None):
    if dashboard is None:
        raise ValueError("An explicit project-root TrainingDashboard is required")
    dashboard.start()

    class Handler(base):
        def do_GET(self):
            url = urlparse(self.path)
            query = parse_qs(url.query)
            route = "/api/" + url.path.split("/api/", 1)[1] if "/api/" in url.path else url.path
            training_page = not route.startswith("/api/") and (
                query.get("page") == ["training"] or route.rstrip("/").endswith("/training")
            )
            if not (
                training_page
                or route.endswith("/training-dashboard.js")
                or route.startswith("/api/training/")
            ):
                return super().do_GET()
            try:
                status = 200
                if training_page:
                    content = Path(__file__).with_name("training_dashboard.html").read_bytes()
                    mime = "text/html; charset=utf-8"
                elif route.endswith("/training-dashboard.js"):
                    content = Path(__file__).with_name("training_dashboard.js").read_bytes()
                    mime = "text/javascript; charset=utf-8"
                elif route == "/api/training/catalog":
                    content = json.dumps(
                        dashboard.catalog(query.get("refresh") == ["1"]), allow_nan=False
                    ).encode()
                    mime = "application/json"
                elif route == "/api/training/history":
                    ids = query.get("runs", [""])[0].split(",")
                    content = json.dumps(dashboard.history(ids), allow_nan=False).encode()
                    mime = "application/json"
                else:
                    self.send_error(404)
                    return
            except (BrokenPipeError, ConnectionResetError):
                return
            except (OSError, ValueError, KeyError) as error:
                status = 400
                content = json.dumps(dict(error=str(error))).encode()
                mime = "application/json"
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(content)
            except (BrokenPipeError, ConnectionResetError):
                pass

    return Handler
