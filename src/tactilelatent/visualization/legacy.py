"""Explicit historical viewer isolation; no legacy imports in the core process."""

import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class LegacyViewer:
    def __init__(self, archive_root, project_root, *, log_root=None):
        self.archive = Path(archive_root).resolve()
        self.project = Path(project_root).resolve()
        self.log_root = Path(log_root or self.archive.parent.parent / "logs/viewer")
        self.process = None
        self.port = None
        self.lock = threading.RLock()

    def start(self):
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                return
            launcher = self.archive / "scripts/view_egotouch_autoencoder.py"
            if not launcher.is_file():
                raise ValueError("The historical viewer archive is unavailable")
            # Links expose existing artifacts in the historical workspace without
            # copying arrays or changing any archived implementation bytes.
            for name in ("processing", "datasets", "mano_models", ".venv"):
                link = self.archive / name
                if not link.exists() and not link.is_symlink() and (self.project / name).exists():
                    link.symlink_to(self.project / name, target_is_directory=True)
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                self.port = listener.getsockname()[1]
            environment = dict(
                os.environ,
                PYTHONPATH=str(self.archive / "src"),
                CUDA_VISIBLE_DEVICES="",
                OMP_NUM_THREADS="1",
                MKL_NUM_THREADS="1",
                OPENBLAS_NUM_THREADS="1",
            )
            self.log_root.mkdir(parents=True, exist_ok=True)
            with (self.log_root / "historical-viewer.log").open("ab") as log:
                self.process = subprocess.Popen(
                    [
                        sys.executable,
                        "-u",
                        str(Path(__file__).with_name("historical_worker.py")),
                        "--archive",
                        str(self.archive),
                        "--project-root",
                        str(self.project),
                        "--host",
                        "127.0.0.1",
                        "--port",
                        str(self.port),
                        "--joint-device",
                        "cpu",
                    ],
                    cwd=self.archive,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            # Wait for the owned HTTP process, never for an unrelated PID.
            deadline = time.monotonic() + 45
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise ValueError(
                        "Historical viewer exited; see logs/viewer/historical-viewer.log"
                    )
                try:
                    with socket.create_connection(("127.0.0.1", self.port), timeout=0.2):
                        return
                except OSError:
                    time.sleep(0.1)
            raise ValueError("Historical viewer is still starting; retry shortly")

    def request(self, tail):
        self.start()
        target = "http://127.0.0.1:" + str(self.port) + "/" + tail.lstrip("/")
        try:
            response = urlopen(Request(target), timeout=120)
        except HTTPError as error:
            response = error
        except URLError as error:
            raise ValueError("Historical viewer is unavailable: " + str(error.reason)) from error
        with response:
            return response.status, dict(response.headers), response.read()

    def close(self):
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # Only the read-only child HTTP server started by this
                    # object is owned here; training processes are never touched.
                    self.process.kill()
                    self.process.wait(timeout=5)
