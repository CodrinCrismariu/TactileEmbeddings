"""Serve native sensor reconstruction and prediction through the shared registry.

Inputs are project/configuration paths and viewer requests; outputs are HTTP
responses and service logs. Historical models use an isolated compatibility
worker. The operational entrypoint is ``tl view --port 8080``.
"""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .legacy import LegacyViewer
from .state import ViewerState
from .training_dashboard import TrainingDashboard, make_training_handler

ASSETS = {"viewer.js": "text/javascript; charset=utf-8", "viewer.css": "text/css; charset=utf-8"}


def make_handler(state, *, legacy=None, dashboard=None):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            if parsed.path.endswith("/legacy"):
                # The archived UI uses relative API and asset URLs. Keep its
                # directory URL, including a surrounding IDE proxy prefix.
                self.send_response(301)
                location = parsed.path + "/" + ("?" + parsed.query if parsed.query else "")
                self.send_header("Location", location)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if "/legacy/" in parsed.path:
                return self._legacy(parsed)
            route = (
                "/api/" + parsed.path.split("/api/", 1)[1]
                if "/api/" in parsed.path
                else parsed.path
            )
            try:
                if route == "/api/catalog":
                    return self._json(state.catalog(query.get("refresh") == ["1"]))
                if route == "/api/episodes":
                    return self._json(
                        state.episodes(
                            self._get(query, "source"),
                            self._get(query, "split"),
                            self._get(query, "filter", ""),
                        )
                    )
                if route == "/api/frame":
                    return self._json(
                        state.frame(
                            self._get(query, "source"),
                            self._get(query, "episode"),
                            int(self._get(query, "frame", "0")),
                            page=self._get(query, "page", "reconstruction"),
                            mirrored=self._get(query, "mirror", "0") == "1",
                            model=self._get(query, "model", None),
                            epoch=self._get(query, "epoch", None),
                            attention=self._get(query, "attention", "0") == "1",
                        )
                    )
                name = parsed.path.rsplit("/", 1)[-1]
                if name in ASSETS:
                    return self._bytes(
                        200, (Path(__file__).parent / name).read_bytes(), ASSETS[name]
                    )
                if route.startswith("/api/"):
                    return self._json(dict(error="Unknown API route"), 404)
                return self._bytes(
                    200,
                    Path(__file__).with_name("viewer.html").read_bytes(),
                    "text/html; charset=utf-8",
                )
            except (OSError, ValueError, KeyError, RuntimeError) as error:
                self._json(dict(error=str(error)), 400)

        @staticmethod
        def _get(query, name, default=""):
            return query.get(name, [default])[0]

        def _json(self, value, status=200):
            self._bytes(status, json.dumps(value, allow_nan=False).encode(), "application/json")

        def _bytes(self, status, content, mime):
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(content)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _legacy(self, parsed):
            if legacy is None:
                return self._json(dict(error="Historical viewer is unavailable"), 503)
            try:
                path = parsed.path.split("/legacy", 1)[1].lstrip("/")
                tail = path + ("?" + parsed.query if parsed.query else "")
                status, headers, body = legacy.request(tail)
                self._bytes(status, body, headers.get("Content-Type", "application/octet-stream"))
            except (OSError, ValueError) as error:
                self._json(dict(error=str(error)), 503)

        def log_message(self, format, *args):
            pass

    return make_training_handler(Handler, dashboard) if dashboard is not None else Handler


def serve(*, project_root, host="127.0.0.1", port=8080, experiment=None):
    """Start the current UI; launch the isolated historical worker on demand."""
    root = Path(project_root).resolve()
    from ..configuration import metadata_root

    code_root = metadata_root(root)
    state = ViewerState(root, experiment)
    legacy = LegacyViewer(code_root / "legacy/v1", root, log_root=code_root / "logs/viewer")
    dashboard = TrainingDashboard(root, journal=code_root / "logs/viewer/training-history")
    server = ThreadingHTTPServer(
        (host, int(port)), make_handler(state, legacy=legacy, dashboard=dashboard)
    )
    print(f"TactileLatent viewer: http://{host}:{server.server_port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        dashboard.close()
        legacy.close()
