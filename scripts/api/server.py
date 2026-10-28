"""Run with python -m scripts.api.server; only the Next server talks to this API."""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from scripts.configuration.store import ConfigStore, PROJECT_ROOT
from scripts.orchestration.stages import STAGES
from scripts.prompting import prompt_presets
from scripts.services.synthetic_api_client import begin_service_session, stop_owned_services
from .jobs import BusyError, JobManager
from .artifacts import ArtifactNotReadyError, ArtifactStore
from .uploads import MAX_UPLOAD_BYTES
from .previews import PreviewStore

MAX_JSON_BYTES = 1024 * 1024


def default_data_dir() -> Path:
    return Path(os.environ.get("ANALYSIS_OUTPUT_DIR", str(PROJECT_ROOT / "data"))).expanduser().resolve()


def ollama_models() -> tuple[list[str], bool]:
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=1) as response:
            data = json.loads(response.read(1024 * 1024))
        return sorted({item["name"] for item in data.get("models", []) if isinstance(item.get("name"), str)}), True
    except (OSError, ValueError, KeyError, TypeError):
        return [], False


class AnalysisServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, manager: JobManager, *, model_query=ollama_models):
        self.manager = manager
        self.artifacts = ArtifactStore(manager)
        self.previews = PreviewStore(manager)
        self.model_query = model_query
        self.model_lock = threading.Lock()
        self.model_cache = ([], False)
        self.model_checked = 0.0
        super().__init__(address, AnalysisHandler)
        try:
            begin_service_session()
        except BaseException:
            super().server_close()
            raise

    def server_close(self):
        try:
            super().server_close()
        finally:
            stop_owned_services()

    def config_response(self):
        settings, prompts = self.manager.config.snapshot()
        with self.model_lock:
            if time.monotonic() - self.model_checked > 15:
                self.model_cache = self.model_query()
                self.model_checked = time.monotonic()
            models, available = self.model_cache
        return dict(stages=[dict(key=s.key, label=s.label, group=s.group) for s in STAGES],
                    settings=settings, prompts=prompts, prompt_presets=prompt_presets(),
                    models=models, ollama_available=available,
                    active_job_id=self.manager.active_job_id)


class RequestError(Exception):
    def __init__(self, status: int, message: str):
        self.status = status
        super().__init__(message)


class AnalysisHandler(BaseHTTPRequestHandler):
    server_version = "LocalAnalysis/1.0"

    def log_message(self, *args):
        # Inference prints belong to the current job; routine polling is silent.
        pass

    def setup(self):
        super().setup()
        self.connection.settimeout(30)

    def _send(self, status, value):
        content = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(content)

    def _send_file(self, path, content_type, size):
        with path.open("rb") as stream:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(size))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            remaining = size
            while remaining:
                chunk = stream.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def _send_preview(self, content):
        self.send_response(200)
        self.send_header("Content-Type", "image/webp")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(content)

    def _body(self, maximum):
        if self.headers.get("Transfer-Encoding"):
            raise RequestError(400, "Chunked bodies are not supported.")
        raw = self.headers.get("Content-Length")
        if raw is None:
            raise RequestError(411, "Content-Length is required.")
        try:
            size = int(raw)
        except ValueError as exc:
            raise RequestError(400, "Invalid Content-Length.") from exc
        if size <= 0:
            raise RequestError(400, "Request body cannot be empty.")
        if size > maximum:
            raise RequestError(413, f"Request exceeds the {maximum // (1024 * 1024)} MB limit.")
        body = self.rfile.read(size)
        if len(body) != size:
            raise RequestError(400, "Incomplete request body.")
        return body

    def _json(self):
        if self.headers.get_content_type() != "application/json":
            raise RequestError(415, "Use application/json for this request.")
        try:
            return json.loads(self._body(MAX_JSON_BYTES), parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Non-finite JSON numbers are not allowed.")))
        except (UnicodeDecodeError, ValueError) as exc:
            raise RequestError(400, "Request body must be valid JSON with finite numbers.") from exc

    def _dispatch(self):
        try:
            hostname = urlparse(f"http://{self.headers.get('Host', '')}").hostname
            if hostname not in {"127.0.0.1", "localhost", "::1"}:
                raise RequestError(403, "This service accepts localhost requests only.")
            if self.command != "GET" and self.headers.get("Origin") is not None:
                raise RequestError(403, "Send browser requests through the Next.js interface.")
            path = urlparse(self.path).path
            manager = self.server.manager
            if self.command == "GET" and path == "/health":
                return self._send(200, dict(ok=True, data_dir=str(manager.data_dir)))
            if self.command == "GET" and path == "/config":
                return self._send(200, self.server.config_response())
            if self.command == "PUT" and path == "/config":
                manager.config.update(self._json())
                return self._send(200, self.server.config_response())
            if self.command == "POST" and path == "/uploads":
                filename = unquote(self.headers.get("X-Filename", ""))
                return self._send(201, manager.uploads.save(self._body(MAX_UPLOAD_BYTES), filename))
            if self.command == "POST" and path == "/jobs":
                return self._send(202, {"job": manager.submit(self._json())})
            if self.command == "GET" and path == "/jobs":
                return self._send(200, {"jobs": manager.list()})
            if self.command == "POST" and path == "/jobs/resume":
                return self._send(202, {"jobs": manager.resume(self._json())})
            if self.command == "DELETE" and path.startswith("/jobs/") and "/" not in path[len("/jobs/"):]:
                job_id = path[len("/jobs/"):]
                manager.delete(job_id)
                return self._send(200, {"deleted": job_id})
            if self.command == "GET" and path.startswith("/jobs/"):
                job_id, _, remainder = path[len("/jobs/"):].partition("/")
                if remainder == "previews":
                    return self._send(200, self.server.previews.list(job_id))
                if remainder.startswith("preview/"):
                    return self._send_preview(self.server.previews.image(job_id, unquote(remainder[len("preview/"):])) )
                if remainder == "artifact":
                    return self._send(200, self.server.artifacts.manifest(job_id))
                if remainder.startswith("files/"):
                    return self._send_file(*self.server.artifacts.file(job_id, unquote(remainder[len("files/"):])) )
                if not remainder:
                    return self._send(200, {"job": manager.get(job_id)})
            raise RequestError(404, "API route not found.")
        except RequestError as exc:
            self._send(exc.status, {"error": str(exc)})
        except (BusyError, ArtifactNotReadyError) as exc:
            self._send(409, {"error": str(exc)})
        except FileNotFoundError as exc:
            self._send(404, {"error": str(exc)})
        except (ValueError, TypeError) as exc:
            self._send(400, {"error": str(exc)})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            self._send(500, {"error": f"Analysis service error: {exc}"})

    do_GET = _dispatch
    do_POST = _dispatch
    do_PUT = _dispatch
    do_DELETE = _dispatch


def main():
    parser = argparse.ArgumentParser(description="Local image analysis API for the Next.js interface.")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    args = parser.parse_args()
    # Bind first: a second launch must not mark the live service's jobs interrupted.
    server = AnalysisServer(("127.0.0.1", args.port), None)
    try:
        config = ConfigStore()
        manager = JobManager(args.data_dir, config)
        server.manager = manager
        server.artifacts = ArtifactStore(manager)
        server.previews = PreviewStore(manager)
        print(f"Analysis API: http://127.0.0.1:{server.server_port}", flush=True)
        print(f"Results folder: {manager.data_dir}", flush=True)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
