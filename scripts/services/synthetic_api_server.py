"""
Lightweight local API server for nonescape synthetic-image inference.

Endpoints:
  GET  /health
  POST /classify   {"image_path": "...", "model_path": "...?", "async": true}
  GET  /tasks/<id> poll preparation/inference, including after a client disconnect
  POST /unload     wait for active inference and release unused allocations
"""
from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from threading import Lock

# Support a configured direct .py command as well as package-based startup.
if not __package__:
    project_root = str(Path(__file__).resolve().parents[2])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from scripts.runtime.gpu import inference_call, release_cuda_memory
from scripts.services.synthetic_tasks import SyntheticTasks
from scripts.services.synthetic_inference_client import probabilities

import torch
from PIL import Image
from huggingface_hub import hf_hub_download
from nonescape import NonescapeClassifier, preprocess_image

DEFAULT_MODEL_PATH = "nonescape-v0.safetensors"
HF_REPO_ID = "e3ntity/nonescape-v0"
HF_FILENAME = "nonescape-v0.safetensors"

_active_model_path: str | None = None
_prediction_lock = Lock()


def _resolve_model_path(model: str) -> str:
    p = Path(model)
    if p.is_file():
        return str(p.resolve())
    if model == DEFAULT_MODEL_PATH or p.name == HF_FILENAME:
        from huggingface_hub import try_to_load_from_cache
        cached = try_to_load_from_cache(repo_id=HF_REPO_ID, filename=HF_FILENAME)
        if isinstance(cached, str) and Path(cached).is_file():
            return cached
        print(f"Preparing {HF_FILENAME} from Hugging Face ({HF_REPO_ID}); an incomplete download must finish first.")
        return hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME)
    raise FileNotFoundError(f"Model file not found: {model}")


def _load_classifier(model_path: str | None = None, report=lambda message: None) -> tuple[NonescapeClassifier, str]:
    global _active_model_path
    requested = model_path or DEFAULT_MODEL_PATH
    report("Preparing model weights; the first download can take several minutes")
    resolved = _resolve_model_path(requested)

    report("Loading the synthetic classifier")
    classifier = NonescapeClassifier.from_pretrained(resolved)
    classifier.eval()
    _active_model_path = resolved
    print(f"Loaded nonescape model: {resolved}")
    return classifier, resolved


def _predict(image_path: Path, model_path: str | None = None, report=lambda message: None) -> dict[str, Any]:
    global _active_model_path
    # Keep loading, inference AND cleanup under one lock. A second HTTP request
    # must not allocate another classifier while the first one is being freed.
    with _prediction_lock:
        try:
            return _predict_once(image_path, model_path, report)
        finally:
            _active_model_path = None


@inference_call
def _predict_once(image_path: Path, model_path: str | None = None, report=lambda message: None) -> dict[str, Any]:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    classifier, resolved_model = _load_classifier(model_path=model_path, report=report)
    report("Classifying the image")
    with Image.open(image_path) as source:
        img = source.convert("RGB")
    x = preprocess_image(img).unsqueeze(0)
    with torch.no_grad():
        probs = classifier(x)

    real_prob = float(probs[0, 0].item())
    synthetic_prob = float(probs[0, 1].item())
    probabilities({"ok": True, "probs": {"real": real_prob, "synthetic": synthetic_prob}})
    is_synthetic = synthetic_prob > 0.5
    confidence = synthetic_prob if is_synthetic else real_prob
    return {
        "ok": True,
        "image_path": str(image_path),
        "model_path": resolved_model,
        "probs": {"real": real_prob, "synthetic": synthetic_prob},
        "is_synthetic": is_synthetic,
        "confidence": float(confidence),
    }


_tasks = SyntheticTasks(_predict)


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


class SyntheticApiHandler(BaseHTTPRequestHandler):
    server_version = "SyntheticAPI/1.0"

    def _send_json(self, code: int, payload: dict[str, Any]) -> None:
        data = _json_bytes(payload)
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        try:
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass  # A disconnected client can retrieve its task on the next Resume.

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {self.address_string()} {fmt % args}")

    def do_GET(self) -> None:
        if self.path.rstrip("/") == "/health":
            self._send_json(
                200,
                {
                    "ok": True,
                    "model_loaded": _active_model_path is not None,
                    "model_path": _active_model_path,
                    "unload_after_request": True,
                    "inference_protocol": 2,
                },
            )
            return
        if self.path.startswith("/tasks/"):
            try:
                self._send_json(200, _tasks.get(self.path.removeprefix("/tasks/")))
            except KeyError:
                self._send_json(404, {"ok": False, "error": "Synthetic task no longer exists; resume the analysis."})
            return
        self._send_json(404, {"ok": False, "error": "Not Found"})

    def do_POST(self) -> None:
        if self.path.rstrip("/") == "/unload":
            with _prediction_lock:
                release_cuda_memory()
            self._send_json(200, {"ok": True, "model_loaded": False})
            return
        if self.path.rstrip("/") != "/classify":
            self._send_json(404, {"ok": False, "error": "Not Found"})
            return

        try:
            content_len = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(content_len) if content_len > 0 else b"{}"
            payload = json.loads(raw.decode("utf-8"))
            image_path = payload.get("image_path")
            model_path = payload.get("model_path") or getattr(self.server, "model_path", None)
            if not image_path:
                self._send_json(400, {"ok": False, "error": "Missing required field: image_path"})
                return

            if payload.get("async") is True:
                try:
                    task = _tasks.submit(Path(str(image_path)), str(model_path) if model_path else None)
                except RuntimeError as exc:
                    self._send_json(409, {"ok": False, "error": str(exc)})
                    return
                self._send_json(202, task)
                return
            result = _predict(Path(str(image_path)), model_path=str(model_path) if model_path else None)
            self._send_json(200, result)
        except FileNotFoundError as exc:
            self._send_json(404, {"ok": False, "error": str(exc)})
        except json.JSONDecodeError:
            self._send_json(400, {"ok": False, "error": "Invalid JSON body"})
        except Exception as exc:
            self._send_json(500, {"ok": False, "error": str(exc)})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local nonescape synthetic inference API.")
    parser.add_argument("--host", default="127.0.0.1", help="Host/interface to bind.")
    parser.add_argument("--port", type=int, default=8765, help="Port to listen on.")
    parser.add_argument(
        "--model-path",
        default=None,
        help="Optional local model path, loaded only while a request is running.",
    )
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), SyntheticApiHandler)
    server.model_path = args.model_path
    print(f"Synthetic API listening on http://{args.host}:{server.server_port}")
    print("Endpoints: GET /health, POST /classify, GET /tasks/<id>, POST /unload")
    server.serve_forever()


if __name__ == "__main__":
    main()
