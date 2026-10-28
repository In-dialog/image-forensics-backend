"""
Lightweight local API server for nonescape synthetic-image inference.

Endpoints:
  GET  /health
  POST /classify   {"image_path": "...", "model_path": "...?"}
"""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from huggingface_hub import hf_hub_download
from nonescape import NonescapeClassifier, preprocess_image

DEFAULT_MODEL_PATH = "nonescape-v0.safetensors"
HF_REPO_ID = "e3ntity/nonescape-v0"
HF_FILENAME = "nonescape-v0.safetensors"

_classifier: NonescapeClassifier | None = None
_classifier_model_path: str | None = None


def _resolve_model_path(model: str) -> str:
    p = Path(model)
    if p.is_file():
        return str(p.resolve())
    if model == DEFAULT_MODEL_PATH or p.name == HF_FILENAME:
        print(f"Downloading {HF_FILENAME} from Hugging Face ({HF_REPO_ID})...")
        return hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME)
    raise FileNotFoundError(f"Model file not found: {model}")


def _get_classifier(model_path: str | None = None) -> tuple[NonescapeClassifier, str]:
    global _classifier, _classifier_model_path

    requested = model_path or DEFAULT_MODEL_PATH
    resolved = _resolve_model_path(requested)

    if _classifier is None or _classifier_model_path != resolved:
        _classifier = NonescapeClassifier.from_pretrained(resolved)
        _classifier_model_path = resolved
        print(f"Loaded nonescape model: {resolved}")
    return _classifier, resolved


def _predict(image_path: Path, model_path: str | None = None) -> dict[str, Any]:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    classifier, resolved_model = _get_classifier(model_path=model_path)
    img = Image.open(image_path).convert("RGB")
    x = preprocess_image(img).unsqueeze(0)
    with torch.no_grad():
        probs = classifier(x)

    real_prob = float(probs[0, 0].item())
    synthetic_prob = float(probs[0, 1].item())
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


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


class SyntheticApiHandler(BaseHTTPRequestHandler):
    server_version = "SyntheticAPI/1.0"

    def _send_json(self, code: int, payload: dict[str, Any]) -> None:
        data = _json_bytes(payload)
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {self.address_string()} {fmt % args}")

    def do_GET(self) -> None:
        if self.path.rstrip("/") == "/health":
            self._send_json(
                200,
                {
                    "ok": True,
                    "model_loaded": _classifier is not None,
                    "model_path": _classifier_model_path,
                },
            )
            return
        self._send_json(404, {"ok": False, "error": "Not Found"})

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/classify":
            self._send_json(404, {"ok": False, "error": "Not Found"})
            return

        try:
            content_len = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(content_len) if content_len > 0 else b"{}"
            payload = json.loads(raw.decode("utf-8"))
            image_path = payload.get("image_path")
            model_path = payload.get("model_path")
            if not image_path:
                self._send_json(400, {"ok": False, "error": "Missing required field: image_path"})
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
        help="Optional local model path to preload. If omitted, default HF model is used on first request.",
    )
    args = parser.parse_args()

    if args.model_path:
        _get_classifier(model_path=args.model_path)

    server = ThreadingHTTPServer((args.host, args.port), SyntheticApiHandler)
    print(f"Synthetic API listening on http://{args.host}:{args.port}")
    print("Endpoints: GET /health, POST /classify")
    server.serve_forever()


if __name__ == "__main__":
    main()
