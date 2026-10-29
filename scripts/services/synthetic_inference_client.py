"""Poll inference independently of HTTP connection timeouts and preserve errors."""
from __future__ import annotations

import json
import math
from pathlib import Path
import time
from urllib import request, error


def probabilities(payload: dict) -> tuple[float, float]:
    if not isinstance(payload, dict) or payload.get("ok") is not True:
        raise ValueError("Synthetic service did not return a successful prediction.")
    values = payload.get("probs")
    if not isinstance(values, dict):
        raise ValueError("Synthetic prediction is missing probabilities.")
    pair = (values.get("real"), values.get("synthetic"))
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) or not 0 <= value <= 1 for value in pair):
        raise ValueError("Synthetic probabilities must be finite numbers between zero and one.")
    if not math.isclose(sum(pair), 1, abs_tol=0.001):
        raise ValueError("Synthetic probabilities do not sum to one.")
    return tuple(float(value) for value in pair)


def _json(url, payload=None, *, timeout=10):
    req = request.Request(url, data=json.dumps(payload).encode() if payload is not None else None,
                          headers={"Content-Type": "application/json"})
    try:
        with request.urlopen(req, timeout=timeout) as response:
            return json.load(response)
    except error.HTTPError as exc:
        try:
            detail = json.loads(exc.read().decode()).get("error")
        except (ValueError, UnicodeError, AttributeError):
            detail = None
        raise RuntimeError(f"Synthetic service error ({exc.code}): {detail or exc.reason}") from exc


def predict(image: Path, *, api_url: str, model_path=None, timeout_s=1800, poll_interval=1,
            report=lambda message: None) -> tuple[float, float]:
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("Synthetic inference timeout must be a positive number.")
    base = api_url.rstrip("/")
    health = _json(f"{base}/health")
    if health.get("inference_protocol") != 2:
        raise RuntimeError("Restart the analysis backend and any separately started synthetic service "
                           "to enable reconnectable inference and model-loading progress.")
    task = _json(f"{base}/classify", {"image_path": str(image), "model_path": str(model_path) if model_path else None,
                                     "async": True})
    task_id = task.get("id")
    if not isinstance(task_id, str) or len(task_id) != 32 or any(c not in "0123456789abcdef" for c in task_id):
        raise ValueError("Synthetic service returned an invalid task identifier.")
    deadline = time.monotonic() + timeout_s
    activity = None
    while True:
        if task.get("status") == "failed":
            raise RuntimeError(f"Synthetic inference failed: {task.get('error') or 'Unknown service error'}")
        if task.get("status") == "succeeded":
            return probabilities(task.get("result"))
        if task.get("status") != "running":
            raise ValueError("Synthetic service returned an invalid task status.")
        if task.get("activity") != activity:
            activity = task.get("activity")
            report(activity or "Processing image")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Synthetic analysis is still working after {timeout_s:g}s ({activity}). "
                               "Resume will reconnect to this task instead of starting another model load.")
        time.sleep(min(poll_interval, max(0, deadline - time.monotonic())))
        task = _json(f"{base}/tasks/{task_id}")
