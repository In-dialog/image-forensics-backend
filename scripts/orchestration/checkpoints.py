"""Per-run model-response checkpoints, separate from downloadable results."""
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import json
from pathlib import Path

from scripts.configuration.store import atomic_json

_directory: ContextVar[Path | None] = ContextVar("analysis_checkpoints", default=None)


@contextmanager
def checkpoint_scope(directory: Path | None):
    token = _directory.set(directory)
    try:
        yield
    finally:
        _directory.reset(token)


def response_checkpoint(image: Path, request: dict) -> Path | None:
    directory = _directory.get()
    if directory is None:
        return None
    digest = hashlib.sha256(image.read_bytes())
    digest.update(json.dumps(request, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8"))
    return directory / f"{digest.hexdigest()}.json"


def load_response(path: Path | None, validate):
    if path is None:
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        if document["version"] != 1:
            return None
        return validate(document["response"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def save_response(path: Path | None, result: dict):
    if path is not None:
        atomic_json(path, {"version": 1, "response": result})
