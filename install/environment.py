"""Shared paths for setup and runtime; no side effects beyond explicit calls."""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE_FILE = ROOT / ".install-state.json"


def python_path(environment: Path) -> Path:
    return environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def read_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {}


def default_cache() -> Path:
    return Path(os.environ.get("IMAGE_FORENSICS_CACHE", Path.home() / ".cache" / "image-forensics")).expanduser()


def configure_environment(cache: Path | None = None) -> dict[str, str]:
    """Respect existing shared model caches; never copy them into a checkout."""
    state = read_state()
    custom = cache is not None or state.get("custom_cache", False)
    cache = cache or Path(state.get("cache_dir", default_cache()))
    values = {
        "IMAGE_FORENSICS_CACHE": str(cache),
        "HF_HOME": str(cache / "huggingface" if custom else Path.home() / ".cache" / "huggingface"),
        "TORCH_HOME": str(cache / "torch" if custom else Path.home() / ".cache" / "torch"),
        "IMAGE_FORENSICS_WEIGHTS": str(cache / "photoholmes"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "MPLBACKEND": "Agg",
    }
    for key, value in values.items():
        os.environ.setdefault(key, value)
    return dict(os.environ)
