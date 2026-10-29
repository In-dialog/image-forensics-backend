"""Check existing synthetic dependencies without importing models or installing."""
from __future__ import annotations

import subprocess
from pathlib import Path

SYNTHETIC_DEPENDENCIES = ("nonescape", "torch", "torchvision", "transformers", "safetensors", "PIL", "huggingface_hub")


def missing_synthetic_dependencies(python: Path) -> str:
    check = (
        "import importlib.util,sys; "
        f"missing=[name for name in {SYNTHETIC_DEPENDENCIES!r} if importlib.util.find_spec(name) is None]; "
        "print(', '.join(missing)); sys.exit(bool(missing))"
    )
    try:
        result = subprocess.run([str(python), "-B", "-c", check], capture_output=True,
                                text=True, timeout=10, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.TimeoutExpired) as exc:
        return str(exc)
    if result.returncode:
        return (result.stderr or result.stdout).strip() or f"Python exited with code {result.returncode}"
    return ""
