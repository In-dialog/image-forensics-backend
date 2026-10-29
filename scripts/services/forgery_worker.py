"""Run one forensic method per process so CUDA libraries cannot retain allocations."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from scripts.runtime.processes import owned_process

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _command(method, image, device, output):
    command = [sys.executable, "-B", "-u", "-m", "scripts.services.forgery_worker",
               "--method", method, "--image", str(image), "--output", str(output)]
    if device is not None:
        command.extend(["--device", device])
    return command


def run_isolated_method(method: str, image: Path, device: str | None = None, *, timeout=3600):
    """Return CPU arrays only, after the child has exited. No model is cached here."""
    import numpy as np

    image = image.resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix="forensic-method-") as directory:
        output = Path(directory) / "result.npz"
        with owned_process(
            _command(method, image, device, output), cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
            env=dict(os.environ, PYTHONUNBUFFERED="1", PYTHONUTF8="1"),
        ) as process:
            log, _ = process.communicate(timeout=timeout)
        if log.strip():
            print(log.rstrip())
        if process.returncode:
            raise RuntimeError(f"{method} worker exited with code {process.returncode}: {log[-4000:].strip()}")
        if not output.is_file():
            raise RuntimeError(f"{method} worker exited without saving results.")
        with np.load(output, allow_pickle=False) as saved:
            if not saved.files or set(saved.files) - {"heatmap", "mask", "detection"}:
                raise ValueError(f"{method} worker returned invalid outputs.")
            result = {key: saved[key].copy() if key in saved else None for key in ("heatmap", "mask", "detection")}
        print(f"  {method}: worker exited; GPU resources released")
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    import ensure_path  # Bootstrap the shared extension modules in this child only.
    import numpy as np
    from scripts.extensions.forgery_detection_extensions import read_image_as_tensor, run_method
    result = run_method(args.method, read_image_as_tensor(args.image), device=args.device)
    np.savez(args.output, **{key: value for key, value in result.items() if value is not None})


if __name__ == "__main__":
    main()
