"""Read-only compatibility checks, safe before any packages are installed."""
from __future__ import annotations

import csv
import io
import platform
import shutil
import subprocess
from pathlib import Path

PROFILES = {
    "cpu": ("2.6.0", "0.21.0", "cpu"),
    "cuda": ("2.6.0", "0.21.0", "cu124"),
    "cuda-modern": ("2.7.1", "0.22.1", "cu128"),
}


def inspect_hardware() -> dict:
    result = {"system": platform.system(), "machine": platform.machine(), "version": platform.version(), "gpus": []}
    binary = shutil.which("nvidia-smi")
    if binary:
        try:
            response = subprocess.run(
                [binary, "--query-gpu=name,memory.total,driver_version,compute_cap", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10, check=True,
            )
            for row in csv.reader(io.StringIO(response.stdout)):
                if len(row) == 4:
                    result["gpus"].append(dict(name=row[0].strip(), memory_mib=int(row[1]), driver=row[2].strip(), capability=float(row[3])))
        except (OSError, ValueError, subprocess.SubprocessError):
            result["gpu_error"] = "NVIDIA driver inspection failed. Update the driver or choose CPU."
    return result


def choose_profile(requested: str, hardware: dict) -> str:
    if requested != "auto":
        return requested
    gpus = hardware.get("gpus", [])
    if not gpus:
        return "cpu"
    return "cuda-modern" if any(gpu["capability"] >= 10 for gpu in gpus) else "cuda"


def compatibility_errors(profile: str, hardware: dict) -> list[str]:
    errors = []
    if hardware["system"] not in {"Windows", "Linux"} or hardware["machine"].lower() not in {"amd64", "x86_64"}:
        errors.append("This backend installer supports Windows/Linux x64. Use the separate web viewer on macOS/ARM.")
    if hardware["system"] == "Windows" and hardware.get("version"):
        try:
            if int(hardware["version"].split(".")[-1]) < 19045:
                errors.append("Ollama requires Windows 10 22H2 (build 19045) or newer.")
        except ValueError:
            errors.append("Could not verify Windows version; Windows 10 22H2 or newer is required.")
    if profile.startswith("cuda"):
        gpus = hardware.get("gpus", [])
        if not gpus:
            errors.append("No supported NVIDIA driver was detected. Install the NVIDIA driver or choose --profile cpu.")
        for gpu in gpus:
            # Conservative release policy matching the CUDA wheel's toolkit generation.
            required = 570 if profile == "cuda-modern" else 551
            if int(gpu["driver"].split(".")[0]) < required:
                errors.append(f"{gpu['name']}: update NVIDIA driver to {required}+ for {profile}.")
            if profile == "cuda" and gpu["capability"] >= 10:
                errors.append("This GPU needs --profile cuda-modern (CUDA 12.8 wheels).")
            if gpu["capability"] < 5:
                errors.append("This GPU is too old for the supported CUDA/Ollama setup. Choose CPU.")
    return errors


def free_gib(path: Path) -> float:
    while not path.exists() and path != path.parent:
        path = path.parent
    return shutil.disk_usage(path).free / 1024**3
