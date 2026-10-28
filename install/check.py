"""Read-only diagnostics. Never loads model weights or starts a service."""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from install.environment import configure_environment, read_state
from install.hardware import choose_profile, compatibility_errors, inspect_hardware


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imports", action="store_true", help="Import ML classes without fetching weights or running inference.")
    args = parser.parse_args(argv)
    hardware = inspect_hardware()
    state = read_state()
    profile = state.get("profile", choose_profile("auto", hardware))
    errors = compatibility_errors(profile, hardware)
    print(json.dumps({"hardware": hardware, "profile": profile, "installation": state}, indent=2))
    for name in ["Pillow", "numpy", "torch", "torchvision", "transformers", "diffusers", "ollama", "deepgaze-pytorch", "controlnet-aux"]:
        try:
            print(f"OK {name}: {importlib.metadata.version(name)}")
        except importlib.metadata.PackageNotFoundError:
            errors.append(f"Missing package: {name}. Run the installer.")
    if args.imports:
        configure_environment()
        # Importing a module must not download. These exact classes are used at runtime.
        for module, names in {
            "transformers": ["Sam3Model", "Sam3Processor"],
            "diffusers": ["MarigoldDepthPipeline"],
            "controlnet_aux.open_pose": ["OpenposeDetector"],
            "deepgaze_pytorch": ["DeepGazeMSDB", "MSDBDataset"],
        }.items():
            try:
                loaded = importlib.import_module(module)
                for name in names:
                    getattr(loaded, name)
                print(f"OK {module} imports")
            except Exception as exc:
                errors.append(f"Cannot import {module}: {exc}")
        if profile.startswith("cuda"):
            try:
                import torch
                if not torch.cuda.is_available():
                    errors.append("CUDA wheel installed but torch cannot use the GPU. Check the NVIDIA driver.")
            except ImportError:
                pass
    for error in errors:
        print("CHECK: " + error)
    print("No model weights were loaded. No services were started.")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
