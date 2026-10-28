"""Guided backend installation. --dry-run performs no downloads or writes."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from install.environment import STATE_FILE, configure_environment, default_cache, python_path
from install.hardware import PROFILES, choose_profile, compatibility_errors, free_gib, inspect_hardware


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["auto", *PROFILES], default="auto")
    parser.add_argument("--dry-run", action="store_true", help="Print the complete plan; no downloads, installs, files or services.")
    parser.add_argument("--yes", action="store_true", help="Use defaults without interactive questions; provide HF_TOKEN separately.")
    parser.add_argument("--skip-models", action="store_true", help="Install software only; fetch models later with install/models.py.")
    parser.add_argument("--with-forgery", action="store_true", help="Advanced PhotoHolmes add-on; native compiler may be needed.")
    parser.add_argument("--with-synthetic", action="store_true", help="Install nonescape in its own CPU environment.")
    parser.add_argument("--cache-dir", type=Path, help="Store new HF/Torch/forgery model downloads on another drive.")
    parser.add_argument("--allow-low-disk", action="store_true", help="Advanced: bypass the conservative disk allowance when reusing existing caches.")
    parser.add_argument("--uv", default=shutil.which("uv"), help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def install_commands(args, profile: str, uv: str) -> list[list[str]]:
    torch, vision, wheel = PROFILES[profile]
    py = str(python_path(ROOT / ".venv"))
    constraints = str(ROOT / "requirements" / f"{profile}.txt")
    commands = [
        [uv, "venv", "--python", "3.11", str(ROOT / ".venv")],
        [uv, "pip", "install", "--python", py, f"torch=={torch}", f"torchvision=={vision}", "--index-url", f"https://download.pytorch.org/whl/{wheel}"],
        [uv, "pip", "install", "--python", py, "-r", str(ROOT / "requirements/core.txt"), "-c", constraints],
    ]
    if args.with_forgery:
        commands.append([sys.executable, "-B", str(ROOT / "install/forgery_source.py")])
        commands.append([uv, "pip", "install", "--python", py, "-r", str(ROOT / "requirements/forgery.txt"), "-c", str(ROOT / "requirements/core.txt"), "-c", constraints])
    if args.with_synthetic:
        synthetic_py = str(python_path(ROOT / ".venv-synthetic"))
        commands.extend([
            [uv, "venv", "--python", "3.11", str(ROOT / ".venv-synthetic")],
            [uv, "pip", "install", "--python", synthetic_py, "torch==2.7.1", "torchvision==0.22.1", "--index-url", "https://download.pytorch.org/whl/cpu"],
            [uv, "pip", "install", "--python", synthetic_py, "-r", str(ROOT / "requirements/synthetic.txt"), "-c", str(ROOT / "requirements/synthetic-torch.txt")],
            [uv, "pip", "check", "--python", synthetic_py],
        ])
    commands.append([uv, "pip", "check", "--python", py])
    return commands


def show_plan(args, profile, hardware, commands):
    print("\nIMAGE FORENSICS ANALYSIS - BACKEND SETUP\n")
    print(f"Computer: {hardware['system']} {hardware['machine']}")
    for gpu in hardware["gpus"]:
        print(f"GPU: {gpu['name']} / {gpu['memory_mib'] // 1024} GiB VRAM / driver {gpu['driver']}")
    print(f"Install profile: {profile}")
    print("Core: 12 stages, including objects, parts, depth, pose, saliency and structured descriptions/rankings.")
    print(f"Optional forgery: {'included' if args.with_forgery else 'off'}; synthetic: {'included in a separate environment' if args.with_synthetic else 'off'}")
    print("CPU runs are supported by these libraries but can be very slow. NVIDIA with 12+ GiB VRAM and 32 GiB RAM is a practical target, not a guaranteed minimum.")
    print("Allow roughly 50 GiB for a fresh software/runtime/core-model setup, plus result files; optional models need more. This is a conservative planning allowance, not a measured download total.")
    print(f"Checkout free space: {free_gib(ROOT):.1f} GiB")
    print(f"Additional model cache: {args.cache_dir or default_cache()}")
    print("Existing HF_HOME/TORCH_HOME are respected. Ollama keeps its own model library (OLLAMA_MODELS).")
    print("Model access: https://huggingface.co/facebook/sam3 (accept access terms in your browser first).")
    print("Get a read token: https://huggingface.co/settings/tokens . The token is never written into this project.")
    print("Ollama: https://ollama.com/download (0.12.7+ required for Qwen3-VL).")
    if args.with_forgery:
        print("PhotoHolmes is built from pinned upstream source with one packaging patch: opencv-python becomes opencv-python-headless. Its source code and LICENSE are retained.")
        print("PhotoHolmes is an advanced add-on. On Linux install build-essential/python3-dev/libjpeg-dev; Windows may need Visual Studio C++ Build Tools. Native builds are not guaranteed.")
    print("\nPackage commands:")
    for command in commands:
        print("  " + subprocess.list2cmdline(command))
    print("\nModels: " + ("SKIPPED; run install/models.py later." if args.skip_models else "SAM3, Marigold, OpenPose, DeepGaze backbones and Qwen3-VL will be fetched into local caches."))


def disk_errors(args) -> list[str]:
    """Planning allowances, deliberately conservative before making large writes."""
    if args.allow_low_disk:
        return []
    package_budget = 10 + (6 if args.with_synthetic else 0) + (2 if args.with_forgery else 0)
    errors = []
    if free_gib(ROOT) < package_budget:
        errors.append(f"At least {package_budget} GiB free is required by the software-install allowance. Choose another drive.")
    if not args.skip_models:
        # Ollama and CLIP can remain on the home drive even with a custom HF cache.
        shared = Path(os.environ.get("HF_HOME", args.cache_dir or Path.home()))
        budget = 50 + (10 if args.with_synthetic or args.with_forgery else 0)
        if free_gib(shared) < budget:
            errors.append(f"Model-cache drive has less than the {budget} GiB planning allowance. Use --skip-models or --cache-dir on a larger drive; existing-cache users may explicitly choose --allow-low-disk.")
        ollama = Path(os.environ.get("OLLAMA_MODELS", Path.home() / ".ollama" / "models"))
        if free_gib(ollama) < 16:
            errors.append("Ollama's model drive needs a 16 GiB allowance for the two Qwen models. Set OLLAMA_MODELS to a larger drive before installing Ollama.")
    return errors


def write_configuration(args, profile, cache):
    from scripts.configuration.store import ConfigStore
    from scripts.orchestration.stages import STAGES
    config = ConfigStore()
    optional = {"forgery_detection": args.with_forgery, "synthetic_analysis": args.with_synthetic}
    selected = [stage.key for stage in STAGES if optional.get(stage.key, True)]
    config.update({"settings": {"device": "cpu" if profile == "cpu" else "cuda", "selected_stages": selected}})
    state = {"version": 1, "profile": profile, "cache_dir": str(cache), "custom_cache": args.cache_dir is not None,
             "forgery": args.with_forgery, "synthetic": args.with_synthetic, "models_prepared": False}
    STATE_FILE.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    args = arguments(argv)
    hardware = inspect_hardware()
    profile = choose_profile(args.profile, hardware)
    commands = install_commands(args, profile, args.uv or "uv")
    show_plan(args, profile, hardware, commands)
    blockers = compatibility_errors(profile, hardware) + disk_errors(args)
    for message in blockers:
        print(f"CHECK: {message}")
    if args.dry_run:
        print("\nDRY RUN COMPLETE. No files changed; no packages/models downloaded; no services started.")
        return 0
    if blockers:
        return 2
    if not args.uv:
        print("Run Install.cmd (Windows) or bash install.sh (Linux) to install the Python bootstrap automatically.")
        return 2
    cache = (args.cache_dir or default_cache()).expanduser().resolve()
    if args.cache_dir and (cache == ROOT or ROOT in cache.parents):
        print("Choose a model cache outside this repository so weights cannot enter Git.")
        return 2
    if not args.yes and input("\nInstall the software" + (" and download models" if not args.skip_models else "") + "? [Y/n] ").strip().lower() not in {"", "y", "yes"}:
        print("Setup cancelled. No packages installed.")
        return 0
    # Avoid creating/resetting an existing environment. Rerunning setup resumes installs.
    env = configure_environment(args.cache_dir)
    env["UV_NO_CACHE"] = "1"
    try:
        for index, command in enumerate(commands, 1):
            if command[1] == "venv" and python_path(Path(command[-1])).is_file():
                print(f"[{index}/{len(commands)}] Reusing {command[-1]}")
                continue
            print(f"\n[{index}/{len(commands)}] " + subprocess.list2cmdline(command), flush=True)
            subprocess.run(command, cwd=ROOT, env=env, check=True)
        write_configuration(args, profile, cache)
        # Offline import checks identify missing classes before any model downloads.
        subprocess.run([str(python_path(ROOT / ".venv")), "-B", str(ROOT / "install/check.py"), "--imports"], cwd=ROOT, env=env, check=True)
        if not args.skip_models:
            command = [str(python_path(ROOT / ".venv")), "-B", str(ROOT / "install/models.py")]
            if args.yes:
                command.append("--yes")
            if args.allow_low_disk:
                command.append("--allow-low-disk")
            subprocess.run(command, cwd=ROOT, env=env, check=True)
            state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            state["models_prepared"] = True
            STATE_FILE.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"\nSetup did not finish: {exc}")
        print("Your completed downloads remain cached. Fix the displayed issue and run the same installer again.")
        print("For a software-only install, add --skip-models. No analysis API was started.")
        return 1
    print("\nSetup complete. Use Start.cmd or bash start.sh when you want to run the backend.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
