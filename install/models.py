"""Explicit, resumable model preparation. --dry-run never contacts the network."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import getpass
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from install.environment import STATE_FILE, configure_environment, read_state
from install.hardware import free_gib

OLLAMA_MODELS = ("qwen3-vl:8b-instruct", "qwen3-vl:8b-thinking")
GROUPS = ("sam3", "depth", "pose", "saliency", "ollama", "forgery", "synthetic")


def selected_groups(group: str, state: dict) -> list[str]:
    if group != "all":
        return [group]
    return ["sam3", "depth", "pose", "saliency", "ollama"] + (["forgery"] if state.get("forgery") else []) + (["synthetic"] if state.get("synthetic") else [])


def ollama_binary() -> str | None:
    if os.name != "nt":
        from install.ollama_runtime import private_binary
        private = private_binary()
        if private.is_file():
            return str(private)
    binary = shutil.which("ollama")
    if binary:
        return binary
    if os.name == "nt":
        candidate = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"
        if candidate.is_file():
            return str(candidate)
    return None


def install_ollama(yes: bool) -> str:
    binary = ollama_binary()
    if binary:
        return binary
    print("Ollama is the local vision-model service. It needs additional disk space for its app and models.")
    if not yes and input("Install Ollama from its official distributor now? [Y/n] ").strip().lower() not in {"", "y", "yes"}:
        raise RuntimeError("Install Ollama from https://ollama.com/download and rerun model setup.")
    if os.name == "nt":
        winget = shutil.which("winget")
        if winget:
            subprocess.run([winget, "install", "--id", "Ollama.Ollama", "--exact", "--accept-package-agreements", "--accept-source-agreements"], check=True)
        else:
            # Official user installer; interactive UI handles Windows permissions.
            with tempfile.TemporaryDirectory(prefix="image-forensics-ollama-") as temporary:
                installer = Path(temporary) / "OllamaSetup.exe"
                urllib.request.urlretrieve("https://ollama.com/download/OllamaSetup.exe", installer)
                subprocess.run([str(installer)], check=True)
    else:
        from install.ollama_runtime import install_linux_runtime
        return install_linux_runtime()
    binary = ollama_binary()
    if not binary:
        raise RuntimeError("Ollama installation finished but the command was not found. Reopen your terminal and rerun model setup.")
    return binary


def ollama_ready() -> bool:
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/version", timeout=2) as response:
            version = json.load(response).get("version", "0")
        numbers = tuple(int(part) for part in version.split("-")[0].split(".")[:3])
        if numbers < (0, 12, 7):
            raise RuntimeError("Update Ollama to version 0.12.7 or newer for Qwen3-VL.")
        return True
    except (OSError, ValueError):
        return False


@contextmanager
def ollama_session(binary: str):
    """Stop only the temporary server we started. Never terminate a user's service."""
    owned = None
    try:
        if not ollama_ready():
            owned = subprocess.Popen([binary, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), start_new_session=os.name != "nt")
            for _ in range(60):
                if ollama_ready():
                    break
                if owned.poll() is not None:
                    raise RuntimeError("Ollama could not start. Open Ollama manually and rerun model setup.")
                time.sleep(1)
            else:
                raise RuntimeError("Ollama was not ready after 60 seconds.")
        yield
    finally:
        if owned is not None and owned.poll() is None:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(owned.pid), "/T", "/F"], capture_output=True, check=False)
            else:
                try:
                    os.killpg(owned.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            try:
                owned.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if os.name == "nt":
                    owned.kill()
                else:
                    try:
                        os.killpg(owned.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                owned.wait()


def prepare_huggingface(yes: bool):
    from huggingface_hub import get_token, login
    if get_token():
        return
    print("1. Open https://huggingface.co/facebook/sam3 and accept/request access using your own account.")
    print("2. Open https://huggingface.co/settings/tokens and create a read token with access to this model.")
    print("3. Paste the token below. It is hidden and stored in Hugging Face's user cache, never this repository.")
    if yes:
        raise RuntimeError("SAM3 needs access and a token. Set HF_TOKEN for this process or run setup interactively.")
    token = getpass.getpass("Hugging Face read token: ").strip()
    if not token:
        raise RuntimeError("No token entered. Run model setup again after obtaining SAM3 access.")
    login(token=token, add_to_git_credential=False)


def prepare_group(group: str, yes: bool):
    if group == "ollama":
        binary = install_ollama(yes)
        with ollama_session(binary):
            for model in OLLAMA_MODELS:
                subprocess.run([binary, "pull", model], check=True)
        return
    from huggingface_hub import hf_hub_download, snapshot_download
    if group == "sam3":
        prepare_huggingface(yes)
        snapshot_download("facebook/sam3", allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken"], ignore_patterns=["*tracker*", "*video*"])
    elif group == "depth":
        snapshot_download("prs-eth/marigold-depth-v1-1", allow_patterns=["*.json", "*.txt", "*.safetensors"], ignore_patterns=["*.fp16.*", "*.non_ema.*"])
    elif group == "pose":
        for filename in ("body_pose_model.pth", "hand_pose_model.pth", "facenet.pth"):
            hf_hub_download("lllyasviel/Annotators", filename)
    elif group == "saliency":
        # Fetch CLIP's checksum-verified asset without constructing its large model.
        from clip.clip import _download, _MODELS
        _download(_MODELS["RN50x64"], str(Path.home() / ".cache" / "clip"))
        import torch
        # Upstream pins DINOv2 source to this revision. The CPU model is released immediately.
        model = torch.hub.load("facebookresearch/dinov2:6a62615", "dinov2_vitb14", skip_validation=True, trust_repo=True)
        del model
        checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / "deepgazemsdb.pth"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        if not checkpoint.is_file():
            temporary = checkpoint.with_suffix(".download")
            try:
                torch.hub.download_url_to_file("https://github.com/matthias-k/DeepGaze/releases/download/v1.2.0/deepgazemsdb.pth", str(temporary))
                temporary.replace(checkpoint)
            finally:
                temporary.unlink(missing_ok=True)
    elif group == "synthetic":
        hf_hub_download("e3ntity/nonescape-v0", "nonescape-v0.safetensors")
    elif group == "forgery":
        import ensure_path  # initializes sys._project_root for the runtime extensions
        from scripts.extensions.forgery_detection_extensions import AVAILABLE_METHODS, ensure_weights
        for method in AVAILABLE_METHODS:
            ensure_weights(method)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=("all", *GROUPS), default="all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--allow-low-disk", action="store_true", help="Advanced: bypass model-cache planning allowance when reusing existing files.")
    args = parser.parse_args(argv)
    groups = selected_groups(args.group, read_state())
    print("Model preparation: " + ", ".join(groups))
    print("Hugging Face/Torch caches and Ollama's library are reused. CLIP uses ~/.cache/clip.")
    if args.dry_run:
        print("DRY RUN: no imports of ML packages, network requests, writes or services.")
        return 0
    configure_environment()
    if not args.allow_low_disk:
        allowances = {"sam3": 5, "depth": 6, "pose": 1, "saliency": 4, "ollama": 16, "forgery": 3, "synthetic": 3}
        locations = {
            "sam3": Path(os.environ["HF_HOME"]), "depth": Path(os.environ["HF_HOME"]),
            "pose": Path(os.environ["HF_HOME"]), "synthetic": Path(os.environ["HF_HOME"]),
            "saliency": Path(os.environ["TORCH_HOME"]),
            "ollama": Path(os.environ.get("OLLAMA_MODELS", Path.home() / ".ollama" / "models")),
            "forgery": Path(os.environ["IMAGE_FORENSICS_WEIGHTS"]),
        }
        for group in groups:
            if free_gib(locations[group]) < allowances[group]:
                print(f"Insufficient planning space for {group}: allow {allowances[group]} GiB on {locations[group]}. Move the cache or explicitly use --allow-low-disk for existing cached files.")
                return 2
    for index, group in enumerate(groups, 1):
        print(f"\n[{index}/{len(groups)}] Preparing {group}", flush=True)
        try:
            prepare_group(group, args.yes)
        except Exception as exc:
            print(f"Could not prepare {group}: {exc}")
            print("Previously downloaded files remain cached. Rerun this command to continue.")
            print("For SAM3 access errors, accept the model's terms with the same Hugging Face account as your token.")
            return 1
    if args.group == "all" and STATE_FILE.exists():
        state = read_state()
        state["models_prepared"] = True
        STATE_FILE.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    print("Models prepared. No image analysis was run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
