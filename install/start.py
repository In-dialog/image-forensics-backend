"""Start the local API and own only the Ollama process this launcher creates."""
from pathlib import Path
import json
import urllib.request
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from install.environment import configure_environment


def main():
    if any(argument in {"--help", "-h"} for argument in sys.argv[1:]):
        print("Start local Image Forensics API and, if needed, a temporary Ollama server.\nOptions: --port NUMBER --data-dir PATH\nCtrl+C stops the API and only the Ollama server this launcher created.")
        return 0
    configure_environment()
    from install.models import ollama_binary, ollama_session
    binary = ollama_binary()
    if not binary:
        print("Ollama was not found. Run the installer or install/models.py to complete setup.")
        return 2
    from scripts.api.server import main as serve
    try:
        with ollama_session(binary):
            from scripts.configuration.store import ConfigStore
            config = ConfigStore()
            settings, _ = config.snapshot()
            from scripts.configuration.schema import PROMPT_STAGES
            required = {settings["stage_options"][PROMPT_STAGES[stage]]["model"] for stage in settings["selected_stages"] if stage in PROMPT_STAGES}
            with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=5) as response:
                installed = {model["name"] for model in json.load(response).get("models", [])}
            missing = sorted(name for name in required if name not in installed and f"{name}:latest" not in installed)
            if missing:
                print("Missing Ollama models: " + ", ".join(missing))
                print("Complete model setup with install/models.py, or use ollama pull MODEL for a custom model.")
                return 2
            serve()
    except (OSError, RuntimeError) as exc:
        print(f"Backend could not start: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
