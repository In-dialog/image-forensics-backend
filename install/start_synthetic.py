"""Portable optional service entrypoint, always using its isolated environment."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from install.environment import configure_environment, python_path


def service_command() -> tuple[list[str], dict[str, str]]:
    """Return the direct command without launching it (also used by the API owner)."""
    python = python_path(ROOT / ".venv-synthetic")
    if not python.is_file():
        raise RuntimeError("Synthetic detection is not installed. Rerun Install.cmd / bash install.sh with --with-synthetic.")
    return [str(python), "-B", str(ROOT / "scripts/services/synthetic_api_server.py")], configure_environment()


def main():
    try:
        command, environment = service_command()
        return subprocess.call([*command, *sys.argv[1:]], cwd=ROOT, env=environment)
    except KeyboardInterrupt:
        return 0
    except RuntimeError as exc:
        print(exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
