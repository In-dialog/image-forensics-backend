"""Portable optional service entrypoint, preferring its isolated environment."""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from install.environment import configure_environment, python_path
from scripts.runtime.python_environment import missing_synthetic_dependencies


def resolve_python() -> Path:
    explicit = os.environ.get("SYNTHETIC_PYTHON")
    candidates = [Path(explicit).expanduser()] if explicit else [
        python_path(ROOT / ".venv-synthetic"), python_path(ROOT / ".venv"),
    ]
    problems = []
    for python in candidates:
        if not python.is_absolute():
            python = ROOT / python
        if not python.is_file():
            problems.append(f"{python}: interpreter not found")
            continue
        missing = missing_synthetic_dependencies(python)
        if not missing:
            # Preserve POSIX venv symlinks so Python finds that venv's packages.
            return python.absolute()
        problems.append(f"{python}: {missing}")
    raise RuntimeError(
        "Synthetic detection has no usable Python environment. "
        "Rerun Install.cmd / bash install.sh with --with-synthetic, "
        "or set SYNTHETIC_PYTHON to an existing interpreter with Nonescape installed. "
        "Nothing was installed or downloaded.\n" + "\n".join(problems)
    )


def service_command() -> tuple[list[str], dict[str, str]]:
    """Return the direct command without launching it (also used by the API owner)."""
    python = resolve_python()
    return [str(python), "-B", "-u", str(ROOT / "scripts/services/synthetic_api_server.py")], configure_environment()


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
