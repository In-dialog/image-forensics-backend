#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# Must happen before curl, mkdir, uv, or any Python bootstrap.
for argument in "$@"; do
  if [[ "$argument" == "--dry-run" || "$argument" == "--help" || "$argument" == "-h" ]]; then
    if command -v python3 >/dev/null 2>&1; then
      exec python3 -B "$PROJECT_ROOT/install/setup.py" "$@"
    fi
    printf '%s\n' 'Inspection only: no writes, downloads, installation or services.' \
      'Plan: private Python 3.11; choose CPU/NVIDIA wheels; install pinned core requirements; guide Ollama, SAM3 access and model downloads.' \
      'Options: --profile cpu|cuda|cuda-modern, --skip-models, --with-forgery, --with-synthetic, --cache-dir PATH.'
    exit 0
  fi
done
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
  printf '%s\n' 'This backend installer supports Linux x64. Use the separate frontend on macOS.' >&2
  exit 2
fi
BOOTSTRAP_FREE_KB="$(df -Pk "$PROJECT_ROOT" | awk 'NR == 2 {print $4}')"
if [[ ! "$BOOTSTRAP_FREE_KB" =~ ^[0-9]+$ ]] || (( BOOTSTRAP_FREE_KB < 1048576 )); then
  printf '%s\n' 'Free at least 1 GiB for the Python bootstrap, or move this folder to a larger drive. Nothing has been downloaded.' >&2
  exit 2
fi
TOOLS_DIR="$PROJECT_ROOT/.tools"
UV="$TOOLS_DIR/uv"
if [[ ! -x "$UV" ]]; then
  command -v curl >/dev/null || { printf '%s\n' 'Install curl using your distribution package manager, then retry.' >&2; exit 2; }
  mkdir -p "$TOOLS_DIR"
  command -v sha256sum >/dev/null || { printf '%s\n' 'Install coreutils (sha256sum), then retry.' >&2; exit 2; }
  UV_ARCHIVE="$TOOLS_DIR/uv-0.9.7.tar.gz"
  curl --proto '=https' --tlsv1.2 -LsSf 'https://github.com/astral-sh/uv/releases/download/0.9.7/uv-x86_64-unknown-linux-gnu.tar.gz' -o "$UV_ARCHIVE"
  printf '%s  %s\n' 'b26fcc8dfa1c39b5a5613445af3be3eefda45d9a39359bee271eafe34913583e' "$UV_ARCHIVE" | sha256sum --check --status
  tar -xzf "$UV_ARCHIVE" -C "$TOOLS_DIR" --strip-components=1
fi
export UV_PYTHON_INSTALL_DIR="$TOOLS_DIR/python"
export UV_NO_CACHE=1
"$UV" python install 3.11
PYTHON_PATH="$("$UV" python find --managed-python 3.11)"
exec "$PYTHON_PATH" -B "$PROJECT_ROOT/install/setup.py" --uv "$UV" "$@"
