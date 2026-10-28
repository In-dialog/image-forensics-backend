"""
Single source of truth: add project root to sys.path so "import scripts" works when
runners are executed directly. Also sets sys._project_root for prompt paths etc.

Runners in scripts/executers/ run this before any "from scripts..." by finding this file
up the directory tree and exec'ing it. Use this block at the top of each runner (after
import sys / pathlib.Path):

  _p = Path(__file__).resolve().parent
  for _ in range(6):
      _ep = _p / "ensure_path.py"
      if _ep.exists():
          exec(open(_ep, encoding="utf-8").read(), {"__file__": str(_ep), "sys": sys, "Path": Path})
          break
      _p = _p.parent
  else:
      raise RuntimeError("ensure_path.py not found above this file")
"""
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
sys._project_root = _PROJECT_ROOT
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
