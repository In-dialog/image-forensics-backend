"""
Runner: find metadata.json in a folder, read the main image path (source_image_path),
run composition/leading lines on that image, save lines as JSON (line1 start [x,y] end [x,y], etc.).
"""
import sys
from pathlib import Path

# Single source of truth: ensure_path.py at project root (see ensure_path.py)
_p = Path(__file__).resolve().parent
for _ in range(6):
    _ep = _p / "ensure_path.py"
    if _ep.exists():
        exec(open(_ep, encoding="utf-8").read(), {"__file__": str(_ep), "sys": sys, "Path": Path})
        break
    _p = _p.parent
else:
    raise RuntimeError("ensure_path.py not found above this file")

from scripts.utilities.save_stage_utils import (
    COMPOSITION_SUBFOLDER,
    METADATA_FILENAME,
    run_composition_lines_and_save,
)
from scripts.extensions import find_metadata_files, get_image_path_from_metadata

DEFAULT_METADATA_FOLDER = Path(sys._project_root) / "outputs"


def execute(
    run_folder: Path | None = None,
    metadata_folder: Path | None = None,
    *,
    window_size: int = 64,
    decimals: int = 2,
) -> list[Path]:
    """
    Run composition lines; save composition_lines.json and composition_lines.webp in run_folder/composition/.
    If run_folder is set, process only that run. Else scan metadata_folder for metadata.json.
    Returns list of run folder paths written.
    """
    if run_folder is not None:
        meta_path = run_folder / METADATA_FILENAME
        meta_files = [meta_path] if meta_path.exists() else []
    else:
        folder = metadata_folder if metadata_folder is not None else DEFAULT_METADATA_FOLDER
        meta_files = find_metadata_files(folder)
    if not meta_files:
        base = run_folder if run_folder is not None else (metadata_folder or DEFAULT_METADATA_FOLDER)
        print(f"No {METADATA_FILENAME} found at {base}")
        return []

    written: list[Path] = []
    for meta_path in meta_files:
        image_path = get_image_path_from_metadata(meta_path)
        if image_path is None:
            print(f"Skip {meta_path}: no valid image path")
            continue
        run_folder = meta_path.parent
        print(f"Composition lines: {image_path} -> {run_folder}")
        out_path = run_composition_lines_and_save(
            image_path,
            run_folder,
            window_size=window_size,
            decimals=decimals,
        )
        comp_dir = run_folder / COMPOSITION_SUBFOLDER
        print(f"  Saved {comp_dir.name}/ (JSON + image)")
        written.append(run_folder)
    return written


def main() -> None:
    run_folders = execute()
    print(f"\nDone. Composition lines (JSON + image) in {len(run_folders)} folder(s).")


if __name__ == "__main__":
    main()
