"""
Runner: for each run folder with metadata.json, compute bbox_composition for every extract
(lines from center of each bbox side to the image edge). Writes updated metadata back to metadata.json.
"""
import json
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

from PIL import Image

from scripts.orchestration.progress import progress_steps, start_step
from scripts.utilities.save_stage_utils import METADATA_FILENAME
from scripts.extensions import (
    find_metadata_files,
    get_image_path_from_metadata,
    add_bbox_composition_to_extracts,
)

DEFAULT_METADATA_FOLDER = Path(sys._project_root) / "outputs"


def execute(
    run_folder: Path | None = None,
    metadata_folder: Path | None = None,
    *,
    decimals: int = 2,
) -> list[Path]:
    """
    Add bbox_composition to each extract; write metadata back.
    If run_folder is set, process only that run. Else scan metadata_folder for metadata.json files.
    Returns list of run folder paths updated.
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
        try:
            with Image.open(image_path) as im:
                image_width, image_height = im.width, im.height
        except (OSError, AttributeError):
            print(f"Skip {meta_path}: could not read image size from {image_path}")
            continue
        with open(meta_path, encoding="utf-8") as f:
            metadata = json.load(f)
        if not metadata.get("extracts"):
            print(f"Skip {meta_path}: no extracts")
            continue
        progress_steps("Measure regions", "Save metadata")
        updated = add_bbox_composition_to_extracts(
            metadata, image_width, image_height, decimals=decimals
        )
        start_step(2)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(updated, f, indent=2, ensure_ascii=False)
        run_folder = meta_path.parent
        print(f"Bbox composition: {image_path} -> {run_folder} (metadata updated)")
        written.append(run_folder)
    return written


def main() -> None:
    run_folders = execute()
    print(f"\nDone. bbox_composition in metadata for {len(run_folders)} folder(s).")


if __name__ == "__main__":
    main()
