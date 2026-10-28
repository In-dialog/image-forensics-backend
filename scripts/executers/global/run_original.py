"""
Runner: for each run folder with metadata.json, take the original (source) image,
convert to WebP, and save to run_folder/original/original.webp.
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

from scripts.utilities.save_stage_utils import METADATA_FILENAME
from scripts.orchestration.progress import progress_steps, start_step
from scripts.utilities.image_utils import ensure_dir, load_rgb
from scripts.extensions import find_metadata_files, get_image_path_from_metadata

DEFAULT_METADATA_FOLDER = Path(sys._project_root) / "outputs"
ORIGINAL_SUBFOLDER = "original"
ORIGINAL_FILENAME = "original.webp"


def execute(
    run_folder: Path | None = None,
    metadata_folder: Path | None = None,
    *,
    quality: int = 90,
) -> list[Path]:
    """
    Copy original image as WebP to run_folder/original/original.webp.
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
        out_dir = run_folder / ORIGINAL_SUBFOLDER
        ensure_dir(out_dir)
        out_path = out_dir / ORIGINAL_FILENAME
        progress_steps("Load original image", "Save preview")
        start_step(1)
        img = load_rgb(image_path)
        start_step(2)
        img.save(out_path, "WEBP", quality=quality)
        print(f"Original: {image_path} -> {out_path}")
        written.append(run_folder)
    return written


def main() -> None:
    run_folders = execute()
    print(f"\nDone. Original WebP in {len(run_folders)} folder(s).")


if __name__ == "__main__":
    main()
