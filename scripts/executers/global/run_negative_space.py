"""
Runner: for each run folder with metadata.json, compute negative-space image (original minus
all segments by alpha) and save to run_folder/negative_space/negative_space.webp.
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

from PIL import Image

from scripts.orchestration.progress import progress_steps, start_step
from scripts.utilities.save_stage_utils import METADATA_FILENAME
from scripts.utilities.image_utils import ensure_dir
from scripts.extensions import (
    find_metadata_files,
    get_image_path_from_metadata,
    compute_negative_space_image,
)

DEFAULT_METADATA_FOLDER = Path(sys._project_root) / "outputs"
NEGATIVE_SPACE_SUBFOLDER = "negative_space"
NEGATIVE_SPACE_FILENAME = "negative_space.webp"


def execute(
    run_folder: Path | None = None,
    metadata_folder: Path | None = None,
) -> list[Path]:
    """
    Compute negative-space image; save to run_folder/negative_space/negative_space.webp.
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
        progress_steps("Combine item masks", "Save negative space")
        start_step(1)
        img_array = compute_negative_space_image(run_folder, image_path)
        if img_array is None:
            print(f"Skip {run_folder}: no extracts or segments")
            continue
        start_step(2)
        out_dir = run_folder / NEGATIVE_SPACE_SUBFOLDER
        ensure_dir(out_dir)
        out_path = out_dir / NEGATIVE_SPACE_FILENAME
        Image.fromarray(img_array, mode="L").convert("RGB").save(out_path, "WEBP", lossless=True)
        print(f"Negative space: {image_path} -> {out_path}")
        written.append(run_folder)
    return written


def main() -> None:
    run_folders = execute()
    print(f"\nDone. Negative space in {len(run_folders)} folder(s).")


if __name__ == "__main__":
    main()
