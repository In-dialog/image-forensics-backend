"""
Runner: find metadata.json in a hardcoded folder, read the main image path (source_image_path),
run depth pass on that image only (not on segment crops), save depth in run_folder/depth/,
then add x, y, z to each extract in metadata using segment masks + depth map.
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
from scripts.extensions import add_z_to_extracts, find_metadata_files, get_image_path_from_metadata, run_depth_and_save

DEFAULT_METADATA_FOLDER = Path(sys._project_root) / "outputs"


def execute(
    run_folder: Path | None = None,
    metadata_folder: Path | None = None,
    *,
    model_id: str = "prs-eth/marigold-depth-v1-1",
    device: str | None = None,
) -> list[Path]:
    """
    Run depth on the main image; save depth.webp in run_folder/depth/; add x,y,z to extracts.
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
        print(f"Depth: {image_path} -> {run_folder}")
        run_depth_and_save(
            image_path,
            run_folder,
            model_id=model_id,
            device=device,
        )
        # Second pass: add x, y, z to each extract using segment masks + depth map
        if add_z_to_extracts(run_folder) is not None:
            print(f"  Updated metadata with x, y, z")
        written.append(run_folder)
    return written


def main() -> None:
    run_folders = execute()
    print(f"\nDone. Depth outputs in {len(run_folders)} folder(s).")


if __name__ == "__main__":
    main()
