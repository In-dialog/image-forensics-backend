"""
Composition lines extensions: discover metadata.json files and read source image path.
Used by the composition lines runner; shared by runners that iterate over run subfolders.
"""
import json
from pathlib import Path

from scripts.utilities.save_stage_utils import METADATA_FILENAME, SOURCE_IMAGE_PATH_KEY


def find_metadata_files(folder: Path) -> list[Path]:
    """Return paths to metadata.json under folder (one per run subfolder)."""
    if not folder.exists():
        return []
    return list(folder.glob("*/" + METADATA_FILENAME))


def get_image_path_from_metadata(metadata_path: Path) -> Path | None:
    """Read metadata.json and return the image path; None if missing or invalid."""
    try:
        with open(metadata_path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    path_str = data.get(SOURCE_IMAGE_PATH_KEY) or data.get("source_image_path")
    if not path_str:
        return None
    p = Path(path_str)
    return p if p.exists() else None
