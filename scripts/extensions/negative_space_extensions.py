"""
Negative space extension: subtract all segments (using their alpha) from the original image.
Output is black-and-white: white = what remains after subtraction (negative space),
black = what was subtracted (segment areas).
"""
import json
from pathlib import Path
from scripts.orchestration.progress import progress_items, skip_item

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation

from scripts.utilities.save_stage_utils import (
    IMAGE_HEIGHT_KEY,
    IMAGE_WIDTH_KEY,
    METADATA_FILENAME,
    SEGMENTS_SUBFOLDER,
)
from scripts.utilities.image_utils import segment_mask_in_full_image


def _image_size(source: Path | np.ndarray) -> tuple[int, int] | None:
    """Return (height, width) of the image. Path or (H,W) or (H,W,C) array."""
    if isinstance(source, np.ndarray):
        arr = np.asarray(source)
        return (int(arr.shape[0]), int(arr.shape[1]))
    try:
        with Image.open(source) as im:
            return (im.height, im.width)
    except (OSError, AttributeError):
        return None


def compute_negative_space_image(
    run_folder: Path,
    source_image: Path | np.ndarray,
    *,
    dilate_pixels: int = 3,
) -> np.ndarray | None:
    """
    Original image minus masked objects → negative space, at original size and proportion.

    Build the negative-space mask as black-and-white: white = pixels left after
    subtracting all segments (background), black = segment areas. The returned image
    has exactly the same size and proportion as the original (no resizing). Use the
    same source image as was used for extraction so bbox alignment is correct.

    Optionally dilate the object mask by dilate_pixels before inverting, so the
    cutout boundaries expand slightly and thin edge artifacts (from soft alpha or
    antialiasing) are removed from the negative space.

    Args:
        run_folder: Run directory containing metadata.json and segments/.
        source_image: Path to the original image or array (used for dimensions only).
        dilate_pixels: Pixels to expand the object mask (0 = no change). Default 2.

    Returns:
        (H, W) uint8 image, same dimensions as original: 255 = negative space, 0 = subtracted.
        None if metadata/extracts missing or no valid segments.
    """
    meta_path = run_folder / METADATA_FILENAME
    if not meta_path.exists():
        return None
    with open(meta_path, encoding="utf-8") as f:
        metadata = json.load(f)
    extracts = metadata.get("extracts")
    if not extracts:
        return None

    # Use extraction-time dimensions when present so canvas matches bbox coordinate space
    # (avoids "pushed left" when the image file was replaced or has different size).
    meta_w = metadata.get(IMAGE_WIDTH_KEY)
    meta_h = metadata.get(IMAGE_HEIGHT_KEY)
    if meta_w is not None and meta_h is not None and meta_w > 0 and meta_h > 0:
        full_h, full_w = int(meta_h), int(meta_w)
    else:
        # Fallback for metadata from before we stored dimensions: use bbox extent so we
        # don't build a canvas wider than the bbox space (which causes left shift).
        max_x2 = max((e["bbox"][2] for e in extracts if e.get("bbox") and len(e["bbox"]) == 4), default=0)
        max_y2 = max((e["bbox"][3] for e in extracts if e.get("bbox") and len(e["bbox"]) == 4), default=0)
        if max_x2 > 0 and max_y2 > 0:
            full_w, full_h = max_x2, max_y2
        else:
            size = _image_size(source_image)
            if size is None:
                return None
            full_h, full_w = size
    segments_dir = run_folder / SEGMENTS_SUBFOLDER

    combined_mask = np.zeros((full_h, full_w), dtype=np.float64)
    for entry in progress_items(extracts, step=1, label=lambda item: item.get("name", "Unnamed item")):
        name = entry.get("name")
        bbox = entry.get("bbox")
        if not name or not bbox or len(bbox) != 4:
            skip_item()
            continue
        seg_path = segments_dir / f"{name}.webp"
        if not seg_path.exists():
            seg_path = segments_dir / f"{name}.png"
        mask = segment_mask_in_full_image(seg_path, bbox, full_h, full_w)
        if mask is not None:
            combined_mask = np.maximum(combined_mask, mask)
        else:
            skip_item()

    combined_mask = np.clip(combined_mask, 0.0, 1.0)
    # Optionally dilate the object mask so cutout edges expand and edge artifacts
    # (soft alpha, antialiasing) are removed from the negative space.
    if dilate_pixels > 0:
        object_mask = (combined_mask > 0.5).astype(np.uint8)
        combined_mask = binary_dilation(object_mask, iterations=dilate_pixels).astype(np.float64)
    # White = left after subtraction (negative space), black = subtracted (segments)
    negative_mask = 1.0 - combined_mask
    # Strict binary: only 0 or 255 (no gray)
    out = ((negative_mask > 0.5).astype(np.uint8)) * 255
    # Out shape is (full_h, full_w) = same size and proportion as original image
    return np.ascontiguousarray(out)
