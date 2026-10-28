"""
Depth extensions: run depth on main image and save; add z (depth) to extract metadata using
segment masks and the main image depth map. Each segment PNG (with transparency) is used as a
mask; we take the highest depth value in that region as z. We keep using existing bbox_position
for (x, y) and only extend it to [x, y, z].
"""
import json
from pathlib import Path
from scripts.orchestration.progress import progress_steps, progress_items, start_step, skip_item
from typing import Any

import numpy as np
from PIL import Image

from scripts.utilities.save_stage_utils import METADATA_FILENAME, SEGMENTS_SUBFOLDER
from scripts.utilities.image_utils import ensure_dir, segment_mask_in_full_image
from scripts.utilities.depth_utils import extract_depth

# Subfolder name for depth map (must match run_depth_map)
DEPTH_SUBFOLDER = "depth"
DEPTH_FILENAME = "depth.webp"

# Alpha threshold: pixel is "item" if alpha >= this (0-255)
ALPHA_THRESHOLD = 128


def run_depth_and_save(
    image_path: Path,
    run_folder: Path,
    *,
    model_id: str = "prs-eth/marigold-depth-v1-1",
    device: str | None = None,
) -> Path:
    """
    Run depth extraction on the main image only (not segment crops). Save grayscale depth
    in run_folder/depth/. Returns run_folder.
    """
    progress_steps("Estimate depth", "Save depth map", "Measure item depths", "Save metadata")
    start_step(1)
    result = extract_depth(image_path, model_id=model_id, device=device)
    start_step(2)
    depth_folder = run_folder / DEPTH_SUBFOLDER
    ensure_dir(depth_folder)
    Image.fromarray(result.depth_u8).save(depth_folder / DEPTH_FILENAME, "WEBP", quality=90)
    return run_folder


def _load_depth_map(run_folder: Path) -> np.ndarray | None:
    """Load grayscale depth map from run_folder/depth/depth.webp. Returns (H, W) uint8 or None."""
    path = run_folder / DEPTH_SUBFOLDER / DEPTH_FILENAME
    if not path.exists():
        return None
    img = Image.open(path)
    arr = np.array(img)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return np.asarray(arr, dtype=np.uint8)


def _max_depth_in_mask(depth: np.ndarray, mask: np.ndarray, alpha_threshold: float = ALPHA_THRESHOLD / 255.0) -> float:
    """Return the maximum depth value where mask >= alpha_threshold. Returns 0.0 if no pixels."""
    where = mask >= alpha_threshold
    if not np.any(where):
        return 0.0
    return float(np.max(depth[where]))


def add_z_to_extracts(
    run_folder: Path,
    *,
    alpha_threshold: int = ALPHA_THRESHOLD,
    in_place: bool = True,
) -> dict[str, Any] | None:
    """
    Load metadata.json and depth map from run_folder; for each extract, use the segment PNG
    as a mask on the depth map, compute z = max depth in that region.
    Updates each extract's "bbox_position" to [x, y, z] using existing bbox center for x,y and depth for z.

    If in_place is True, overwrites metadata.json. Returns the updated metadata dict, or None
    if depth or metadata is missing.
    """
    meta_path = run_folder / METADATA_FILENAME
    if not meta_path.exists():
        return None
    with open(meta_path, encoding="utf-8") as f:
        metadata = json.load(f)
    extracts = metadata.get("extracts")
    if not extracts:
        return metadata

    depth = _load_depth_map(run_folder)
    if depth is None:
        return None
    full_h, full_w = depth.shape
    depth_norm = depth.astype(np.float64) / 255.0
    alpha_th = alpha_threshold / 255.0

    segments_dir = run_folder / SEGMENTS_SUBFOLDER
    for entry in progress_items(extracts, step=3, label=lambda item: item.get("name", "Unnamed item")):
        name = entry.get("name")
        bbox = entry.get("bbox")
        if not name or not bbox or len(bbox) != 4:
            skip_item()
            continue
        # Segments can be stored as WebP or PNG images.
        seg_path = segments_dir / f"{name}.webp"
        if not seg_path.exists():
            seg_path = segments_dir / f"{name}.png"
        mask = segment_mask_in_full_image(seg_path, bbox, full_h, full_w)
        # Use existing bbox_position (center) for x,y when present; else derive from bbox
        existing_pos = entry.get("bbox_position")
        if existing_pos and len(existing_pos) >= 2:
            x, y = float(existing_pos[0]), float(existing_pos[1])
        else:
            x = (bbox[0] + bbox[2]) / 2.0
            y = (bbox[1] + bbox[3]) / 2.0
        if mask is None:
            z = 0.0
        else:
            z = round(_max_depth_in_mask(depth_norm, mask, alpha_th), 6)
        # Single canonical position: [x, y, z] for layout / next pass
        entry["bbox_position"] = [round(x, 2), round(y, 2), round(z, 6)]

    if in_place:
        start_step(4)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    return metadata
