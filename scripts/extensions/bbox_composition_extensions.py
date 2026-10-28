"""
Bbox composition: lines from the center of each side of a bounding box to the image edge.
Used by the bbox composition runner to add bbox_composition to each extract in metadata.
"""
import copy
from scripts.orchestration.progress import progress_items, skip_item
from typing import Any

# Bbox format: [x_min, y_min, x_max, y_max]


def bbox_to_edge_lines(
    bbox: list[float | int],
    image_width: int,
    image_height: int,
    *,
    decimals: int = 2,
) -> dict[str, Any]:
    """
    For a single bbox, compute the line from the center of each side (left, right, top, bottom)
    to the corresponding image edge. Returns start [x,y], end [x,y], and distance in pixels.

    Args:
        bbox: [x_min, y_min, x_max, y_max]
        image_width: Full image width in pixels.
        image_height: Full image height in pixels.
        decimals: Round coordinates and distance to this many decimal places.

    Returns:
        Dict with keys "left", "right", "top", "bottom". Each value:
        {"start": [x, y], "end": [x, y], "distance_px": float}.
    """
    if len(bbox) != 4:
        return {}
    x_min, y_min, x_max, y_max = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2

    def r(v: float) -> float:
        return round(v, decimals)

    def pt(x: float, y: float) -> list[float]:
        return [r(x), r(y)]

    return {
        "left": {
            "start": pt(x_min, cy),
            "end": pt(0, cy),
            "distance_px": r(x_min),
        },
        "right": {
            "start": pt(x_max, cy),
            "end": pt(image_width, cy),
            "distance_px": r(image_width - x_max),
        },
        "top": {
            "start": pt(cx, y_min),
            "end": pt(cx, 0),
            "distance_px": r(y_min),
        },
        "bottom": {
            "start": pt(cx, y_max),
            "end": pt(cx, image_height),
            "distance_px": r(image_height - y_max),
        },
    }


def add_bbox_composition_to_extracts(
    metadata: dict[str, Any],
    image_width: int,
    image_height: int,
    *,
    decimals: int = 2,
) -> dict[str, Any]:
    """
    Add bbox_composition to each extract in metadata. Each extract must have a "bbox" key
    [x_min, y_min, x_max, y_max]. Modifies a copy of metadata; does not mutate the input.

    Returns:
        New metadata dict with each extract having a "bbox_composition" key (left/right/top/bottom
        lines with start, end, distance_px).
    """
    out = copy.deepcopy(metadata)
    extracts = out.get("extracts")
    if not extracts:
        return out
    for entry in progress_items(extracts, step=1, label=lambda item: item.get("name", "Unnamed item")):
        bbox = entry.get("bbox")
        if not bbox or len(bbox) != 4:
            skip_item()
            continue
        entry["bbox_composition"] = bbox_to_edge_lines(
            bbox, image_width, image_height, decimals=decimals
        )
    return out
