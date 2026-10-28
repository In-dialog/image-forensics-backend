"""Bounded detail views and coordinate merging for the existing vision detector."""
from __future__ import annotations

import math
import re


MAX_SCENE_DETECTIONS = 16
MAX_DETAIL_DETECTIONS = 8


def detail_view_boxes(width: int, height: int) -> list[tuple[int, int, int, int]]:
    """Four overlapping views for images with long side >=512 and short side >=256.

    Each original-pixel crop is at least 128px per side. Adjacent views overlap
    by about 15%, including landscape and portrait images below 512px on one side.
    """
    if max(width, height) < 512 or min(width, height) < 256:
        return []
    tile_width = max(128, math.ceil(width / 1.85))
    tile_height = max(128, math.ceil(height / 1.85))
    return [(left, top, left + tile_width, top + tile_height)
            for top in (0, height - tile_height) for left in (0, width - tile_width)]


def remap_box(box, view, width: int, height: int) -> tuple[int, int, int, int]:
    """Map a crop-normalized XYXY box back to full-image normalized coordinates."""
    left, top, right, bottom = view
    x1 = math.floor((left + box[0] * (right - left) / 1000) * 1000 / width)
    y1 = math.floor((top + box[1] * (bottom - top) / 1000) * 1000 / height)
    x2 = math.ceil((left + box[2] * (right - left) / 1000) * 1000 / width)
    y2 = math.ceil((top + box[3] * (bottom - top) / 1000) * 1000 / height)
    return (max(0, min(999, x1)), max(0, min(999, y1)),
            max(x1 + 1, min(1000, x2)), max(y1 + 1, min(1000, y2)))


def box_iou(first, second) -> float:
    overlap = max(0, min(first[2], second[2]) - max(first[0], second[0])) * max(0, min(first[3], second[3]) - max(first[1], second[1]))
    area_first = (first[2] - first[0]) * (first[3] - first[1])
    area_second = (second[2] - second[0]) * (second[3] - second[1])
    union = area_first + area_second - overlap
    return overlap / union if union > 0 else 0.0


def _label(value: str) -> str:
    return re.sub(r"[^\w]+", " ", value.casefold()).strip()


def merge_calls(existing: list, incoming: list, maximum: int) -> list:
    """Merge the same label at IoU >=0.65; preserve separate and small nested regions."""
    merged = list(existing[:maximum])
    for call in incoming:
        if len(merged) >= maximum:
            break
        group, label, box = call
        duplicate = any(old_group == group and _label(old_label) == _label(label) and box_iou(old_box, box) >= 0.65
                        for old_group, old_label, old_box in merged)
        if not duplicate:
            merged.append(call)
    return merged
