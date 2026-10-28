"""Compare occupied pixels in a shared image frame, independent of labels or crop padding."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


MASK_DUPLICATE_IOU = 0.85


@dataclass
class MaskRegion:
    box: tuple[int, int, int, int]
    pixels: np.ndarray
    area: int


def mask_region(mask, origin=(0, 0), *, threshold=0.5) -> MaskRegion | None:
    """Store only a mask's occupied extent; do not discard one-pixel-wide regions."""
    if mask is None:
        return None
    pixels = np.asarray(mask)
    if pixels.ndim > 2:
        pixels = pixels.squeeze()
    if pixels.ndim != 2:
        return None
    pixels = pixels > threshold
    rows = np.flatnonzero(pixels.any(axis=1))
    columns = np.flatnonzero(pixels.any(axis=0))
    if not len(rows) or not len(columns):
        return None
    left, top = int(columns[0]), int(rows[0])
    right, bottom = int(columns[-1]) + 1, int(rows[-1]) + 1
    cropped = pixels[top:bottom, left:right].copy()
    return MaskRegion((origin[0] + left, origin[1] + top, origin[0] + right, origin[1] + bottom),
                      cropped, int(np.count_nonzero(cropped)))


def extract_region(extract, offset=(0, 0), *, threshold=0.5) -> MaskRegion | None:
    """An extraction's full-size mask is relative to its source image or parent crop."""
    mask = getattr(extract, "mask_original", None)
    if mask is None:
        return None
    pixels = np.asarray(mask)
    if pixels.ndim > 2:
        pixels = pixels.squeeze()
    if pixels.ndim != 2:
        return None
    left, top, right, bottom = (int(value) for value in extract.bbox)
    left, top = max(0, left), max(0, top)
    right, bottom = min(pixels.shape[1], right), min(pixels.shape[0], bottom)
    if left >= right or top >= bottom:
        return None
    return mask_region(pixels[top:bottom, left:right], (offset[0] + left, offset[1] + top), threshold=threshold)


def saved_region(image_path: Path | None, box, *, threshold=0.5) -> MaskRegion | None:
    """Read saved alpha at its recorded position; missing alpha is not evidence of a mask."""
    if image_path is None or not image_path.is_file() or not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    left, top, right, bottom = (int(value) for value in box)
    with Image.open(image_path) as image:
        if image.size != (right - left, bottom - top) or ("A" not in image.getbands() and "transparency" not in image.info):
            return None
        alpha = np.asarray(image.convert("RGBA"))[:, :, 3]
        return mask_region(alpha, (left, top), threshold=threshold * 255)


def mask_iou(first: MaskRegion, second: MaskRegion) -> float:
    left, top = max(first.box[0], second.box[0]), max(first.box[1], second.box[1])
    right, bottom = min(first.box[2], second.box[2]), min(first.box[3], second.box[3])
    if left >= right or top >= bottom:
        return 0.0
    a = first.pixels[top - first.box[1]:bottom - first.box[1], left - first.box[0]:right - first.box[0]]
    b = second.pixels[top - second.box[1]:bottom - second.box[1], left - second.box[0]:right - second.box[0]]
    intersection = int(np.count_nonzero(a & b))
    return intersection / (first.area + second.area - intersection)


class RegionIndex:
    """Canonical regions only: a discarded mask cannot bridge two distinct objects."""

    def __init__(self):
        self.regions: list[tuple[object, MaskRegion]] = []

    def add(self, identity, region: MaskRegion | None) -> None:
        if region is not None:
            self.regions.append((identity, region))

    def match(self, region: MaskRegion | None):
        if region is not None:
            for identity, prior in self.regions:
                if min(region.area, prior.area) / max(region.area, prior.area) < MASK_DUPLICATE_IOU:
                    continue
                if mask_iou(region, prior) >= MASK_DUPLICATE_IOU:
                    return identity
        return None


def repeated_parent(parent: dict, label: str, global_box) -> bool:
    """Reject re-extracting the same named parent, including OCR with a different mask.

    This applies only to a direct parent and a similarly sized box. Smaller named
    parts and separate occurrences of the same text remain distinct.
    """
    from .detection_views import box_iou

    parent_box = parent.get("bbox")
    parent_label = parent.get("label")
    if not parent_label or not isinstance(parent_box, (tuple, list)) or len(parent_box) != 4:
        return False
    same_label = " ".join(parent_label.casefold().split()) == " ".join(label.casefold().split())
    return same_label and box_iou(parent_box, global_box) >= 0.65
