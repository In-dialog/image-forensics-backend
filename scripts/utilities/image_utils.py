"""
Image utilities shared across pipelines (pose, depth, VP, saliency, SAM, composition).
Consolidates: load RGB, resize by long side, PIL/numpy conversion, normalize, depth colorize.
"""
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Load / save
# ---------------------------------------------------------------------------


def load_rgb(path: Path | str) -> Image.Image:
    """Load image from path as PIL Image in RGB mode."""
    return Image.open(path).convert("RGB")


# Checkerboard colors (light grays) to imitate transparency; avoids confusing with image white
_CHECKER_LIGHT = (240, 240, 240)
_CHECKER_DARK = (192, 192, 192)
_CHECKER_TILE = 12


def _checkerboard_image(width: int, height: int) -> Image.Image:
    """RGB PIL Image filled with a neutral gray checkerboard (transparency-style)."""
    tile = _CHECKER_TILE
    # One tile = 2x2 squares; create a 2*tile x 2*tile pattern and repeat
    pattern = Image.new("RGB", (2 * tile, 2 * tile))
    pattern.paste(_CHECKER_LIGHT, (0, 0, tile, tile))
    pattern.paste(_CHECKER_DARK, (tile, 0, 2 * tile, tile))
    pattern.paste(_CHECKER_DARK, (0, tile, tile, 2 * tile))
    pattern.paste(_CHECKER_LIGHT, (tile, tile, 2 * tile, 2 * tile))
    out = Image.new("RGB", (width, height))
    for y in range(0, height, 2 * tile):
        for x in range(0, width, 2 * tile):
            out.paste(pattern, (x, y))
    return out


def load_rgb_flat(path: Path | str, background: str | tuple[int, int, int] = "checker") -> Image.Image:
    """
    Load image as RGB, compositing transparency onto a background.
    Use for WebP/PNG segments so SAM does not see alpha as black or artifacts.
    background: "checker" (default) = gray checkerboard to imitate transparency;
      or (R, G, B) for a solid color.
    """
    with Image.open(path) as im:
        im = im.convert("RGBA")
    w, h = im.size
    if background == "checker":
        out = _checkerboard_image(w, h)
    else:
        out = Image.new("RGB", (w, h), background)
    out.paste(im, (0, 0), im)
    return out


def ensure_dir(path: Path) -> Path:
    """Create directory (and parents) if needed. Return path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Resize (common in pose, depth, deepgaze)
# ---------------------------------------------------------------------------


def resize_long_side(
    img: Image.Image,
    long_side: int,
    *,
    multiple_of: Optional[int] = None,
    resample: int = Image.Resampling.LANCZOS,
) -> Image.Image:
    """
    Resize so the longer side equals long_side (or stays under if already smaller).
    If multiple_of is set (e.g. 64), dimensions are rounded to multiples of it.
    """
    w, h = img.size
    if max(w, h) <= long_side and multiple_of is None:
        return img

    if max(w, h) > long_side:
        scale = long_side / max(w, h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
    else:
        new_w, new_h = w, h

    if multiple_of is not None:
        new_w = max(multiple_of, (new_w + multiple_of - 1) // multiple_of * multiple_of)
        new_h = max(multiple_of, (new_h + multiple_of - 1) // multiple_of * multiple_of)

    if (new_w, new_h) == (w, h):
        return img
    return img.resize((new_w, new_h), resample=resample)


# ---------------------------------------------------------------------------
# PIL <-> numpy (pose, depth, deepgaze, sam3)
# ---------------------------------------------------------------------------


def pil_to_numpy(img: Image.Image) -> np.ndarray:
    """Convert PIL Image to uint8 HWC numpy array (RGB)."""
    return np.array(img, dtype=np.uint8)


def numpy_to_pil(arr: np.ndarray) -> Image.Image:
    """
    Convert numpy array to PIL Image.
    Handles: uint8 HWC, float [0,1] HWC, and CHW (C,H,W) with 1 or 3 channels.
    """
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        # CHW -> HWC
        arr = np.transpose(arr, (1, 2, 0))
    if np.issubdtype(arr.dtype, np.floating):
        if arr.max() <= 1.0:
            arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        return Image.fromarray(arr)
    return Image.fromarray(arr.astype(np.uint8))


# ---------------------------------------------------------------------------
# Normalize (composition_lines npNormalise, depth process_depth)
# ---------------------------------------------------------------------------


def normalize_to_01(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize array to [0, 1]. Constant arrays become 0."""
    arr = np.nan_to_num(arr, nan=0.0)
    lo, hi = arr.min(), arr.max()
    if hi > lo:
        return (arr - lo) / (hi - lo)
    return np.zeros_like(arr)


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 255] and return uint8. Constant arrays become 0."""
    n = normalize_to_01(arr)
    return (n * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Depth / 1-channel visualization (depth extraction colorize)
# ---------------------------------------------------------------------------


def colorize_depth(depth: np.ndarray) -> np.ndarray:
    """Map single-channel depth to viridis-like RGB uint8 for visualization."""
    d = normalize_to_01(np.nan_to_num(depth, nan=0.0))
    r = np.clip(0.267 + 1.5 * d - 2.0 * d**2 + 0.5 * d**3, 0, 1)
    g = np.clip(0.005 + 1.4 * d - 0.5 * d**2, 0, 1)
    b = np.clip(0.329 + 2.0 * d - 2.0 * d**2, 0, 1)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Segment mask (place RGBA crop alpha at bbox in full image)
# ---------------------------------------------------------------------------


def segment_mask_in_full_image(
    segment_path: Path,
    bbox: list[int],
    full_h: int,
    full_w: int,
) -> np.ndarray | None:
    """
    Load segment PNG (RGBA), use alpha as mask, place at bbox position in full image.
    Returns (full_h, full_w) float 0-1 mask, or None on error.
    bbox is [x1, y1, x2, y2] (position from metadata). Segment image is used at its
    native size — no resizing. Extraction pipeline keeps segment size equal to bbox.
    """
    if not segment_path.exists():
        return None
    img = Image.open(segment_path).convert("RGBA")
    arr = np.array(img)
    if arr.ndim != 3 or arr.shape[2] != 4:
        return None
    alpha = arr[:, :, 3].astype(np.float64) / 255.0  # (H_crop, W_crop)
    alpha = np.clip(alpha, 0.0, 1.0)
    x1, y1, x2, y2 = bbox
    crop_h, crop_w = alpha.shape
    # Place at (x1, y1) using segment's native size — no resize (extraction didn't change size)
    full_mask = np.zeros((full_h, full_w), dtype=np.float64)
    y1_c = max(0, y1)
    y2_c = min(full_h, y1 + crop_h)
    x1_c = max(0, x1)
    x2_c = min(full_w, x1 + crop_w)
    sy = y1_c - y1
    sx = x1_c - x1
    h = y2_c - y1_c
    w = x2_c - x1_c
    if h > 0 and w > 0:
        full_mask[y1_c:y2_c, x1_c:x2_c] = alpha[sy : sy + h, sx : sx + w]
    return np.clip(full_mask, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Border artifact removal (resize / pipeline edges)
# ---------------------------------------------------------------------------


def strip_border_artifact(arr: np.ndarray, width: int = 1) -> np.ndarray:
    """
    Overwrite the outer `width` row/column with the adjacent inner row/column to remove
    thin border artifacts from resize or pipeline padding. Works in-place on a copy;
    use width=1 for a single-pixel artifact line. Best practice after resizing depth/masks.
    """
    if width <= 0 or arr.ndim not in (2, 3):
        return arr
    out = np.asarray(arr, copy=True)
    h, w = out.shape[0], out.shape[1]
    if h <= 2 * width or w <= 2 * width:
        return out
    # Replicate inner edge onto border: top <- row width, bottom <- row h-1-width, etc.
    for i in range(width):
        out[i, :] = out[width, :]
        out[-(i + 1), :] = out[-(width + 1), :]
        out[:, i] = out[:, width]
        out[:, -(i + 1)] = out[:, -(width + 1)]
    return out


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------


def ensure_rgb_hwc(arr: np.ndarray) -> np.ndarray:
    """Ensure array is HWC uint8 RGB (e.g. duplicate channel if grayscale)."""
    if arr.ndim == 2:
        return np.stack([arr] * 3, axis=-1)
    return np.asarray(arr, dtype=np.uint8)
