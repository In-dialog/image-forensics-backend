"""
Save stage outputs: write images and optional metadata into a parent folder.
"""
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .image_utils import ensure_dir, load_rgb, numpy_to_pil

METADATA_FILENAME = "metadata.json"
LLM_LIST_FILENAME = "llm_list.txt"
SEGMENTS_SUBFOLDER = "segments"
COMPOSITION_SUBFOLDER = "composition"
COMPOSITION_LINES_FILENAME = "composition_lines.json"
COMPOSITION_LINES_IMAGE_FILENAME = "composition_lines.webp"
# Default base for run folders: outputs (relative to cwd when running from project root)
DEFAULT_BASE_DIR = Path("outputs")


def _safe_filename(name: str) -> str:
    """Sanitize to a safe filename (no path separators, no empty)."""
    name = re.sub(r'[<>:"/\\|?*]', "_", name).strip()
    return name or "image"


def _image_to_pil(img: Image.Image | np.ndarray) -> Image.Image:
    if isinstance(img, Image.Image):
        return img
    return numpy_to_pil(img)


# Key in metadata for the initial image filename (stem); used as folder name for this run.
SOURCE_IMAGE_KEY = "source_image"
# Key in metadata for the full original image path (absolute path as string).
SOURCE_IMAGE_PATH_KEY = "source_image_path"
# Keys for image dimensions at extraction time (used so negative space / masks use same canvas as bboxes).
IMAGE_WIDTH_KEY = "image_width"
IMAGE_HEIGHT_KEY = "image_height"
# Key written into metadata.json: path to the folder where images and this JSON were saved.
OUTPUT_FOLDER_KEY = "output_folder"


def save_stage(
    *,
    images: list[tuple[str, Image.Image | np.ndarray]] | None = None,
    metadata: dict[str, Any] | None = None,
    llm_list: list[str] | None = None,
    base_dir: Path | str | None = None,
    source_image_path: Path | str | None = None,
) -> Path:
    """
    Save stage outputs: run folder named by image stem (no stage name), segments in a subfolder.

    Layout:
      base / {source_image} /
        metadata.json
        llm_list.txt   (if llm_list is provided)
        segments /
          Person_0.webp
          Books_0.webp
          ...

    Args:
        images: Optional list of (filename_stem_or_name, image). Saved under segments/.
        metadata: Optional dict (saved as metadata.json in run folder). Can be augmented by source_image_path.
        llm_list: Optional list of strings (e.g. labels/prompts from vision model). Saved as one per line in llm_list.txt.
        base_dir: Base directory; when None, uses DEFAULT_BASE_DIR.
        source_image_path: If provided, fills source_image, source_image_path, image_width, image_height in metadata.

    Returns:
        The run folder path (parent of segments/).
    """
    base = Path(base_dir) if base_dir is not None else DEFAULT_BASE_DIR
    meta = dict(metadata) if metadata else {}
    if source_image_path is not None:
        path = Path(source_image_path)
        meta[SOURCE_IMAGE_KEY] = path.stem
        meta[SOURCE_IMAGE_PATH_KEY] = str(path.resolve())
        with Image.open(path) as im:
            meta[IMAGE_WIDTH_KEY] = im.width
            meta[IMAGE_HEIGHT_KEY] = im.height
    source = meta.get(SOURCE_IMAGE_KEY)
    if source:
        run_folder = base / source
    else:
        run_folder = base / "output"
    ensure_dir(run_folder)

    if images:
        segments_folder = run_folder / SEGMENTS_SUBFOLDER
        ensure_dir(segments_folder)
        image_paths = {}
        for name, img in images:
            fname = _safe_filename(name)
            if not any(fname.endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp")):
                fname = f"{fname}.webp"
            path = segments_folder / fname
            pil = _image_to_pil(img)
            if pil.mode in ("RGBA", "LA"):
                pil = pil.convert("RGBA")
                pil.save(path, "WEBP", quality=90, method=6)
            else:
                pil = pil.convert("RGB") if pil.mode != "RGB" else pil
                pil.save(path, "WEBP", quality=90)
            image_paths[name] = path.relative_to(run_folder).as_posix()
        if isinstance(meta.get("extracts"), list):
            meta["extracts"] = [
                {**entry, "image_path": image_paths[entry["name"]]}
                if entry.get("name") in image_paths else entry
                for entry in meta["extracts"]
            ]

    if meta:
        path = run_folder / METADATA_FILENAME
        meta[OUTPUT_FOLDER_KEY] = str(run_folder.resolve())
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    if llm_list:
        (run_folder / LLM_LIST_FILENAME).write_text("\n".join(llm_list), encoding="utf-8")

    return run_folder


def composition_lines_to_dict(lines_array: np.ndarray, *, decimals: int = 2) -> dict[str, Any]:
    """
    Convert (N, 2, 2) lines array to JSON-serializable dict: line1, line2, ... with start/end [x,y].
    """
    out: dict[str, Any] = {}
    for i, line in enumerate(lines_array, start=1):
        start_xy = [round(float(line[0][0]), decimals), round(float(line[0][1]), decimals)]
        end_xy = [round(float(line[1][0]), decimals), round(float(line[1][1]), decimals)]
        out[f"line{i}"] = {"start": start_xy, "end": end_xy}
    return out


def draw_composition_lines_on_image(image_path: Path, lines: np.ndarray) -> Image.Image:
    """
    Load the image and draw composition lines on it. Lines are (N, 2, 2) in pixel coords [x, y].
    Returns PIL Image (RGB) with lines drawn.
    """
    from PIL import ImageDraw

    img = load_rgb(image_path)
    draw = ImageDraw.Draw(img)
    w, h = img.size
    line_width = max(1, min(w, h) // 400)
    for line in lines:
        x0, y0 = int(round(float(line[0][0]))), int(round(float(line[0][1])))
        x1, y1 = int(round(float(line[1][0]))), int(round(float(line[1][1])))
        draw.line([(x0, y0), (x1, y1)], fill=(255, 0, 0), width=line_width)
    return img


def draw_composition_lines_transparent(image_path: Path, lines: np.ndarray) -> Image.Image:
    """
    Draw composition lines on transparent background (RGBA). Lines are (N, 2, 2) in pixel coords.
    Returns PIL Image (RGBA) for overlay use.
    """
    from PIL import ImageDraw

    img = load_rgb(image_path)
    w, h = img.size
    rgba = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(rgba)
    line_width = max(1, min(w, h) // 400)
    for line in lines:
        x0, y0 = int(round(float(line[0][0]))), int(round(float(line[0][1])))
        x1, y1 = int(round(float(line[1][0]))), int(round(float(line[1][1])))
        draw.line([(x0, y0), (x1, y1)], fill=(255, 0, 0, 255), width=line_width)
    return rgba


def run_composition_lines_and_save(
    image_path: Path,
    run_folder: Path,
    *,
    window_size: int = 64,
    decimals: int = 2,
    filename: str | None = None,
) -> Path:
    """
    Run composition/leading lines on the image; save JSON and an image with lines in run_folder/composition/.
    Returns path to the written JSON file.
    """
    from .composition_lines_utils import get_composition_lines
    from scripts.orchestration.progress import progress_steps, start_step

    progress_steps("Find composition lines", "Save geometry and overlay")
    start_step(1)
    lines = get_composition_lines(image_path, window_size=window_size)
    start_step(2)
    data = composition_lines_to_dict(lines, decimals=decimals)
    composition_dir = run_folder / COMPOSITION_SUBFOLDER
    ensure_dir(composition_dir)
    out_path = composition_dir / (filename or COMPOSITION_LINES_FILENAME)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    img_with_lines = draw_composition_lines_transparent(image_path, lines)
    img_path = composition_dir / COMPOSITION_LINES_IMAGE_FILENAME
    img_with_lines.save(img_path, "WEBP", quality=90, method=6)
    return out_path
