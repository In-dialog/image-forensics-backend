"""Enumerate top-level and granular image items in one original-image coordinate frame."""
from __future__ import annotations

import json
import re
import hashlib
from pathlib import Path

from PIL import Image


def safe_component(value: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', "_", value).strip() or "image"


def granular_directory(run_folder: Path, name: str, group: str) -> Path:
    return run_folder / "granular_segments" / safe_component(group) / safe_component(name)


def _box(value) -> list[int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    if any(isinstance(number, bool) or not isinstance(number, (int, float)) for number in value):
        return None
    left, top, right, bottom = (int(number) for number in value)
    return [left, top, right, bottom] if left < right and top < bottom else None


def _image_path(folder: Path, name: str, run_folder: Path) -> str | None:
    for extension in (".webp", ".png", ".jpg", ".jpeg"):
        candidate = folder / (safe_component(name) + extension)
        if candidate.is_file() and candidate.resolve().is_relative_to(run_folder.resolve()):
            return candidate.relative_to(run_folder).as_posix()
    return None


def enumerate_analysis_items(run_folder: Path, metadata: dict | None = None) -> list[dict]:
    """Expose child masks without copying them into the top-level segment collection."""
    if metadata is None:
        metadata = json.loads((run_folder / "metadata.json").read_text(encoding="utf-8"))
    items: list[dict] = []
    parents: dict[str, dict] = {}
    for entry in metadata.get("extracts", []):
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            continue
        item = {"name": name, "label": entry.get("label") or name,
                "group": entry.get("group") or "objects", "source": "item"}
        box = _box(entry.get("bbox"))
        if box:
            item["bbox"] = box
        path = _image_path(run_folder / "segments", name, run_folder)
        if path:
            item["image_path"] = path
        parents[name] = item
        items.append(item)
    seen = set(parents)
    for metadata_path in sorted((run_folder / "granular_segments").glob("**/segment_metadata.json")):
        granular = json.loads(metadata_path.read_text(encoding="utf-8"))
        parent_name = granular.get("parent_name")
        parent = parents.get(parent_name)
        if parent is None:
            continue
        parent_box = _box(granular.get("parent_bbox")) or parent.get("bbox")
        for entry in granular.get("extracts", []):
            child_name = entry.get("name")
            if not isinstance(child_name, str) or not child_name:
                continue
            name = f"{parent_name}::{child_name}"
            if name in seen:
                continue
            local_box = _box(entry.get("bbox"))
            global_box = _box(entry.get("bbox_global"))
            if global_box is None and local_box and parent_box:
                global_box = [local_box[0] + parent_box[0], local_box[1] + parent_box[1],
                              local_box[2] + parent_box[0], local_box[3] + parent_box[1]]
            item = {"name": name, "label": entry.get("label") or child_name,
                    "parent_name": parent_name, "group": entry.get("group") or parent["group"], "source": "part"}
            if global_box:
                item["bbox"] = global_box
            path = _image_path(metadata_path.parent, child_name, run_folder)
            if path:
                item["image_path"] = path
            items.append(item)
            seen.add(name)
    return items


def result_identity(item: dict) -> dict:
    return {key: item[key] for key in ("name", "label", "parent_name", "image_path", "group", "bbox", "source") if key in item}


def original_image_path(run_folder: Path, metadata: dict) -> Path | None:
    source = metadata.get("source_image_path")
    if isinstance(source, str) and Path(source).is_file():
        return Path(source)
    exported = run_folder / "original" / "original.webp"
    return exported if exported.is_file() else None


def item_evidence_board(run_folder: Path, item: dict, source: Path | None) -> Path | None:
    """Cache marked Scene/Nearby views and an unmarked Target in one evidence image.

    Nearby expands every target edge by its longest side, retaining local context.
    Scene remains complete at a maximum 1280px side; every view preserves aspect
    ratio. Neutral headings and cyan outlines are annotations, not source content.
    """
    from PIL import ImageDraw, ImageFont

    try:
        box = _box(item.get("bbox"))
    except (ValueError, OverflowError):
        return None
    if source is None or not source.is_file() or box is None:
        return None
    source_stat = source.stat()
    identity = json.dumps(
        ["evidence_board_v2_three_views", item.get("name"), str(source.resolve()),
         source_stat.st_mtime_ns, source_stat.st_size, box], ensure_ascii=False,
    ).encode("utf-8")
    folder = run_folder / "analysis_evidence"
    target = folder / ("board-" + hashlib.sha256(identity).hexdigest()[:24] + ".png")
    if not target.resolve().is_relative_to(run_folder.resolve()):
        raise ValueError("Evidence board must stay inside the analysis folder")
    if target.is_file():
        return target

    def mark(view: Image.Image, coordinates: tuple[int, int, int, int], frame: tuple[int, int]) -> None:
        x1, y1, x2, y2 = coordinates
        x1 = min(view.width - 1, round(x1 * view.width / frame[0]))
        y1 = min(view.height - 1, round(y1 * view.height / frame[1]))
        x2 = min(view.width - 1, max(x1, round(x2 * view.width / frame[0]) - 1))
        y2 = min(view.height - 1, max(y1, round(y2 * view.height / frame[1]) - 1))
        ImageDraw.Draw(view).rectangle(
            (x1, y1, x2, y2), outline=(0, 255, 255), width=max(3, round(min(view.size) / 180)),
        )

    with Image.open(source) as original:
        left, top, right, bottom = box
        left, top = max(0, left), max(0, top)
        right, bottom = min(original.width, right), min(original.height, bottom)
        if left >= right or top >= bottom:
            return None
        rgb = original.convert("RGB")
        crop = rgb.crop((left, top, right, bottom))
        padding = max(right - left, bottom - top)
        nearby_box = (max(0, left - padding), max(0, top - padding),
                      min(original.width, right + padding), min(original.height, bottom + padding))
        nearby = rgb.crop(nearby_box)
        nearby_frame = nearby.size
        scene_scale = min(1.0, 1280 / max(original.size))
        scene_size = (max(1, round(original.width * scene_scale)), max(1, round(original.height * scene_scale)))
        scene = rgb.resize(scene_size, Image.Resampling.LANCZOS)
        mark(scene, (left, top, right, bottom), original.size)

    margin, gap, heading_height = 12, 12, 32
    nearby_width, target_width = max(96, round(scene.width / 2)), max(64, round(scene.width / 3))
    available = 2200 - margin * 2 - gap * 2 - scene.width
    if nearby_width + target_width > available:
        nearby_width = round(available * nearby_width / (nearby_width + target_width))
        target_width = available - nearby_width

    def fit(view: Image.Image, width: int) -> Image.Image:
        scale = min(width / view.width, scene.height / view.height)
        return view.resize((max(1, round(view.width * scale)), max(1, round(view.height * scale))),
                           Image.Resampling.LANCZOS)

    nearby, crop = fit(nearby, nearby_width), fit(crop, target_width)
    mark(nearby, (left - nearby_box[0], top - nearby_box[1],
                  right - nearby_box[0], bottom - nearby_box[1]), nearby_frame)
    board = Image.new("RGB", (margin * 2 + scene.width + gap * 2 + nearby_width + target_width,
                              margin * 2 + heading_height + scene.height), (242, 244, 247))
    image_top = margin + heading_height
    draw = ImageDraw.Draw(board)
    font = ImageFont.load_default(size=18)
    panel_left = margin
    for heading, view, width in (("Scene", scene, scene.width), ("Nearby", nearby, nearby_width),
                                 ("Target", crop, target_width)):
        board.paste(view, (panel_left + (width - view.width) // 2,
                           image_top + (scene.height - view.height) // 2))
        draw.text((panel_left, margin), heading, fill=(30, 35, 40), font=font)
        panel_left += width + gap
    folder.mkdir(parents=True, exist_ok=True)
    board.save(target, "PNG")
    return target
