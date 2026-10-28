"""Structured detection, coordinate conversion, and shared SAM extraction."""
import json
import math
from pathlib import Path
from scripts.orchestration.progress import progress_items, progress_activity, skip_item
from typing import Any, List, Optional, Tuple

import numpy as np

VisionBox = Optional[Tuple[int, int, int, int]]
DETECTION_GROUPS = ("people", "objects", "texts")

# Qwen-style normalized coordinates are in [0, 1000].
_QWEN_COORD_NORM_MAX = 1000
_QWEN_COORD_NORM_SCALE = 1000.0


def sam_box_candidates(
    box: VisionBox,
    orig_w: int,
    orig_h: int,
) -> List[Tuple[int, int, int, int]]:
    """
    Build SAM box candidates from one vision-model box.
    Convert one vision-model box (normalized XYXY in [0, 1000]) to image pixel XYXY.
    This is strict normalized mode (no absolute-pixel fallback).
    """
    if box is None:
        return []

    x1n, y1n, x2n, y2n = (int(v) for v in box)
    x1n = max(0, min(_QWEN_COORD_NORM_MAX, x1n))
    y1n = max(0, min(_QWEN_COORD_NORM_MAX, y1n))
    x2n = max(0, min(_QWEN_COORD_NORM_MAX, x2n))
    y2n = max(0, min(_QWEN_COORD_NORM_MAX, y2n))
    if x1n > x2n:
        x1n, x2n = x2n, x1n
    if y1n > y2n:
        y1n, y2n = y2n, y1n

    x1 = int((x1n / _QWEN_COORD_NORM_SCALE) * orig_w)
    y1 = int((y1n / _QWEN_COORD_NORM_SCALE) * orig_h)
    x2 = int(math.ceil((x2n / _QWEN_COORD_NORM_SCALE) * orig_w))
    y2 = int(math.ceil((y2n / _QWEN_COORD_NORM_SCALE) * orig_h))

    x1 = max(0, min(x1, orig_w - 1))
    y1 = max(0, min(y1, orig_h - 1))
    x2 = max(0, min(x2, orig_w))
    y2 = max(0, min(y2, orig_h))
    if x2 <= x1:
        x2 = min(orig_w, x1 + 1)
    if y2 <= y1:
        y2 = min(orig_h, y1 + 1)
    return [(x1, y1, x2, y2)]



def labels_from_plan(
    calls: List[Tuple[str, str, VisionBox]],
) -> List[str]:
    """One label per SAM call, preserving duplicates so the list reflects every item that will be processed."""
    return [label for _, label, _ in calls]


MAX_CALLS_PER_GROUP = 15
MIN_BOX_PIXEL_SPAN_FOR_SAM = 4
DEFAULT_MASK_CHOICE_MODE = "score"
DEFAULT_MASK_CHOICE_THRESHOLD = 0.5
MASK_CHOICE_MODES = {"score", "fill", "score_x_fill", "balanced"}



def _normalized_box_too_small_for_image(
    box: VisionBox,
    image_w: int,
    image_h: int,
    min_pixel_span: int = MIN_BOX_PIXEL_SPAN_FOR_SAM,
) -> bool:
    """
    True if a normalized 0..1000 box is too thin for SAM on this image size.
    Keeps behavior stable across different resolutions.
    """
    if box is None:
        return False
    x1, y1, x2, y2 = (int(v) for v in box)
    span_x = abs(x2 - x1)
    span_y = abs(y2 - y1)
    min_norm_x = max(1, int(math.ceil((min_pixel_span * 1000.0) / max(1, image_w))))
    min_norm_y = max(1, int(math.ceil((min_pixel_span * 1000.0) / max(1, image_h))))
    return span_x < min_norm_x or span_y < min_norm_y


def vision_calls_for_group(
    image_path: Path,
    group: str,
    *,
    prompt_profile: dict | None = None,
    max_calls_for_group: int | None = None,
    vision_model: str = "qwen3-vl:8b-instruct",
    vision_device: str | None = None,
    num_ctx: int | None = 8192,
    num_predict: int | None = 2048,
    temperature: float | None = 0,
    top_k: int | None = 30,
    top_p: float | None = 0.75,
    repeat_penalty: float | None = 1.25,
    presence_penalty: float | None = 0,
    repeat_last_n: int | None = None,
    seed: int | None = 42,
    think: bool | str | None = False,
    keep_alive: int | str = 0,
    print_response: bool = False,
    print_raw_response: bool = False,
    scene_detail_pass: bool = False,
    detail_dir: Path | None = None,
    detection_scope: str = "scene",
) -> List[Tuple[str, str, VisionBox]]:
    """Use bounded calls to the same detector, optionally inspecting scene-object detail views."""
    from contextlib import nullcontext
    from dataclasses import replace
    from PIL import Image
    from scripts.prompting import compile_detection
    from scripts.utilities.structured_results import resolve_profile, run_compiled
    from scripts.utilities.ollama_vision import vision_model_session
    from scripts.configuration.store import atomic_json
    from scripts.utilities.detection_views import (
        MAX_SCENE_DETECTIONS, MAX_DETAIL_DETECTIONS, detail_view_boxes, remap_box, merge_calls,
    )

    profile = resolve_profile("object_detect", prompt_profile)
    cap = MAX_CALLS_PER_GROUP if max_calls_for_group is None else max(1, int(max_calls_for_group))
    views = []
    width = height = 0
    if detail_dir is not None or (scene_detail_pass and group == "objects"):
        with Image.open(image_path) as original:
            width, height = original.size
    if scene_detail_pass and group == "objects":
        views = detail_view_boxes(width, height)
        if views and detail_dir is None:
            raise ValueError("Scene detail passes require an evidence directory inside the run folder.")
    traces = []

    def detect(source, maximum, request_keep_alive, view=None):
        compiled = compile_detection(profile, group, max_items=maximum, scope=detection_scope)
        if view is not None:
            compiled = replace(compiled, user=compiled.user + (
                "\nThis is an overlapping detail crop from a larger scene. Locate only confidently visible "
                "instances in this crop. Use coordinates relative to this crop; do not infer hidden repetitions "
                "or complete objects beyond its edges."
            ))
        trace = {
            "source_image": str(source.resolve()),
            "source_crop": source.name if view is not None else None,
            "crop_pixel_box": list(view) if view else [0, 0, width, height],
            "max_items": maximum, "status": "running",
            "compiled": {"system": compiled.system, "user": compiled.user, "schema": compiled.schema},
        }
        traces.append(trace)
        try:
            response = run_compiled(
                compiled, source, model=vision_model, vision_device=vision_device, keep_alive=request_keep_alive,
                num_ctx=num_ctx, num_predict=num_predict if num_predict is not None else 2048,
                temperature=temperature, top_k=top_k, top_p=top_p, repeat_penalty=repeat_penalty, presence_penalty=presence_penalty,
                repeat_last_n=repeat_last_n, seed=seed, think=think, debug_raw_response=print_raw_response,
            )
        except Exception as error:
            trace.update(status="failed", error=str(error))
            raise
        calls = [(group, item["label"], tuple(item["box"])) for item in response["items"]]
        remapped = [(category, label, remap_box(box, view, width, height) if view else box)
                    for category, label, box in calls]
        trace.update(
            status="completed", local_detections=response["items"],
            full_image_detections=[{"label": label, "box": list(box)} for _, label, box in remapped],
        )

        if print_response:
            print(f"Validated vision response: {json.dumps(response, ensure_ascii=False)}")
        return remapped

    session = vision_model_session(vision_model) if views else nullcontext(keep_alive)
    calls = []
    try:
        with session as request_keep_alive:
            progress_activity("Inspecting full view")
            scene_calls = detect(image_path, min(cap, MAX_SCENE_DETECTIONS), request_keep_alive)
            calls = merge_calls([], scene_calls, cap)
            removed = len(scene_calls) - len(calls)
            traces[-1]["duplicates_removed"] = removed
            if removed:
                print(f"   Merged {removed} repeated {group} detection(s) from the full view.")
            if views:
                detail_dir.mkdir(parents=True, exist_ok=True)
                with Image.open(image_path) as source_image:
                    original = source_image.convert("RGB")
                    for index, view in enumerate(views, 1):
                        if len(calls) >= cap:
                            break
                        source = detail_dir / f"object_detail_{index:02d}.png"
                        original.crop(view).save(source, "PNG")
                        maximum = min(MAX_DETAIL_DETECTIONS, cap - len(calls))
                        progress_activity(f"Detail view {index} of {len(views)}")
                        print(f"   Vision object detail {index}/{len(views)} (maximum {maximum})...")
                        incoming = detect(source, maximum, request_keep_alive, view)
                        before = len(calls)
                        calls = merge_calls(calls, incoming, cap)
                        removed = len(incoming) - (len(calls) - before)
                        traces[-1]["duplicates_removed"] = removed
                        if removed:
                            print(f"   Merged {removed} repeated object detection(s) from detail view {index}.")
    finally:
        if detail_dir is not None:
            payload = {
                "coordinate_space": "normalized_full_image_0_1000", "image_size": [width, height],
                "group": group, "scope": detection_scope,
                "configured_maximum": cap, "merged_count": len(calls), "passes": traces,
                "merged_detections": [{"label": label, "box": list(box)} for _, label, box in calls],
            }
            stem = "object" if group == "objects" else group
            atomic_json(detail_dir / f"{stem}_detection_passes.json", payload)
    return calls


# ---------------------------------------------------------------------------
# Text-only extraction (no bounding box): take all instances
# ---------------------------------------------------------------------------


def extract_by_text_only_all(
    group: str,
    label: str,
    image_path: Path,
    model_and_processor: Any,
    sam_mask_threshold: float = 0.5,
    min_sam_score: float = 0.0,
) -> List[Tuple[str, str, Any]]:
    """
    Run SAM with text prompt only (no box). Returns all extracted instances as
    (group, label, extract) so the runner can keep every segment, not just one.
    Use when the vision model did not return a valid bounding box.
    """
    from scripts.utilities import extract_by_prompt

    extracts = extract_by_prompt(
        label,
        image_path,
        model_and_processor=model_and_processor,
        mask_threshold=sam_mask_threshold,
        threshold=min_sam_score if min_sam_score > 0 else 0.5,
    )
    return [(group, label, ex) for ex in extracts]


def _mask_fill_ratio(
    ex: Any,
    box: Tuple[int, int, int, int],
    mask_threshold: float,
) -> float:
    """
    Fraction of pixels inside `box` where SAM mask >= threshold.
    Used as a cheap quality signal when multiple masks are returned.
    """
    mask = getattr(ex, "mask_original", None)
    if mask is None:
        return 0.0
    arr = np.asarray(mask, dtype=np.float32)
    if arr.ndim > 2:
        arr = np.squeeze(arr)
    if arr.ndim != 2 or arr.size == 0:
        return 0.0
    h, w = arr.shape[:2]
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w))
    y2 = max(0, min(int(y2), h))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    crop = arr[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0
    return float((crop > float(mask_threshold)).mean())


def _mask_choice_value(
    ex: Any,
    box: Tuple[int, int, int, int],
    *,
    mode: str,
    mask_threshold: float,
) -> Tuple[float, float, float]:
    """
    Score one SAM extract for selection.
    Returns (primary_value, sam_score, fill_ratio).
    """
    sam_score = float(getattr(ex, "score", 0.0))
    fill = _mask_fill_ratio(ex, box, mask_threshold)
    if mode == "fill":
        primary = fill
    elif mode == "score_x_fill":
        primary = sam_score * max(fill, 1e-6)
    elif mode == "balanced":
        primary = (0.7 * sam_score) + (0.3 * fill)
    else:
        primary = sam_score
    return (primary, sam_score, fill)


def _mask_fraction_in_box(ex: Any, box: Tuple[int, int, int, int], threshold: float) -> float:
    """How much of an instance actually belongs to the requested image region."""
    mask = getattr(ex, "mask_original", None)
    if mask is None:
        return 0.0
    foreground = np.asarray(mask).squeeze() > threshold
    if foreground.ndim != 2:
        return 0.0
    total = int(np.count_nonzero(foreground))
    if not total:
        return 0.0
    height, width = foreground.shape
    left, top, right, bottom = box
    inside = foreground[max(0, top):min(height, bottom), max(0, left):min(width, right)]
    return float(np.count_nonzero(inside)) / total


def _select_best_extract(
    extracts: List[Any],
    box: Tuple[int, int, int, int],
    *,
    mode: str,
    mask_threshold: float,
) -> Tuple[Any, float, float]:
    """
    Select one extract from a boxed SAM call.
    Tie-break order: primary value -> SAM score -> fill ratio.
    """
    best = None
    best_key: Tuple[float, float, float] | None = None
    for ex in extracts:
        key = _mask_choice_value(ex, box, mode=mode, mask_threshold=mask_threshold)
        if best_key is None or key > best_key:
            best = ex
            best_key = key
    if best is None:
        raise ValueError("No extracts to select from.")
    _, sam_score, fill = best_key if best_key is not None else (0.0, 0.0, 0.0)
    return (best, sam_score, fill)


def run_sam_calls_with_fallback(
    image_path: Path,
    calls: List[Tuple[str, str, VisionBox]],
    model_and_processor: Any,
    *,
    bbox_dedupe_tolerance: int = 5,
    mask_choice_mode: str = DEFAULT_MASK_CHOICE_MODE,
    mask_choice_threshold: float = DEFAULT_MASK_CHOICE_THRESHOLD,
    min_sam_score: float = 0.0,
    sam_mask_threshold: float = 0.5,
    report_box_failures: bool = False,
    debug: bool = False,
) -> List[Tuple[str, str, Any]]:
    """
    Shared SAM extraction loop for Vision+SAM runners.
    For each (group, label, box):
    - if box is valid: run SAM with the converted Qwen3-VL XYXY box
      and choose one mask by `mask_choice_mode`:
        score | fill | score_x_fill | balanced
    - if boxed extraction fails: fallback to text-only extraction (all instances)
    - if no valid box: text-only extraction (all instances)
    Final results are deduplicated by occupied masks, keeping the highest SAM score.
    """
    from PIL import Image
    from scripts.utilities import extract_by_prompt
    from scripts.utilities.sam3_utils import deduplicate_results, is_valid_box

    mode = (mask_choice_mode or DEFAULT_MASK_CHOICE_MODE).strip().lower()
    if mode not in MASK_CHOICE_MODES:
        if debug:
            print(
                f"    [DEBUG] invalid mask_choice_mode={mask_choice_mode!r}; "
                f"using {DEFAULT_MASK_CHOICE_MODE!r}"
            )
        mode = DEFAULT_MASK_CHOICE_MODE
    try:
        mask_th = float(mask_choice_threshold)
    except (TypeError, ValueError):
        mask_th = DEFAULT_MASK_CHOICE_THRESHOLD
    mask_th = max(0.0, min(1.0, mask_th))
    try:
        min_score = float(min_sam_score)
    except (TypeError, ValueError):
        min_score = 0.0
    min_score = max(0.0, min(1.0, min_score))
    try:
        sam_mask_th = float(sam_mask_threshold)
    except (TypeError, ValueError):
        sam_mask_th = 0.5
    sam_mask_th = max(0.0, min(1.0, sam_mask_th))

    with Image.open(image_path) as im:
        img_w, img_h = im.size

    results: List[Tuple[str, str, Any]] = []
    seen_attempts: set[Tuple[str, str, Optional[Tuple[int, int, int, int]]]] = set()
    for group, label, box in progress_items(calls, label=lambda call: call[1]):
        too_small_norm = _normalized_box_too_small_for_image(box, img_w, img_h)
        # Normalized coordinates are not pixels: a width of 2 can span 12 pixels
        # in a 6000px source. Apply the minimum size after conversion.
        positive_box = is_valid_box(box, min_size=1)
        valid_box = positive_box and not too_small_norm
        candidates = sam_box_candidates(box, img_w, img_h) if positive_box else []
        valid_candidates = [c for c in candidates if is_valid_box(c)]
        dedupe_box = valid_candidates[0] if valid_candidates else (candidates[0] if candidates else None)
        dedupe_key = (group.lower().strip(), label.lower().strip(), dedupe_box)
        if dedupe_key in seen_attempts:
            skip_item()
            if debug:
                print(f"    [DEBUG] duplicate call skipped: group={group} label={label} box={dedupe_box}")
            continue
        seen_attempts.add(dedupe_key)

        if debug:
            print(f"    [DEBUG] label={repr(label)} box={box} valid_box={valid_box}")
            if valid_box:
                print(f"    [DEBUG] candidate boxes: {candidates}")
                print(f"    [DEBUG] valid candidate boxes: {valid_candidates}")

        best = None
        best_mode_value = float("-inf")
        best_score = 0.0
        best_fill = 0.0
        if valid_box and valid_candidates:
            for candidate_box in valid_candidates:
                extracts = extract_by_prompt(
                    label,
                    image_path,
                    box=candidate_box,
                    model_and_processor=model_and_processor,
                    mask_threshold=sam_mask_th,
                    threshold=min_score if min_score > 0 else 0.5,
                )
                if min_score > 0.0:
                    extracts = [ex for ex in extracts if float(getattr(ex, "score", 0.0)) >= min_score]
                if not extracts:
                    continue
                candidate_best, candidate_score, candidate_fill = _select_best_extract(
                    extracts,
                    candidate_box,
                    mode=mode,
                    mask_threshold=mask_th,
                )
                candidate_mode_value, _, _ = _mask_choice_value(
                    candidate_best,
                    candidate_box,
                    mode=mode,
                    mask_threshold=mask_th,
                )
                if (
                    best is None
                    or candidate_mode_value > best_mode_value
                    or (
                        candidate_mode_value == best_mode_value
                        and candidate_score > best_score
                    )
                    or (
                        candidate_mode_value == best_mode_value
                        and candidate_score == best_score
                        and candidate_fill > best_fill
                    )
                ):
                    best = candidate_best
                    best_mode_value = candidate_mode_value
                    best_score = candidate_score
                    best_fill = candidate_fill

        if best is not None:
            if debug:
                print(
                    f"    [DEBUG] EXTRACTED: label={repr(label)} "
                    f"mode={mode} value={best_mode_value:.4f} "
                    f"score={best_score:.4f} fill={best_fill:.4f}"
                )
            results.append((group, label, best))
            continue

        if report_box_failures:
            if valid_box:
                if candidates and not valid_candidates:
                    print(
                        f"   Box for '{label}' ({group}) became too small after scaling: "
                        f"{candidates} — fallback to text-only"
                    )
                else:
                    print(
                        f"   Boxed extraction failed for '{label}' ({group}) "
                        f"with candidates={valid_candidates} — fallback to text-only"
                    )
            else:
                if too_small_norm and box is not None:
                    print(
                        f"   Normalized box for '{label}' ({group}) is too small for this image: "
                        f"{box} — fallback to text-only"
                    )
                else:
                    print(f"   No valid box for '{label}' ({group}) — text-only, taking all instances")

        progress_activity("Trying text-guided mask extraction")
        text_only = extract_by_text_only_all(
            group,
            label,
            image_path,
            model_and_processor,
            sam_mask_threshold=sam_mask_th,
            min_sam_score=min_score,
        )
        if min_score > 0.0:
            text_only = [row for row in text_only if float(getattr(row[2], "score", 0.0)) >= min_score]
        if candidates:
            # One localized detection must not turn into every similarly named
            # instance elsewhere in the image when the boxed SAM call fails.
            region = candidates[0]
            text_only = [row for row in text_only if _mask_fraction_in_box(row[2], region, sam_mask_th) >= 0.5]
            if text_only:
                selected, _, _ = _select_best_extract(
                    [row[2] for row in text_only], region, mode=mode, mask_threshold=mask_th
                )
                text_only = [(group, label, selected)]
        if debug:
            tag = "fallback text_only" if valid_box else "text_only"
            print(f"    [DEBUG] {tag}: label={repr(label)} -> {len(text_only)} result(s)")
        if not text_only:
            skip_item()
        results.extend(text_only)

    unique = deduplicate_results(results, tolerance=bbox_dedupe_tolerance, mask_threshold=sam_mask_th)
    if len(unique) < len(results):
        print(f"   Merged {len(results) - len(unique)} repeated segment(s); {len(unique)} unique regions remain.")
    return unique


def save_vision_calls_overlay(
    image_path: Path,
    calls: List[Tuple[str, str, VisionBox]],
    output_path: Path,
) -> Path:
    """
    Save a debug image with all Vision-model boxes and labels drawn on top of the source image.
    """
    from PIL import Image, ImageDraw, ImageFont

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(image_path) as im:
        canvas = im.convert("RGB")
    draw = ImageDraw.Draw(canvas)

    img_w, img_h = canvas.size
    stroke = max(2, min(img_w, img_h) // 400)
    font_size = max(12, min(img_w, img_h) // 45)
    try:
        font = ImageFont.truetype("arial.ttf", size=font_size)
    except Exception:
        font = ImageFont.load_default()

    colors = {
        "people": (220, 20, 60),
        "objects": (60, 125, 255),
        "texts": (0, 140, 80),
        "symbols": (200, 140, 0),
    }

    for idx, (group, label, box) in enumerate(calls, start=1):
        if box is None or len(box) != 4:
            continue
        x1, y1, x2, y2 = sam_box_candidates(box, img_w, img_h)[0]
        x1 = max(0, min(x1, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        x2 = max(0, min(x2, img_w))
        y2 = max(0, min(y2, img_h))
        if x2 <= x1 or y2 <= y1:
            continue

        color = colors.get(group.lower(), (255, 90, 0))
        draw.rectangle((x1, y1, x2, y2), outline=color, width=stroke)

        text = f"#{idx} {group}: {label}"
        tx0, ty0, tx1, ty1 = draw.textbbox((0, 0), text, font=font)
        tw, th = (tx1 - tx0), (ty1 - ty0)
        text_x = x1
        text_y = max(0, y1 - th - 6)
        draw.rectangle((text_x, text_y, text_x + tw + 6, text_y + th + 4), fill=color)
        draw.text((text_x + 3, text_y + 2), text, fill=(0, 0, 0), font=font)

    canvas.save(output_path)
    return output_path


def save_sam_results_overlay(
    image_path: Path,
    results: List[Tuple[str, str, Any]],
    output_path: Path,
    *,
    mask_threshold: float = 0.5,
    mask_alpha: float = 0.35,
) -> Path:
    """
    Save a debug image with SAM results overlaid on the source image.
    - Colored mask fill from ex.mask_original (full-image mask)
    - Bbox and label from each (group, label, ex) row
    """
    from PIL import Image, ImageDraw, ImageFont

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(image_path) as im:
        base = im.convert("RGB")

    img = np.array(base, dtype=np.float32)
    h, w = img.shape[:2]

    # Keep group colors aligned with vision overlay.
    colors = {
        "people": np.array((220, 20, 60), dtype=np.float32),
        "objects": np.array((60, 125, 255), dtype=np.float32),
        "texts": np.array((0, 140, 80), dtype=np.float32),
        "symbols": np.array((200, 140, 0), dtype=np.float32),
    }

    # 1) Blend SAM masks on image.
    for group, _label, ex in results:
        mask = getattr(ex, "mask_original", None)
        if mask is None:
            continue
        arr = np.asarray(mask, dtype=np.float32)
        if arr.ndim > 2:
            arr = np.squeeze(arr)
        if arr.shape != (h, w):
            arr = np.array(
                Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).resize(
                    (w, h), Image.Resampling.LANCZOS
                ),
                dtype=np.float32,
            ) / 255.0
        m = np.clip(arr, 0.0, 1.0)
        m = (m >= float(mask_threshold)).astype(np.float32)[..., None]
        if not np.any(m):
            continue
        c = colors.get(group.lower(), np.array((255, 90, 0), dtype=np.float32))
        img = img * (1.0 - (m * mask_alpha)) + c[None, None, :] * (m * mask_alpha)

    img_u8 = np.clip(img, 0, 255).astype(np.uint8)
    canvas = Image.fromarray(img_u8)
    draw = ImageDraw.Draw(canvas)

    # 2) Draw bbox + labels.
    stroke = max(2, min(w, h) // 400)
    font_size = max(12, min(w, h) // 45)
    try:
        font = ImageFont.truetype("arial.ttf", size=font_size)
    except Exception:
        font = ImageFont.load_default()

    for idx, (group, label, ex) in enumerate(results, start=1):
        bbox = getattr(ex, "bbox", None)
        if bbox is None or len(bbox) != 4:
            continue
        score = float(getattr(ex, "score", 0.0))
        x1, y1, x2, y2 = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            continue
        color_np = colors.get(group.lower(), np.array((255, 90, 0), dtype=np.float32))
        color = (int(color_np[0]), int(color_np[1]), int(color_np[2]))
        draw.rectangle((x1, y1, x2, y2), outline=color, width=stroke)
        text = f"#{idx} {group}: {label} ({score:.2f})"
        tx0, ty0, tx1, ty1 = draw.textbbox((0, 0), text, font=font)
        tw, th = (tx1 - tx0), (ty1 - ty0)
        text_x = x1
        text_y = max(0, y1 - th - 6)
        draw.rectangle((text_x, text_y, text_x + tw + 6, text_y + th + 4), fill=color)
        draw.text((text_x + 3, text_y + 2), text, fill=(0, 0, 0), font=font)

    canvas.save(output_path)
    return output_path


def save_vision_calls_debug_json(
    image_path: Path,
    calls: List[Tuple[str, str, VisionBox]],
    output_path: Path,
) -> Path:
    """
    Save a JSON trace of raw vision boxes and converted pixel boxes used by SAM/overlay.
    This is the ground truth for debugging coordinate alignment issues.
    """
    from PIL import Image

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as im:
        img_w, img_h = im.size

    rows: list[dict[str, object]] = []
    for idx, (group, label, box) in enumerate(calls, start=1):
        pixel_box: list[int] | None = None
        if box is not None:
            candidates = sam_box_candidates(box, img_w, img_h)
            if candidates:
                pixel_box = list(candidates[0])
        rows.append(
            {
                "index": idx,
                "group": group,
                "label": label,
                "raw_box_norm_0_1000": list(box) if box is not None else None,
                "pixel_box_xyxy": pixel_box,
            }
        )

    payload = {
        "source_image": str(image_path),
        "image_size": {"width": img_w, "height": img_h},
        "calls": rows,
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path
