"""
Runner: for each segment in metadata.json, run vision + SAM to extract granular sub-segments.
Two clear stages:
  Stage 1 – Vision only: vision model describes each segment and produces instructions for SAM3.
  Stage 2 – SAM only: load SAM once, run extraction from instructions, save sub-segments.
Sub-segment images are saved to granular_segments/{group}/{item_name}/ with a segment_metadata.json.
"""
import gc
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, List, Tuple

# Per-segment SAM instructions from stage 1: (name, group, seg_path, calls)
SamInstructions = Tuple[str, str, Path, List[Tuple[str, str, Any]]]

_p = Path(__file__).resolve().parent
for _ in range(6):
    _ep = _p / "ensure_path.py"
    if _ep.exists():
        exec(_ep.read_text(encoding="utf-8"), {"__file__": str(_ep), "sys": sys, "Path": Path})
        break
    _p = _p.parent
else:
    raise RuntimeError("ensure_path.py not found above this file")

from PIL import Image

from scripts.orchestration.progress import progress_steps, progress_items, progress_activity, start_step, skip_item
from scripts.utilities import load_sam3, offload_vision_model, unload_sam3
from scripts.utilities.image_utils import ensure_dir
from scripts.utilities.structured_results import resolve_profile
from scripts.utilities.analysis_items import granular_directory, safe_component
from scripts.utilities.save_stage_utils import (
    METADATA_FILENAME,
    SEGMENTS_SUBFOLDER,
    _safe_filename,
)
from scripts.utilities.sam3_utils import (
    Sam3Extract,
    build_extracts_metadata,
    named_extracts_for_images,
)
from scripts.extensions import (
    run_sam_calls_with_fallback,
    vision_calls_for_group,
    save_vision_calls_debug_json,
)

GRANULAR_SEGMENTS_SUBFOLDER = "granular_segments"
SEGMENT_METADATA_FILENAME = "segment_metadata.json"
BBOX_DEDUPE_TOLERANCE = 5
MIN_SEGMENT_SIZE = 12  # Skip segment if width or height is smaller than this (px per side)
SKIP_REASON_TOO_SMALL = "Object too small"

# Set True to print why some labels are not extracted (vision parsing, box validity, empty SAM)
DEBUG_EXTRACTION = False

_GROUP_TO_PROMPT_KEY = {
    "people": "people",
    "texts": "texts",
}


def _prompt_key_for_group(group: str) -> str:
    """Map a segment's group to the prompt section key (people, texts, or objects)."""
    return _GROUP_TO_PROMPT_KEY.get(group.lower(), "objects")


def _segment_image_path(run_dir: Path, name: str) -> Path | None:
    segments_dir = run_dir / SEGMENTS_SUBFOLDER
    for ext in (".webp", ".png"):
        p = segments_dir / f"{safe_component(name)}{ext}"
        if p.exists():
            return p
    return None


def _segment_too_small(seg_path: Path, min_side: int = MIN_SEGMENT_SIZE) -> bool:
    """True if image width or height is smaller than min_side (px)."""
    with Image.open(seg_path) as im:
        w, h = im.size
    return w < min_side or h < min_side


def _write_skip_output(run_folder: Path, name: str, group: str, reason: str) -> None:
    """Write segment_metadata.json for a skipped item (e.g. too small)."""
    out_dir = granular_directory(run_folder, name, group)
    ensure_dir(out_dir)
    meta = {"parent_name": name, "parent_group": group, "skip_reason": reason}
    with open(out_dir / SEGMENT_METADATA_FILENAME, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def _original_crop(run_folder: Path, entry: dict, group: str, source: Path | None) -> Path | None:
    """Keep all original pixels inside the parent, including printed and nested details."""
    box = entry.get("bbox")
    if source is None or not isinstance(box, list) or len(box) != 4:
        return None
    with Image.open(source) as image:
        left, top, right, bottom = (int(value) for value in box)
        left, top = max(0, left), max(0, top)
        right, bottom = min(image.width, right), min(image.height, bottom)
        if left >= right or top >= bottom:
            return None
        crop = image.convert("RGB").crop((left, top, right, bottom))
    folder = granular_directory(run_folder, entry["name"], group)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "source.png"
    crop.save(path, "PNG")
    return path


def stage1_vision_instructions(
    segment_image_path: Path,
    parent_group: str,
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
    print_raw_vision_response: bool = True,
) -> List[Tuple[str, str, Any]]:
    """
    Stage 1 – Vision only. Run vision model on segment, parse response into SAM3 instructions (calls).
    Returns list of (group, label, box) calls; empty if no prompt or no valid instructions.
    """
    prompt_key = _prompt_key_for_group(parent_group)
    profile = resolve_profile("analyze_segments", prompt_profile)
    calls = vision_calls_for_group(
        image_path=segment_image_path,
        group=prompt_key,
        prompt_profile=profile,
        max_calls_for_group=max_calls_for_group,
        vision_model=vision_model,
        vision_device=vision_device,
        num_ctx=num_ctx,
        num_predict=num_predict,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repeat_penalty=repeat_penalty, presence_penalty=presence_penalty,
        repeat_last_n=repeat_last_n,
        seed=seed,
        think=think,
        print_response=False,
        print_raw_response=print_raw_vision_response,
        detection_scope="parts",
    )
    if DEBUG_EXTRACTION:
        print(f"    [DEBUG] parsed calls: {len(calls)} -> {calls}")
    return calls


def _run_sam_extraction(
    segment_image_path: Path,
    calls: List[Tuple[str, str, Any]],
    sam: object,
    bbox_dedupe_tolerance: int,
    mask_choice_mode: str = "score",
    mask_choice_threshold: float = 0.5,
    min_sam_score: float = 0.0,
    sam_mask_threshold: float = 0.5,
) -> List[Tuple[str, str, Sam3Extract]]:
    """Run SAM extraction for one segment's instructions. Used in stage 2."""
    return run_sam_calls_with_fallback(
        image_path=segment_image_path,
        calls=calls,
        model_and_processor=sam,
        bbox_dedupe_tolerance=bbox_dedupe_tolerance,
        mask_choice_mode=mask_choice_mode,
        mask_choice_threshold=mask_choice_threshold,
        min_sam_score=min_sam_score,
        sam_mask_threshold=sam_mask_threshold,
        report_box_failures=False,
        debug=DEBUG_EXTRACTION,
    )


def stage2_run_sam(
    instructions: List[SamInstructions],
    run_folder: Path,
    *,
    bbox_dedupe_tolerance: int = BBOX_DEDUPE_TOLERANCE,
    mask_choice_mode: str = "score",
    mask_choice_threshold: float = 0.5,
    min_sam_score: float = 0.0,
    sam_mask_threshold: float = 0.5,
    device: str | None = None,
) -> None:
    """
    Stage 2 – SAM only. Load SAM once, run extraction for each segment's instructions, save outputs.
    """
    if not instructions:
        return
    from scripts.utilities.segmentation_identity import RegionIndex, extract_region, saved_region, repeated_parent

    parent_entries = {
        entry["name"]: entry for entry in json.loads((run_folder / METADATA_FILENAME).read_text(encoding="utf-8")).get("extracts", [])
        if entry.get("name")
    }
    known = RegionIndex()
    for name, entry in parent_entries.items():
        known.add(name, saved_region(_segment_image_path(run_folder, name), entry.get("bbox")))
    sam = load_sam3(device=device)
    try:
        for name, group, seg_path, calls in instructions:
            progress_activity(f"Extracting parts of {name}")
            results = _run_sam_extraction(
                seg_path,
                calls,
                sam,
                bbox_dedupe_tolerance=bbox_dedupe_tolerance,
                mask_choice_mode=mask_choice_mode,
                mask_choice_threshold=mask_choice_threshold,
                min_sam_score=min_sam_score,
                sam_mask_threshold=sam_mask_threshold,
            )
            out_dir = granular_directory(run_folder, name, group)
            ensure_dir(out_dir)
            seg_meta = build_extracts_metadata(results)
            seg_meta["parent_name"] = name
            seg_meta["parent_group"] = group
            seg_meta["coordinate_space"] = "parent_crop"
            seg_meta["source_image"] = seg_path.relative_to(run_folder).as_posix()
            with Image.open(seg_path) as source_crop:
                seg_meta["image_width"], seg_meta["image_height"] = source_crop.size
            parent = parent_entries.get(name, {})
            parent_box = parent.get("bbox")
            offset = None
            if isinstance(parent_box, list) and len(parent_box) == 4:
                left, top = max(0, int(parent_box[0])), max(0, int(parent_box[1]))
                offset = (left, top)
                seg_meta["parent_bbox"] = [left, top, left + seg_meta["image_width"], top + seg_meta["image_height"]]
            images = named_extracts_for_images(results)
            kept, duplicates = [], []
            for child, (img_name, img), (_, label, extract) in zip(seg_meta["extracts"], images, results, strict=True):
                region = None
                if offset is not None:
                    x1, y1, x2, y2 = child["bbox"]
                    child["bbox_global"] = [left + x1, top + y1, left + x2, top + y2]
                    region = extract_region(extract, offset, threshold=sam_mask_threshold)
                    duplicate = known.match(region)
                    reason = "same_mask_region"
                    if duplicate is None and repeated_parent(parent, label, child["bbox_global"]):
                        duplicate, reason = name, "repeated_parent"
                    if duplicate is not None:
                        duplicates.append({**child, "duplicate_of": duplicate, "reason": reason})
                        continue
                fname = _safe_filename(img_name)
                if not fname.endswith((".png", ".webp", ".jpg", ".jpeg", ".bmp")):
                    fname = f"{fname}.webp"
                path = out_dir / fname
                if img.mode in ("RGBA", "LA"):
                    img = img.convert("RGBA")
                else:
                    img = img.convert("RGB")
                img.save(path, "WEBP", quality=90, method=6)
                child["image_path"] = path.relative_to(run_folder).as_posix()
                kept.append(child)
                known.add(f"{name}::{child['name']}", region)
            seg_meta["extracts"] = kept
            if duplicates:
                seg_meta["duplicate_detections"] = duplicates
                print(f"  Merged {len(duplicates)} repeated part(s) for {name} with existing image regions.")
            meta_out = out_dir / SEGMENT_METADATA_FILENAME
            with open(meta_out, "w", encoding="utf-8") as f:
                json.dump(seg_meta, f, indent=2, ensure_ascii=False)
            print(f"  Saved {len(kept)} sub-segment(s) + {SEGMENT_METADATA_FILENAME} -> {out_dir.relative_to(run_folder)}")
            gc.collect()
    finally:
        unload_sam3(sam)
        sam = None


def execute(
    run_folder: Path,
    *,
    vision_model: str = "qwen3-vl:8b-instruct",
    vision_device: str | None = None,
    prompt_profile: dict | None = None,
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
    bbox_dedupe_tolerance: int | None = None,
    mask_choice_mode: str | None = None,
    mask_choice_threshold: float | None = None,
    min_sam_score: float | None = None,
    sam_mask_threshold: float | None = None,
    max_items_people: int | None = None,
    max_items_objects: int | None = None,
    max_items_texts: int | None = None,
    print_raw_vision_response: bool = True,
    device: str | None = None,
    debug_extraction: bool | None = None,
    min_item_size: int | None = None,
) -> Path:
    """
    Analyze every segment: Stage 1 = vision only (instructions for SAM3), Stage 2 = load SAM and extract.
    Saves sub-segments to granular_segments/{group}/{item_name}/ with segment_metadata.json each.
    If debug_extraction is True (or DEBUG_EXTRACTION is True), prints why some labels are not extracted.
    Returns run_folder.
    """
    prev_debug = DEBUG_EXTRACTION
    if debug_extraction is not None:
        globals()["DEBUG_EXTRACTION"] = debug_extraction
    try:
        return _execute_impl(
            run_folder,
            vision_model=vision_model,
            vision_device=vision_device,
            prompt_profile=prompt_profile,
            num_ctx=num_ctx,
            num_predict=num_predict,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repeat_penalty=repeat_penalty, presence_penalty=presence_penalty,
            repeat_last_n=repeat_last_n,
            seed=seed,
            think=think,
            bbox_dedupe_tolerance=bbox_dedupe_tolerance,
            mask_choice_mode=mask_choice_mode,
            mask_choice_threshold=mask_choice_threshold,
            min_sam_score=min_sam_score,
            sam_mask_threshold=sam_mask_threshold,
            max_items_people=max_items_people,
            max_items_objects=max_items_objects,
            max_items_texts=max_items_texts,
            print_raw_vision_response=print_raw_vision_response,
            device=device,
            min_item_size=min_item_size,
        )
    finally:
        if debug_extraction is not None:
            globals()["DEBUG_EXTRACTION"] = prev_debug


def _execute_impl(
    run_folder: Path,
    *,
    vision_model: str = "qwen3-vl:8b-instruct",
    vision_device: str | None = None,
    prompt_profile: dict | None = None,
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
    bbox_dedupe_tolerance: int | None = None,
    mask_choice_mode: str | None = None,
    mask_choice_threshold: float | None = None,
    min_sam_score: float | None = None,
    sam_mask_threshold: float | None = None,
    max_items_people: int | None = None,
    max_items_objects: int | None = None,
    max_items_texts: int | None = None,
    print_raw_vision_response: bool = True,
    device: str | None = None,
    min_item_size: int | None = None,
) -> Path:
    """Inner implementation of execute (used so debug_extraction can be toggled without threading it everywhere)."""
    min_side = min_item_size if min_item_size is not None else MIN_SEGMENT_SIZE
    meta_path = run_folder / METADATA_FILENAME
    if not meta_path.exists():
        print(f"No {METADATA_FILENAME} found at {run_folder}")
        return run_folder

    with open(meta_path, encoding="utf-8") as f:
        metadata = json.load(f)

    extracts = metadata.get("extracts") or []
    progress_steps("Prepare items", "Find parts", "Extract part masks", "Save metadata")
    if not extracts:
        start_step(1, total=0)
        print(f"No extracts in {meta_path}")
        return run_folder

    profile = resolve_profile("analyze_segments", prompt_profile)
    source_value = metadata.get("source_image_path")
    original_source = Path(source_value) if isinstance(source_value, str) and Path(source_value).is_file() else None
    if original_source is None:
        exported = run_folder / "original" / "original.webp"
        original_source = exported if exported.is_file() else None
    max_items_map: dict[str, int | None] = {
        "people": max_items_people,
        "objects": max_items_objects,
        "texts": max_items_texts,
    }

    # Stage 1 – Vision only, per group; offload vision model after each group
    print("  --- Stage 1: Vision (instructions for SAM3) ---")
    by_group: dict[str, List[Tuple[str, str, Path]]] = defaultdict(list)
    for entry in progress_items(extracts, step=1, label=lambda item: item.get("name", "Unnamed item")):
        name = entry.get("name")
        group = entry.get("group") or "objects"
        if not name:
            skip_item()
            continue
        seg_path = _segment_image_path(run_folder, name)
        if seg_path is None:
            skip_item()
            print(f"  Skip {name}: segment image not found")
            continue
        if _segment_too_small(seg_path, min_side=min_side):
            skip_item()
            _write_skip_output(run_folder, name, group, SKIP_REASON_TOO_SMALL)
            print(f"  Skip {name}: {SKIP_REASON_TOO_SMALL}")
            continue
        seg_path = _original_crop(run_folder, entry, group, original_source) or seg_path
        by_group[group].append((name, group, seg_path))

    instructions: List[SamInstructions] = []
    start_step(2, total=sum(len(items) for items in by_group.values()))
    for group_key, segment_list in by_group.items():
        print(f"  --- Group: {group_key} ---")
        for name, group, seg_path in progress_items(segment_list, label=lambda item: item[0]):
            print(f"  --- {name} ---")
            prompt_key = _prompt_key_for_group(group)
            max_calls = max_items_map.get(prompt_key)
            calls = stage1_vision_instructions(
                seg_path, group,
                prompt_profile=profile,
                max_calls_for_group=max_calls,
                vision_model=vision_model,
                vision_device=vision_device,
                num_ctx=num_ctx,
                num_predict=num_predict,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repeat_penalty=repeat_penalty, presence_penalty=presence_penalty,
                repeat_last_n=repeat_last_n,
                seed=seed,
                think=think,
                print_raw_vision_response=print_raw_vision_response,
            )
            trace_dir = granular_directory(run_folder, name, group)
            save_vision_calls_debug_json(seg_path, calls, trace_dir / "vision_calls_debug.json")
            if not calls:
                skip_item()
                print(f"  No SAM instructions for {name}")
                _write_skip_output(run_folder, name, group, "No visible parts detected")
                continue
            instructions.append((name, group, seg_path, calls))
        offload_vision_model(vision_model)

    # Stage 2 – Load SAM once, run extraction, save
    start_step(3, total=sum(len(calls) for _, _, _, calls in instructions), unit="regions", activity="Preparing segmentation")
    if instructions:
        print("  --- Stage 2: SAM extraction ---")
        tol = BBOX_DEDUPE_TOLERANCE if bbox_dedupe_tolerance is None else bbox_dedupe_tolerance
        stage2_run_sam(
            instructions,
            run_folder,
            bbox_dedupe_tolerance=tol,
            mask_choice_mode=(mask_choice_mode or "score"),
            mask_choice_threshold=(0.5 if mask_choice_threshold is None else float(mask_choice_threshold)),
            min_sam_score=(0.0 if min_sam_score is None else float(min_sam_score)),
            sam_mask_threshold=(0.5 if sam_mask_threshold is None else float(sam_mask_threshold)),
            device=device,
        )
    else:
        print("  No segments with instructions; skipping SAM.")

    # Display categories may change independently of these stored asset locations.
    start_step(4)
    for entry in extracts:
        if entry.get("name"):
            part_metadata = granular_directory(run_folder, entry["name"], entry.get("group") or "objects") / SEGMENT_METADATA_FILENAME
            if part_metadata.is_file():
                entry["parts_metadata_path"] = part_metadata.relative_to(run_folder).as_posix()
    from scripts.configuration.store import atomic_json
    atomic_json(meta_path, metadata)
    return run_folder


def main() -> None:
    from scripts.utilities.save_stage_utils import DEFAULT_BASE_DIR
    import os
    os.chdir(sys._project_root)
    for run_dir in sorted(DEFAULT_BASE_DIR.iterdir()):
        meta = run_dir / METADATA_FILENAME
        if meta.exists():
            print(f"\n=== {run_dir.name} ===")
            execute(run_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
