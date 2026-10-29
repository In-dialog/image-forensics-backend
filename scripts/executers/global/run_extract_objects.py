"""
Runner: vision (Ollama) lists items + boxes [x1,y1,x2,y2] → SAM 3 extracts segments per call.
Uses scripts.utilities (vision_chat, sam3) and scripts.extensions (planning + shared SAM loop).
"""
import gc
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

_p = Path(__file__).resolve().parent
for _ in range(6):
    _ep = _p / "ensure_path.py"
    if _ep.exists():
        exec(_ep.read_text(encoding="utf-8"), {"__file__": str(_ep), "sys": sys, "Path": Path})
        break
    _p = _p.parent
else:
    raise RuntimeError("ensure_path.py not found above this file")

from scripts.orchestration.progress import progress_steps, progress_items, start_step
from scripts.runtime.gpu import release_after_call
from scripts.utilities import load_sam3, unload_sam3
from scripts.extensions import (
    DETECTION_GROUPS,
    labels_from_plan,
    run_sam_calls_with_fallback,
    save_sam_results_overlay,
    save_vision_calls_debug_json,
    save_vision_calls_overlay,
    vision_calls_for_group,
)
from scripts.utilities.sam3_utils import (
    Sam3Extract,
    build_extracts_metadata,
    named_extracts_for_images,
)
from scripts.utilities.save_stage_utils import save_stage
from scripts.utilities.structured_results import resolve_profile
from scripts.prompting import compile_detection
from scripts.extensions.extractor_extensions import MAX_CALLS_PER_GROUP
from scripts.utilities.detection_views import MAX_SCENE_DETECTIONS

BBOX_DEDUPE_TOLERANCE = 5
ENV_MASK_CHOICE_MODE = "SAM_MASK_CHOICE_MODE"
ENV_MASK_CHOICE_THRESHOLD = "SAM_MASK_CHOICE_THRESHOLD"
ENV_MIN_SAM_SCORE = "SAM_MIN_SCORE"
ENV_SAM_MASK_THRESHOLD = "SAM_MASK_THRESHOLD"


def _unique_labels(calls: List[Tuple[str, str, tuple[int, int, int, int] | None]]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for _, label, _ in calls:
        key = label.lower().strip()
        if key in seen:
            continue
        seen.add(key)
        out.append(label)
    return out


@release_after_call
def run_extract_objects(
    image_path: Path,
    *,
    prompt_profile: dict | None = None,
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
    bbox_dedupe_tolerance: int | None = None,
    mask_choice_mode: str | None = None,
    mask_choice_threshold: float | None = None,
    min_sam_score: float | None = None,
    sam_mask_threshold: float | None = None,
    max_items_people: int | None = None,
    max_items_objects: int | None = None,
    max_items_texts: int | None = None,
    print_raw_vision_response: bool = False,
    debug_vision_logs: bool = False,
    load_sam_before: bool = False,
    unload_sam_after: bool = True,
    device: str | None = None,
    detail_dir: Path | None = None,
) -> Tuple[List[Tuple[str, str, Sam3Extract]], List[str], List[Tuple[str, str, tuple[int, int, int, int] | None]]]:
    """Vision lists items with boxes [x1,y1,x2,y2] → SAM 3 extracts; returns (results, labels_for_sam, calls)."""
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    print(f"1. Vision: list items by type (per group) — {image_path}")

    # 1. Vision per group (single shared parser flow per group)
    profile = resolve_profile("object_detect", prompt_profile)
    calls: List[Tuple[str, str, tuple[int, int, int, int] | None]] = []
    max_items_map: dict[str, int | None] = {
        "people": max_items_people,
        "objects": max_items_objects,
        "texts": max_items_texts,
    }
    progress_steps("Find regions", "Extract masks", "Save outputs")
    for group in progress_items(DETECTION_GROUPS, step=1, unit="groups"):
        print(f"   Vision ({group})...")
        group_calls = vision_calls_for_group(
            image_path=image_path,
            group=group,
            prompt_profile=profile,
            max_calls_for_group=max_items_map.get(group),
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
            print_response=debug_vision_logs,
            print_raw_response=print_raw_vision_response,
            scene_detail_pass=detail_dir is not None,
            detail_dir=detail_dir,
        )
        calls.extend(group_calls)
        if not debug_vision_logs:
            labels = _unique_labels(group_calls)
            print(f"   Detected ({group}): {', '.join(labels) if labels else '(none)'}")
        gc.collect()

    labels_for_sam = labels_from_plan(calls)
    print(f"Labels for SAM: {labels_for_sam}")
    mode_eff = (mask_choice_mode or os.environ.get(ENV_MASK_CHOICE_MODE, "score")).strip().lower()
    threshold_eff: float
    if mask_choice_threshold is not None:
        threshold_eff = float(mask_choice_threshold)
    else:
        raw = os.environ.get(ENV_MASK_CHOICE_THRESHOLD, "0.5").strip()
        try:
            threshold_eff = float(raw)
        except ValueError:
            threshold_eff = 0.5
    threshold_eff = max(0.0, min(1.0, threshold_eff))
    min_score_eff: float
    if min_sam_score is not None:
        min_score_eff = float(min_sam_score)
    else:
        raw = os.environ.get(ENV_MIN_SAM_SCORE, "0.0").strip()
        try:
            min_score_eff = float(raw)
        except ValueError:
            min_score_eff = 0.0
    min_score_eff = max(0.0, min(1.0, min_score_eff))
    sam_mask_th_eff: float
    if sam_mask_threshold is not None:
        sam_mask_th_eff = float(sam_mask_threshold)
    else:
        raw = os.environ.get(ENV_SAM_MASK_THRESHOLD, "0.5").strip()
        try:
            sam_mask_th_eff = float(raw)
        except ValueError:
            sam_mask_th_eff = 0.5
    sam_mask_th_eff = max(0.0, min(1.0, sam_mask_th_eff))
    print(
        f"SAM mask selection: mode={mode_eff}, threshold={threshold_eff:.2f}, "
        f"min_score={min_score_eff:.2f}, sam_mask_th={sam_mask_th_eff:.2f}"
    )

    start_step(2, total=len(calls), unit="regions", activity="Preparing segmentation" if calls else None)
    if not calls:
        return ([], [], [])

    # 3. SAM per call (box: one best extract; no box: text-only, take all)
    print("2. SAM 3: one call per (label, box) or text-only when no box...")
    sam = load_sam3(device=device) if load_sam_before else None
    try:
        tol = BBOX_DEDUPE_TOLERANCE if bbox_dedupe_tolerance is None else bbox_dedupe_tolerance
        results = run_sam_calls_with_fallback(
            image_path=image_path,
            calls=calls,
            model_and_processor=sam,
            bbox_dedupe_tolerance=tol,
            mask_choice_mode=mode_eff,
            mask_choice_threshold=threshold_eff,
            min_sam_score=min_score_eff,
            sam_mask_threshold=sam_mask_th_eff,
            report_box_failures=debug_vision_logs,
            debug=debug_vision_logs,
        )
        return (results, labels_for_sam, calls)
    finally:
        if load_sam_before and unload_sam_after and sam is not None:
            unload_sam3(sam)
            sam = None


def execute(
    image_path: Path,
    *,
    prompt_profile: dict | None = None,
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
    bbox_dedupe_tolerance: int | None = None,
    mask_choice_mode: str | None = None,
    mask_choice_threshold: float | None = None,
    min_sam_score: float | None = None,
    sam_mask_threshold: float | None = None,
    max_items_people: int | None = None,
    max_items_objects: int | None = None,
    max_items_texts: int | None = None,
    print_raw_vision_response: bool = False,
    debug_vision_logs: bool = False,
    load_sam_before: bool = False,
    unload_sam_after: bool = True,
    device: str | None = None,
    base_dir: Path | str | None = None,
    save_vision_overlay: bool = True,
) -> Path:
    """Run pipeline and save to outputs/{stem}/segments/ + metadata.json. Returns run folder path."""
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    print(f"Image: {image_path}")
    from scripts.utilities.save_stage_utils import DEFAULT_BASE_DIR
    expected_folder = (Path(base_dir) if base_dir is not None else DEFAULT_BASE_DIR) / image_path.stem
    detail_dir = expected_folder / "vision_details"
    if not detail_dir.resolve().is_relative_to(expected_folder.resolve()):
        raise ValueError("Detection evidence must stay inside the run folder")
    results, labels_for_sam, calls = run_extract_objects(
        image_path,
        prompt_profile=prompt_profile,
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
        bbox_dedupe_tolerance=bbox_dedupe_tolerance,
        mask_choice_mode=mask_choice_mode,
        mask_choice_threshold=mask_choice_threshold,
        min_sam_score=min_sam_score,
        sam_mask_threshold=sam_mask_threshold,
        max_items_people=max_items_people,
        max_items_objects=max_items_objects,
        max_items_texts=max_items_texts,
        print_raw_vision_response=print_raw_vision_response,
        debug_vision_logs=debug_vision_logs,
        load_sam_before=load_sam_before,
        unload_sam_after=unload_sam_after,
        device=device,
        detail_dir=detail_dir,
    )

    images = named_extracts_for_images(results)
    metadata = build_extracts_metadata(results)
    start_step(3)
    run_folder = save_stage(
        images=images,
        metadata=metadata,
        llm_list=labels_for_sam,
        base_dir=base_dir,
        source_image_path=image_path,
    )
    if save_vision_overlay:
        overlay_path = run_folder / "vision_boxes_overlay.png"
        save_vision_calls_overlay(image_path, calls, overlay_path)
        print(f"Saved vision overlay: {overlay_path}")
        sam_overlay_path = run_folder / "sam_results_overlay.png"
        save_sam_results_overlay(image_path, results, sam_overlay_path)
        print(f"Saved SAM overlay: {sam_overlay_path}")
    # Preserve the exact application-owned contract alongside editable analysis goals.
    prompts_dir = run_folder / "vision_prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    profile = resolve_profile("object_detect", prompt_profile)
    caps = {"people": max_items_people, "objects": max_items_objects, "texts": max_items_texts}
    for group in DETECTION_GROUPS:
        cap = caps[group] if caps[group] is not None else MAX_CALLS_PER_GROUP
        compiled = compile_detection(profile, group, max_items=min(cap, MAX_SCENE_DETECTIONS))
        snapshot = {"system": compiled.system, "user": compiled.user, "schema": compiled.schema}
        (prompts_dir / f"{group}.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    print(f"Saved vision prompts: {prompts_dir}")
    debug_json_path = run_folder / "vision_calls_debug.json"
    save_vision_calls_debug_json(image_path, calls, debug_json_path)
    print(f"Saved vision calls debug: {debug_json_path}")
    return run_folder


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Extract objects from an image using Ollama and SAM 3.")
    parser.add_argument("image", type=Path, help="Path to the image to analyze")
    parser.add_argument("--output", type=Path, default=Path("data"), help="Result directory (default: data)")
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error(f"Image not found: {args.image}")
    execute(args.image.resolve(), base_dir=args.output.resolve(), load_sam_before=True, unload_sam_after=True)


if __name__ == "__main__":
    main()
