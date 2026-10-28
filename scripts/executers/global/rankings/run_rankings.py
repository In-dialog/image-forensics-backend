"""Rank each item using its appearance, nearby relationships, and whole-scene context."""
import json
import sys
from pathlib import Path


_p = Path(__file__).resolve().parent
for _ in range(6):
    _ep = _p / "ensure_path.py"
    if _ep.exists():
        exec(_ep.read_text(encoding="utf-8"), {"__file__": str(_ep), "sys": sys, "Path": Path})
        break
    _p = _p.parent
else:
    raise RuntimeError("ensure_path.py not found above this file")

from scripts.orchestration.progress import progress_steps, progress_items, start_step, skip_item
from scripts.prompting import ITEM_CONTEXT, compile_rankings
from scripts.utilities.structured_results import resolve_profile, run_compiled, save_rankings
from scripts.utilities.analysis_items import enumerate_analysis_items, result_identity, item_evidence_board, original_image_path
from scripts.utilities.ollama_vision import vision_model_session
from scripts.utilities.save_stage_utils import METADATA_FILENAME
from scripts.extensions import get_image_path_from_metadata

ORIGINAL_SUBFOLDER = "original"
ORIGINAL_FILENAME = "original.webp"
MIN_REGION_SIZE = 12  # Skip region if width or height is smaller (px)
def _get_full_image_path(run_folder: Path, meta_path: Path) -> Path | None:
    """Prefer run_folder/original/original.webp; else source image from metadata."""
    original = run_folder / ORIGINAL_SUBFOLDER / ORIGINAL_FILENAME
    if original.exists():
        return original
    return get_image_path_from_metadata(meta_path)


def _region_too_small(bbox: list[int], min_side: int = MIN_REGION_SIZE) -> bool:
    """True if bbox (left, top, right, bottom) width or height is smaller than min_side."""
    if len(bbox) != 4:
        return True
    left, top, right, bottom = bbox
    w = right - left
    h = bottom - top
    return w < min_side or h < min_side


def execute(
    run_folder: Path,
    *,
    vision_model: str = "qwen3-vl:8b-instruct",
    vision_device: str | None = None,
    prompt_profile: dict | None = None,
    num_ctx: int | None = 8192,
    num_predict: int | None = 2048,
    temperature: float | None = 0.7,
    top_k: int | None = 20,
    top_p: float | None = 0.8,
    repeat_penalty: float | None = 1,
    presence_penalty: float | None = 1.5,
    repeat_last_n: int | None = None,
    seed: int | None = 42,
    think: bool | str | None = False,
    min_region_size: int | None = None,
) -> Path:
    min_side = min_region_size if min_region_size is not None else MIN_REGION_SIZE
    meta_path = run_folder / METADATA_FILENAME
    if not meta_path.exists():
        print(f"No {METADATA_FILENAME} at {run_folder}")
        return run_folder

    with open(meta_path, encoding="utf-8") as f:
        metadata = json.load(f)
    extracts = enumerate_analysis_items(run_folder, metadata)
    progress_steps("Rank items", "Save rankings")
    if not extracts:
        start_step(1, total=0)
        return run_folder

    image_path = _get_full_image_path(run_folder, meta_path)
    if not image_path or not image_path.exists():
        print(f"No full image found for {run_folder}")
        return run_folder

    profile = resolve_profile("per_item_rankings", prompt_profile)
    items: list[dict] = []
    source = original_image_path(run_folder, metadata) or image_path
    with vision_model_session(vision_model) as keep_alive:
        for entry in progress_items(extracts, step=1, label=lambda item: item.get("name", "Unnamed item")):
            name, bbox = entry.get("name"), entry.get("bbox")
            if not name or not bbox or len(bbox) != 4:
                skip_item()
                continue
            if _region_too_small(bbox, min_side=min_side):
                print(f"  Skip {name}: region smaller than {min_side}px")
                skip_item()
                continue
            board = item_evidence_board(run_folder, entry, source)
            if board is None:
                print(f"  Skip {name}: original image region unavailable")
                skip_item()
                continue
            compiled = compile_rankings(profile, context=ITEM_CONTEXT)
            print(f"  Ranking {name} against {len(profile['criteria'])} criteria")
            result = run_compiled(
                compiled, board, model=vision_model, vision_device=vision_device, num_ctx=num_ctx,
                num_predict=num_predict if num_predict is not None else 2048,
                temperature=temperature, top_k=top_k, top_p=top_p,
                repeat_penalty=repeat_penalty, presence_penalty=presence_penalty, repeat_last_n=repeat_last_n,
                seed=seed, think=think, keep_alive=keep_alive,
            )
            items.append({**result_identity(entry), "ratings": result["ratings"]})
            save_rankings(run_folder, profile, items, scope="per_item")
            for key, rating in result["ratings"].items():
                print(f"    {key}: {rating['value']} - {rating['reason']}")
    start_step(2)
    save_rankings(run_folder, profile, items, scope="per_item")
    print(f"  Wrote per-item ranking results ({len(items)} entries)")
    return run_folder


def main() -> None:
    import os
    from scripts.utilities.save_stage_utils import DEFAULT_BASE_DIR
    from scripts.extensions import find_metadata_files
    os.chdir(sys._project_root)
    meta_files = find_metadata_files(DEFAULT_BASE_DIR)
    for meta_path in sorted(meta_files):
        run_dir = meta_path.parent
        print(f"\n=== {run_dir.name} ===")
        execute(run_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
