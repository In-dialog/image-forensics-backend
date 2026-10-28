"""Describe image items with application-owned structured output contracts."""
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
from scripts.prompting import ITEM_CONTEXT, compile_description
from scripts.utilities.structured_results import resolve_profile, run_compiled, save_descriptions
from scripts.utilities.analysis_items import enumerate_analysis_items, result_identity, item_evidence_board, original_image_path
from scripts.utilities.ollama_vision import vision_model_session
from scripts.utilities.save_stage_utils import METADATA_FILENAME

MIN_SEGMENT_SIZE = 12  # Skip segment if width or height is smaller than this (px)
SKIP_REASON_TOO_SMALL = "Object too small"


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
    min_item_size: int | None = None,
) -> Path:
    min_side = min_item_size if min_item_size is not None else MIN_SEGMENT_SIZE
    meta_path = run_folder / METADATA_FILENAME
    if not meta_path.exists():
        print(f"No {METADATA_FILENAME} at {run_folder}")
        return run_folder

    with open(meta_path, encoding="utf-8") as f:
        metadata = json.load(f)
    extracts = enumerate_analysis_items(run_folder, metadata)
    progress_steps("Describe items", "Save descriptions")
    if not extracts:
        start_step(1, total=0)
        return run_folder

    profile = resolve_profile("image_description", prompt_profile)
    items: list[dict] = []
    source = original_image_path(run_folder, metadata)
    with vision_model_session(vision_model) as keep_alive:
        for entry in progress_items(extracts, step=1, label=lambda item: item.get("name", "Unnamed item")):
            name, bbox = entry.get("name"), entry.get("bbox")
            if not name or not bbox or len(bbox) != 4:
                skip_item()
                continue
            if min(bbox[2] - bbox[0], bbox[3] - bbox[1]) < min_side:
                print(f"  Skip {name}: {SKIP_REASON_TOO_SMALL}")
                skip_item()
                continue
            board = item_evidence_board(run_folder, entry, source)
            if board is None:
                print(f"  Skip {name}: original image region unavailable")
                skip_item()
                continue
            print(f"  Describing {name}")
            group = "nested image region" if entry.get("parent_name") else "image region"
            compiled = compile_description(profile, group, context=ITEM_CONTEXT)
            result = run_compiled(
                compiled, board, model=vision_model, vision_device=vision_device, num_ctx=num_ctx,
                num_predict=num_predict if num_predict is not None else 2048,
                temperature=temperature, top_k=top_k, top_p=top_p,
                repeat_penalty=repeat_penalty, presence_penalty=presence_penalty, repeat_last_n=repeat_last_n,
                seed=seed, think=think, keep_alive=keep_alive,
            )
            items.append({**result_identity(entry), "answers": result["answers"]})
            save_descriptions(run_folder, profile, items)
            for key, answer in result["answers"].items():
                print(f"    {key}: {answer if answer is not None else 'Not determinable'}")
    start_step(2)
    save_descriptions(run_folder, profile, items)
    print(f"  Wrote description results ({len(items)} entries)")
    return run_folder


def main() -> None:
    import os
    from scripts.utilities.save_stage_utils import DEFAULT_BASE_DIR
    os.chdir(sys._project_root)
    for run_dir in sorted(DEFAULT_BASE_DIR.iterdir()):
        if (run_dir / METADATA_FILENAME).exists():
            print(f"\n=== {run_dir.name} ===")
            execute(run_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
