"""Rank a complete image with protected typed result contracts."""
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

from scripts.orchestration.progress import progress_steps, start_step
from scripts.prompting import compile_rankings
from scripts.utilities.structured_results import resolve_profile, run_compiled, save_rankings
from scripts.utilities.image_utils import load_rgb
from scripts.utilities.save_stage_utils import METADATA_FILENAME
from scripts.extensions import get_image_path_from_metadata

ORIGINAL_SUBFOLDER = "original"
ORIGINAL_FILENAME = "original.webp"
def _get_full_image_path(run_folder: Path, meta_path: Path) -> Path | None:
    """Prefer run_folder/original/original.webp; else source image from metadata."""
    original = run_folder / ORIGINAL_SUBFOLDER / ORIGINAL_FILENAME
    if original.exists():
        return original
    return get_image_path_from_metadata(meta_path)


def execute(
    run_folder: Path,
    *,
    vision_model: str = "qwen3-vl",
    vision_device: str | None = None,
    prompt_profile: dict | None = None,
    num_ctx: int | None = 12288,
    num_predict: int | None = 6144,
    temperature: float | None = 0.6,
    top_k: int | None = 20,
    top_p: float | None = 0.95,
    repeat_penalty: float | None = 1,
    presence_penalty: float | None = 0,
    repeat_last_n: int | None = None,
    seed: int | None = 42,
    think: bool | str | None = True,
) -> Path:
    meta_path = run_folder / METADATA_FILENAME
    if not meta_path.exists():
        print(f"No {METADATA_FILENAME} at {run_folder}")
        return run_folder

    image_path = _get_full_image_path(run_folder, meta_path)
    if not image_path or not image_path.exists():
        print(f"No full image found for {run_folder}")
        return run_folder

    try:
        load_rgb(image_path)
    except (OSError, ValueError) as e:
        print(f"Could not load image {image_path}: {e}")
        return run_folder

    profile = resolve_profile("full_image_rankings", prompt_profile)
    progress_steps("Assess whole image", "Save rankings")
    start_step(1, activity=f"Assessing {len(profile['criteria'])} criteria together")
    compiled = compile_rankings(profile, context="The entire image composition as a whole.")
    print(f"  Ranking the full image against {len(profile['criteria'])} criteria")
    result = run_compiled(
        compiled, image_path, model=vision_model, vision_device=vision_device, num_ctx=num_ctx,
        num_predict=num_predict if num_predict is not None else 6144,
        temperature=temperature, top_k=top_k, top_p=top_p,
        repeat_penalty=repeat_penalty, presence_penalty=presence_penalty, repeat_last_n=repeat_last_n,
        seed=seed, think=think,
    )
    start_step(2)
    save_rankings(
        run_folder, profile, [{"name": "full_image", "ratings": result["ratings"]}],
        scope="full_image",
    )
    for key, rating in result["ratings"].items():
        print(f"  {key}: {rating['value']} - {rating['reason']}")
    print("  Wrote full-image ranking results")
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
