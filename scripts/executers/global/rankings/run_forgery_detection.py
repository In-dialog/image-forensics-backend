"""
Runner: for each run folder with metadata.json, run all available PhotoHolmes
forgery-detection methods on the source image and save heatmaps / masks /
detection scores into run_folder/forgery_detection/<method>/.
Writes forgery_detection.json with human-readable results.
"""
import json
import sys
from pathlib import Path
from typing import Any

_p = Path(__file__).resolve().parent
for _ in range(6):
    _ep = _p / "ensure_path.py"
    if _ep.exists():
        exec(open(_ep, encoding="utf-8").read(), {"__file__": str(_ep), "sys": sys, "Path": Path})
        break
    _p = _p.parent
else:
    raise RuntimeError("ensure_path.py not found above this file")

import torch
from scripts.orchestration.progress import progress_steps, progress_items, start_step, skip_item, fail_item
from scripts.utilities.save_stage_utils import METADATA_FILENAME
from scripts.utilities.image_utils import ensure_dir
from scripts.extensions import (
    find_metadata_files,
    get_image_path_from_metadata,
    FORGERY_METHODS,
    FORGERY_METHOD_LABELS,
    read_image_as_tensor,
    run_forgery_method,
    forgery_score_from_output,
    save_heatmap,
    save_mask,
    should_skip_forgery_method,
)

DEFAULT_METADATA_FOLDER = Path(sys._project_root) / "outputs"
FORGERY_SUBFOLDER = "forgery_detection"
RESULTS_FILENAME = "forgery_detection.json"


def _process_image(
    image_path: Path,
    out_dir: Path,
    methods: list[str],
    existing: dict[str, Any],
    device: str | None = None,
) -> dict[str, Any]:
    """Run all requested methods on a single image. Returns metadata dict."""
    progress_steps("Prepare image", "Run forensic checks", "Save findings")
    start_step(1)
    image_tensor = read_image_as_tensor(image_path)
    results_meta: dict[str, Any] = dict(existing)

    for name in progress_items(methods, step=2, unit="checks", label=lambda name: FORGERY_METHOD_LABELS[name]["label"]):
        if name in existing and "error" not in existing[name]:
            skip_item()
            print(f"  {name}: already done, skipping")
            continue

        if should_skip_forgery_method(name, image_path):
            skip_item()
            print(f"  Skipping {name} (JPEG-only, image is {image_path.suffix})")
            continue

        method_dir = out_dir / name
        ensure_dir(method_dir)
        print(f"  Running {name}...")
        try:
            result = run_forgery_method(name, image_tensor, device=device)
        except Exception as exc:
            fail_item()
            print(f"  {name} failed: {exc}")
            results_meta[name] = {"error": str(exc)}
            continue

        info = FORGERY_METHOD_LABELS[name]
        entry: dict[str, Any] = {
            "label": info["label"],
            "description": info["description"],
        }

        heatmap = result["heatmap"]
        if heatmap is not None:
            heatmap_path = method_dir / "heatmap.png"
            save_heatmap(heatmap, heatmap_path)
            entry["heatmap_file"] = heatmap_path.name

        mask = result["mask"]
        if mask is not None:
            mask_path = method_dir / "mask.png"
            save_mask(mask, mask_path)
            entry["mask_file"] = mask_path.name

        detection = result["detection"]
        # TruFor: use heatmap-derived score so full-image (e.g. AI) forgeries are reflected;
        # the detection scalar is tuned for local tampering and often underestimates.
        if name == "trufor" and heatmap is not None:
            score = round(forgery_score_from_output(heatmap, mask), 4)
            entry["score_derived_from"] = "heatmap"
        elif detection is not None:
            score = round(float(detection.item()) if detection.size == 1 else float(detection.mean()), 4)
        else:
            score = round(forgery_score_from_output(heatmap, mask), 4)
            entry["score_derived_from"] = "heatmap" if heatmap is not None else "mask"

        entry["score"] = score
        entry["range"] = info["range"]
        entry["meaning_0"] = info["meaning_0"]
        entry["meaning_1"] = info["meaning_1"]

        results_meta[name] = entry
        print(f"  {info['label']}: {score}  ({name})")

    return results_meta


def execute(
    run_folder: Path | None = None,
    metadata_folder: Path | None = None,
    *,
    methods: list[str] | None = None,
    device: str | None = None,
) -> list[Path]:
    """
    Run PhotoHolmes forgery-detection methods on the source image in each run folder.
    Saves heatmaps/masks to run_folder/forgery_detection/<method>/ and writes
    forgery_detection.json with human-readable scores and labels.
    """
    if methods is None:
        methods = list(FORGERY_METHODS)
    for m in methods:
        if m not in FORGERY_METHODS:
            raise ValueError(f"Unknown method '{m}'. Choose from: {FORGERY_METHODS}")

    if run_folder is not None:
        meta_path = run_folder / METADATA_FILENAME
        meta_files = [meta_path] if meta_path.exists() else []
    else:
        folder = metadata_folder if metadata_folder is not None else DEFAULT_METADATA_FOLDER
        meta_files = find_metadata_files(folder)

    if not meta_files:
        base = run_folder if run_folder is not None else (metadata_folder or DEFAULT_METADATA_FOLDER)
        print(f"No {METADATA_FILENAME} found at {base}")
        return []

    # Resolve "auto" (None) to cuda when available so GPU is used by default
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Forgery detection device: {device}")

    written: list[Path] = []
    for meta_path in meta_files:
        image_path = get_image_path_from_metadata(meta_path)
        if image_path is None:
            print(f"Skip {meta_path}: no valid image path")
            continue

        run_dir = meta_path.parent
        out_dir = run_dir / FORGERY_SUBFOLDER
        ensure_dir(out_dir)

        results_path = run_dir / RESULTS_FILENAME
        existing: dict[str, Any] = {}
        if results_path.exists():
            with open(results_path, "r", encoding="utf-8") as f:
                existing = json.load(f)

        print(f"Forgery detection: {image_path}")
        results_meta = _process_image(image_path, out_dir, methods, existing, device=device)
        start_step(3)

        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results_meta, f, indent=2, ensure_ascii=False)

        print(f"  Wrote {RESULTS_FILENAME}")
        written.append(run_dir)

    return written


def main() -> None:
    run_folders = execute()
    print(f"\nDone. Forgery detection in {len(run_folders)} folder(s).")


if __name__ == "__main__":
    main()
