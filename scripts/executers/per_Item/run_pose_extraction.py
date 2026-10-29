"""
Runner: find metadata.json in a folder, read extracts with group "people" (person segments),
run OpenPose pose extraction on each segment image, save pose maps and pose.json to
run_folder/pose/. If person_vision.json exists, updates it with pose_image (path to pose
image) per item so the main person vision metadata stays in one place.
"""
import json
import sys
import warnings

import numpy as np
from pathlib import Path
from PIL import Image

# Single source of truth: ensure_path.py at project root (see ensure_path.py)
_p = Path(__file__).resolve().parent
for _ in range(6):
    _ep = _p / "ensure_path.py"
    if _ep.exists():
        exec(open(_ep, encoding="utf-8").read(), {"__file__": str(_ep), "sys": sys, "Path": Path})
        break
    _p = _p.parent
else:
    raise RuntimeError("ensure_path.py not found above this file")

# Suppress noisy warnings from controlnet_aux
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")
warnings.filterwarnings("ignore", message="Overwriting .* in registry")
warnings.filterwarnings("ignore", message="The module 'mediapipe' is not installed")

from scripts.orchestration.progress import progress_steps, progress_items, start_step, skip_item
from scripts.runtime.gpu import inference_call
from scripts.configuration.store import atomic_json
from scripts.utilities.save_stage_utils import METADATA_FILENAME, SEGMENTS_SUBFOLDER
from scripts.utilities.image_utils import ensure_dir
from scripts.utilities.pose_extraction_utils import (
    load_detector,
    extract_pose_and_poses,
    pose_result_to_json,
)
from scripts.extensions import find_metadata_files

DEFAULT_METADATA_FOLDER = Path(sys._project_root) / "outputs"
POSE_SUBFOLDER = "pose"
# Single JSON inside pose/ folder (pose/pose.json)
POSE_JSON_FILENAME = "pose.json"
# Master metadata updated with pose_image per item (same file as run_person_vision)
PERSON_VISION_FILENAME = "person_vision.json"

# Group value in metadata for person segments (canonical from extractor)
PEOPLE_GROUP = "people"


def _person_extracts(metadata: dict) -> list[dict]:
    """Return extracts whose group is 'people' (person segments)."""
    extracts = metadata.get("extracts") or []
    return [e for e in extracts if (e.get("group") or "").lower() == PEOPLE_GROUP]


def _segment_image_path(run_dir: Path, name: str) -> Path | None:
    """Resolve segment image path: segments/{name}.webp or .png. Returns None if missing."""
    segments_dir = run_dir / SEGMENTS_SUBFOLDER
    for ext in (".webp", ".png"):
        p = segments_dir / f"{name}{ext}"
        if p.exists():
            return p
    return None


@inference_call
def execute(
    run_folder: Path | None = None,
    metadata_folder: Path | None = None,
    *,
    model_id: str = "lllyasviel/Annotators",
    hand_and_face: bool = True,
    detect_resolution: int = 512,
    device: str | None = None,
) -> list[Path]:
    """
Run OpenPose pose extraction per person segment; save pose map images and one pose.json
(with all items together) in run_folder/pose/. Skips runs with no person segments.
    If run_folder is set, process only that run. Else scan metadata_folder for metadata.json.
    Returns list of run folder paths written.
    """
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

    progress_steps("Prepare pose model", "Extract poses", "Save pose data")
    start_step(1)
    detector, device_used = load_detector(model_id, device=device)
    print(f"OpenPose on {device_used}")

    written: list[Path] = []
    for meta_path in meta_files:
        run_dir = meta_path.parent
        try:
            with open(meta_path, encoding="utf-8") as f:
                metadata = json.load(f)
        except (json.JSONDecodeError, OSError):
            print(f"Skip {meta_path}: could not load metadata")
            continue

        person_extracts = _person_extracts(metadata)
        if not person_extracts:
            start_step(2, total=0, unit="people")
            atomic_json(run_dir / POSE_SUBFOLDER / POSE_JSON_FILENAME, {"items": []})
            print(f"Skip {run_dir}: no segments labeled person/people")
            continue

        out_dir = run_dir / POSE_SUBFOLDER
        ensure_dir(out_dir)
        print(f"Pose extraction: {run_dir} -> {out_dir} ({len(person_extracts)} person segment(s))")
        items: list[dict] = []

        for entry in progress_items(person_extracts, step=2, unit="people", label=lambda item: item.get("name", "Unnamed person")):
            name = entry.get("name")
            if not name:
                skip_item()
                continue
            seg_path = _segment_image_path(run_dir, name)
            if seg_path is None:
                skip_item()
                print(f"  Skip {name}: segment image not found")
                continue
            pose_pil, poses, W, H = extract_pose_and_poses(
                seg_path,
                detector,
                hand_and_face=hand_and_face,
                detect_resolution=detect_resolution,
            )
            img_path = out_dir / f"{name}_pose.webp"
            arr = np.array(pose_pil.convert("RGBA"))
            # draw_poses renders on a black canvas; make near-black pixels transparent
            black_mask = (arr[:, :, 0] <= 5) & (arr[:, :, 1] <= 5) & (arr[:, :, 2] <= 5)
            arr[black_mask, 3] = 0
            Image.fromarray(arr).save(img_path, "WEBP", quality=90, method=6)
            data = pose_result_to_json(poses, W, H)
            # One segment = one person; flatten so JSON has name first, then points, then rest
            person = (data.get("people") or [{}])[0]
            item = {
                "name": name,
                "points": person.get("points", []),
                "lines": person.get("lines", []),
                "image_width": data.get("image_width"),
                "image_height": data.get("image_height"),
            }
            items.append(item)
            print(f"  Saved {img_path.name} (segment {name})")

        start_step(3)
        out_path = out_dir / POSE_JSON_FILENAME
        atomic_json(out_path, {"items": items})
        print(f"  Saved {out_path.name} ({len(items)} item(s))")

        # Update person_vision.json with pose image name per item (same metadata as segment/person vision)
        pv_path = run_dir / PERSON_VISION_FILENAME
        if pv_path.exists():
            try:
                with open(pv_path, encoding="utf-8") as f:
                    person_vision = json.load(f)
                pose_by_name = {it["name"]: f"{POSE_SUBFOLDER}/{it['name']}_pose.webp" for it in items}
                for item in person_vision.get("items") or []:
                    name = item.get("name")
                    if name in pose_by_name:
                        item["pose_image"] = pose_by_name[name]
                with open(pv_path, "w", encoding="utf-8") as f:
                    json.dump(person_vision, f, indent=2, ensure_ascii=False)
                print(f"  Updated {PERSON_VISION_FILENAME} with pose_image per item")
            except (json.JSONDecodeError, OSError) as e:
                print(f"  Skip update {PERSON_VISION_FILENAME}: {e}")
        written.append(run_dir)
    return written


def main() -> None:
    run_folders = execute()
    print(f"\nDone. Pose extraction in {len(run_folders)} folder(s).")


if __name__ == "__main__":
    main()
