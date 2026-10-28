"""
Runner: find metadata.json in a folder, read the main image path (source_image_path),
run DeepGaze MSDB saliency on that image, save grayscale heatmap as saliency_deepgaze_msdb.webp in run_folder/saliency_deepgaze_msdb/.
"""
import sys
from pathlib import Path

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

from scripts.orchestration.progress import start_step
from scripts.utilities.save_stage_utils import METADATA_FILENAME
from scripts.utilities.image_utils import ensure_dir
from scripts.utilities.deepgaze_msdb_utils import run_deepgaze_msdb, save_heatmap
from scripts.extensions import find_metadata_files, get_image_path_from_metadata

DEFAULT_METADATA_FOLDER = Path(sys._project_root) / "outputs"
SALIENCY_DEEPGAZE_MSDB_SUBFOLDER = "saliency_deepgaze_msdb"


def execute(
    run_folder: Path | None = None,
    metadata_folder: Path | None = None,
    *,
    long_side: int = 1024,
    pixel_per_dva: float = 35.0,
    dataset: str | None = None,
    w_deepgaze: float = 0.5,
    w_text: float = 0.2,
    w_faces: float = 0.25,
    w_objects: float = 0.05,
    device: str | None = None,
) -> list[Path]:
    """
    Run DeepGaze MSDB saliency; save grayscale heatmap as saliency_deepgaze_msdb.webp in run_folder/saliency_deepgaze_msdb/.
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

    written: list[Path] = []
    for meta_path in meta_files:
        image_path = get_image_path_from_metadata(meta_path)
        if image_path is None:
            print(f"Skip {meta_path}: no valid image path")
            continue
        run_folder = meta_path.parent
        print(f"DeepGaze MSDB: {image_path} -> {run_folder}")
        prob_final, _ = run_deepgaze_msdb(
            image_path,
            long_side=long_side,
            pixel_per_dva=pixel_per_dva,
            dataset=dataset,
            w_deepgaze=w_deepgaze,
            w_text=w_text,
            w_faces=w_faces,
            w_objects=w_objects,
            device=device,
        )
        start_step(3)
        out_dir = run_folder / SALIENCY_DEEPGAZE_MSDB_SUBFOLDER
        ensure_dir(out_dir)
        heat_path = out_dir / "saliency_deepgaze_msdb.webp"
        save_heatmap(prob_final, heat_path, grayscale=True)
        print(f"  Saved {out_dir.name}/{heat_path.name}")
        written.append(run_folder)
    return written


def main() -> None:
    run_folders = execute()
    print(f"\nDone. DeepGaze MSDB saliency in {len(run_folders)} folder(s).")


if __name__ == "__main__":
    main()
