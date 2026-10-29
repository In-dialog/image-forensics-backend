"""
Runner: find metadata.json in a folder, read the main image path (source_image_path),
run Nonescape synthetic classifier on that image. Write result to synthetic_analysis.json
in the run folder. Does not write to metadata.
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest

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

from scripts.orchestration.progress import progress_steps, start_step
from scripts.runtime.gpu import inference_call
from scripts.runtime.ollama import prepare_torch_model
from scripts.utilities.save_stage_utils import METADATA_FILENAME
from scripts.extensions import find_metadata_files, get_image_path_from_metadata

# "global" is a reserved keyword; use importlib to import from scripts.executers.global
_run_forgery = __import__("scripts.executers.global.rankings.run_forgery_detection", fromlist=["execute"])
run_forgery_detection = _run_forgery.execute

DEFAULT_METADATA_FOLDER = Path(sys._project_root) / "outputs"
SYNTHETIC_ANALYSIS_FILENAME = "synthetic_analysis.json"
DEFAULT_MODEL_PATH = "nonescape-v0.safetensors"
HF_REPO_ID = "e3ntity/nonescape-v0"
HF_FILENAME = "nonescape-v0.safetensors"
DEFAULT_SYNTHETIC_API_URL = os.getenv("SYNTHETIC_API_URL", "http://127.0.0.1:8765")


def _resolve_model_path(model: str) -> str:
    """Return path to safetensors file: use local path if it exists, else download from Hugging Face."""
    from huggingface_hub import hf_hub_download

    p = Path(model)
    if p.is_file():
        return str(p.resolve())
    # Default: fetch from Hugging Face
    if model == DEFAULT_MODEL_PATH or p.name == HF_FILENAME:
        print(f"Downloading {HF_FILENAME} from Hugging Face ({HF_REPO_ID})...")
        return hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME)
    raise FileNotFoundError(f"Model file not found: {model}")


def _load_local_classifier(resolved_model_path: str) -> Any:
    from nonescape import NonescapeClassifier

    return NonescapeClassifier.from_pretrained(resolved_model_path)


@inference_call
def _predict_local(resolved_model_path: str, image_path: Path) -> tuple[float, float]:
    import torch
    from PIL import Image
    from nonescape import preprocess_image

    prepare_torch_model()
    classifier = _load_local_classifier(resolved_model_path)
    img = Image.open(image_path).convert("RGB")
    x = preprocess_image(img).unsqueeze(0)
    with torch.no_grad():
        probs = classifier(x)
    return float(probs[0, 0].item()), float(probs[0, 1].item())


def _join_api_url(base_url: str, endpoint: str) -> str:
    return f"{base_url.rstrip('/')}{endpoint}"


def _api_health(api_url: str, timeout_s: float = 5.0) -> dict[str, Any]:
    url = _join_api_url(api_url, "/health")
    with urlrequest.urlopen(url, timeout=timeout_s) as resp:
        data = resp.read().decode("utf-8")
    return json.loads(data)


def _predict_via_api(
    image_path: Path,
    *,
    api_url: str,
    model_path: str | None = None,
    timeout_s: float = 30.0,
) -> tuple[float, float]:
    payload: dict[str, Any] = {"image_path": str(image_path)}
    if model_path:
        payload["model_path"] = model_path
    body = json.dumps(payload).encode("utf-8")
    req = urlrequest.Request(
        _join_api_url(api_url, "/classify"),
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlrequest.urlopen(req, timeout=timeout_s) as resp:
        data = resp.read().decode("utf-8")
    out = json.loads(data)
    probs = out.get("probs", {})
    real_prob = float(probs.get("real", 0.0))
    synthetic_prob = float(probs.get("synthetic", 0.0))
    return real_prob, synthetic_prob


def execute(
    run_folder: Path | None = None,
    metadata_folder: Path | None = None,
    *,
    model_path: str | Path | None = None,
    run_forgery_after: bool = True,
    api_url: str | None = DEFAULT_SYNTHETIC_API_URL,
) -> list[Path]:
    """
    Run Nonescape synthetic classifier on the source image; write result to
    synthetic_analysis.json in each run folder. Does not modify metadata. Returns list of run folder paths updated.
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

    model = str(model_path) if model_path is not None else DEFAULT_MODEL_PATH
    progress_steps("Prepare classifier", "Assess image", "Save findings")
    start_step(1)
    use_api = bool(api_url and str(api_url).strip())

    if use_api:
        try:
            health = _api_health(str(api_url))
            if health.get("unload_after_request") is not True:
                raise RuntimeError("Restart the synthetic API to enable automatic model unloading.")
            print(f"Synthetic API ready: {api_url} (ok={health.get('ok', True)})")
        except Exception as exc:
            raise RuntimeError(
                f"Synthetic API unavailable at {api_url}: {exc}. "
                "Start the optional service with python install/start_synthetic.py or run with --no-api."
            ) from exc

    if not use_api:
        resolved = _resolve_model_path(model)

    written: list[Path] = []
    for meta_path in meta_files:
        image_path = get_image_path_from_metadata(meta_path)
        if image_path is None:
            print(f"Skip {meta_path}: no valid image path")
            continue
        run_folder = meta_path.parent
        print(f"Synthetic analysis: {image_path} -> {run_folder}")
        start_step(2)
        if use_api:
            try:
                real_prob, synthetic_prob = _predict_via_api(
                    image_path,
                    api_url=str(api_url),
                    model_path=model_path if model_path is not None else None,
                )
            except (urlerror.URLError, TimeoutError, ValueError, KeyError) as exc:
                raise RuntimeError(f"API inference failed for {image_path}: {exc}") from exc
        else:
            real_prob, synthetic_prob = _predict_local(resolved, image_path)

        # probs: [real, synthetic]; index 1 = synthetic
        is_synthetic = synthetic_prob > 0.5
        confidence = float(synthetic_prob if is_synthetic else real_prob)
        # Same structure as other full_image_rankings: label, description, score, raw, range, meaning_0/5/10
        synthetic_object = {
            "label": "AI / Synthetic Detection",
            "description": "Detects whether the image is AI-generated or synthetic (vs real/camera-captured).",
            "score": round(10 * synthetic_prob, 1),
            "raw": f"AI Generated: {str(is_synthetic).lower()}",
            "raw_probs": {"real": round(real_prob, 6), "synthetic": round(synthetic_prob, 6)},
            "inference_source": "api" if use_api else "local",
            "range": "0 to 10",
            "meaning_0": "real / not AI-generated",
            "meaning_5": "ambiguous",
            "meaning_10": "synthetic / AI-generated",
        }
        out_path = run_folder / SYNTHETIC_ANALYSIS_FILENAME
        start_step(3)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(synthetic_object, f, indent=2, ensure_ascii=False)
        print(f"  AI Generated: {is_synthetic}, Confidence: {confidence}")
        print(f"  Wrote {SYNTHETIC_ANALYSIS_FILENAME}")
        written.append(run_folder)

    # Optional: run forgery detection on the same run folder(s)
    if run_forgery_after:
        for run_folder in written:
            run_forgery_detection(run_folder=run_folder)

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Run synthetic analysis with optional forgery chaining.")
    parser.add_argument(
        "--api-url",
        default=DEFAULT_SYNTHETIC_API_URL,
        help="Synthetic inference API base URL (default: env SYNTHETIC_API_URL or http://127.0.0.1:8765).",
    )
    parser.add_argument(
        "--no-api",
        action="store_true",
        help="Disable API mode and force local nonescape inference.",
    )
    parser.add_argument(
        "--no-forgery",
        action="store_true",
        help="Do not run forgery detection after synthetic analysis.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Optional local safetensors path for nonescape model.",
    )
    args = parser.parse_args()

    run_folders = execute(
        run_forgery_after=not args.no_forgery,
        model_path=args.model_path,
        api_url=None if args.no_api else args.api_url,
    )
    print(f"\nDone. {SYNTHETIC_ANALYSIS_FILENAME} written for {len(run_folders)} folder(s).")


if __name__ == "__main__":
    main()
