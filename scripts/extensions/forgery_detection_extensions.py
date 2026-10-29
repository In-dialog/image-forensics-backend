"""
Forgery detection extensions: PhotoHolmes method configs, weight management,
image I/O helpers, and method execution.
"""
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from scripts.runtime.gpu import clear_exception_frames, inference_call, release_cuda_memory
from scripts.runtime.ollama import prepare_torch_model

import contextlib
import cv2
import numpy as np
import torch

WEIGHTS_DIR = Path(os.environ.get("IMAGE_FORENSICS_WEIGHTS", Path.home() / ".cache" / "image-forensics" / "photoholmes"))
# Avoid joblib/loky physical-core probing warnings on some Windows setups.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))


@contextlib.contextmanager
def _torch_load_weights_only_false():
    """Temporarily use weights_only=False for torch.load so PhotoHolmes checkpoints
    (e.g. TruFor) that contain numpy objects load under PyTorch 2.6+."""
    orig_load = torch.load
    def _load(*args, **kwargs):
        kwargs["weights_only"] = False
        return orig_load(*args, **kwargs)
    torch.load = _load
    try:
        yield
    finally:
        torch.load = orig_load


def _to_device(obj: Any, device: str) -> Any:
    """Move object to device when supported by either .to() or .to_device()."""
    if hasattr(obj, "to_device"):
        obj.to_device(device)
        return obj
    if hasattr(obj, "to"):
        return obj.to(device)
    return obj


def _preprocessed_to_device(preprocessed: dict[str, Any], device: str) -> dict[str, Any]:
    """Move any tensors in preprocessed (recursively) to device so they match the model."""
    out: dict[str, Any] = {}
    for k, v in preprocessed.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        elif isinstance(v, dict):
            out[k] = _preprocessed_to_device(v, device)
        elif isinstance(v, (list, tuple)):
            moved = [x.to(device) if isinstance(x, torch.Tensor) else x for x in v]
            out[k] = type(v)(moved)
        else:
            out[k] = v
    return out


def _is_torch_device_mismatch_error(exc: Exception) -> bool:
    """Detect common PyTorch CPU/CUDA tensor mismatch runtime errors."""
    msg = str(exc)
    return (
        "Input type (" in msg
        and "weight type (" in msg
        and "should be the same" in msg
    )

JPEG_EXTENSIONS = {".jpg", ".jpeg"}

# Execution order: TruFor (main), then the rest. Focal excluded from default pipeline.
AVAILABLE_METHODS = [
    "trufor",
    "zero",
    "splicebuster",
    "noisesniffer",
    "adaptive_cfa_net",
    "exif_as_language",
    "psccnet",
]

JPEG_ONLY_METHODS = {"zero"}

METHOD_LABELS = {
    "zero": {
        "label": "JPEG Grid Tampering",
        "description": "Detects splicing by finding misaligned JPEG compression grids",
        "range": "0.0 to 1.0",
        "meaning_0": "no tampering detected",
        "meaning_1": "forgery detected (misaligned JPEG grid)",
    },
    "splicebuster": {
        "label": "Noise Residual Anomaly",
        "description": "Detects splicing by finding inconsistencies in the image noise pattern",
        "range": "heatmap (0.0 to 1.0)",
        "meaning_0": "consistent noise (likely authentic)",
        "meaning_1": "anomalous noise (likely spliced)",
    },
    "noisesniffer": {
        "label": "Noise-Based Forgery",
        "description": "Detects forgery by modelling background noise and finding local deviations",
        "range": "0.0 to 1.0",
        "meaning_0": "no forgery detected",
        "meaning_1": "forgery detected (noise anomaly)",
    },
    "adaptive_cfa_net": {
        "label": "CFA Pattern Anomaly",
        "description": "Detects forgery by finding inconsistencies in the camera sensor color filter array pattern",
        "range": "0.0 to 1.0",
        "meaning_0": "consistent CFA pattern (likely authentic)",
        "meaning_1": "CFA anomaly detected (likely forged)",
    },
    "exif_as_language": {
        "label": "EXIF Metadata Consistency",
        "description": "Detects forgery by finding mismatches between image content and expected camera metadata",
        "range": "0.0 to 1.0",
        "meaning_0": "content matches expected metadata",
        "meaning_1": "content inconsistent with metadata (likely manipulated)",
    },
    "trufor": {
        "label": "TruFor (Trust the Forensics)",
        "description": "State-of-the-art detector combining noise fingerprints and visual features for all-round forgery detection",
        "range": "0.0 to 1.0",
        "meaning_0": "authentic image",
        "meaning_1": "forged image",
    },
    "focal": {
        "label": "FOCAL Contrastive Forgery",
        "description": "Detects forgery using contrastive learning and unsupervised clustering, effective on unseen manipulation types",
        "range": "0.0 to 1.0",
        "meaning_0": "authentic image",
        "meaning_1": "forged image",
    },
    "psccnet": {
        "label": "PSCC-Net Spatial Correlation",
        "description": "Detects and localizes manipulation via progressive spatial-channel correlation analysis",
        "range": "0.0 to 1.0",
        "meaning_0": "authentic image",
        "meaning_1": "manipulated image",
    },
}

WEIGHT_URLS = {
    "adaptive_cfa_net": [
        ("weights.pth", "https://raw.githubusercontent.com/qbammey/adaptive_cfa_forensics/master/src/models/pretrained.pt"),
    ],
    "exif_as_language": [
        ("weights.pth", "https://drive.usercontent.google.com/download?id=1EdEzBhCY9gO3qYvoNMibk8SwAxerDtQp&export=download&authuser=0&confirm=t"),
    ],
    "trufor": [
        ("trufor.pth.tar", "https://www.grip.unina.it/download/prog/TruFor/TruFor_weights.zip"),
    ],
    "focal": [
        ("VIT_weights.pth", "https://drive.usercontent.google.com/download?id=15bkxN0jFUnmVL8vBMbhYdlSFsO8jdebe&export=download&authuser=0&confirm=t"),
        ("HRNET_weights.pth", "https://drive.usercontent.google.com/download?id=1FTeFqF0kjjfoB7U6FLxfKER45oeYOoIP&export=download&authuser=0&confirm=t"),
    ],
    "psccnet": [
        ("FENet.pth", "https://github.com/proteus1991/PSCC-Net/raw/main/checkpoint/HRNet_checkpoint/HRNet.pth"),
        ("SegNet.pth", "https://github.com/proteus1991/PSCC-Net/raw/main/checkpoint/NLCDetection_checkpoint/NLCDetection.pth"),
        ("ClsNet.pth", "https://github.com/proteus1991/PSCC-Net/raw/main/checkpoint/DetectionHead_checkpoint/DetectionHead.pth"),
    ],
}


# ── Weight management ─────────────────────────────────────────────────────────

def download_weight_file(url: str, dest: Path) -> None:
    print(f"    Downloading {dest.name} ...")
    urllib.request.urlretrieve(url, str(dest))


def ensure_weights(method_name: str) -> Path:
    """Download weights if not present. Returns the method's weight directory."""
    if method_name not in WEIGHT_URLS:
        return WEIGHTS_DIR / method_name
    wdir = WEIGHTS_DIR / method_name
    os.makedirs(wdir, exist_ok=True)

    for filename, url in WEIGHT_URLS[method_name]:
        dest = wdir / filename
        if dest.exists():
            continue

        if url.endswith(".zip"):
            zip_path = wdir / (filename + ".tmp.zip")
            download_weight_file(url, zip_path)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(wdir)
            inner = wdir / "weights" / filename
            if inner.exists():
                shutil.move(str(inner), str(dest))
            inner_dir = wdir / "weights"
            if inner_dir.is_dir():
                shutil.rmtree(inner_dir)
            zip_path.unlink(missing_ok=True)
        else:
            download_weight_file(url, dest)

    return wdir


# ── Image I/O ─────────────────────────────────────────────────────────────────

def read_image_as_tensor(path: Path) -> torch.Tensor:
    """Read image as CHW uint8 Tensor (photoholmes format)."""
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb.transpose(2, 0, 1))


def save_heatmap(arr: np.ndarray, out_path: Path) -> None:
    """Save heatmap as grayscale (0–255). Coloring can be applied outside."""
    if arr is None:
        return
    norm = cv2.normalize(arr.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite(str(out_path), norm.astype(np.uint8))


def save_mask(arr: np.ndarray, out_path: Path) -> None:
    if arr is None:
        return
    mask_uint8 = (arr.astype(np.float32) * 255).clip(0, 255).astype(np.uint8)
    cv2.imwrite(str(out_path), mask_uint8)


def tensor_to_numpy(t: torch.Tensor | np.ndarray | None) -> np.ndarray | None:
    if t is None:
        return None
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return t


# ── Method execution ──────────────────────────────────────────────────────────

@inference_call
def run_method(
    method_name: str,
    image_tensor: torch.Tensor,
    device: str | None = None,
) -> dict[str, Any]:
    """Run a single PhotoHolmes method and return raw numpy results."""
    prepare_torch_model()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    # Move image to target device so preprocessing and model run on GPU when device is cuda
    torch_methods = ("adaptive_cfa_net", "exif_as_language", "trufor", "focal", "psccnet")
    if method_name in torch_methods:
        image_tensor = image_tensor.to(device)
    image_data = {"image": image_tensor}

    if method_name == "zero":
        from photoholmes.methods.zero import Zero, zero_preprocessing  # type: ignore[import-untyped]
        preprocessed = zero_preprocessing(**image_data)
        result = Zero().benchmark(**preprocessed)

    elif method_name == "splicebuster":
        from photoholmes.methods.splicebuster import Splicebuster, splicebuster_preprocessing  # type: ignore[import-untyped]
        preprocessed = splicebuster_preprocessing(**image_data)
        result = Splicebuster().benchmark(**preprocessed)

    elif method_name == "noisesniffer":
        from photoholmes.methods.noisesniffer import Noisesniffer, noisesniffer_preprocessing  # type: ignore[import-untyped]
        preprocessed = noisesniffer_preprocessing(**image_data)
        result = Noisesniffer().benchmark(**preprocessed)

    elif method_name == "adaptive_cfa_net":
        from photoholmes.methods.adaptive_cfa_net import AdaptiveCFANet, adaptive_cfa_net_preprocessing  # type: ignore[import-untyped]
        wdir = ensure_weights(method_name)
        preprocessed = adaptive_cfa_net_preprocessing(**image_data)
        preprocessed = _preprocessed_to_device(preprocessed, device)
        with _torch_load_weights_only_false():
            method = _to_device(AdaptiveCFANet(weights=str(wdir / "weights.pth")), device)
        try:
            result = method.benchmark(**preprocessed)
        except RuntimeError as exc:
            # Upstream PhotoHolmes adaptive_cfa_net may create CPU index tensors
            # during predict(), which can break CUDA inference. Retry on CPU only
            # for this specific mismatch.
            if device != "cpu" and _is_torch_device_mismatch_error(exc):
                print("    adaptive_cfa_net CUDA mismatch detected, retrying on CPU...")
                preprocessed = _preprocessed_to_device(preprocessed, "cpu")
                image_tensor = image_tensor.cpu()
                image_data = {"image": image_tensor}
                method = None
                clear_exception_frames(exc)
                release_cuda_memory()
                with _torch_load_weights_only_false():
                    method_cpu = _to_device(AdaptiveCFANet(weights=str(wdir / "weights.pth")), "cpu")
                result = method_cpu.benchmark(**preprocessed)
            else:
                raise

    elif method_name == "exif_as_language":
        from photoholmes.methods.exif_as_language import EXIFAsLanguage, exif_as_language_preprocessing  # type: ignore[import-untyped]
        wdir = ensure_weights(method_name)
        preprocessed = exif_as_language_preprocessing(**image_data)
        preprocessed = _preprocessed_to_device(preprocessed, device)
        with _torch_load_weights_only_false():
            method = _to_device(EXIFAsLanguage(weights=str(wdir / "weights.pth"), device=device), device)
        result = method.benchmark(**preprocessed)

    elif method_name == "trufor":
        from photoholmes.methods.trufor import TruFor, trufor_preprocessing  # type: ignore[import-untyped]
        wdir = ensure_weights(method_name)
        preprocessed = trufor_preprocessing(**image_data)
        # use_confidence=False: use raw forgery heatmap (confidence weighting can dampen scores
        # on AI-generated images). Pass device so TruFor loads on GPU when available.
        with _torch_load_weights_only_false():
            method = TruFor(
                weights=str(wdir / "trufor.pth.tar"),
                use_confidence=False,
                device=device,
            )
            result = method.benchmark(**preprocessed)

    elif method_name == "focal":
        from photoholmes.methods.focal import Focal, focal_preprocessing  # type: ignore[import-untyped]
        wdir = ensure_weights(method_name)
        preprocessed = focal_preprocessing(**image_data)
        preprocessed = _preprocessed_to_device(preprocessed, device)
        with _torch_load_weights_only_false():
            weights = {"ViT": str(wdir / "VIT_weights.pth"), "HRNet": str(wdir / "HRNET_weights.pth")}
            method = _to_device(Focal(weights=weights), device)
        result = method.benchmark(**preprocessed)

    elif method_name == "psccnet":
        from photoholmes.methods.psccnet import PSCCNet, psccnet_preprocessing  # type: ignore[import-untyped]
        wdir = ensure_weights(method_name)
        preprocessed = psccnet_preprocessing(**image_data)
        preprocessed = _preprocessed_to_device(preprocessed, device)
        weights = {"FENet": str(wdir / "FENet.pth"), "SegNet": str(wdir / "SegNet.pth"), "ClsNet": str(wdir / "ClsNet.pth")}
        with _torch_load_weights_only_false():
            method = _to_device(PSCCNet(weights=weights), device)
        try:
            result = method.benchmark(**preprocessed)
        except RuntimeError as exc:
            if device != "cpu" and _is_torch_device_mismatch_error(exc):
                print("    psccnet CUDA mismatch detected, retrying on CPU...")
                preprocessed = _preprocessed_to_device(preprocessed, "cpu")
                image_tensor = image_tensor.cpu()
                image_data = {"image": image_tensor}
                method = None
                clear_exception_frames(exc)
                release_cuda_memory()
                with _torch_load_weights_only_false():
                    method_cpu = _to_device(PSCCNet(weights=weights), "cpu")
                result = method_cpu.benchmark(**preprocessed)
            else:
                raise

    else:
        raise ValueError(f"Unknown method: {method_name}")

    return {
        "heatmap": tensor_to_numpy(result.get("heatmap")),
        "mask": tensor_to_numpy(result.get("mask")),
        "detection": tensor_to_numpy(result.get("detection")),
    }


def score_from_output(
    heatmap: np.ndarray | None,
    mask: np.ndarray | None,
    top_percentile: float = 95,
) -> float:
    """Derive a 0-1 forgery score from a heatmap or mask when the method
    doesn't return an explicit detection score.

    Strategy: take the mean of the top N percentile of pixel values.
    This captures "how suspicious is the most suspicious region" rather than
    averaging over the entire (mostly clean) image.
    """
    arr = heatmap if heatmap is not None else mask
    if arr is None:
        return 0.0
    flat = arr.astype(np.float64).ravel()
    if flat.max() == flat.min():
        return float(flat.mean())
    threshold = np.percentile(flat, top_percentile)
    top_values = flat[flat >= threshold]
    return float(np.clip(top_values.mean(), 0.0, 1.0))


def is_jpeg(image_path: Path) -> bool:
    return image_path.suffix.lower() in JPEG_EXTENSIONS


def should_skip(method_name: str, image_path: Path) -> bool:
    """Return True if method should be skipped for this image format."""
    return method_name in JPEG_ONLY_METHODS and not is_jpeg(image_path)
