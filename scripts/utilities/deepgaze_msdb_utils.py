"""
DeepGaze MSDB saliency: run model, optional text/face/object priors, fusion, and save heatmap/overlay.
Reusable by tests and by the run_deepgaze_msdb executor. Uses image_utils.resize_long_side.
"""
import io
from pathlib import Path
from scripts.runtime.gpu import inference_call, release_cuda_memory
from scripts.runtime.ollama import prepare_torch_model
from scripts.orchestration.progress import progress_steps, progress_items, progress_activity, start_step

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter

from .image_utils import load_rgb, resize_long_side

# Optional at import; required when running model or priors
try:
    import deepgaze_pytorch
    from deepgaze_pytorch import MSDBDataset
except ImportError:
    deepgaze_pytorch = None
    MSDBDataset = None


def logsumexp_np(x: np.ndarray) -> float:
    m = np.max(x)
    return float(m + np.log(np.sum(np.exp(x - m))))


def heat_for_display(prob: np.ndarray, percentile: float = 99.5, gamma: float = 0.35) -> np.ndarray:
    """Normalize with percentile, clip, then gamma boost for visibility."""
    p = np.percentile(prob, percentile)
    heat = prob / (p + 1e-12)
    heat = np.clip(heat, 0.0, 1.0)
    heat = heat**gamma
    return heat


def mass_mask(prob: np.ndarray, mass: float = 0.8) -> np.ndarray:
    """Smallest set of pixels that contains the given fraction of probability mass."""
    flat = prob.ravel()
    idx = np.argsort(-flat)
    c = np.cumsum(flat[idx])
    cut = idx[c <= mass]
    mask = np.zeros_like(flat, dtype=bool)
    mask[cut] = True
    return mask.reshape(prob.shape)


def _bboxes_to_prob(h: int, w: int, bboxes: list, sigma_scale: float = 0.25) -> np.ndarray:
    """Turn list of (x, y, box_w, box_h) into a normalized probability map (Gaussian at each center)."""
    out = np.zeros((h, w), dtype=np.float64)
    for (x, y, bw, bh) in bboxes:
        cx = x + bw / 2.0
        cy = y + bh / 2.0
        sigma = max(bw, bh) * sigma_scale
        sigma = max(sigma, 4.0)
        ys = np.arange(h, dtype=np.float64)
        xs = np.arange(w, dtype=np.float64)
        gx = np.exp(-0.5 * ((xs - cx) / sigma) ** 2)
        gy = np.exp(-0.5 * ((ys - cy) / sigma) ** 2)
        out += np.outer(gy, gx)
    if out.sum() > 0:
        out = out / out.sum()
    else:
        out = np.ones((h, w), dtype=np.float64) / (h * w)
    return out


def get_text_prior(rgb: np.ndarray, h: int, w: int) -> np.ndarray:
    """Text prior: slogans, captions, watermarks. Returns (h,w) prob summing to 1."""
    try:
        import pytesseract

        pil_img = Image.fromarray(rgb)
        data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT)
        bboxes = []
        for i in range(len(data["left"])):
            if data["text"][i].strip():
                bboxes.append(
                    (
                        data["left"][i],
                        data["top"][i],
                        data["width"][i],
                        data["height"][i],
                    )
                )
        return _bboxes_to_prob(h, w, bboxes)
    except Exception:
        return np.ones((h, w), dtype=np.float64) / (h * w)


def get_face_prior(rgb: np.ndarray, h: int, w: int) -> np.ndarray:
    """Face prior: faces/eyes dominate gaze. Returns (h,w) prob summing to 1."""
    try:
        import cv2

        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(path)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        bboxes = [tuple(int(z) for z in f) for f in faces]
        return _bboxes_to_prob(h, w, bboxes)
    except Exception:
        return np.ones((h, w), dtype=np.float64) / (h * w)


def get_object_prior(rgb: np.ndarray, h: int, w: int) -> np.ndarray:
    """Optional object prior. Placeholder: uniform."""
    return np.ones((h, w), dtype=np.float64) / (h * w)


def save_heatmap(prob: np.ndarray, out_path: Path, *, grayscale: bool = False) -> None:
    """Save saliency heatmap as WebP. prob is HxW, sums to 1. If grayscale, save as L mode."""
    heat = heat_for_display(prob)
    if grayscale:
        arr = (np.clip(heat, 0.0, 1.0) * 255).astype(np.uint8)
        Image.fromarray(arr, mode="L").save(out_path, "WEBP", quality=90)
        return
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 6))
    plt.axis("off")
    plt.imshow(heat, interpolation="nearest")
    plt.tight_layout(pad=0)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close()
    buf.seek(0)
    Image.open(buf).convert("RGB").save(out_path, "WEBP", quality=90)


def save_overlay(
    rgb: np.ndarray, prob: np.ndarray, out_path: Path, alpha: float = 0.45
) -> None:
    """Save RGB with saliency overlay as WebP (grayscale base + heat on mass mask)."""
    import matplotlib.pyplot as plt

    gray = np.dot(rgb[..., :3].astype(np.float32), [0.299, 0.587, 0.114]).astype(np.uint8)
    rgb_bw = np.stack([gray, gray, gray], axis=-1)
    heat = heat_for_display(prob)
    mask = mass_mask(prob, mass=0.8)
    heat_masked = np.where(mask, heat, np.nan)
    plt.figure(figsize=(6, 6))
    plt.axis("off")
    plt.imshow(rgb_bw)
    plt.imshow(heat_masked, alpha=alpha, interpolation="nearest")
    plt.tight_layout(pad=0)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close()
    buf.seek(0)
    Image.open(buf).convert("RGB").save(out_path, "WEBP", quality=90)


@inference_call
def run_deepgaze_msdb(
    image_path: Path,
    *,
    long_side: int = 1024,
    pixel_per_dva: float = 35.0,
    dataset: str | None = None,
    w_deepgaze: float = 0.5,
    w_text: float = 0.2,
    w_faces: float = 0.25,
    w_objects: float = 0.05,
    device: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run DeepGaze MSDB on the image; fuse with optional text/face/object priors.
    Returns (prob_final, rgb) at resized size (long_side), both numpy.
    """
    if deepgaze_pytorch is None or MSDBDataset is None:
        raise ImportError("deepgaze_pytorch is required for run_deepgaze_msdb")

    progress_steps("Estimate attention", "Inspect text, faces and objects", "Save heatmap")
    start_step(1, activity="Preparing attention model")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    img = load_rgb(image_path)
    img = resize_long_side(img, long_side)
    rgb = np.array(img, dtype=np.uint8)
    h, w, _ = rgb.shape

    prepare_torch_model()
    centerbias = np.zeros((h, w), dtype=np.float32)
    centerbias = centerbias - logsumexp_np(centerbias)
    centerbias_tensor = torch.tensor(centerbias[None, ...], dtype=torch.float32, device=device)
    image_tensor = torch.tensor(
        rgb.transpose(2, 0, 1)[None, ...], dtype=torch.float32, device=device
    )

    dataset_arg = None
    if dataset is not None and MSDBDataset is not None:
        dataset_arg = getattr(MSDBDataset, str(dataset).upper(), None)

    model = deepgaze_pytorch.DeepGazeMSDB(pretrained=True).to(device)
    model.eval()
    progress_activity("Estimating visual attention")

    with torch.no_grad():
        log_density = model(
            image_tensor,
            centerbias_tensor,
            pixel_per_dva=pixel_per_dva,
            dataset=dataset_arg,
        )

    if log_density.ndim == 4:
        log_density = log_density[0, 0]
    elif log_density.ndim == 3:
        log_density = log_density[0]
    else:
        raise RuntimeError(f"Unexpected log_density shape {tuple(log_density.shape)}")

    if tuple(log_density.shape) != (h, w):
        log_density = F.interpolate(
            log_density[None, None, ...],
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )[0, 0]

    log_density_np = log_density.detach().cpu().numpy()
    del model, image_tensor, centerbias_tensor, log_density
    release_cuda_memory()
    m = np.max(log_density_np)
    prob = np.exp(log_density_np - m)
    prob = prob / (prob.sum() + 1e-12)
    prob_deepgaze = gaussian_filter(prob, sigma=12.0)
    prob_deepgaze = prob_deepgaze / (prob_deepgaze.sum() + 1e-12)

    priors = [("Text", get_text_prior), ("Faces", get_face_prior), ("Objects", get_object_prior)]
    prob_text, prob_faces, prob_objects = [
        compute(rgb, h, w) for _, compute in progress_items(priors, step=2, unit="checks", label=lambda item: item[0])
    ]

    prob_final = (
        w_deepgaze * prob_deepgaze
        + w_text * prob_text
        + w_faces * prob_faces
        + w_objects * prob_objects
    )
    prob_final = prob_final / (prob_final.sum() + 1e-12)

    return prob_final, rgb
