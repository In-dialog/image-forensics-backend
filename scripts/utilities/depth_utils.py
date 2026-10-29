"""
Depth extraction utility using Marigold (diffusers). Returns depth map and derived views.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from scripts.runtime.gpu import inference_call, release_cuda_memory
from scripts.runtime.ollama import prepare_torch_model

import numpy as np
from PIL import Image

from .image_utils import (
    colorize_depth,
    load_rgb,
    normalize_to_uint8,
    strip_border_artifact,
)

# Defaults; callers do not need to change these unless tuning.
DEFAULT_MODEL_ID = "prs-eth/marigold-depth-v1-1"
DEFAULT_NUM_INFERENCE_STEPS = 2


@dataclass
class DepthExtractResult:
    """Result of depth extraction: float map and derived uint8 views."""

    depth: np.ndarray  # (H, W) float, higher = closer to camera
    depth_u8: np.ndarray  # (H, W) grayscale uint8, min–max normalized
    depth_color: np.ndarray  # (H, W, 3) RGB uint8, colorized for visualization


@inference_call
def extract_depth(
    image: Path | str | Image.Image,
    *,
    model_id: str = DEFAULT_MODEL_ID,
    num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
    device: Optional[str] = None,
) -> DepthExtractResult:
    """
    Run Marigold depth estimation on the original image (no resizing). Returns depth map
    (same size as input) and grayscale/colorized views. Convention: higher depth value = closer.

    Args:
        image: Image path (Path or str) or PIL Image (RGB).
        model_id: Hugging Face model id (default: prs-eth/marigold-depth-v1-1).
        num_inference_steps: Inference steps (1–4 recommended for v1-1).
        device: "cuda" or "cpu"; None = auto.

    Returns:
        DepthExtractResult with .depth (float), .depth_u8 (grayscale), .depth_color (RGB).

    Raises:
        FileNotFoundError: If image is a path that does not exist.
        ImportError: If diffusers/torch are not installed.
    """
    if isinstance(image, (Path, str)):
        path = Path(image)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        pil_image = load_rgb(path)
    else:
        pil_image = image.convert("RGB")

    try:
        import torch
        from diffusers import DDIMScheduler, MarigoldDepthPipeline
    except ImportError as err:
        raise ImportError("Install: pip install diffusers torch Pillow") from err

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = (
        torch.bfloat16
        if (device == "cuda" and torch.cuda.is_bf16_supported())
        else torch.float32
    )

    prepare_torch_model()
    pipe = MarigoldDepthPipeline.from_pretrained(model_id)
    pipe.scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing"
    )
    pipe = pipe.to(device=device, dtype=dtype)
    pipe.set_progress_bar_config(disable=True)

    w, h = pil_image.size
    output = pipe(pil_image, num_inference_steps=num_inference_steps)
    depth = output.prediction
    if hasattr(depth, "cpu"):
        depth = depth.cpu().numpy()
    depth = np.squeeze(depth)
    # Invert so higher = closer (foreground = high)
    depth_np = 1.0 - depth.astype(np.float64)
    del pipe, output, depth
    release_cuda_memory()

    # Pipeline may pad output to multiples of 8; crop back to original size
    if depth_np.shape[0] > h or depth_np.shape[1] > w:
        depth_np = depth_np[:h, :w].copy()

    # Remove thin border artifacts (left/top often from pipeline edge)
    depth_np = strip_border_artifact(depth_np, width=1)

    depth_u8 = normalize_to_uint8(depth_np)
    depth_color = colorize_depth(depth_np)

    return DepthExtractResult(depth=depth_np, depth_u8=depth_u8, depth_color=depth_color)
