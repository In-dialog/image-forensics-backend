"""Shared utilities for pipelines (image, depth, SAM, save, vision). Extensions live in scripts.extensions."""
from .image_utils import (
    colorize_depth,
    ensure_dir,
    ensure_rgb_hwc,
    load_rgb,
    normalize_to_01,
    normalize_to_uint8,
    numpy_to_pil,
    pil_to_numpy,
    resize_long_side,
)
from .depth_utils import DepthExtractResult, extract_depth
from .ollama_vision import offload_vision_model, vision_chat
from .save_stage_utils import save_stage
from .sam3_utils import (
    Sam3Extract,
    Sam3Loaded,
    build_extracts_metadata,
    extract_by_prompt,
    is_valid_box,
    load_sam3,
    unload_all_sam3,
    unload_sam3,
)
from .composition_lines_utils import get_base_line_indices, get_composition_lines
from .runtime_cleanup import cleanup_runtime

__all__ = [
    "colorize_depth",
    "DepthExtractResult",
    "ensure_dir",
    "extract_depth",
    "ensure_rgb_hwc",
    "load_rgb",
    "normalize_to_01",
    "normalize_to_uint8",
    "numpy_to_pil",
    "pil_to_numpy",
    "resize_long_side",
    "save_stage",
    "Sam3Extract",
    "Sam3Loaded",
    "build_extracts_metadata",
    "extract_by_prompt",
    "is_valid_box",
    "load_sam3",
    "unload_all_sam3",
    "unload_sam3",
    "offload_vision_model",
    "vision_chat",
    "cleanup_runtime",
    "get_base_line_indices",
    "get_composition_lines",
]
