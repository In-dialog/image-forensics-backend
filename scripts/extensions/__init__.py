"""Extensions that build on scripts.utilities (depth, structured extraction, etc.)."""
from .depth_extensions import add_z_to_extracts, run_depth_and_save
from .extractor_extensions import (
    extract_by_text_only_all,
    DETECTION_GROUPS,
    labels_from_plan,
    run_sam_calls_with_fallback,
    sam_box_candidates,
    save_sam_results_overlay,
    save_vision_calls_debug_json,
    save_vision_calls_overlay,
    vision_calls_for_group,
)
from .composition_lines_extensions import find_metadata_files, get_image_path_from_metadata
from .negative_space_extensions import compute_negative_space_image
from .bbox_composition_extensions import (
    bbox_to_edge_lines,
    add_bbox_composition_to_extracts,
)
from .forgery_detection_extensions import (
    AVAILABLE_METHODS as FORGERY_METHODS,
    METHOD_LABELS as FORGERY_METHOD_LABELS,
    ensure_weights as ensure_forgery_weights,
    read_image_as_tensor,
    run_method as run_forgery_method,
    save_heatmap,
    save_mask,
    score_from_output as forgery_score_from_output,
    should_skip as should_skip_forgery_method,
    tensor_to_numpy,
)

__all__ = [
    "add_z_to_extracts",
    "run_depth_and_save",
    "extract_by_text_only_all",
    "DETECTION_GROUPS",
    "labels_from_plan",
    "compute_negative_space_image",
    "find_metadata_files",
    "get_image_path_from_metadata",
    "run_sam_calls_with_fallback",
    "sam_box_candidates",
    "save_sam_results_overlay",
    "save_vision_calls_debug_json",
    "save_vision_calls_overlay",
    "vision_calls_for_group",
    "bbox_to_edge_lines",
    "add_bbox_composition_to_extracts",
    "FORGERY_METHODS",
    "FORGERY_METHOD_LABELS",
    "ensure_forgery_weights",
    "read_image_as_tensor",
    "run_forgery_method",
    "save_heatmap",
    "save_mask",
    "forgery_score_from_output",
    "should_skip_forgery_method",
    "tensor_to_numpy",
]
