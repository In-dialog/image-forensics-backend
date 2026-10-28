"""Canonical execution order and backend names, without UI or model imports."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Stage:
    key: str
    label: str
    group: str
    runner: str


STAGES = (
    Stage("extract_objects", "Extract Objects", "global", "run_extract_objects"),
    Stage("bbox_composition", "Bounding Box Composition", "global", "run_bbox_composition"),
    Stage("depth_map", "Depth Map", "global", "run_depth_map"),
    Stage("negative_space", "Negative Space", "global", "run_negative_space"),
    Stage("composition_lines", "Composition Lines", "global", "run_composition_lines"),
    Stage("deepgaze_msdb", "Visual Attention (DeepGaze)", "global", "run_deepgaze_msdb"),
    Stage("original_export", "Original Image Export", "global", "run_original"),
    Stage("analyze_segments", "Analyze Segments", "per_item", "run_analyze_segments"),
    Stage("pose_extraction", "Pose Extraction", "per_item", "run_pose_extraction"),
    Stage("image_description", "Image Description", "per_item", "run_image_description"),
    Stage("forgery_detection", "Forgery Detection", "ranking", "run_forgery_detection"),
    Stage("full_image_rankings", "Full Image Rankings", "ranking", "run_full_image_rankings"),
    Stage("per_item_rankings", "Per-Item Rankings", "ranking", "run_rankings"),
    Stage("synthetic_analysis", "Synthetic Analysis", "ranking", "run_synthetic_analysis"),
)
STAGE_BY_KEY = {stage.key: stage for stage in STAGES}
