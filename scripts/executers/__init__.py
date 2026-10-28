# Runners / executers that use the utilities library.
# Re-export execute from each so callers can: from scripts.executers import run_extract_objects, ...
# Note: subpackage is named "global" (reserved word), so we import it via __import__.

_g = __import__("scripts.executers.global", fromlist=[
    "run_extract_objects", "run_bbox_composition", "run_depth_map", "run_negative_space",
    "run_composition_lines", "run_deepgaze_msdb", "run_original", "run_rankings",
    "run_forgery_detection", "run_full_image_rankings", "run_synthetic_analysis",
])
run_extract_objects = _g.run_extract_objects
run_bbox_composition = _g.run_bbox_composition
run_depth_map = _g.run_depth_map
run_negative_space = _g.run_negative_space
run_composition_lines = _g.run_composition_lines
run_deepgaze_msdb = _g.run_deepgaze_msdb
run_original = _g.run_original
run_rankings = _g.run_rankings
run_forgery_detection = _g.run_forgery_detection
run_full_image_rankings = _g.run_full_image_rankings
run_synthetic_analysis = _g.run_synthetic_analysis
from scripts.executers.per_Item import run_pose_extraction, run_analyze_segments, run_image_description

__all__ = [
    "run_extract_objects",
    "run_bbox_composition",
    "run_depth_map",
    "run_negative_space",
    "run_composition_lines",
    "run_deepgaze_msdb",
    "run_pose_extraction",
    "run_analyze_segments",
    "run_image_description",
    "run_original",
    "run_rankings",
    "run_forgery_detection",
    "run_full_image_rankings",
    "run_synthetic_analysis",
]
