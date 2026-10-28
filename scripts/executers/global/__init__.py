# Global pipeline executers (extract objects, bbox, depth, negative space, composition lines, deepgaze, original).
# Re-exports are done from scripts.executers so callers use: from scripts.executers import run_* .

from .run_extract_objects import execute as run_extract_objects
from .run_bbox_composition import execute as run_bbox_composition
from .run_depth_map import execute as run_depth_map
from .run_negative_space import execute as run_negative_space
from .run_composition_lines import execute as run_composition_lines
from .run_deepgaze_msdb import execute as run_deepgaze_msdb
from .run_original import execute as run_original
from .rankings.run_rankings import execute as run_rankings
from .rankings.run_full_image_rankings import execute as run_full_image_rankings
from .rankings.run_forgery_detection import execute as run_forgery_detection
from .rankings.run_synthetic_analysis import execute as run_synthetic_analysis

__all__ = [
    "run_extract_objects",
    "run_bbox_composition",
    "run_depth_map",
    "run_negative_space",
    "run_composition_lines",
    "run_deepgaze_msdb",
    "run_original",
    "run_rankings",
    "run_full_image_rankings",
    "run_forgery_detection",
    "run_synthetic_analysis",
]
