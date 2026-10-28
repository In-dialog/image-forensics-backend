"""Translate validated settings and immutable profile snapshots to the ordered pipeline."""
import json
from pathlib import Path

from scripts.orchestration.execution import RunPlan, StageCall
from scripts.orchestration.stages import STAGE_BY_KEY
from scripts.services.synthetic_api_client import SyntheticApiConfig
from scripts.prompting import validate_profile
from .schema import LLM_KEYS, PROMPT_STAGES, SAM_KEYS
from .store import PROJECT_ROOT

DEVICE_STAGES = {"extract_objects", "depth_map", "deepgaze_msdb", "analyze_segments", "pose_extraction", "forgery_detection"}


def build_plan(folder: Path, image: Path, stages: list[str], settings: dict, prompts_dir: Path) -> RunPlan:
    calls = []
    for key in stages:
        stage = STAGE_BY_KEY[key]
        options = {}
        if key in DEVICE_STAGES:
            device = settings["device"]
            options["device"] = None if device == "auto" else device
        if key in PROMPT_STAGES:
            config_key = PROMPT_STAGES[key]
            cfg = settings["stage_options"][config_key]
            options.update({name: cfg[name] for name in LLM_KEYS})
            profile = json.loads((prompts_dir / f"{config_key}.json").read_text(encoding="utf-8"))
            options.update(vision_model=cfg["model"], vision_device=settings["device"], prompt_profile=validate_profile(config_key, profile))
        if key in {"extract_objects", "analyze_segments"}:
            options.update({name: cfg[name] for name in SAM_KEYS})
        if key == "extract_objects":
            options.update(load_sam_before=True, unload_sam_after=True,
                           save_vision_overlay=cfg["save_vision_overlay"],
                           print_raw_vision_response=cfg["debug_logs"], debug_vision_logs=cfg["debug_logs"])
        if key in {"analyze_segments", "image_description"}:
            options["min_item_size"] = settings["min_item_size_px"]
        if key == "per_item_rankings":
            options["min_region_size"] = cfg["ignore_items_smaller_than"] or 12
        if key == "synthetic_analysis":
            options.update(api_url=settings["synthetic_api"]["url"], run_forgery_after=False)
        calls.append(StageCall(key, options))
    return RunPlan(folder, tuple(calls), image)


def synthetic_config(settings: dict, command: str) -> SyntheticApiConfig:
    api = settings["synthetic_api"]
    return SyntheticApiConfig(api["url"], api["auto_start"], command, api["start_timeout_seconds"], PROJECT_ROOT)
