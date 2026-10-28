"""JSON configuration accepted by the local analysis service."""
from __future__ import annotations

import copy
import math
from urllib.parse import urlparse

from scripts.orchestration.stages import STAGES, STAGE_BY_KEY
from scripts.prompting import validate_profiles
from scripts.prompting.profiles import STAGE_KINDS

PROMPT_KEYS = tuple(STAGE_KINDS)
PROMPT_STAGES = {"extract_objects": "object_detect", **{key: key for key in PROMPT_KEYS if key != "object_detect"}}
LLM_KEYS = ("num_ctx", "num_predict", "temperature", "top_k", "top_p", "repeat_penalty", "presence_penalty", "repeat_last_n", "seed", "think")
SAM_KEYS = ("bbox_dedupe_tolerance", "mask_choice_mode", "mask_choice_threshold", "min_sam_score", "sam_mask_threshold", "max_items_people", "max_items_objects", "max_items_texts")


def defaults() -> dict:
    llm = dict(model="qwen3-vl:8b-instruct", num_ctx=8192, num_predict=2048, temperature=0,
               top_k=30, top_p=0.75, repeat_penalty=1.25, presence_penalty=0, repeat_last_n=256, seed=42, think=False)
    options = {key: copy.deepcopy(llm) for key in PROMPT_KEYS}
    for key in ("object_detect", "analyze_segments"):
        options[key].update(bbox_dedupe_tolerance=5, mask_choice_mode="score",
                            mask_choice_threshold=0.5, min_sam_score=0.1, sam_mask_threshold=0.5,
                            max_items_people=15, max_items_objects=15, max_items_texts=15)
    options["object_detect"].update(bbox_dedupe_tolerance=20, mask_choice_mode="fill",
                                    mask_choice_threshold=0.4, sam_mask_threshold=0.2,
                                    max_items_people=25, max_items_objects=50,
                                    save_vision_overlay=True, debug_logs=False)
    for key in ("image_description", "per_item_rankings"):
        options[key].update(temperature=0.7, top_p=0.8, top_k=20, repeat_penalty=1.0, presence_penalty=1.5)
    options["full_image_rankings"].update(model="qwen3-vl:8b-thinking", num_ctx=12288, num_predict=6144, think=True,
                                          temperature=0.6, top_p=0.95, top_k=20, repeat_penalty=1.0, presence_penalty=0)
    options["per_item_rankings"]["ignore_items_smaller_than"] = 12
    return dict(device="auto",
                min_item_size_px=12, selected_stages=[stage.key for stage in STAGES if stage.key not in {"forgery_detection", "synthetic_analysis"}],
                stage_options=options,
                synthetic_api=dict(url="http://127.0.0.1:8765", auto_start=True, start_timeout_seconds=20))


def default_settings() -> dict:
    return defaults()


def merge_known(base: dict, patch: object, path: str = "settings") -> dict:
    if not isinstance(patch, dict):
        raise ValueError(f"{path} must be an object.")
    result = copy.deepcopy(base)
    for key, value in patch.items():
        if key not in result:
            raise ValueError(f"Unknown setting: {path}.{key}")
        result[key] = merge_known(result[key], value, f"{path}.{key}") if isinstance(result[key], dict) else value
    return result


def number(value, name: str, *, integer=False, minimum=None, maximum=None, nullable=True):
    if value is None or value == "":
        if nullable:
            return None
        raise ValueError(f"{name} is required.")
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"{name} must be a number.")
    try:
        parsed = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a number.") from exc
    if not math.isfinite(parsed) or (integer and not parsed.is_integer()):
        raise ValueError(f"{name} must be a finite {'integer' if integer else 'number'}.")
    if (minimum is not None and parsed < minimum) or (maximum is not None and parsed > maximum):
        raise ValueError(f"{name} is outside its allowed range ({minimum} to {maximum}).")
    return int(parsed) if integer else parsed


def selected_stages(value: object, *, require_extraction=False) -> list[str]:
    if not isinstance(value, list) or not value or any(not isinstance(key, str) or key not in STAGE_BY_KEY for key in value):
        raise ValueError("Select at least one known analysis stage.")
    if len(value) != len(set(value)):
        raise ValueError("Duplicate analysis stages are not allowed.")
    if require_extraction and "extract_objects" not in value:
        raise ValueError("Extract Objects is required when starting an analysis from a new image.")
    return [stage.key for stage in STAGES if stage.key in value]


def validate_settings(raw: object) -> dict:
    result = merge_known(default_settings(), raw)
    if not isinstance(result["device"], str) or result["device"] not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda.")
    result["selected_stages"] = selected_stages(result["selected_stages"])
    result["min_item_size_px"] = number(result["min_item_size_px"], "min_item_size_px", integer=True, minimum=1, maximum=100000, nullable=False)
    ranges = {
        "num_ctx": (True, 1, 10000000), "num_predict": (True, -2, 10000000),
        "temperature": (False, 0, 100), "top_k": (True, 0, 1000000), "top_p": (False, 0, 1),
        "repeat_penalty": (False, 0, 100), "presence_penalty": (False, -2, 2), "repeat_last_n": (True, -1, 10000000),
        "seed": (True, -(2**53 - 1), 2**53 - 1), "bbox_dedupe_tolerance": (True, 0, 100000),
        "mask_choice_threshold": (False, 0, 1), "min_sam_score": (False, 0, 1), "sam_mask_threshold": (False, 0, 1),
        "max_items_people": (True, 1, 100000), "max_items_objects": (True, 1, 100000),
        "max_items_texts": (True, 1, 100000), "ignore_items_smaller_than": (True, 1, 100000),
    }
    for key, options in result["stage_options"].items():
        for name, value in options.items():
            label = f"{key}.{name}"
            if name in ranges:
                integer, minimum, maximum = ranges[name]
                options[name] = number(value, label, integer=integer, minimum=minimum, maximum=maximum)
            elif name == "model":
                if not isinstance(value, str) or not value.strip() or len(value) > 200 or any(ord(c) < 32 for c in value):
                    raise ValueError(f"{label} must be a model name.")
                options[name] = value.strip()
            elif name == "think":
                if isinstance(value, str):
                    value = {"true": True, "false": False, "": None}.get(value.lower(), value.lower())
                if value is not None and not isinstance(value, bool) and value not in ("low", "medium", "high"):
                    raise ValueError(f"{label} must be true, false, low, medium, high, or null.")
                options[name] = value
            elif name == "mask_choice_mode":
                if value not in (None, "score", "fill", "score_x_fill", "balanced"):
                    raise ValueError(f"{label} is not a supported mask mode.")
            elif not isinstance(value, bool):
                raise ValueError(f"{label} must be a boolean.")
    api = result["synthetic_api"]
    if not isinstance(api["url"], str):
        raise ValueError("synthetic_api.url must be a local HTTP URL.")
    parsed = urlparse(api["url"])
    try:
        valid = parsed.scheme in {"http", "https"} and parsed.hostname in {"127.0.0.1", "localhost", "::1"} and parsed.port is not None
    except ValueError:
        valid = False
    if not valid or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in ("", "/"):
        raise ValueError("synthetic_api.url must be a local HTTP URL with a port.")
    api["url"] = api["url"].rstrip("/")
    if not isinstance(api["auto_start"], bool):
        raise ValueError("synthetic_api.auto_start must be a boolean.")
    api["start_timeout_seconds"] = number(api["start_timeout_seconds"], "synthetic_api.start_timeout_seconds", minimum=1, maximum=600, nullable=False)
    return result


def validate_prompts(raw: object, current: dict) -> dict:
    return validate_profiles(raw, current)
