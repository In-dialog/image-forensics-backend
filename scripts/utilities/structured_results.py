"""Validated model execution and canonical structured result persistence."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from scripts.configuration.store import atomic_json
from scripts.prompting import default_profiles, validate_profile, validate_response
from scripts.orchestration.checkpoints import response_checkpoint, load_response, save_response
from scripts.orchestration.progress import progress_activity
from .ollama_vision import structured_vision_chat


def resolve_profile(stage: str, profile: dict | None) -> dict:
    return validate_profile(stage, profile if profile is not None else default_profiles()[stage])


def run_compiled(compiled: Any, image: Path, **model_options: Any) -> dict:
    checkpoint = response_checkpoint(image, {
        "system": compiled.system, "user": compiled.user, "schema": compiled.schema,
        "options": {key: value for key, value in model_options.items() if key != "keep_alive"},
    })
    cached = load_response(checkpoint, lambda result: validate_response(compiled, result))
    if cached is not None:
        progress_activity("Using saved result")
        return cached
    result = structured_vision_chat(
        compiled.user,
        image,
        system_prompt=compiled.system,
        response_schema=compiled.schema,
        validate=lambda data: validate_response(compiled, data),
        **model_options,
    )
    save_response(checkpoint, result)
    return result


def _descriptors(entries: list[dict], prefix: str) -> list[dict]:
    return [{**entry, "id": f"{prefix}_{index + 1}"} for index, entry in enumerate(entries)]


def save_rankings(run_folder: Path, profile: dict, items: list[dict], *, scope: str) -> None:
    """Persist the canonical typed ranking document."""
    filenames = {"full_image": "full_image_rankings.v2.json", "per_item": "per_item_rankings.v2.json"}
    if scope not in filenames:
        raise ValueError(f"Unknown ranking scope: {scope}")
    envelope = {
        "schema_version": 2, "scope": scope,
        "criteria": _descriptors(profile["criteria"], "criterion"), "items": items,
    }
    atomic_json(run_folder / filenames[scope], envelope)


def save_descriptions(run_folder: Path, profile: dict, items: list[dict]) -> None:
    envelope = {
        "schema_version": 2, "style": profile["style"],
        "questions": _descriptors(profile["questions"], "question"), "items": items,
    }
    atomic_json(run_folder / "descriptions.v2.json", envelope)
