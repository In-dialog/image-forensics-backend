"""Run a selected sequence against one backend output folder."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping

from scripts.orchestration.stages import STAGES, STAGE_BY_KEY
from scripts.orchestration.progress import stage_progress
from scripts.orchestration.checkpoints import checkpoint_scope


@dataclass(frozen=True)
class StageCall:
    key: str
    options: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class RunPlan:
    run_folder: Path
    stages: tuple[StageCall, ...]
    image_path: Path | None = None


def validate_run_folder(folder: Path) -> None:
    path = folder / "metadata.json"
    if not path.is_file():
        raise ValueError(
            f"No metadata.json in {folder}. Enable Extract Objects for a new image "
            "or choose an existing run folder."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("extracts"), list):
        raise ValueError(f"Invalid extraction metadata: {path}")


def validate_plan(plan: RunPlan) -> None:
    if not plan.stages:
        raise ValueError("Select at least one analysis stage.")
    keys = [call.key for call in plan.stages]
    if len(set(keys)) != len(keys) or any(key not in STAGE_BY_KEY for key in keys):
        raise ValueError("The execution plan contains duplicate or unknown stages.")
    if keys != [stage.key for stage in STAGES if stage.key in keys]:
        raise ValueError("Stages must run in Global, Per Item, Ranking order.")
    if plan.run_folder.exists() and not plan.run_folder.is_dir():
        raise ValueError(f"Run folder is not a directory: {plan.run_folder}")
    if plan.image_path is not None and not plan.image_path.is_file():
        raise ValueError(f"Input image not found: {plan.image_path}")
    if "extract_objects" in keys:
        if plan.image_path is None:
            raise ValueError("Extract Objects needs an input image. Uncheck it to reuse an existing run.")
    else:
        validate_run_folder(plan.run_folder)


def load_runners() -> dict[str, Callable]:
    # Serving configuration or validating a plan must not import ML dependencies.
    from scripts import executers
    return {stage.key: getattr(executers, stage.runner) for stage in STAGES}


def execute_plan(
    plan: RunPlan,
    *,
    runners: Mapping[str, Callable] | None = None,
    on_progress: Callable[[str, int, int, str], None] | None = None,
    on_subprogress: Callable[[str, dict], None] | None = None,
    before_synthetic: Callable[[], object] | None = None,
    checkpoint_dir: Path | None = None,
    cleanup: Callable[[], None] | None = None,
) -> Path:
    """Stop on raised errors; report progress before/after each backend call.

    Some analysis runners handle individual item failures internally; those details
    remain in their logs/output files. Completion here means the runner returned.
    """
    validate_plan(plan)
    backend = runners if runners is not None else load_runners()
    from scripts.runtime.gpu import clear_exception_frames
    if cleanup is None:
        if runners is None:
            from scripts.utilities.runtime_cleanup import cleanup_runtime
            cleanup = lambda: cleanup_runtime(strict=True, report=True)
        else:
            # Injected runners also support offline tests without services.
            cleanup = lambda: None
    cleanup()
    folder = plan.run_folder
    total = len(plan.stages)
    for index, call in enumerate(plan.stages, 1):
        label = STAGE_BY_KEY[call.key].label
        if on_progress:
            on_progress(call.key, index, total, "running")
        try:
            with stage_progress((lambda update: on_subprogress(call.key, update)) if on_subprogress else None), \
                    checkpoint_scope(checkpoint_dir / call.key if checkpoint_dir is not None else None):
                if call.key == "synthetic_analysis" and before_synthetic:
                    before_synthetic()
                if call.key == "extract_objects":
                    folder = Path(backend[call.key](
                        image_path=plan.image_path, base_dir=folder.parent, **call.options
                    ))
                    validate_run_folder(folder)
                else:
                    backend[call.key](run_folder=folder, **call.options)
        except BaseException as exc:
            clear_exception_frames(exc)
            try:
                cleanup()
            except Exception as cleanup_error:
                print(f"Runtime cleanup warning: {cleanup_error}")
            if not isinstance(exc, Exception):
                raise
            raise RuntimeError(f"{label} failed: {exc}") from exc
        else:
            # The runner has returned; its model/tensor locals can be freed.
            cleanup()
        if on_progress:
            on_progress(call.key, index, total, "finished")
    return folder
