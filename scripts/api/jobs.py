"""Single-worker pipeline jobs with durable state and per-job prompt snapshots."""
from __future__ import annotations

import contextlib
import copy
import io
import hashlib
import json
import re
import shutil
import sys
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping

from PIL import Image

from scripts.configuration.plan import build_plan, synthetic_config
from scripts.configuration.schema import selected_stages
from scripts.configuration.store import ConfigStore, atomic_json
from scripts.orchestration.execution import execute_plan
from scripts.orchestration.stages import STAGE_BY_KEY
from scripts.services.synthetic_api_client import ensure_available
from .uploads import UploadStore
from .recovery import RunRecovery, RECOVERABLE


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class BusyError(RuntimeError):
    pass


class _WorkerLog(io.TextIOBase):
    """Capture only this worker's prints; concurrent HTTP/server output passes through."""
    def __init__(self, original, append: Callable[[str], None]):
        self.original = original
        self.append = append
        self.thread_id = threading.get_ident()
        self.pending = ""

    @property
    def encoding(self):
        return "utf-8"

    def writable(self):
        return True

    def isatty(self):
        return False

    def write(self, value):
        if threading.get_ident() != self.thread_id:
            return self.original.write(value)
        self.pending += value.replace("\r", "\n")
        while "\n" in self.pending or len(self.pending) > 4096:
            end = self.pending.find("\n")
            end = min(end, 4096) if end >= 0 else 4096
            line, self.pending = self.pending[:end], self.pending[end + (self.pending[end:end+1] == "\n"):]
            if line.strip():
                self.append(line)
        return len(value)

    def flush(self):
        if threading.get_ident() != self.thread_id:
            self.original.flush()
        elif self.pending:
            self.append(self.pending)
            self.pending = ""


def runtime_cleanup():
    # Keep importing this module, serving config, and validating uploads free of ML imports.
    from scripts.utilities.runtime_cleanup import cleanup_runtime
    cleanup_runtime(report=True)


class JobManager:
    def __init__(self, data_dir: Path, config: ConfigStore, *, runners: Mapping[str, Callable] | None = None,
                 cleanup: Callable | None = None, prepare_synthetic: Callable = ensure_available):
        self.data_dir = data_dir.resolve()
        self.root = self.data_dir / ".jobs"
        self.root.mkdir(parents=True, exist_ok=True)
        self.uploads = UploadStore(self.data_dir / ".inputs")
        self.config = config
        self.runners = runners
        self.cleanup = cleanup if cleanup is not None else (runtime_cleanup if runners is None else lambda: None)
        self.prepare_synthetic = prepare_synthetic
        self.lock = threading.RLock()
        self.jobs: dict[str, dict] = {}
        self._last_persist_attempt: dict[str, float] = {}
        self._io_warnings: set[tuple[str, str]] = set()
        self.active_job_id: str | None = None
        self.worker: threading.Thread | None = None
        self.queue: list[tuple] = []
        self.recovery = RunRecovery(self.data_dir, self.root, self.uploads.root)
        self._recover()

    def _recover(self):
        for path in self.root.glob("*/status.json"):
            try:
                job = json.loads(path.read_text(encoding="utf-8"))
                if job["id"] != path.parent.name or not re.fullmatch(r"[a-z0-9_-]+-[0-9a-f]{12}", job["id"]):
                    continue
                if job["status"] in {"queued", "running"}:
                    job.update(status="interrupted", error="The analysis service stopped before this job finished.", updated_at=now())
                    for stage in job["stages"]:
                        if stage["status"] in {"pending", "running"}:
                            stage["status"] = "skipped"
                        for step in stage.get("substeps", []):
                            if step["status"] == "running":
                                step["status"] = "interrupted"
                            elif step["status"] == "pending":
                                step["status"] = "skipped"
                    atomic_json(path, job)
                self.jobs[job["id"]] = job
            except (OSError, ValueError, KeyError, TypeError):
                continue

    def _persist(self, job: dict):
        job["updated_at"] = now()
        self._last_persist_attempt[job["id"]] = time.monotonic()
        atomic_json(self.root / job["id"] / "status.json", job)

    def _warn_io(self, job: dict, context: str, error: OSError):
        key = (job["id"], context)
        if key in self._io_warnings:
            return
        self._io_warnings.add(key)
        message = f"{context} could not be saved: {error}. In-memory status remains available; disk status may be stale."
        job["logs"] = (job["logs"] + [message])[-200:]
        # Bypass the worker's captured stderr to avoid recursively invoking this logger.
        try:
            if sys.__stderr__ is not None:
                sys.__stderr__.write(message + "\n")
                sys.__stderr__.flush()
        except (OSError, ValueError):
            pass

    def _persist_progress(self, job: dict, context: str = "Progress state"):
        try:
            self._persist(job)
        except OSError as exc:
            self._warn_io(job, context, exc)

    def get(self, job_id: str) -> dict:
        with self.lock:
            if job_id not in self.jobs:
                raise FileNotFoundError("Analysis job not found.")
            return self._snapshot(self.jobs[job_id])

    def _snapshot(self, job: dict) -> dict:
        result = copy.deepcopy(job)
        result["queue_position"] = next((i for i, entry in enumerate(self.queue, 1) if entry[0] == job["id"]), None)
        return result

    def list(self) -> list[dict]:
        with self.lock:
            ordered = sorted(self.jobs.values(), key=lambda item: item["created_at"], reverse=True)
            recent_ids = {job["id"] for job in ordered[:100]}
            return [self._snapshot(job) for job in ordered if job["id"] in recent_ids or job["status"] != "succeeded"]

    def _start_next(self):
        """Called with the manager lock, after the prior worker restored stdout."""
        if self.active_job_id is not None or not self.queue:
            return
        job_id, image, settings, prompt_dir = self.queue.pop(0)
        self.active_job_id = job_id
        self.worker = threading.Thread(target=self._run, args=(job_id, image, settings, prompt_dir),
                                       name=f"analysis-{job_id}", daemon=True)
        self.worker.start()

    def resume(self, payload: object) -> list[dict]:
        if not isinstance(payload, dict) or set(payload) != {"job_ids"}:
            raise ValueError("Resume accepts job_ids only.")
        ids = payload["job_ids"]
        if not isinstance(ids, list) or not 1 <= len(ids) <= 1000 or any(not isinstance(key, str) for key in ids) or len(set(ids)) != len(ids):
            raise ValueError("Choose one or more distinct unfinished analyses.")
        with self.lock:
            prepared = []
            for job_id in ids:
                if job_id not in self.jobs:
                    raise FileNotFoundError("Analysis job not found.")
                job = self.jobs[job_id]
                if job["status"] not in RECOVERABLE or self.active_job_id == job_id:
                    raise BusyError("Only failed or interrupted analyses can be resumed.")
                image, settings, prompts = self.recovery.inputs(job)
                first, repair = self.recovery.completed_prefix(job, image, settings, prompts)
                keys = [stage["key"] for stage in job["stages"]][first:]
                if keys:
                    build_plan(self.data_dir / job_id, image, keys, settings, prompts)
                prepared.append((job, first, image, settings, prompts, repair))
            prepared.sort(key=lambda item: item[0]["created_at"])
            originals = {job["id"]: copy.deepcopy(job) for job, *_ in prepared}
            try:
                for job, first, *_ in prepared:
                    job.update(status="queued", error=None, result_id=None, stage_key=None, completed_stages=first,
                               resume_attempts=job.get("resume_attempts", 0) + 1)
                    for stage in job["stages"][:first]:
                        stage["status"] = "succeeded"
                    for stage in job["stages"][first:]:
                        stage["status"] = "pending"
                        stage.pop("substeps", None)
                        stage.pop("current_substep", None)
                    self._persist(job)
            except OSError:
                for key, original in originals.items():
                    self.jobs[key] = original
                    self._persist_progress(original, "Resume rollback")
                raise
            for job, first, _, _, _, repair in prepared:
                self._append_log(job["id"], repair or f"Resume validation: verified outputs for {first} completed stage(s).")
            self.queue.extend((job["id"], image, settings, prompts) for job, _, image, settings, prompts, _ in prepared)
            self._start_next()
            return self.list()

    def delete(self, job_id: str):
        with self.lock:
            if job_id not in self.jobs:
                raise FileNotFoundError("Analysis job not found.")
            job = self.jobs[job_id]
            if self.active_job_id == job_id or job["status"] not in RECOVERABLE | {"queued"}:
                raise BusyError("Only unfinished analyses that are not running can be deleted.")
            self.recovery.delete(job, [other for key, other in self.jobs.items() if key != job_id])
            self.queue = [entry for entry in self.queue if entry[0] != job_id]
            del self.jobs[job_id]
            self._last_persist_attempt.pop(job_id, None)
            self._io_warnings = {entry for entry in self._io_warnings if entry[0] != job_id}

    def submit(self, payload: object) -> dict:
        if not isinstance(payload, dict) or set(payload) - {"upload_id", "stages", "settings", "prompts"}:
            raise ValueError("Job accepts upload_id, stages, settings, and prompts only.")
        stages = selected_stages(payload.get("stages"), require_extraction=True)
        image, upload = self.uploads.get(payload.get("upload_id"))
        settings, prompts = self.config.snapshot(payload.get("settings"), payload.get("prompts"))
        with self.lock:
            if self.active_job_id is not None:
                raise BusyError("Another analysis is already running. Wait for it to finish.")
            stem = re.sub(r"[^a-z0-9_-]+", "-", Path(upload["filename"]).stem.lower()).strip("-_")[:60] or "image"
            job_id = f"{stem}-{uuid.uuid4().hex[:12]}"
            job_dir = self.root / job_id
            prompt_dir = job_dir / "prompts"
            prompt_dir.mkdir(parents=True)
            for key, profile in prompts.items():
                atomic_json(prompt_dir / f"{key}.json", profile)
            settings["selected_stages"] = stages
            atomic_json(job_dir / "settings.json", settings)
            # Existing extraction runners derive output folder from the input stem.
            job_image = self.uploads.root / f"{job_id}{image.suffix}"
            shutil.copyfile(image, job_image)
            job = dict(id=job_id, status="queued", original_filename=upload["filename"], stage_key=None,
                       completed_stages=0, total_stages=len(stages),
                       stages=[dict(key=key, label=STAGE_BY_KEY[key].label, status="pending") for key in stages],
                       created_at=now(), updated_at=now(), error=None, result_id=None, logs=[], upload_id=upload["upload_id"],
                       source_sha256=hashlib.sha256(job_image.read_bytes()).hexdigest())
            self.jobs[job_id] = job
            try:
                self._persist(job)
            except OSError as exc:
                job.update(status="failed", error=f"Could not save the new analysis: {exc}")
                self._persist_progress(job, "New job state")
                raise
            self.queue.append((job_id, job_image, settings, prompt_dir))
            self._start_next()
            return copy.deepcopy(job)

    def _append_log(self, job_id: str, message: str):
        with self.lock:
            job = self.jobs[job_id]
            job["logs"].append(message)
            job["logs"] = job["logs"][-200:]
            try:
                with (self.root / job_id / "output.log").open("a", encoding="utf-8") as stream:
                    stream.write(message + "\n")
            except OSError as exc:
                self._warn_io(job, "Analysis log", exc)
            # Model-loading progress can emit thousands of lines per second.
            if time.monotonic() - self._last_persist_attempt.get(job_id, 0) >= 1:
                self._persist_progress(job)

    def _progress(self, job_id: str, key: str, index: int, total: int, state: str):
        with self.lock:
            job = self.jobs[job_id]
            job["stage_key"] = key
            stage = next(stage for stage in job["stages"] if stage["key"] == key)
            stage["status"] = "running" if state == "running" else "succeeded"
            job["completed_stages"] = sum(item["status"] == "succeeded" for item in job["stages"])
            self._persist_progress(job)
        self._append_log(job_id, f"{STAGE_BY_KEY[key].label}: {'started' if state == 'running' else 'finished'}")

    def _subprogress(self, job_id: str, key: str, update: dict):
        with self.lock:
            job = self.jobs[job_id]
            stage = next(stage for stage in job["stages"] if stage["key"] == key)
            if stage["status"] != "running":
                return
            previous_step = stage.get("current_substep")
            previous_item = stage["substeps"][previous_step - 1]["current"] if previous_step else None
            stage.update(copy.deepcopy(update))
            job["updated_at"] = now()
            steps = stage["substeps"]
            current = stage["current_substep"]
            terminal = current is not None and steps[current - 1]["status"] != "running"
            next_item = steps[current - 1]["current"] if current else None
            # Persist before the next possibly long model call, even if preceding
            # items completed within the normal disk-write throttle interval.
            item_started = next_item is not None and next_item != previous_item
            due = time.monotonic() - self._last_persist_attempt.get(job_id, 0) >= 1
            if current != previous_step or terminal or item_started or due:
                self._persist_progress(job)

    def _run(self, job_id: str, image: Path, settings: dict, prompt_dir: Path):
        try:
            self._execute_job(job_id, image, settings, prompt_dir)
        finally:
            with self.lock:
                self.active_job_id = None
                self._start_next()

    def _execute_job(self, job_id: str, image: Path, settings: dict, prompt_dir: Path):
        output = _WorkerLog(sys.stdout, lambda message: self._append_log(job_id, message))
        errors = _WorkerLog(sys.stderr, lambda message: self._append_log(job_id, message))
        failure = None
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            try:
                with self.lock:
                    # Queued runs can wait: validate their inputs and completed
                    # outputs again immediately before executing any model.
                    job = self.jobs[job_id]
                    image, settings, prompt_dir = self.recovery.inputs(job)
                    first, repair = self.recovery.completed_prefix(job, image, settings, prompt_dir)
                    if repair:
                        self._append_log(job_id, repair)
                    for stage in job["stages"][:first]:
                        stage["status"] = "succeeded"
                    for stage in job["stages"][first:]:
                        stage["status"] = "pending"
                    job["completed_stages"] = first
                    self.jobs[job_id]["status"] = "running"
                    self._persist(self.jobs[job_id])
                folder = self.data_dir / job_id
                original_dir = folder / "original"
                original_dir.mkdir(parents=True, exist_ok=True)
                from scripts.orchestration.artifacts import validate_image
                try:
                    validate_image(original_dir / "original.webp")
                    original_valid = True
                except ValueError:
                    original_valid = False
                if not original_valid:
                    with Image.open(image) as source:
                        source.convert("RGB").save(original_dir / "original.webp", "WEBP", quality=90)
                remaining = [stage["key"] for stage in self.jobs[job_id]["stages"] if stage["status"] != "succeeded"]
                if not remaining:
                    return
                plan = build_plan(folder, image, remaining, settings, prompt_dir)
                execute_plan(plan, runners=self.runners,
                             checkpoint_dir=self.root / job_id / "checkpoints",
                             on_progress=lambda *args: self._progress(job_id, *args),
                             on_subprogress=lambda *args: self._subprogress(job_id, *args),
                             before_synthetic=lambda: self.prepare_synthetic(synthetic_config(settings, self.config.synthetic_start_command)))
            except Exception as exc:
                failure = str(exc) or type(exc).__name__
                traceback.print_exc()
            finally:
                try:
                    self.cleanup()
                except Exception as exc:
                    print(f"Runtime cleanup warning: {exc}")
                output.flush()
                errors.flush()
                with self.lock:
                    job = self.jobs[job_id]
                    job.update(status="failed" if failure else "succeeded", error=failure, result_id=None if failure else job_id)
                    for stage in job["stages"]:
                        if stage["status"] == "running":
                            stage["status"] = "failed" if failure else "succeeded"
                        elif stage["status"] == "pending":
                            stage["status"] = "skipped"
                    self._persist_progress(job, "Final job state")
