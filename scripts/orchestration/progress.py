"""Thread-local stage/substep progress; usable by runners without the web service."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy


_tracker: ContextVar[StageProgress | None] = ContextVar("stage_progress", default=None)


class StageProgress:
    def __init__(self, publish):
        self.publish = publish
        self.steps = []
        self.current = None
        self.outcome = None

    def emit(self):
        self.publish(deepcopy({"substeps": self.steps, "current_substep": self.current}))

    def define(self, labels):
        self.steps = [dict(label=label, status="pending", completed=0, total=None,
                           unit="items", current=None, activity=None, skipped=0, failed=0) for label in labels]
        self.current = None
        self.emit()

    @property
    def step(self):
        return self.steps[self.current - 1] if self.current is not None else None

    def start(self, index, total, unit, activity):
        if not 1 <= index <= len(self.steps):
            raise ValueError("Progress substep is outside the declared stage steps.")
        if total is not None and (not isinstance(total, int) or isinstance(total, bool) or total < 0):
            raise ValueError("Progress total must be a nonnegative integer or unknown.")
        if self.step and self.step["status"] == "running":
            self.finish()
        self.current, self.outcome = index, None
        self.step.update(status="running", completed=0, total=total, unit=unit,
                         current=None, activity=activity, skipped=0, failed=0)
        self.emit()

    def finish(self):
        if self.step and self.step["status"] == "running":
            step = self.step
            if step["total"] is None:
                step.update(total=1, completed=1, unit="steps")
            elif step["completed"] < step["total"]:
                step["skipped"] += step["total"] - step["completed"]
                step["completed"] = step["total"]
            step.update(status="succeeded", current=None, activity=None)
            self.outcome = None
            self.emit()

    def advance(self):
        if self.step is None:
            return
        if self.step["total"] is not None and self.step["completed"] >= self.step["total"]:
            raise ValueError("Progress exceeds the declared work count.")
        self.step["completed"] += 1
        if self.outcome in ("skipped", "failed"):
            self.step[self.outcome] += 1
        self.step.update(current=None, activity=None)
        self.outcome = None
        self.emit()


@contextmanager
def stage_progress(publish):
    tracker = StageProgress(publish) if publish is not None else None
    token = _tracker.set(tracker)
    try:
        if tracker:
            tracker.define(["Processing"])
            tracker.start(1, None, "steps", None)
        yield
    except BaseException:
        if tracker and tracker.step:
            tracker.step["status"] = "failed"
            tracker.emit()
        raise
    else:
        if tracker:
            tracker.finish()
            for step in tracker.steps:
                if step["status"] == "pending":
                    step["status"] = "skipped"
            tracker.emit()
    finally:
        _tracker.reset(token)


def progress_steps(*labels):
    if tracker := _tracker.get():
        tracker.define(labels)


def start_step(index, *, total=None, unit="items", activity=None):
    if tracker := _tracker.get():
        tracker.start(index, total, unit, activity)


def finish_step():
    if tracker := _tracker.get():
        tracker.finish()


def progress_activity(text):
    if (tracker := _tracker.get()) and tracker.step:
        tracker.step["activity"] = text
        tracker.emit()


def skip_item():
    if tracker := _tracker.get():
        tracker.outcome = "skipped"


def fail_item():
    if tracker := _tracker.get():
        tracker.outcome = "failed"


def progress_items(items, *, step=None, unit="items", label=str):
    """Advance only after a loop body returns, including explicit skip/failure outcomes.

    Without a step number, this contributes to an already-declared total (e.g.
    SAM calls across several parent crops), rather than resetting its counter.
    """
    if step is not None:
        start_step(step, total=len(items), unit=unit)
    tracker = _tracker.get()
    for item in items:
        if tracker and tracker.step:
            tracker.outcome = None
            tracker.step.update(current=str(label(item)), activity=None)
            tracker.emit()
        yield item
        if tracker:
            tracker.advance()
    if step is not None:
        finish_step()
