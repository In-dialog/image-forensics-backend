"""Bounded, reconnectable synthetic inference tasks; never retain model objects."""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
from threading import Lock, Thread
import traceback
from uuid import uuid4


class SyntheticTasks:
    def __init__(self, predict, *, retained=16):
        self.predict = predict
        self.retained = retained
        self.lock = Lock()
        self.tasks = {}

    def submit(self, image: Path, model: str | None = None) -> dict:
        image = image.resolve(strict=True)
        identity = (str(image), hashlib.sha256(image.read_bytes()).hexdigest(), model)
        with self.lock:
            for task in self.tasks.values():
                if task["identity"] == identity and task["status"] != "failed":
                    return self._public(task)
            if any(task["status"] == "running" for task in self.tasks.values()):
                raise RuntimeError("Synthetic analysis is already processing another image. Wait for it to finish.")
            while len(self.tasks) >= self.retained:
                del self.tasks[next(iter(self.tasks))]
            task = dict(id=uuid4().hex, identity=identity, status="running",
                        activity="Preparing model files", error=None, result=None)
            self.tasks[task["id"]] = task
            snapshot = self._public(task)
        Thread(target=self._run, args=(task["id"], image, model), daemon=True,
               name=f"synthetic-{task['id']}").start()
        return snapshot

    @staticmethod
    def _public(task):
        return deepcopy({key: value for key, value in task.items() if key != "identity"})

    def get(self, task_id: str) -> dict:
        with self.lock:
            return self._public(self.tasks[task_id])

    def _activity(self, task_id, message):
        with self.lock:
            self.tasks[task_id]["activity"] = message

    def _run(self, task_id, image, model):
        try:
            result = self.predict(image, model, lambda message: self._activity(task_id, message))
        except Exception as exc:
            traceback.print_exc()
            with self.lock:
                self.tasks[task_id].update(status="failed", error=str(exc) or type(exc).__name__)
        else:
            with self.lock:
                self.tasks[task_id].update(status="succeeded", activity="Complete", result=result)
