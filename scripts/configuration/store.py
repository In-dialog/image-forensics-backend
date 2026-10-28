"""Atomic persistence for the current structured analysis configuration."""
from __future__ import annotations

import copy
import json
import os
import threading
import time
import uuid
from pathlib import Path

from .schema import PROMPT_KEYS, default_settings, merge_known, validate_prompts, validate_settings
from scripts.prompting import default_profiles

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        # Windows readers/antivirus can briefly hold the destination without delete sharing.
        delays = (0.01, 0.025, 0.05, 0.1, 0.2, 0.4)
        for attempt in range(len(delays) + 1):
            try:
                os.replace(temporary, path)
                break
            except PermissionError as exc:
                if getattr(exc, "winerror", None) not in {5, 32} or attempt == len(delays):
                    raise
                time.sleep(delays[attempt])
    finally:
        temporary.unlink(missing_ok=True)


class ConfigStore:
    def __init__(self, root: Path | None = None, *, project_root: Path = PROJECT_ROOT):
        self.root = root or Path(__file__).resolve().parent
        self.project_root = project_root
        self.path = self.root / "settings.json"
        self.lock = threading.RLock()
        if self.path.exists():
            saved = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(saved, dict) or saved.get("version") != 2:
                raise ValueError("Configuration must use the current structured format (version 2).")
            if set(saved) != {"version", "settings", "prompts", "local"}:
                raise ValueError("Configuration must contain exactly version, settings, prompts and local.")
            local = saved["local"]
            if (not isinstance(local, dict) or set(local) != {"synthetic_start_command"}
                    or not isinstance(local["synthetic_start_command"], str)
                    or not local["synthetic_start_command"].strip() or "\0" in local["synthetic_start_command"]):
                raise ValueError("local must contain a nonempty synthetic_start_command string.")
            self.data = dict(version=2, settings=validate_settings(saved["settings"]),
                             prompts=validate_prompts(saved["prompts"], {}), local=copy.deepcopy(local))
            if set(self.data["prompts"]) != set(PROMPT_KEYS):
                raise ValueError("Saved configuration must contain every analysis profile.")
        else:
            self.data = dict(version=2, settings=validate_settings(default_settings()), prompts=default_profiles(),
                             local={"synthetic_start_command": str(self.project_root / "install" / "start_synthetic.py")})
            atomic_json(self.path, self.data)

    def snapshot(self, settings: object = None, prompts: object = None) -> tuple[dict, dict]:
        with self.lock:
            return (validate_settings(merge_known(self.data["settings"], {} if settings is None else settings)),
                    validate_prompts({} if prompts is None else prompts, self.data["prompts"]))

    def update(self, payload: object) -> tuple[dict, dict]:
        if not isinstance(payload, dict) or set(payload) - {"settings", "prompts"}:
            raise ValueError("Configuration accepts only settings and prompts.")
        with self.lock:
            settings, prompts = self.snapshot(payload.get("settings"), payload.get("prompts"))
            next_data = {**self.data, "settings": settings, "prompts": prompts}
            atomic_json(self.path, next_data)
            self.data = next_data
            return copy.deepcopy(settings), copy.deepcopy(prompts)

    @property
    def synthetic_start_command(self) -> str:
        return self.data["local"]["synthetic_start_command"]
