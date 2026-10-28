"""Read immutable run inputs and remove only files owned by a selected run."""
import json
from pathlib import Path
import re
import shutil

from scripts.configuration.schema import validate_settings
from .uploads import FORMAT_EXTENSIONS

RECOVERABLE = {"failed", "interrupted"}


def owned_path(root: Path, name: str) -> Path:
    root = root.resolve()
    candidate = root / name
    if not name or Path(name).name != name or candidate.resolve() != candidate or candidate.parent != root:
        raise ValueError("Run files must stay inside their own storage folder.")
    return candidate


class RunRecovery:
    def __init__(self, data_dir: Path, jobs_root: Path, inputs_root: Path):
        self.data_dir, self.jobs_root, self.inputs_root = data_dir, jobs_root, inputs_root

    def inputs(self, job: dict):
        owned_path(self.data_dir, job["id"])
        folder = owned_path(self.jobs_root, job["id"])
        settings = validate_settings(json.loads((folder / "settings.json").read_text(encoding="utf-8")))
        images = [owned_path(self.inputs_root, job["id"] + extension) for extension in set(FORMAT_EXTENSIONS.values())]
        images = [path for path in images if path.is_file()]
        if len(images) != 1:
            raise ValueError("The saved original image is unavailable. Start a new analysis with the image.")
        return images[0], settings, folder / "prompts"

    def delete(self, job: dict, other_jobs: list[dict]):
        # Resolve every deletion target before changing anything. The settings and
        # status folder is last so a failed removal remains visible and retryable.
        targets = [owned_path(self.data_dir, job["id"])]
        targets += [owned_path(self.inputs_root, job["id"] + extension) for extension in set(FORMAT_EXTENSIONS.values())]
        upload_id = job.get("upload_id")
        if upload_id and re.fullmatch(r"[0-9a-f]{32}", upload_id) and not any(other.get("upload_id") == upload_id for other in other_jobs):
            targets += [owned_path(self.inputs_root, upload_id + suffix) for suffix in {*FORMAT_EXTENSIONS.values(), ".json"}]
        targets.append(owned_path(self.jobs_root, job["id"]))
        for path in targets:
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
