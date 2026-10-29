"""Read immutable run inputs and remove only files owned by a selected run."""
import json
import hashlib
from pathlib import Path
import re
import shutil

from scripts.configuration.schema import validate_settings
from scripts.configuration.plan import build_plan
from scripts.orchestration.execution import validate_plan
from scripts.orchestration.artifacts import validate_image, validate_stage_outputs
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
        validate_image(images[0])
        if job.get("source_sha256") and hashlib.sha256(images[0].read_bytes()).hexdigest() != job["source_sha256"]:
            raise ValueError("The saved input image has changed since this analysis started. Start a new analysis.")
        return images[0], settings, owned_path(folder, "prompts")

    def completed_prefix(self, job: dict, image: Path, settings: dict, prompts: Path):
        """Check every completed stage; repair from the first invalid dependency."""
        keys = [stage["key"] for stage in job["stages"]]
        if keys != settings["selected_stages"]:
            raise ValueError("Saved stage history does not match this analysis's settings.")
        # Validate all saved profiles, including stages that previously succeeded.
        plan = build_plan(self.data_dir / job["id"], image, keys, settings, prompts)
        validate_plan(plan)
        for index, (stage, call) in enumerate(zip(job["stages"], plan.stages)):
            if stage["status"] != "succeeded":
                return index, None
            try:
                validate_stage_outputs(call.key, plan.run_folder, call.options)
                if call.key == "extract_objects":
                    from scripts.orchestration.artifacts import read_document
                    metadata = read_document(plan.run_folder / "metadata.json")
                    if Path(metadata.get("source_image_path", "")).resolve() != image.resolve():
                        raise ValueError("Extraction metadata refers to a different source image")
                    if (metadata["image_width"], metadata["image_height"]) != validate_image(image):
                        raise ValueError("Extraction dimensions do not match the saved input image")
            except (ValueError, OSError, TypeError, KeyError) as exc:
                return index, f"Resume validation: {stage['label']} needs repair: {exc}. Re-running this stage and those after it."
        return len(keys), None

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
