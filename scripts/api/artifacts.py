"""Export completed analysis folders as a checked manifest and individual files."""
from __future__ import annotations

import hashlib
import json
import re
import stat
from pathlib import Path, PurePosixPath

from scripts.configuration.store import atomic_json
from .jobs import JobManager

ALLOWED_TYPES = {
    ".json": "application/json", ".txt": "text/plain; charset=utf-8",
    ".webp": "image/webp", ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".bmp": "image/bmp", ".tif": "image/tiff", ".tiff": "image/tiff", ".gif": "image/gif",
}
MAX_FILES = 10000
MAX_FILE_BYTES = 128 * 1024 * 1024
MAX_ARTIFACT_BYTES = 1024 * 1024 * 1024


class ArtifactNotReadyError(RuntimeError):
    pass


def _is_link(path: Path) -> bool:
    info = path.lstat()
    return path.is_symlink() or bool(getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def checked_file(folder: Path, relative: str) -> Path:
    parts = PurePosixPath(relative).parts
    if not relative or relative.startswith("/") or re.search(r'[\\:\x00-\x1f]', relative) or not parts or any(part.startswith(".") for part in relative.split("/")):
        raise ValueError("Invalid artifact file path.")
    if any(not part for part in relative.split("/")):
        raise ValueError("Invalid artifact file path.")
    path = folder
    if _is_link(folder):
        raise ValueError("Artifact folders cannot be symbolic links or junctions.")
    for part in parts:
        path = path / part
        if _is_link(path):
            raise ValueError("Artifact files cannot contain symbolic links or junctions.")
    path.resolve().relative_to(folder.resolve())
    if not path.is_file() or path.suffix.lower() not in ALLOWED_TYPES:
        raise ValueError("Unsupported artifact file.")
    return path


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ArtifactStore:
    def __init__(self, manager: JobManager):
        self.manager = manager

    def manifest(self, job_id: str) -> dict:
        job = self.manager.get(job_id)
        if job["status"] != "succeeded":
            raise ArtifactNotReadyError("Only completed analyses can be downloaded.")
        stored = self.manager.root / job_id / "artifact.json"
        if stored.is_file():
            return json.loads(stored.read_text(encoding="utf-8"))
        folder = self.manager.data_dir / job_id
        checked_file(folder, "metadata.json")
        checked_file(folder, "original/original.webp")
        files = []
        total_size = 0
        # rglob does not follow symlink directories; reject links explicitly, including junctions.
        pending = [folder]
        while pending:
            directory = pending.pop()
            for path in directory.iterdir():
                if path.name.startswith(".") or _is_link(path):
                    raise ValueError("Artifact contains a private or linked path.")
                if path.is_dir():
                    pending.append(path)
                    if len(pending) + len(files) > MAX_FILES:
                        raise ValueError("Artifact contains too many files or folders.")
                    continue
                relative = path.relative_to(folder).as_posix()
                checked_file(folder, relative)
                size = path.stat().st_size
                total_size += size
                if size > MAX_FILE_BYTES or total_size > MAX_ARTIFACT_BYTES or len(files) >= MAX_FILES:
                    raise ValueError("Artifact exceeds its transfer size limit.")
                files.append(dict(path=relative, size=size, sha256=_digest(path)))
        artifact = dict(result_id=job_id, files=sorted(files, key=lambda item: item["path"]),
                        name=Path(job["original_filename"]).stem, created_at=job["created_at"], total_size=total_size)
        atomic_json(stored, artifact)
        return artifact

    def file(self, job_id: str, relative: str) -> tuple[Path, str, int]:
        manifest = self.manifest(job_id)
        entry = next((item for item in manifest["files"] if item["path"] == relative), None)
        if entry is None:
            raise FileNotFoundError("Artifact file not found.")
        path = checked_file(self.manager.data_dir / job_id, relative)
        size = path.stat().st_size
        if size != entry["size"] or size > MAX_FILE_BYTES:
            raise ArtifactNotReadyError("Artifact changed after completion. Its saved manifest no longer matches.")
        return path, ALLOWED_TYPES[path.suffix.lower()], size
