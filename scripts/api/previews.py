"""Small, read-only image previews from a run that may still be producing files."""
from __future__ import annotations

import io
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from .artifacts import MAX_FILE_BYTES, _is_link, checked_file

IMAGE_EXTENSIONS = {'.webp', '.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.gif'}


class PreviewStore:
    def __init__(self, manager):
        self.manager = manager

    def list(self, job_id: str) -> dict:
        self.manager.get(job_id)
        folder = self.manager.data_dir / job_id
        if not folder.exists():
            return {'previews': []}
        if _is_link(folder):
            raise ValueError('Linked preview folders are not allowed.')
        pending, candidates, scanned = [folder], [], 0
        while pending and scanned < 4000:
            directory = pending.pop()
            try:
                for entry in directory.iterdir():
                    scanned += 1
                    if scanned > 4000:
                        break
                    if entry.name.startswith('.') or _is_link(entry):
                        continue
                    if entry.is_dir():
                        pending.append(entry)
                    elif entry.suffix.lower() in IMAGE_EXTENSIONS:
                        stat = entry.stat()
                        if 0 < stat.st_size <= MAX_FILE_BYTES:
                            relative = entry.relative_to(folder).as_posix()
                            candidates.append((stat.st_mtime_ns, relative, stat.st_size))
            except OSError:
                # A stage can still be creating or replacing this directory.
                continue
        previews = []
        for modified, relative, size in sorted(candidates, reverse=True)[:24]:
            label = relative.replace('_', ' ').rsplit('.', 1)[0]
            previews.append(dict(path=relative, label=label, version=f'{modified}-{size}'))
        return {'previews': previews}

    def image(self, job_id: str, relative: str) -> bytes:
        self.manager.get(job_id)
        if Path(relative).suffix.lower() not in IMAGE_EXTENSIONS:
            raise FileNotFoundError('Preview image not found.')
        source = checked_file(self.manager.data_dir / job_id, relative)
        if source.stat().st_size > MAX_FILE_BYTES:
            raise ValueError('Preview image is too large.')
        try:
            with Image.open(source) as image:
                if image.width * image.height > 40_000_000:
                    raise ValueError('Preview image is too large.')
                image.thumbnail((1024, 1024))
                output = io.BytesIO()
                image.convert('RGBA' if 'A' in image.getbands() else 'RGB').save(output, 'WEBP', quality=80)
                return output.getvalue()
        except (OSError, UnidentifiedImageError) as exc:
            raise FileNotFoundError('Preview is not ready yet.') from exc
