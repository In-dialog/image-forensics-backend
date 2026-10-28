"""Validate images while retaining the original encoded bytes for forensic stages."""
from __future__ import annotations

import io
import json
import re
import uuid
import warnings
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from scripts.configuration.store import atomic_json

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
FORMAT_EXTENSIONS = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "TIFF": ".tiff", "BMP": ".bmp", "GIF": ".gif"}


class UploadStore:
    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)

    def save(self, data: bytes, filename: str) -> dict:
        if not isinstance(filename, str) or not filename or len(filename) > 180 or re.search(r'[\\/<>:"|?*\x00-\x1f]', filename):
            raise ValueError("Use a filename without folders or special path characters.")
        if filename in {".", ".."} or filename.endswith((".", " ")):
            raise ValueError("Invalid image filename.")
        if not data or len(data) > MAX_UPLOAD_BYTES:
            raise ValueError("Image upload must be between 1 byte and 20 MB.")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(data)) as source:
                    width, height = source.size
                    if source.format not in FORMAT_EXTENSIONS:
                        raise ValueError("Use a PNG, JPEG, WebP, TIFF, BMP, or GIF image.")
                    extension = FORMAT_EXTENSIONS[source.format]
                    if width * height > MAX_IMAGE_PIXELS or max(width, height) > 16383:
                        raise ValueError("Image exceeds 40 megapixels or 16383 pixels on one side.")
                    source.verify()
                with Image.open(io.BytesIO(data)) as source:
                    source.load()
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
            raise ValueError("Upload is not a supported, readable image.") from exc
        upload_id = uuid.uuid4().hex
        path = self.root / f"{upload_id}{extension}"
        path.write_bytes(data)
        atomic_json(self.root / f"{upload_id}.json", {"upload_id": upload_id, "filename": filename, "extension": extension})
        return {"upload_id": upload_id, "filename": filename}

    def get(self, upload_id: object) -> tuple[Path, dict]:
        if not isinstance(upload_id, str) or not re.fullmatch(r"[0-9a-f]{32}", upload_id):
            raise ValueError("Invalid upload ID.")
        metadata = self.root / f"{upload_id}.json"
        if not metadata.is_file():
            raise FileNotFoundError("Upload not found. Upload the image again.")
        info = json.loads(metadata.read_text(encoding="utf-8"))
        extension = info.get("extension")
        if extension not in FORMAT_EXTENSIONS.values():
            raise ValueError("Stored upload has an invalid image format.")
        image = self.root / f"{upload_id}{extension}"
        if not image.is_file():
            raise FileNotFoundError("Upload not found. Upload the image again.")
        return image, info
