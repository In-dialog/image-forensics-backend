"""Prepare PhotoHolmes with one real packaging change: use headless OpenCV.

The pinned upstream code and LICENSE are retained. Its built wheel declares
opencv-python-headless honestly, so uv/pip checks do not require two cv2 wheels.
This module does not install packages, build wheels, load models or run services.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import urllib.request
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REVISION = "223762ccd6ca23d75a32b96bbf1e26c2c39b3880"
SOURCE_URL = f"https://codeload.github.com/photoholmes/photoholmes/zip/{REVISION}"
SOURCE_FOLDER = ".tools/photoholmes-headless"
MARKER = ".image-forensics-patch.json"
MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
MAX_EXTRACTED_BYTES = 128 * 1024 * 1024


def private_directory(path: Path) -> None:
    """Reject both Unix links and Windows junctions before writes or removal."""
    try:
        info = path.lstat()
    except FileNotFoundError:
        return
    if (stat.S_ISLNK(info.st_mode)
            or getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
            or not stat.S_ISDIR(info.st_mode)):
        raise ValueError(f"Expected an ordinary directory, not a link or file: {path}")


def dependencies(source: str) -> tuple[re.Match, list[str]]:
    section = re.search(r"(?ms)^\[project\]\s*$\n(.*?)(?=^\[|\Z)", source)
    if not section or not re.search(r'^name\s*=\s*"photoholmes"\s*$', section[1], re.M):
        raise ValueError("The source does not declare the expected PhotoHolmes project.")
    if not re.search(r'^version\s*=\s*"0\.0\.1"\s*$', section[1], re.M):
        raise ValueError("Unexpected PhotoHolmes version; review the pinned source before changing its dependencies.")
    match = re.search(r"(?m)^dependencies\s*=\s*(\[[^\]]*\])", source)
    if not match or not section.start(1) <= match.start() < section.end(1):
        raise ValueError("PhotoHolmes dependency declaration was not found in [project].")
    values = ast.literal_eval(match[1])
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise ValueError("Unexpected PhotoHolmes dependency declaration.")
    return match, values


def patch_metadata(source: str) -> str:
    match, values = dependencies(source)
    if values.count("opencv-python") != 1 or any(value.startswith("opencv-") and value != "opencv-python" for value in values):
        raise ValueError("Expected exactly one upstream opencv-python dependency. Refusing an unreviewed source patch.")
    updated = re.sub(r'(["\'])opencv-python\1', '"opencv-python-headless"', match[1], count=1)
    result = source[:match.start(1)] + updated + source[match.end(1):]
    _, actual = dependencies(result)
    if actual != ["opencv-python-headless" if value == "opencv-python" else value for value in values]:
        raise ValueError("The packaging patch changed an unexpected dependency.")
    return result


def extract_source(archive: Path, target: Path) -> None:
    """Validate the entire archive before extracting regular source files."""
    prefix = f"photoholmes-{REVISION}"
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        if len(members) > 10000 or sum(member.file_size for member in members) > MAX_EXTRACTED_BYTES:
            raise ValueError("PhotoHolmes source archive exceeds the source size limit.")
        names = set()
        selected = []
        for member in members:
            raw = member.orig_filename.rstrip('/')
            parts = raw.split('/')
            if (not raw or '\\' in raw or any(part in {'', '.', '..'} for part in parts)
                    or any(re.search(r'[\x00-\x1f<>:"|?*]', part) or part.endswith((' ', '.'))
                           or re.match(r'^(con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\.|$)', part, re.I) for part in parts)
                    or PurePosixPath(raw).is_absolute() or parts[0] != prefix):
                raise ValueError(f"Unsafe source archive path: {member.filename}")
            if stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError("Source archive links are not allowed.")
            relative = '/'.join(parts[1:])
            if not relative:
                if member.is_dir():
                    continue
                raise ValueError("The archive root must be a directory.")
            key = relative.casefold()
            if key in names:
                raise ValueError("Duplicate source archive path.")
            names.add(key)
            selected.append((member, parts[1:]))
        private_directory(target)
        target.mkdir(parents=True, exist_ok=False)
        for member, parts in selected:
            filename = target.joinpath(*parts)
            if member.is_dir():
                filename.mkdir(parents=True, exist_ok=True)
            else:
                filename.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(member) as source, filename.open('xb') as output:
                    shutil.copyfileobj(source, output)


def download_source(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "Image-Forensics-Installer"})
    received = 0
    with urllib.request.urlopen(request, timeout=60) as response, destination.open('xb') as output:
        while chunk := response.read(1024 * 1024):
            received += len(chunk)
            if received > MAX_ARCHIVE_BYTES:
                raise ValueError("PhotoHolmes source download exceeds its size allowance.")
            output.write(chunk)


def prepared_source(destination: Path) -> bool:
    if not destination.exists():
        return False
    private_directory(destination)
    marker = json.loads((destination / MARKER).read_text(encoding='utf-8'))
    metadata = (destination / 'pyproject.toml').read_bytes()
    license_text = (destination / 'LICENSE').read_bytes()
    if (marker.get('revision') != REVISION
            or marker.get('metadata_sha256') != hashlib.sha256(metadata).hexdigest()
            or marker.get('license_sha256') != hashlib.sha256(license_text).hexdigest()):
        raise ValueError("Existing PhotoHolmes source differs from its packaging record. Inspect or move it before reinstalling.")
    _, values = dependencies(metadata.decode('utf-8'))
    if values.count('opencv-python-headless') != 1 or 'opencv-python' in values:
        raise ValueError("The prepared PhotoHolmes source has unexpected OpenCV dependencies.")
    return True


def prepare_source(root: Path = ROOT, *, downloader=None) -> Path:
    root = root.resolve()
    tools = root / '.tools'
    destination = root / SOURCE_FOLDER
    private_directory(tools)
    private_directory(destination)
    if prepared_source(destination):
        print("Reusing PhotoHolmes source with the recorded headless OpenCV packaging patch.")
        return destination
    tools.mkdir(exist_ok=True)
    temporary = tools / f'forgery-download-{uuid.uuid4().hex}'
    temporary.mkdir()
    try:
        archive = temporary / 'source.zip'
        (downloader or download_source)(SOURCE_URL, archive)
        staged = temporary / 'source'
        extract_source(archive, staged)
        license_text = (staged / 'LICENSE').read_bytes()
        metadata_path = staged / 'pyproject.toml'
        original = metadata_path.read_text(encoding='utf-8')
        updated = patch_metadata(original).encode('utf-8')
        metadata_path.write_bytes(updated)
        record = {
            'revision': REVISION, 'source_url': SOURCE_URL,
            'source_archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
            'patch': 'project.dependencies: opencv-python -> opencv-python-headless; all other upstream files retained',
            'metadata_sha256': hashlib.sha256(updated).hexdigest(),
            'license_sha256': hashlib.sha256(license_text).hexdigest(),
        }
        (staged / MARKER).write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
        private_directory(tools)
        private_directory(destination)
        if destination.exists():
            raise ValueError("PhotoHolmes source appeared during setup. Retry after the other installer finishes.")
        staged.rename(destination)
        print("Prepared pinned PhotoHolmes source; only its OpenCV dependency declaration changed. Upstream LICENSE retained.")
        return destination
    finally:
        private_directory(tools)
        private_directory(temporary)
        if temporary.exists():
            if temporary.resolve().parent != tools.resolve() or not temporary.name.startswith('forgery-download-'):
                raise ValueError("Refusing cleanup outside the source preparation directory.")
            shutil.rmtree(temporary)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if args.dry_run:
        print(f"Would obtain {SOURCE_URL}, retain its LICENSE, and change only its OpenCV dependency to opencv-python-headless.")
        print("DRY RUN: no downloads, files, package installation or model work.")
        return 0
    prepare_source()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
