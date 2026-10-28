"""Private Linux Ollama runtime: verified archive, no system installer or service."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import tarfile
import tempfile
import urllib.request

from install.environment import ROOT
from install.hardware import free_gib

# Official release metadata checked 2026-10-29. Updating this runtime requires
# updating the version, size and digest together; no floating "latest" downloads.
# https://api.github.com/repos/ollama/ollama/releases/tags/v0.40.12
VERSION = "0.40.12"
ARCHIVE_NAME = "ollama-linux-amd64.tar.zst"
ARCHIVE_URL = f"https://github.com/ollama/ollama/releases/download/v{VERSION}/{ARCHIVE_NAME}"
ARCHIVE_BYTES = 1_525_425_005
ARCHIVE_SHA256 = "f8adcb7e08d272283a4bd3abc18cf581afe37e64b2aee04fda4f407f215485ba"
MAX_EXPANDED_BYTES = 12 * 1024**3


def private_binary(root: Path = ROOT) -> Path:
    return root / ".tools" / "ollama" / "bin" / "ollama"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_verified(url: str, path: Path, expected_hash: str, expected_size: int) -> None:
    """Verify bytes before extraction; never execute an unchecked download."""
    if path.exists():
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"Unexpected linked/non-file archive: {path}")
        if path.stat().st_size == expected_size and sha256_file(path) == expected_hash:
            print("Reusing verified Ollama archive.")
            return
        raise RuntimeError(f"Cached Ollama archive failed verification. Remove this file and retry: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".part", dir=path.parent)
    temporary = Path(temporary_name)
    digest = hashlib.sha256()
    written = 0
    previous_percent = -1
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "image-forensics-backend-installer"})
        with os.fdopen(descriptor, "wb") as output, urllib.request.urlopen(request, timeout=60) as response:
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                written += len(chunk)
                if written > expected_size:
                    raise RuntimeError("Ollama archive is larger than the pinned release metadata.")
                digest.update(chunk)
                output.write(chunk)
                percent = written * 100 // expected_size
                if percent // 10 != previous_percent // 10:
                    print(f"Ollama runtime download: {percent}%", flush=True)
                    previous_percent = percent
        if written != expected_size or digest.hexdigest() != expected_hash:
            raise RuntimeError("Ollama archive checksum/size mismatch. Nothing was extracted or executed.")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _relative_path(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name or any(ord(char) < 32 for char in name):
        raise ValueError(f"Unsafe archive path: {name!r}")
    return path


def extract_runtime(stream, destination: Path) -> None:
    """Extract a TAR stream into a fresh staging directory with bounded paths."""
    root = destination.resolve()
    if not root.is_dir() or any(root.iterdir()):
        raise ValueError("Runtime extraction requires an empty staging directory.")
    links = []
    seen = set()
    total = 0
    with tarfile.open(fileobj=stream, mode="r|") as archive:
        for member in archive:
            relative = _relative_path(member.name)
            if len(seen) >= 20000:
                raise ValueError("Ollama runtime contains too many archive entries.")
            if not relative.parts:
                if member.isdir():
                    continue
                raise ValueError("Archive contains a file without a name.")
            target = root.joinpath(*relative.parts)
            resolved = target.resolve()
            if not resolved.is_relative_to(root):
                raise ValueError("Archive member escapes its staging directory.")
            if relative in seen:
                raise ValueError(f"Duplicate archive entry: {member.name}")
            seen.add(relative)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                total += member.size
                if member.size < 0 or total > MAX_EXPANDED_BYTES:
                    raise ValueError("Ollama runtime exceeds its expanded-size allowance.")
                target.parent.mkdir(parents=True, exist_ok=True)
                content = archive.extractfile(member)
                if content is None:
                    raise ValueError(f"Unreadable archive member: {member.name}")
                with content, target.open("xb") as output:
                    shutil.copyfileobj(content, output, length=1024 * 1024)
                target.chmod(member.mode & 0o755)
            elif member.issym() or member.islnk():
                # Defer links until all regular files are written, so later
                # entries cannot use a symlink to redirect extraction writes.
                links.append((target, member))
            else:
                raise ValueError(f"Unsupported archive entry type: {member.name}")
    for target, member in links:
        link = PurePosixPath(member.linkname)
        if link.is_absolute() or "\\" in member.linkname or ":" in member.linkname or any(ord(char) < 32 for char in member.linkname):
            raise ValueError(f"Unsafe archive link: {member.name}")
        referent = ((target.parent if member.issym() else root).joinpath(*link.parts)).resolve()
        if not referent.is_relative_to(root):
            raise ValueError(f"Archive link escapes its staging directory: {member.name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if member.issym():
            target.symlink_to(member.linkname)
        else:
            if not referent.is_file():
                raise ValueError(f"Archive hard link has no regular target: {member.name}")
            os.link(referent, target)
    for target, _member in links:
        if not target.resolve().is_relative_to(root):
            raise ValueError("Archive link chain escapes its staging directory.")


def install_linux_runtime(root: Path = ROOT) -> str:
    """Install only under this checkout's ignored .tools folder; never use sudo."""
    if platform.system() != "Linux" or platform.machine().lower() not in {"x86_64", "amd64"}:
        raise RuntimeError("The private Ollama package supports Linux x64. See https://ollama.com/download for other platforms.")
    root = root.resolve()
    tools = root / ".tools"
    if tools.is_symlink() or not tools.resolve().is_relative_to(root):
        raise RuntimeError("The private tools folder must stay inside the project.")
    binary = private_binary(root)
    if binary.is_file():
        return str(binary)
    destination = tools / "ollama"
    if destination.exists() or destination.is_symlink():
        raise RuntimeError(f"An incomplete/private runtime already exists. Review or remove this folder before retrying: {destination}")
    if free_gib(root) < 14:
        raise RuntimeError("Allow 14 GiB free on the project drive for the private Ollama archive and extraction, or install Ollama separately on a larger drive.")
    try:
        import zstandard
    except ImportError as exc:
        raise RuntimeError("The small zstandard package is missing. Rerun the backend installer to prepare archive support.") from exc
    tools.mkdir(exist_ok=True)
    archive_path = tools / f"ollama-{VERSION}-linux-amd64.tar.zst"
    print(f"Downloading private Ollama {VERSION} (about 1.53 GB compressed). No system service, driver, repository or user will be installed.")
    download_verified(ARCHIVE_URL, archive_path, ARCHIVE_SHA256, ARCHIVE_BYTES)
    with tempfile.TemporaryDirectory(prefix=".ollama-staging-", dir=tools) as temporary:
        staging = Path(temporary) / "runtime"
        staging.mkdir()
        with archive_path.open("rb") as compressed, zstandard.ZstdDecompressor().stream_reader(compressed) as reader:
            extract_runtime(reader, staging)
        staged_binary = staging / "bin" / "ollama"
        if not staged_binary.is_file() or not (staging / "lib" / "ollama").is_dir():
            raise RuntimeError("Official runtime layout was not found. No executable was started.")
        (staging / ".image-forensics-runtime.json").write_text(json.dumps({"version": VERSION, "source": ARCHIVE_URL, "sha256": ARCHIVE_SHA256}, indent=2), encoding="utf-8")
        # Both resolved paths are known children of the checked private tools dir.
        staging.rename(destination)
    archive_path.unlink()
    # Upstream ml/path.go locates ../lib/ollama relative to bin/ollama. Preserve
    # that complete layout rather than copying only the executable onto PATH.
    return str(binary)
