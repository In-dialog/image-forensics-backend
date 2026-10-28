"""Private runtime safety using tiny generated TAR fixtures, never real downloads."""
import hashlib
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import sys
import tarfile
import tempfile
import types
import unittest
from unittest.mock import patch

from install import ollama_runtime as runtime


def tar_fixture(entries):
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w") as archive:
        for name, kind, value in entries:
            member = tarfile.TarInfo(name)
            member.mode = 0o755
            if kind == "file":
                member.size = len(value)
                archive.addfile(member, io.BytesIO(value))
            else:
                member.type = {"directory": tarfile.DIRTYPE, "symlink": tarfile.SYMTYPE, "hardlink": tarfile.LNKTYPE}[kind]
                member.linkname = value
                archive.addfile(member)
    return data.getvalue()


class PrivateOllamaTests(unittest.TestCase):
    def test_extracts_binary_and_library_layout_without_running_anything(self):
        fixture = tar_fixture([(".", "directory", ""), ("bin/ollama", "file", b"test fixture"), ("lib/ollama/library.so", "file", b"data")])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime.extract_runtime(io.BytesIO(fixture), root)
            self.assertEqual((root / "bin/ollama").read_bytes(), b"test fixture")
            self.assertEqual((root / "lib/ollama/library.so").read_bytes(), b"data")

    def test_rejects_traversal_absolute_and_windows_archive_paths(self):
        for path in ("../escape", "/etc/escape", "bin\\escape", "C:/escape", "bin/\x00escape"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                runtime._relative_path(path)

    def test_rejects_links_that_escape_the_staging_directory(self):
        for kind in ("symlink", "hardlink"):
            fixture = tar_fixture([("lib/escape", kind, "../../../outside")])
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                with self.assertRaisesRegex(ValueError, "escapes"):
                    runtime.extract_runtime(io.BytesIO(fixture), Path(temporary))

    def test_rejects_oversized_or_duplicate_archive_members(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(runtime, "MAX_EXPANDED_BYTES", 1), self.assertRaisesRegex(ValueError, "expanded-size"):
                runtime.extract_runtime(io.BytesIO(tar_fixture([("large", "file", b"xx")])), root)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                runtime.extract_runtime(io.BytesIO(tar_fixture([("same", "file", b"a"), ("same", "file", b"b")])), root)

    def test_download_checks_digest_before_publishing_the_archive(self):
        data = b"tiny archive fixture"
        with tempfile.TemporaryDirectory() as temporary, patch.object(runtime.urllib.request, "urlopen", return_value=io.BytesIO(data)), redirect_stdout(io.StringIO()):
            path = Path(temporary) / "runtime.tar.zst"
            with self.assertRaisesRegex(RuntimeError, "checksum"):
                runtime.download_verified("https://example.invalid/fixture", path, "0" * 64, len(data))
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_validated_cached_archive_does_not_use_the_network(self):
        data = b"fixture"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "runtime.tar.zst"
            path.write_bytes(data)
            with patch.object(runtime.urllib.request, "urlopen", side_effect=AssertionError("network forbidden")), redirect_stdout(io.StringIO()):
                runtime.download_verified("https://example.invalid/fixture", path, hashlib.sha256(data).hexdigest(), len(data))

    def test_private_install_has_no_system_commands_or_services(self):
        fixture = tar_fixture([("bin/ollama", "file", b"not executable test data"), ("lib/ollama/library.so", "file", b"data")])
        decompressor = types.SimpleNamespace(ZstdDecompressor=lambda: types.SimpleNamespace(stream_reader=lambda source: source))
        def fake_download(_url, target, _digest, _size):
            target.write_bytes(fixture)
        with tempfile.TemporaryDirectory() as temporary, \
             patch.object(runtime.platform, "system", return_value="Linux"), \
             patch.object(runtime.platform, "machine", return_value="x86_64"), \
             patch.object(runtime, "free_gib", return_value=30), \
             patch.object(runtime, "download_verified", side_effect=fake_download), \
             patch.dict(sys.modules, {"zstandard": decompressor}), \
             patch("subprocess.run", side_effect=AssertionError("system command forbidden")), \
             patch("subprocess.Popen", side_effect=AssertionError("service forbidden")), \
             redirect_stdout(io.StringIO()):
            root = Path(temporary)
            binary = Path(runtime.install_linux_runtime(root))
            self.assertEqual(binary, root / ".tools/ollama/bin/ollama")
            marker = json.loads((root / ".tools/ollama/.image-forensics-runtime.json").read_text())
            self.assertEqual(marker["sha256"], runtime.ARCHIVE_SHA256)
            self.assertEqual(list((root / ".tools").iterdir()), [root / ".tools/ollama"])

    def test_low_disk_stops_before_download_or_writes(self):
        with tempfile.TemporaryDirectory() as temporary, \
             patch.object(runtime.platform, "system", return_value="Linux"), \
             patch.object(runtime.platform, "machine", return_value="x86_64"), \
             patch.object(runtime, "free_gib", return_value=1), \
             patch.object(runtime, "download_verified") as download:
            root = Path(temporary)
            with self.assertRaisesRegex(RuntimeError, "14 GiB"):
                runtime.install_linux_runtime(root)
            download.assert_not_called()
            self.assertFalse((root / ".tools").exists())


if __name__ == "__main__":
    unittest.main()
