"""PhotoHolmes packaging tests use generated tiny archives, never downloads."""
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import shutil
import stat
import tempfile
import unittest
from unittest.mock import Mock, patch
import zipfile

from install import forgery_source, setup


SOURCE_METADATA = '''[build-system]
requires = ["setuptools"]
build-backend = "setuptools.build_meta"

[project]
name = "photoholmes"
version = "0.0.1"
license = { file = "LICENSE" }
dependencies = [
    "numpy",
    "opencv-python",
    "torch >= 2.1",
    "jpegio",
]

[project.optional-dependencies]
test = ["pytest"]
'''


class ForgeryInstallTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.prefix = f'photoholmes-{forgery_source.REVISION}/'

    def archive(self, extra=None):
        destination = self.root / 'fixture.zip'
        with zipfile.ZipFile(destination, 'w') as bundle:
            bundle.writestr(self.prefix + 'pyproject.toml', SOURCE_METADATA)
            bundle.writestr(self.prefix + 'LICENSE', 'Unmodified upstream license fixture.\n')
            bundle.writestr(self.prefix + 'src/photoholmes/__init__.py', '__version__ = "0.0.1"\n')
            if extra:
                filename, content = extra
                if isinstance(filename, str):
                    entry = zipfile.ZipInfo()
                    entry.filename = filename  # Retain deliberately invalid separators on Windows.
                    filename = entry
                bundle.writestr(filename, content)
        return destination

    def test_only_expected_dependency_is_patched(self):
        result = forgery_source.patch_metadata(SOURCE_METADATA)
        self.assertEqual(result, SOURCE_METADATA.replace('"opencv-python"', '"opencv-python-headless"'))
        self.assertIn('"torch >= 2.1"', result)
        for unexpected in [
            SOURCE_METADATA.replace('opencv-python', 'opencv-contrib-python'),
            SOURCE_METADATA.replace('"opencv-python",', '"opencv-python", "opencv-python",'),
            SOURCE_METADATA.replace('version = "0.0.1"', 'version = "0.0.2"'),
            result,
        ]:
            with self.assertRaises(ValueError):
                forgery_source.patch_metadata(unexpected)

    def test_preparation_retains_license_and_code_and_reuses_recorded_source(self):
        archive = self.archive()
        downloader = Mock(side_effect=lambda url, destination: shutil.copyfile(archive, destination))
        checkout = self.root / 'checkout'
        checkout.mkdir()
        with redirect_stdout(StringIO()):
            source = forgery_source.prepare_source(checkout, downloader=downloader)
            again = forgery_source.prepare_source(checkout, downloader=downloader)
        self.assertEqual(source, again)
        self.assertEqual(downloader.call_count, 1)
        self.assertEqual(downloader.call_args.args[0], forgery_source.SOURCE_URL)
        self.assertIn(forgery_source.REVISION, forgery_source.SOURCE_URL)
        self.assertEqual((source / 'LICENSE').read_text(), 'Unmodified upstream license fixture.\n')
        self.assertEqual((source / 'src/photoholmes/__init__.py').read_text(), '__version__ = "0.0.1"\n')
        self.assertEqual((source / 'pyproject.toml').read_text(), forgery_source.patch_metadata(SOURCE_METADATA))
        marker = json.loads((source / forgery_source.MARKER).read_text())
        self.assertEqual(marker['revision'], forgery_source.REVISION)
        self.assertEqual(list((checkout / '.tools').iterdir()), [source])
        (source / 'LICENSE').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'packaging record'):
            forgery_source.prepare_source(checkout, downloader=downloader)
        self.assertEqual(downloader.call_count, 1)

    def test_unsafe_archive_is_rejected_before_writing_any_source(self):
        for filename in [self.prefix + '../escape.py', self.prefix + 'a/../../escape.py',
                         self.prefix + 'C:/escape.py', self.prefix + 'NUL.txt',
                         '/absolute.py', 'wrong-revision/source.py', self.prefix + 'a\\escape.py']:
            with self.subTest(filename=filename):
                archive = self.archive((filename, 'unsafe'))
                destination = self.root / 'extracted'
                with self.assertRaises(ValueError):
                    forgery_source.extract_source(archive, destination)
                self.assertFalse(destination.exists())
                self.assertFalse((self.root / 'escape.py').exists())

    def test_archive_links_and_case_collisions_are_rejected(self):
        linked = zipfile.ZipInfo(self.prefix + 'linked')
        linked.external_attr = (stat.S_IFLNK | 0o777) << 16
        for entry in [(linked, '../../outside'), (self.prefix + 'license', 'case collision')]:
            with self.subTest(entry=str(entry[0])):
                archive = self.archive(entry)
                with self.assertRaises(ValueError):
                    forgery_source.extract_source(archive, self.root / 'extracted')
                self.assertFalse((self.root / 'extracted').exists())

    def test_size_allowance_and_failed_preparation_leave_no_partial_source(self):
        archive = self.archive()
        with patch.object(forgery_source, 'MAX_EXTRACTED_BYTES', 1):
            with self.assertRaisesRegex(ValueError, 'size limit'):
                forgery_source.extract_source(archive, self.root / 'extracted')
        checkout = self.root / 'checkout'
        checkout.mkdir()
        downloader = Mock(side_effect=OSError('fixture network failure'))
        with self.assertRaises(OSError):
            forgery_source.prepare_source(checkout, downloader=downloader)
        self.assertEqual(list((checkout / '.tools').iterdir()), [])

    def test_dry_run_has_no_network_or_files(self):
        with patch.object(forgery_source, 'prepare_source', side_effect=AssertionError('prepare forbidden')), \
             patch('urllib.request.urlopen', side_effect=AssertionError('network forbidden')), \
             patch('pathlib.Path.mkdir', side_effect=AssertionError('write forbidden')), \
             redirect_stdout(StringIO()) as output:
            self.assertEqual(forgery_source.main(['--dry-run']), 0)
        self.assertIn(forgery_source.REVISION, output.getvalue())

    def test_uv_build_follows_preparation_and_preserves_torch_and_core_constraints(self):
        for profile in ['cpu', 'cuda', 'cuda-modern']:
            commands = setup.install_commands(setup.arguments(['--with-forgery']), profile, 'uv')
            preparation = next(index for index, command in enumerate(commands) if any('forgery_source.py' in part for part in command))
            install = next(index for index, command in enumerate(commands) if any('forgery.txt' in part for part in command))
            self.assertLess(preparation, install)
            self.assertIn(str(setup.ROOT / 'requirements/core.txt'), commands[install])
            self.assertIn(str(setup.ROOT / 'requirements' / f'{profile}.txt'), commands[install])
            self.assertEqual(commands[install].count('-c'), 2)
        core_only = setup.install_commands(setup.arguments([]), 'cpu', 'uv')
        self.assertFalse(any('forgery' in part for command in core_only for part in command))


if __name__ == '__main__':
    unittest.main()
