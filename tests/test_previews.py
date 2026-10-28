"""Small image-only preview checks; no model runners or inference."""
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from scripts.api.previews import PreviewStore


class PreviewTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.job_id = 'running-preview'

        def get(job_id):
            if job_id != self.job_id:
                raise FileNotFoundError('Unknown job')
            return {'status': 'running'}

        self.previews = PreviewStore(SimpleNamespace(data_dir=self.root, get=get))

    def test_running_job_previews_are_small_and_do_not_expose_private_files(self):
        self.assertEqual(self.previews.list(self.job_id), {'previews': []})
        original = self.root / self.job_id / 'original' / 'original.png'
        original.parent.mkdir(parents=True)
        Image.new('RGB', (2048, 512), 'green').save(original)
        private = original.parent.parent / '.private'
        private.mkdir()
        Image.new('RGB', (10, 10)).save(private / 'hidden.png')
        listed = self.previews.list(self.job_id)['previews']
        self.assertEqual([item['path'] for item in listed], ['original/original.png'])
        self.assertTrue(listed[0]['version'])
        with Image.open(io.BytesIO(self.previews.image(self.job_id, listed[0]['path']))) as thumbnail:
            self.assertEqual(thumbnail.format, 'WEBP')
            self.assertEqual(thumbnail.size, (1024, 256))

    def test_previews_reject_traversal_private_paths_and_unknown_jobs(self):
        folder = self.root / self.job_id
        folder.mkdir()
        for relative in ('../secret.png', '.private/secret.png', 'C:/secret.png'):
            with self.subTest(relative=relative), self.assertRaises(ValueError):
                self.previews.image(self.job_id, relative)
        with self.assertRaises(FileNotFoundError):
            self.previews.image(self.job_id, 'metadata.json')
        with self.assertRaises(FileNotFoundError):
            self.previews.list('unknown-job')

    def test_image_still_being_written_is_not_a_pipeline_error(self):
        folder = self.root / self.job_id
        folder.mkdir()
        image = folder / 'partial.png'
        image.write_bytes(b'incomplete image')
        with self.assertRaisesRegex(FileNotFoundError, 'not ready'):
            self.previews.image(self.job_id, 'partial.png')
        self.assertEqual(image.read_bytes(), b'incomplete image')


if __name__ == '__main__':
    unittest.main()
