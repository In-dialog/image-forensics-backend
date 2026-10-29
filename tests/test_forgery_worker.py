"""Exercise real subprocess cleanup with small CPU fixtures, without loading models."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts.services import forgery_worker as worker
from scripts.runtime import processes


class ForgeryWorkerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.image = self.root / "source.png"
        self.image.write_bytes(b"fixture")

    def command(self, script):
        return patch.object(worker, "_command", side_effect=lambda method, image, device, output:
                            [sys.executable, "-B", "-u", "-c", script, str(output)])

    def test_arrays_are_read_only_after_successful_process_exit(self):
        with self.command("import numpy as np,sys; np.savez(sys.argv[1],heatmap=np.ones((3,4)),detection=np.array(.5))"):
            result = worker.run_isolated_method("trufor", self.image)
        self.assertEqual(result["heatmap"].shape, (3, 4))
        self.assertEqual(float(result["detection"]), .5)
        self.assertIsNone(result["mask"])

    def test_failed_process_output_is_not_accepted_as_success(self):
        script = "import numpy as np,sys; np.savez(sys.argv[1],detection=np.array(.5)); print('CUDA method failure'); sys.exit(7)"
        with self.command(script), self.assertRaisesRegex(RuntimeError, "CUDA method failure"):
            worker.run_isolated_method("trufor", self.image)

    def test_missing_or_pickle_results_fail(self):
        for script, error in (("pass", RuntimeError),
                              ("import numpy as np,sys; np.savez(sys.argv[1],heatmap=np.array([{}],dtype=object))", ValueError)):
            with self.subTest(script=script), self.command(script), self.assertRaises(error):
                worker.run_isolated_method("trufor", self.image)

    def test_timeout_reaps_the_worker_before_returning(self):
        children = []
        original = subprocess.Popen

        def spawn(*args, **kwargs):
            child = original(*args, **kwargs)
            children.append(child)
            return child

        with self.command("import time; time.sleep(60)"), patch.object(worker.subprocess, "Popen", side_effect=spawn):
            with self.assertRaises(subprocess.TimeoutExpired):
                worker.run_isolated_method("trufor", self.image, timeout=.15)
        self.assertTrue(children)
        self.assertIsNotNone(children[0].poll())

    def test_next_method_can_run_after_a_failure(self):
        with self.command("raise RuntimeError('model unavailable')"), self.assertRaises(RuntimeError):
            worker.run_isolated_method("trufor", self.image)
        with self.command("import numpy as np,sys; np.savez(sys.argv[1],detection=np.array(.2))"):
            self.assertAlmostEqual(float(worker.run_isolated_method("psccnet", self.image)["detection"]), .2)

    def test_backend_shutdown_reaps_an_active_worker_and_blocks_new_ones(self):
        with patch.object(processes, "_closing", False):
            with processes.owned_process([sys.executable, "-B", "-c", "import time; time.sleep(60)"]) as child:
                processes._shutdown_workers()
                self.assertIsNotNone(child.poll())
                with self.assertRaisesRegex(RuntimeError, "shutting down"):
                    with processes.owned_process([sys.executable, "-c", "pass"]):
                        self.fail("Worker started after shutdown")
        self.assertEqual(processes._children, set())


if __name__ == "__main__":
    unittest.main()
