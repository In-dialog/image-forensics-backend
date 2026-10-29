"""Real HTTP/task lifecycle with a tiny fake classifier; no models or downloads."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch
from http.server import ThreadingHTTPServer

from scripts.services.synthetic_tasks import SyntheticTasks
from scripts.services import synthetic_inference_client as client


class SyntheticInferenceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.image = Path(temporary.name) / "image.png"
        self.image.write_bytes(b"test image bytes")
        self.result = {"ok": True, "probs": {"real": 0.25, "synthetic": 0.75}}
        self.gate = threading.Event()
        self.entered = threading.Event()
        self.calls = []

        def predict(image, model, report):
            self.calls.append(image)
            report("Loading cached model")
            self.entered.set()
            self.gate.wait(3)
            return self.result

        self.tasks = SyntheticTasks(predict)
        spec = importlib.util.spec_from_file_location("synthetic_http_test",
                Path(__file__).resolve().parents[1] / "scripts/services/synthetic_api_server.py")
        service = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"torch": Mock(), "nonescape": Mock(), "huggingface_hub": Mock()}):
            spec.loader.exec_module(service)
        service._tasks = self.tasks
        service.SyntheticApiHandler.log_message = lambda *args: None
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), service.SyntheticApiHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        self.addCleanup(self.stop)

    def stop(self):
        self.gate.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(3)

    def test_timeout_and_resume_reconnect_without_duplicate_inference(self):
        activities = []
        with self.assertRaisesRegex(TimeoutError, "Resume will reconnect"):
            client.predict(self.image, api_url=self.url, timeout_s=0.08, poll_interval=0.01, report=activities.append)
        self.assertTrue(self.entered.is_set())
        self.assertIn("Loading cached model", activities)
        self.gate.set()
        self.assertEqual(client.predict(self.image, api_url=self.url, poll_interval=0.01), (0.25, 0.75))
        # A second resume can also retrieve a result whose HTTP response was lost.
        self.assertEqual(client.predict(self.image, api_url=self.url, poll_interval=0.01), (0.25, 0.75))
        self.assertEqual(len(self.calls), 1)

    def test_other_images_are_rejected_while_a_task_is_active(self):
        self.tasks.submit(self.image)
        self.assertTrue(self.entered.wait(1))
        other = self.image.with_name("other.png")
        other.write_bytes(b"different image")
        with self.assertRaisesRegex(RuntimeError, "already processing another image"):
            client.predict(other, api_url=self.url)
        self.assertEqual(len(self.calls), 1)

    def test_service_error_is_preserved_and_failed_task_can_be_retried(self):
        self.tasks.predict = Mock(side_effect=[ValueError("Incomplete model file"), self.result])
        with patch("scripts.services.synthetic_tasks.traceback.print_exc"):
            with self.assertRaisesRegex(RuntimeError, "Incomplete model file"):
                client.predict(self.image, api_url=self.url, poll_interval=0.01)
        self.assertEqual(client.predict(self.image, api_url=self.url, poll_interval=0.01), (0.25, 0.75))
        self.assertEqual(self.tasks.predict.call_count, 2)

    def test_changed_image_does_not_reuse_an_old_prediction(self):
        self.gate.set()
        client.predict(self.image, api_url=self.url, poll_interval=0.01)
        self.image.write_bytes(b"changed pixels")
        client.predict(self.image, api_url=self.url, poll_interval=0.01)
        self.assertEqual(len(self.calls), 2)

    def test_old_service_gets_an_actionable_restart_message(self):
        with patch.object(client, "_json", return_value={"ok": True}):
            with self.assertRaisesRegex(RuntimeError, "Restart the analysis backend"):
                client.predict(self.image, api_url=self.url)

    def test_invalid_probabilities_are_never_saved_as_zero(self):
        for values in ({}, {"real": 0, "synthetic": 0}, {"real": True, "synthetic": 0},
                       {"real": 0.2, "synthetic": float("nan")}, {"real": -1, "synthetic": 2}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                client.probabilities({"ok": True, "probs": values})


if __name__ == "__main__":
    unittest.main()
