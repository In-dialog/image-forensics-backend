"""Small persistence regressions without HTTP servers, model loading, or inference."""
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image

from scripts.api.jobs import JobManager, _WorkerLog
from scripts.configuration.schema import default_settings
from scripts.configuration.store import atomic_json


def sharing_error(code):
    error = PermissionError(13, "Destination temporarily locked")
    error.winerror = code
    return error


class AtomicPersistenceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_windows_sharing_errors_retry_then_replace_successfully(self):
        real_replace = os.replace
        for code in (5, 32):
            with self.subTest(winerror=code):
                path = self.root / f"status-{code}.json"
                path.write_text('{"previous": true}')
                attempts = []
                def replace(source, destination):
                    attempts.append((source, destination))
                    if len(attempts) == 1:
                        self.assertEqual(json.loads(path.read_text()), {"previous": True})
                        raise sharing_error(code)
                    return real_replace(source, destination)
                with patch("scripts.configuration.store.os.replace", side_effect=replace), patch("scripts.configuration.store.time.sleep") as sleep:
                    atomic_json(path, {"updated": True})
                self.assertEqual(len(attempts), 2)
                sleep.assert_called_once_with(0.01)
                self.assertEqual(json.loads(path.read_text()), {"updated": True})
        self.assertEqual(list(self.root.glob("*.tmp")), [])

    def test_permanent_denial_is_bounded_and_keeps_previous_json(self):
        path = self.root / "status.json"
        path.write_text('{"previous": true}')
        with patch("scripts.configuration.store.os.replace", side_effect=sharing_error(5)) as replace, patch("scripts.configuration.store.time.sleep") as sleep:
            with self.assertRaises(PermissionError):
                atomic_json(path, {"updated": True})
        self.assertEqual(replace.call_count, 7)
        self.assertEqual(sleep.call_count, 6)
        self.assertLess(sum(call.args[0] for call in sleep.call_args_list), 1)
        self.assertEqual(json.loads(path.read_text()), {"previous": True})
        self.assertEqual(list(self.root.glob("*.tmp")), [])

    def test_unrelated_permission_error_is_not_retried(self):
        with patch("scripts.configuration.store.os.replace", side_effect=sharing_error(1314)) as replace, patch("scripts.configuration.store.time.sleep") as sleep:
            with self.assertRaises(PermissionError):
                atomic_json(self.root / "status.json", {})
        replace.assert_called_once()
        sleep.assert_not_called()

    def manager_with_job(self):
        manager = JobManager(self.root / "data", Mock(), runners={})
        job = dict(id="test-0123456789ab", status="running", logs=[], stages=[dict(key="extract_objects", status="running")], completed_stages=0)
        manager.jobs[job["id"]] = job
        (manager.root / job["id"]).mkdir()
        manager.active_job_id = job["id"]
        return manager, job

    def test_logging_flush_and_progress_do_not_propagate_persistence_failure(self):
        manager, job = self.manager_with_job()
        output = _WorkerLog(io.StringIO(), lambda line: manager._append_log(job["id"], line))
        with patch.object(manager, "_persist", side_effect=sharing_error(5)), patch("sys.__stderr__", io.StringIO()):
            output.write("Loading weights: 38%")
            output.flush()
            manager._progress(job["id"], "extract_objects", 1, 1, "finished")
        self.assertIn("Loading weights: 38%", job["logs"])
        self.assertTrue(any("Progress state could not be saved" in message for message in job["logs"]))
        self.assertEqual(job["completed_stages"], 1)
        self.assertEqual(job["stages"][0]["status"], "succeeded")

    def test_final_status_failure_releases_active_job_and_keeps_memory_state(self):
        manager, job = self.manager_with_job()
        image = self.root / "input.png"
        Image.new("RGB", (2, 2)).save(image)
        settings = default_settings()
        settings["selected_stages"] = ["extract_objects"]
        with patch.object(manager, "_persist", side_effect=[None, sharing_error(5)]), patch("scripts.api.jobs.build_plan"), patch("scripts.api.jobs.execute_plan"), patch("sys.__stderr__", io.StringIO()):
            manager._run(job["id"], image, settings, self.root / "prompts")
        self.assertIsNone(manager.active_job_id)
        self.assertEqual(job["status"], "succeeded")
        self.assertEqual(job["result_id"], job["id"])
        self.assertTrue(any("Final job state could not be saved" in message for message in job["logs"]))


if __name__ == "__main__":
    unittest.main()
