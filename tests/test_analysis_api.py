"""Real local HTTP contracts with injected stages; no inference or model downloads."""
from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image

from scripts.api.jobs import JobManager
from scripts.api.server import AnalysisServer, default_data_dir
from scripts.configuration.plan import build_plan
from scripts.configuration.schema import default_settings, validate_settings
from scripts.configuration.store import ConfigStore, atomic_json
from scripts.orchestration.stages import STAGES
from scripts.orchestration.progress import progress_steps, progress_items, start_step, skip_item, fail_item
from scripts.prompting import default_profiles


class AnalysisApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.project = self.root / "project"
        settings = default_settings()
        settings["stage_options"]["object_detect"].update(model="saved-model", num_ctx="8192", temperature="0.25")
        atomic_json(self.root / "configuration" / "settings.json", dict(version=2, settings=settings,
                    prompts=default_profiles(), local={"synthetic_start_command": "trusted-local-command"}))
        self.config = ConfigStore(self.root / "configuration", project_root=self.project)
        self.calls = []
        self.runners = {stage.key: self.record_stage(stage.key) for stage in STAGES}
        self.runners["extract_objects"] = self.extract
        self.cleanup = Mock()
        self.manager = JobManager(self.root / "data", self.config, runners=self.runners, cleanup=self.cleanup, prepare_synthetic=Mock())
        self.server = AnalysisServer(("127.0.0.1", 0), self.manager, model_query=lambda: (["installed-model"], True))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"
        self.addCleanup(self.stop)

    def stop(self):
        if self.manager.worker:
            self.manager.worker.join(5)
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(5)

    def record_stage(self, key):
        def run(**kwargs):
            self.calls.append((key, kwargs))
            print(f"Executed {key}")
        return run

    def extract(self, image_path, base_dir, **kwargs):
        self.calls.append(("extract_objects", dict(image_path=image_path, base_dir=base_dir, **kwargs)))
        folder = base_dir / image_path.stem
        folder.mkdir(parents=True, exist_ok=True)
        atomic_json(folder / "metadata.json", dict(extracts=[], image_width=12, image_height=8, source_image_path=str(image_path)))
        print("Extraction completed")
        return folder

    def request(self, method, path, payload=None, *, data=None, headers=None):
        headers = dict(headers or {})
        if payload is not None:
            data = json.dumps(payload).encode()
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(self.base + path, data=data, headers=headers, method=method)
        try:
            response = urllib.request.urlopen(req, timeout=5)
        except urllib.error.HTTPError as exc:
            response = exc
        with response:
            body = response.read()
            if response.headers.get_content_type() == "application/json":
                body = json.loads(body)
            return response.status, body

    def upload(self, filename="source.png"):
        buf = io.BytesIO()
        Image.new("RGB", (12, 8), "red").save(buf, "PNG")
        status, body = self.request("POST", "/uploads", data=buf.getvalue(), headers={"X-Filename": urllib.parse.quote(filename)})
        self.assertEqual(status, 201, body)
        return body

    def submit(self, **extra):
        return self.request("POST", "/jobs", dict(upload_id=self.upload()["upload_id"], stages=["extract_objects", "depth_map"], **extra))

    def wait_job(self, job_id):
        self.manager.worker.join(5)
        self.assertFalse(self.manager.worker.is_alive())
        status, body = self.request("GET", f"/jobs/{job_id}")
        self.assertEqual(status, 200)
        return body["job"]

    def test_config_loads_saved_preferences_and_does_not_expose_local_paths(self):
        status, config = self.request("GET", "/config")
        self.assertEqual(status, 200)
        self.assertEqual(config["settings"]["stage_options"]["object_detect"]["num_ctx"], 8192)
        self.assertEqual(config["settings"]["stage_options"]["object_detect"]["model"], "saved-model")
        self.assertTrue(config["ollama_available"])
        self.assertEqual(config["models"], ["installed-model"])
        self.assertNotIn("trusted-local-command", json.dumps(config))
        self.assertNotIn("prompt_path", json.dumps(config))
        self.assertEqual(self.config.synthetic_start_command, "trusted-local-command")
        self.assertNotIn("migration", config)

    def test_settings_and_prompt_updates_are_atomic_and_persisted(self):
        profile = default_profiles()["object_detect"]
        profile["positive"] = "Find religious symbols and decorative objects."
        status, body = self.request("PUT", "/config", {"settings": {"stage_options": {"object_detect": {"num_ctx": "4096", "seed": "", "presence_penalty": "1.5"}}}, "prompts": {"object_detect": profile}})
        self.assertEqual(status, 200, body)
        restored = ConfigStore(self.config.root)
        settings, prompts = restored.snapshot()
        self.assertEqual(settings["stage_options"]["object_detect"]["num_ctx"], 4096)
        self.assertIsNone(settings["stage_options"]["object_detect"]["seed"])
        self.assertEqual(settings["stage_options"]["object_detect"]["presence_penalty"], 1.5)
        self.assertEqual(prompts["object_detect"], profile)
        before = self.config.path.read_bytes()
        for invalid in ({"synthetic_api": {"start_command": "evil"}}, {"stage_options": {"object_detect": {"prompt_path": "evil"}}}, {"min_item_size_px": True}, {"min_item_size_px": "NaN"}, {"devices": {"global": "gpu"}}, {"stage_options": {"object_detect": {"temperature": []}}}, {"stage_options": {"image_description": {"presence_penalty": 2.01}}}):
            with self.subTest(invalid=invalid):
                self.assertEqual(self.request("PUT", "/config", {"settings": invalid})[0], 400)
                self.assertEqual(self.config.path.read_bytes(), before)
        self.assertEqual(self.request("PUT", "/config", {"prompts": {"object_detect": "Unstructured prompt"}})[0], 400)
        self.assertEqual(self.config.path.read_bytes(), before)

    def test_success_runs_in_order_keeps_original_and_captures_logs(self):
        upload = self.upload("éxample.png")
        status, body = self.request("POST", "/jobs", {"upload_id": upload["upload_id"], "stages": ["depth_map", "extract_objects"]})
        self.assertEqual(status, 202, body)
        job = self.wait_job(body["job"]["id"])
        self.assertEqual(job["status"], "succeeded", job)
        self.assertEqual(job["original_filename"], "éxample.png")
        self.assertEqual(job["completed_stages"], 2)
        self.assertEqual([stage["status"] for stage in job["stages"]], ["succeeded", "succeeded"])
        self.assertEqual([key for key, _ in self.calls], ["extract_objects", "depth_map"])
        folder = self.manager.data_dir / job["result_id"]
        self.assertTrue((folder / "original" / "original.webp").is_file())
        self.assertEqual(self.calls[0][1]["image_path"].stem, folder.name)
        self.assertEqual(self.calls[1][1]["run_folder"], folder)
        self.assertIn("Extraction completed", job["logs"])
        self.assertEqual(json.loads((self.manager.root / job["id"] / "status.json").read_text())["status"], "succeeded")
        self.cleanup.assert_called_once()

    def test_active_job_rejects_second_start_and_snapshots_prompts(self):
        entered, release = threading.Event(), threading.Event()
        def blocked_extract(**kwargs):
            entered.set()
            if not release.wait(5):
                raise RuntimeError("Test release timed out")
            return self.extract(**kwargs)
        self.runners["extract_objects"] = blocked_extract
        profile = default_profiles()["object_detect"]
        profile["positive"] = "Job-specific intent"
        try:
            status, body = self.submit(prompts={"object_detect": profile})
            self.assertEqual(status, 202)
            self.assertTrue(entered.wait(2))
            self.assertEqual(self.submit()[0], 409)
            self.assertEqual(self.request("GET", "/config")[1]["active_job_id"], body["job"]["id"])
            changed = {**profile, "positive": "Changed global intent"}
            self.assertEqual(self.request("PUT", "/config", {"prompts": {"object_detect": changed}})[0], 200)
        finally:
            release.set()
        job = self.wait_job(body["job"]["id"])
        self.assertEqual(job["status"], "succeeded")
        self.assertEqual(self.calls[0][1]["prompt_profile"]["positive"], "Job-specific intent")
        snapshot = self.manager.root / job["id"] / "prompts" / "object_detect.json"
        self.assertEqual(json.loads(snapshot.read_text())["positive"], "Job-specific intent")

    def test_failed_stage_stops_later_work_and_persists_error(self):
        self.runners["depth_map"] = Mock(side_effect=RuntimeError("Model unavailable"))
        upload = self.upload()
        status, body = self.request("POST", "/jobs", {"upload_id": upload["upload_id"], "stages": ["extract_objects", "depth_map", "image_description"]})
        self.assertEqual(status, 202)
        job = self.wait_job(body["job"]["id"])
        self.assertEqual(job["status"], "failed")
        self.assertIn("Model unavailable", job["error"])
        self.assertEqual([stage["status"] for stage in job["stages"]], ["succeeded", "failed", "skipped"])
        self.assertEqual(len(self.calls), 1)
        self.assertIsNone(self.manager.active_job_id)
        self.cleanup.assert_called_once()
        self.assertEqual(self.request("GET", f"/jobs/{job['id']}/artifact")[0], 409)

    def test_upload_rejects_bad_images_and_paths(self):
        for name in ("../escape.png", "C:\\secret.png", "%2fescape.png", "invalid.png"):
            status, _ = self.request("POST", "/uploads", data=b"not-an-image", headers={"X-Filename": name})
            self.assertEqual(status, 400)
        self.assertEqual(list(self.manager.uploads.root.iterdir()), [])

    def test_jpeg_encoding_is_preserved_for_forensic_analysis(self):
        buffer = io.BytesIO()
        Image.new("RGB", (12, 8), "blue").save(buffer, "JPEG", quality=73)
        encoded = buffer.getvalue()
        status, upload = self.request("POST", "/uploads", data=encoded, headers={"X-Filename": "forensics.jpg"})
        self.assertEqual(status, 201)
        image, _ = self.manager.uploads.get(upload["upload_id"])
        self.assertEqual(image.suffix, ".jpg")
        self.assertEqual(image.read_bytes(), encoded)
        status, body = self.request("POST", "/jobs", {"upload_id": upload["upload_id"], "stages": ["extract_objects"]})
        self.assertEqual(status, 202)
        self.assertEqual(self.wait_job(body["job"]["id"])["status"], "succeeded")
        self.assertEqual(self.calls[0][1]["image_path"].read_bytes(), encoded)

    def test_jobs_reject_missing_extraction_unknown_stages_and_upload_paths(self):
        upload = self.upload()
        for payload in (
            {"upload_id": upload["upload_id"], "stages": ["depth_map"]},
            {"upload_id": upload["upload_id"], "stages": ["extract_objects", "unknown"]},
            {"upload_id": "../outside", "stages": ["extract_objects"]},
            {"upload_id": upload["upload_id"], "stages": ["extract_objects"], "run_folder": "C:/outside"},
            {"upload_id": upload["upload_id"], "stages": ["extract_objects"], "prompts": {"object_detect": "Unstructured prompt client"}},
        ):
            self.assertEqual(self.request("POST", "/jobs", payload)[0], 400)
        self.assertEqual(self.manager.list(), [])

    def test_requests_reject_browser_origin_wrong_host_and_nonfinite_json(self):
        self.assertEqual(self.request("PUT", "/config", {}, headers={"Origin": "https://example.com"})[0], 403)
        self.assertEqual(self.request("GET", "/health", headers={"Host": "attacker.example"})[0], 403)
        self.assertEqual(self.request("POST", "/jobs", data=b'{"stages": NaN}', headers={"Content-Type": "application/json"})[0], 400)
        self.assertEqual(self.request("POST", "/jobs", data=b"{}", headers={"Content-Type": "text/plain"})[0], 415)
        self.assertEqual(self.request("GET", "/jobs/missing")[0], 404)

    def test_artifact_manifest_hashes_and_downloads_do_not_include_private_job_data(self):
        status, body = self.submit()
        self.assertEqual(status, 202)
        job = self.wait_job(body["job"]["id"])
        status, artifact = self.request("GET", f"/jobs/{job['id']}/artifact")
        self.assertEqual(status, 200, artifact)
        self.assertEqual(artifact["result_id"], job["id"])
        self.assertEqual({file["path"] for file in artifact["files"]}, {"metadata.json", "original/original.webp"})
        self.assertEqual(artifact["total_size"], sum(file["size"] for file in artifact["files"]))
        for file in artifact["files"]:
            status, content = self.request("GET", f"/jobs/{job['id']}/files/{file['path']}")
            if isinstance(content, dict):
                # request helper decodes application/json, so hash actual bytes with urllib here.
                with urllib.request.urlopen(f"{self.base}/jobs/{job['id']}/files/{file['path']}") as response:
                    content = response.read()
            self.assertEqual(status, 200)
            self.assertEqual(hashlib.sha256(content).hexdigest(), file["sha256"])
        for path in ("../status.json", "%2e%2e/status.json", ".jobs/status.json", "C:/secret.json"):
            self.assertEqual(self.request("GET", f"/jobs/{job['id']}/files/{path}")[0], 404)
        (self.manager.data_dir / job["id"] / "metadata.json").write_text("changed")
        self.assertEqual(self.request("GET", f"/jobs/{job['id']}/files/metadata.json")[0], 409)

    def test_restart_marks_abandoned_job_interrupted(self):
        job_id = "old-image-0123456789ab"
        atomic_json(self.manager.root / job_id / "status.json", dict(id=job_id, status="running", created_at="2026-01-01", updated_at="2026-01-01", stages=[dict(key="depth_map", status="running")]))
        restored = JobManager(self.manager.data_dir, self.config, runners={})
        job = restored.get(job_id)
        self.assertEqual(job["status"], "interrupted")
        self.assertEqual(job["stages"][0]["status"], "skipped")
        self.assertIsNone(restored.active_job_id)

    def test_item_progress_is_live_durable_and_preserved_after_completion(self):
        waiting, release = threading.Event(), threading.Event()

        def process(**kwargs):
            progress_steps("Measure regions", "Save results")
            for item in progress_items(["done", "small", "unavailable", "last"], step=1):
                if item == "small":
                    skip_item()
                elif item == "unavailable":
                    fail_item()
                elif item == "last":
                    waiting.set()
                    if not release.wait(5):
                        raise TimeoutError("Test did not release the worker")
            start_step(2)

        self.runners["depth_map"] = process
        status, body = self.submit()
        self.assertEqual(status, 202)
        job_id = body["job"]["id"]
        try:
            self.assertTrue(waiting.wait(3))
            _, response = self.request("GET", f"/jobs/{job_id}")
            stage = response["job"]["stages"][1]
            self.assertEqual(stage["current_substep"], 1)
            step = stage["substeps"][0]
            self.assertEqual((step["completed"], step["total"], step["skipped"], step["failed"], step["current"]),
                             (3, 4, 1, 1, "last"))
            disk = json.loads((self.manager.root / job_id / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(disk["stages"][1]["substeps"][0], step)
            recovered = JobManager(self.manager.data_dir, self.config, runners={}).get(job_id)
            recovered_step = recovered["stages"][1]["substeps"][0]
            self.assertEqual((recovered_step["status"], recovered_step["completed"]), ("interrupted", 3))
        finally:
            release.set()
        job = self.wait_job(job_id)
        self.assertEqual(job["status"], "succeeded")
        self.assertEqual(job["stages"][1]["current_substep"], 2)
        self.assertEqual(job["stages"][1]["substeps"][0]["completed"], 4)
        self.assertEqual(job["stages"][0]["substeps"][0]["label"], "Processing")

    def failed_job(self):
        self.runners["depth_map"] = Mock(side_effect=RuntimeError("Model unavailable"))
        status, body = self.submit()
        self.assertEqual(status, 202)
        return self.wait_job(body["job"]["id"])

    def wait_queue(self):
        deadline = time.monotonic() + 5
        while self.manager.active_job_id is not None and time.monotonic() < deadline:
            self.manager.worker.join(.05)
        self.assertIsNone(self.manager.active_job_id)

    def test_resume_preserves_completed_work_and_original_settings(self):
        job = self.failed_job()
        original = self.manager.data_dir / job["id"] / "original" / "original.webp"
        original_time = original.stat().st_mtime_ns
        self.config.update({"settings": {"device": "cpu"}})
        self.runners["depth_map"] = self.record_stage("depth_map")
        status, response = self.request("POST", "/jobs/resume", {"job_ids": [job["id"]]})
        self.assertEqual(status, 202, response)
        resumed = self.wait_job(job["id"])
        self.assertEqual((resumed["status"], resumed["completed_stages"], resumed["resume_attempts"]), ("succeeded", 2, 1))
        self.assertEqual([call[0] for call in self.calls], ["extract_objects", "depth_map"])
        self.assertIsNone(self.calls[-1][1]["device"])
        self.assertEqual(resumed["stages"][0], job["stages"][0])
        self.assertEqual(original.stat().st_mtime_ns, original_time)

    def test_resume_queue_runs_oldest_first_and_moves_on_after_a_failure(self):
        jobs = [self.failed_job() for _ in range(3)]
        started, release = threading.Event(), threading.Event()
        order = []

        def process(run_folder, **kwargs):
            order.append(run_folder.name)
            if len(order) == 1:
                started.set()
                release.wait(5)
                raise RuntimeError("Still unavailable")

        self.runners["depth_map"] = process
        status, response = self.request("POST", "/jobs/resume", {"job_ids": [job["id"] for job in reversed(jobs)]})
        self.assertEqual(status, 202, response)
        try:
            self.assertTrue(started.wait(3))
            self.assertEqual(self.manager.get(jobs[1]["id"])["queue_position"], 1)
            self.assertEqual(self.manager.get(jobs[2]["id"])["queue_position"], 2)
            self.assertEqual(self.request("DELETE", f"/jobs/{jobs[0]['id']}")[0], 409)
            self.assertEqual(self.request("POST", "/jobs/resume", {"job_ids": [jobs[1]["id"]]})[0], 409)
        finally:
            release.set()
            self.wait_queue()
        self.assertEqual(order, [job["id"] for job in jobs])
        self.assertEqual([self.manager.get(job["id"])["status"] for job in jobs], ["failed", "succeeded", "succeeded"])

    def test_delete_removes_only_the_selected_unfinished_run(self):
        first, second = self.failed_job(), self.failed_job()
        checkpoint = self.manager.root / first["id"] / "checkpoints" / "sample.json"
        atomic_json(checkpoint, {"response": "saved"})
        status, response = self.request("DELETE", f"/jobs/{first['id']}")
        self.assertEqual(status, 200, response)
        self.assertEqual(self.request("GET", f"/jobs/{first['id']}")[0], 404)
        self.assertFalse((self.manager.root / first["id"]).exists())
        self.assertFalse((self.manager.data_dir / first["id"]).exists())
        self.assertEqual(list(self.manager.uploads.root.glob(first["id"] + ".*")), [])
        self.assertEqual(list(self.manager.uploads.root.glob(first["upload_id"] + ".*")), [])
        self.assertTrue((self.manager.root / second["id"] / "settings.json").is_file())
        self.assertTrue((self.manager.data_dir / second["id"] / "metadata.json").is_file())
        self.assertEqual(self.request("DELETE", "/jobs/..")[0], 404)

    def test_queued_run_can_be_deleted_without_interrupting_the_worker(self):
        first, second = self.failed_job(), self.failed_job()
        entered, release = threading.Event(), threading.Event()
        order = []

        def process(run_folder, **kwargs):
            order.append(run_folder.name)
            entered.set()
            release.wait(5)

        self.runners["depth_map"] = process
        self.request("POST", "/jobs/resume", {"job_ids": [first["id"], second["id"]]})
        try:
            self.assertTrue(entered.wait(3))
            self.assertEqual(self.request("DELETE", f"/jobs/{second['id']}")[0], 200)
        finally:
            release.set()
            self.wait_queue()
        self.assertEqual(order, [first["id"]])
        self.assertEqual(self.manager.get(first["id"])["status"], "succeeded")
        self.assertEqual(self.request("DELETE", f"/jobs/{first['id']}")[0], 409)

    def test_resume_preflight_rejects_missing_inputs_without_queuing_other_runs(self):
        first, second = self.failed_job(), self.failed_job()
        for path in self.manager.uploads.root.glob(second["id"] + ".*"):
            path.unlink()
        status, _ = self.request("POST", "/jobs/resume", {"job_ids": [first["id"], second["id"]]})
        self.assertEqual(status, 400)
        self.assertEqual(self.manager.get(first["id"])["status"], "failed")
        self.assertEqual(self.manager.queue, [])
        self.assertIsNone(self.manager.active_job_id)

    def test_older_unfinished_runs_are_not_hidden_by_the_recent_history_limit(self):
        failed = self.failed_job()
        for index in range(110):
            self.manager.jobs[f"recent-{index}"] = {**failed, "id": f"recent-{index}", "status": "succeeded", "created_at": f"2099-{index:03}"}
        self.assertIn(failed["id"], [job["id"] for job in self.manager.list()])

    def test_resume_disk_failure_rolls_back_the_whole_batch(self):
        first, second = self.failed_job(), self.failed_job()
        persist = self.manager._persist
        attempts = 0

        def write(job):
            nonlocal attempts
            attempts += 1
            if attempts == 2:
                raise OSError("Disk temporarily unavailable")
            persist(job)

        with patch.object(self.manager, "_persist", side_effect=write):
            status, _ = self.request("POST", "/jobs/resume", {"job_ids": [first["id"], second["id"]]})
        self.assertEqual(status, 500)
        self.assertEqual(self.manager.queue, [])
        self.assertIsNone(self.manager.active_job_id)
        for job in (first, second):
            self.assertEqual(self.manager.get(job["id"])["status"], "failed")
            saved = json.loads((self.manager.root / job["id"] / "status.json").read_text())
            self.assertEqual(saved["status"], "failed")

    def test_submit_disk_failure_leaves_a_recoverable_run_instead_of_a_stuck_queue(self):
        persist = self.manager._persist
        attempts = 0

        def write(job):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise OSError("Disk temporarily unavailable")
            persist(job)

        with patch.object(self.manager, "_persist", side_effect=write):
            status, _ = self.submit()
        self.assertEqual(status, 500)
        self.assertEqual(self.manager.list()[0]["status"], "failed")
        self.assertIsNone(self.manager.active_job_id)
        self.assertEqual(self.manager.queue, [])

    def test_full_plan_translates_web_settings_without_ui_objects(self):
        settings = default_settings()
        settings["device"] = "cpu"
        for key, profile in default_profiles().items():
            atomic_json(self.root / "prompts" / f"{key}.json", profile)
        plan = build_plan(self.root / "result", self.root / "source.png", [stage.key for stage in STAGES], settings, self.root / "prompts")
        by_key = {call.key: call.options for call in plan.stages}
        self.assertEqual(by_key["extract_objects"]["prompt_profile"], default_profiles()["object_detect"])
        self.assertNotIn("prompt_path", by_key["extract_objects"])
        self.assertTrue(by_key["extract_objects"]["unload_sam_after"])
        self.assertEqual(by_key["per_item_rankings"]["min_region_size"], 12)
        for key, penalty in (("extract_objects", 0), ("analyze_segments", 0), ("image_description", 1.5),
                             ("per_item_rankings", 1.5), ("full_image_rankings", 0)):
            self.assertEqual(by_key[key]["presence_penalty"], penalty)
        self.assertFalse(by_key["synthetic_analysis"]["run_forgery_after"])
        for key in ("extract_objects", "depth_map", "deepgaze_msdb", "analyze_segments", "pose_extraction", "forgery_detection"):
            self.assertEqual(by_key[key]["device"], "cpu")

    def test_backend_output_environment_is_independent_from_frontend_data(self):
        with patch.dict(os.environ, {"ANALYSIS_OUTPUT_DIR": str(self.root / "backend-results"), "ANALYSIS_DATA_DIR": str(self.root / "frontend-data")}):
            self.assertEqual(default_data_dir(), self.root / "backend-results")


if __name__ == "__main__":
    unittest.main()
