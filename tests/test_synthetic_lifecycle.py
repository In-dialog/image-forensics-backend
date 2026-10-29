"""Mocked process ownership tests. No sockets, models or services are started."""
from pathlib import Path
import importlib.util
import signal
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, call, patch

from scripts.services import synthetic_api_client as client


class SyntheticLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.assertEqual(client._owned_processes, [])
        client.begin_service_session()
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = client.SyntheticApiConfig("http://127.0.0.1:8765", True, "trusted-foreground-command", 1, self.root)
        self.process = Mock(pid=43210, returncode=0)
        self.process.poll.return_value = None
        self.process.wait.return_value = 0

    def tearDown(self):
        # A failed assertion must never leave a fake PID for the atexit hook.
        with patch.object(client, "_stop_process_tree"):
            client.stop_owned_services()

    def test_external_healthy_service_is_never_owned_or_stopped(self):
        with patch.object(client, "is_healthy", return_value=True), \
             patch.object(client.subprocess, "Popen") as spawn, \
             patch.object(client, "_stop_process_tree") as stop:
            self.assertFalse(client.ensure_available(self.config))
            client.stop_owned_services()
            spawn.assert_not_called()
            stop.assert_not_called()

    def test_success_is_kept_until_shutdown_and_stopped_once(self):
        with patch.object(client, "is_healthy", side_effect=[False, True]), \
             patch.object(client.subprocess, "Popen", return_value=self.process) as spawn, \
             patch.object(client, "_stop_process_tree") as stop:
            self.assertTrue(client.ensure_available(self.config))
            stop.assert_not_called()
            self.assertEqual(spawn.call_args.kwargs["start_new_session"], not client.IS_WINDOWS)
            client.stop_owned_services()
            client.stop_owned_services()
            stop.assert_called_once_with(self.process)

    def test_timeout_stops_the_owned_process_before_raising(self):
        with patch.object(client, "is_healthy", return_value=False), \
             patch.object(client.subprocess, "Popen", return_value=self.process), \
             patch.object(client.time, "monotonic", side_effect=[0, 2]), \
             patch.object(client, "_stop_process_tree") as stop:
            with self.assertRaisesRegex(RuntimeError, "not ready"):
                client.ensure_available(self.config)
            self.assertEqual(client._owned_processes, [])
            stop.assert_called_once_with(self.process)

    def test_early_exit_cleans_the_registered_process(self):
        self.process.poll.return_value = 23
        self.process.returncode = 23
        with patch.object(client, "is_healthy", return_value=False), \
             patch.object(client.subprocess, "Popen", return_value=self.process), \
             patch.object(client.time, "monotonic", side_effect=[0, 0]), \
             patch.object(client, "_stop_process_tree") as stop:
            with self.assertRaisesRegex(RuntimeError, "exit code 23"):
                client.ensure_available(self.config)
            stop.assert_called_once_with(self.process)

    def test_shutdown_blocks_a_late_worker_from_starting_another_process(self):
        client.stop_owned_services()
        with patch.object(client.subprocess, "Popen") as spawn:
            with self.assertRaisesRegex(RuntimeError, "shutting down"):
                client.ensure_available(self.config)
            spawn.assert_not_called()

    def test_shutdown_during_startup_cannot_leak_a_new_process(self):
        def health(_url):
            if client._owned_processes:
                client.stop_owned_services()
                return True
            return False
        with patch.object(client, "is_healthy", side_effect=health), \
             patch.object(client.subprocess, "Popen", return_value=self.process), \
             patch.object(client, "_stop_process_tree") as stop:
            with self.assertRaisesRegex(RuntimeError, "shut down"):
                client.ensure_available(self.config)
            stop.assert_called_once_with(self.process)
            self.assertEqual(client._owned_processes, [])

    def test_windows_stops_only_the_exact_owned_pid_tree(self):
        with patch.object(client, "IS_WINDOWS", True), patch.object(client.subprocess, "run") as run:
            client._stop_process_tree(self.process)
            run.assert_called_once_with(["taskkill", "/PID", "43210", "/T", "/F"], capture_output=True, check=False, timeout=15)
            self.process.wait.assert_called_once_with(timeout=5)

    def test_linux_stops_the_owned_group_including_remaining_children(self):
        with patch.object(client, "IS_WINDOWS", False), patch.object(client.os, "killpg", create=True) as kill:
            client._stop_process_tree(self.process)
            self.assertEqual(kill.call_args_list, [call(43210, signal.SIGTERM), call(43210, client.FORCE_KILL)])

    def test_shipped_launcher_uses_isolated_python_directly(self):
        root = self.root
        (root / "install").mkdir()
        (root / "install/start_synthetic.py").touch()
        config = client.SyntheticApiConfig(self.config.url, True, str(root / "install/start_synthetic.py"), 1, root)
        direct = ["isolated-python", "-B", "synthetic_api_server.py"]
        with patch("install.start_synthetic.service_command", return_value=(direct, {"SAFE": "environment"})), \
             patch.object(client, "is_healthy", side_effect=[False, True]), \
             patch.object(client.subprocess, "Popen", return_value=self.process) as spawn:
            self.assertTrue(client.ensure_available(config))
            self.assertEqual(spawn.call_args.args[0], direct)
            self.assertFalse(spawn.call_args.kwargs["shell"])
            self.assertEqual(spawn.call_args.kwargs["env"], {"SAFE": "environment", "PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1"})

    def test_server_close_always_stops_owned_services(self):
        # Execute the real server module under an isolated name. The image/job
        # dependencies are unused by server_close and must not require Pillow
        # in this standard-library-only lifecycle test. No server is constructed.
        dependencies = {
            "scripts.api.jobs": {"BusyError": RuntimeError, "JobManager": object},
            "scripts.api.artifacts": {"ArtifactNotReadyError": RuntimeError, "ArtifactStore": object},
            "scripts.api.uploads": {"MAX_UPLOAD_BYTES": 1},
            "scripts.api.previews": {"PreviewStore": object},
        }
        stubs = {}
        for name, attributes in dependencies.items():
            stub = types.ModuleType(name)
            vars(stub).update(attributes)
            stubs[name] = stub
        path = Path(__file__).resolve().parents[1] / "scripts/api/server.py"
        spec = importlib.util.spec_from_file_location("scripts.api._lifecycle_test_server", path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, stubs):
            spec.loader.exec_module(module)
            server = object.__new__(module.AnalysisServer)
            with patch.object(module.ThreadingHTTPServer, "server_close", side_effect=OSError("close failed")) as close, \
                 patch.object(module, "stop_owned_services") as stop:
                with self.assertRaisesRegex(OSError, "close failed"):
                    server.server_close()
                close.assert_called_once_with()
                stop.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
