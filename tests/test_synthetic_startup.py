"""Existing environment selection and subprocess startup error reporting."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from install import start_synthetic as launcher
from scripts.services import synthetic_api_client as client


class SyntheticStartupTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="synthetic startup ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        root_patch = patch.object(launcher, "ROOT", self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)
        environment = patch.dict(os.environ)
        environment.start()
        self.addCleanup(environment.stop)
        os.environ.pop("SYNTHETIC_PYTHON", None)
        client.begin_service_session()
        self.addCleanup(client.stop_owned_services)

    def python(self, name):
        path = launcher.python_path(self.root / name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return path

    def test_isolated_environment_preferred(self):
        isolated = self.python(".venv-synthetic")
        self.python(".venv")
        with patch.object(launcher, "missing_synthetic_dependencies", return_value=""):
            self.assertEqual(launcher.resolve_python(), isolated)

    def test_existing_main_environment_can_be_reused(self):
        main = self.python(".venv")
        with patch.object(launcher, "missing_synthetic_dependencies", return_value=""):
            self.assertEqual(launcher.resolve_python(), main)

    def test_incomplete_environment_is_skipped(self):
        isolated = self.python(".venv-synthetic")
        main = self.python(".venv")
        with patch.object(launcher, "missing_synthetic_dependencies", side_effect=lambda path: "nonescape" if path == isolated else ""):
            self.assertEqual(launcher.resolve_python(), main)

    def test_virtual_environment_symlink_is_preserved(self):
        base = self.root / "base-python"
        base.touch()
        interpreter = launcher.python_path(self.root / ".venv-synthetic")
        interpreter.parent.mkdir(parents=True)
        try:
            interpreter.symlink_to(base)
        except (OSError, NotImplementedError):
            self.skipTest("File symlinks unavailable on this machine")
        with patch.object(launcher, "missing_synthetic_dependencies", return_value=""):
            self.assertEqual(launcher.resolve_python(), interpreter)

    def test_explicit_environment_cannot_fall_back_silently(self):
        self.python(".venv")
        os.environ["SYNTHETIC_PYTHON"] = str(self.root / "missing interpreter")
        with self.assertRaisesRegex(RuntimeError, "interpreter not found"):
            launcher.resolve_python()

    def test_missing_environment_gives_install_command_without_installing(self):
        with self.assertRaisesRegex(RuntimeError, "--with-synthetic") as error:
            launcher.resolve_python()
        self.assertIn("Nothing was installed or downloaded", str(error.exception))

    def test_service_uses_selected_python_without_a_shell_wrapper(self):
        python = self.python("external environment")
        os.environ["SYNTHETIC_PYTHON"] = str(python)
        with patch.object(launcher, "missing_synthetic_dependencies", return_value=""), \
                patch.object(launcher, "configure_environment", return_value={"TEST": "value"}):
            command, environment = launcher.service_command()
        self.assertEqual(command[:3], [str(python), "-B", "-u"])
        self.assertEqual(command[-1], str(self.root / "scripts/services/synthetic_api_server.py"))
        self.assertEqual(environment, {"TEST": "value"})

    def test_actual_failed_process_detail_is_in_the_analysis_error(self):
        script = self.root / "failed service.py"
        script.write_text("import sys\nprint('startup output', flush=True)\nprint('missing dependency detail', file=sys.stderr)\nsys.exit(1)\n")
        config = client.SyntheticApiConfig("http://127.0.0.1:8765", True, str(script), 5, self.root)
        with patch.object(client, "is_healthy", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "exit code 1") as error:
                client.ensure_available(config)
        self.assertIn("missing dependency detail", str(error.exception))
        self.assertIn("startup output", str(error.exception))
        self.assertIn("Startup log:", str(error.exception))
        self.assertEqual(client._owned_processes, [])


if __name__ == "__main__":
    unittest.main()
