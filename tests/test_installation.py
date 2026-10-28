"""Installer safety and profile tests: no dependencies or downloads required."""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from install import models, setup
from install import start
from install.hardware import choose_profile, compatibility_errors


class InstallationTests(unittest.TestCase):
    def test_profiles_follow_hardware(self):
        self.assertEqual(choose_profile("auto", {"gpus": []}), "cpu")
        self.assertEqual(choose_profile("auto", {"gpus": [{"capability": 8.9}]}), "cuda")
        self.assertEqual(choose_profile("auto", {"gpus": [{"capability": 12.0}]}), "cuda-modern")
        self.assertEqual(choose_profile("cpu", {"gpus": [{"capability": 12.0}]}), "cpu")

    def test_mac_and_arm_are_not_promised_as_backends(self):
        for system, machine in [("Darwin", "arm64"), ("Windows", "ARM64")]:
            self.assertTrue(compatibility_errors("cpu", {"system": system, "machine": machine, "gpus": []}))

    def test_old_driver_and_wrong_blackwell_profile_block(self):
        hardware = {"system": "Windows", "machine": "AMD64", "gpus": [{"name": "NVIDIA", "driver": "550.1", "capability": 12.0}]}
        self.assertEqual(len(compatibility_errors("cuda", hardware)), 2)

    def test_dry_run_never_installs_writes_or_fetches(self):
        hardware = {"system": "Windows", "machine": "AMD64", "gpus": []}
        with patch.object(setup, "inspect_hardware", return_value=hardware), \
             patch.object(setup, "free_gib", return_value=1), \
             patch("subprocess.run", side_effect=AssertionError("subprocess forbidden")), \
             patch("pathlib.Path.write_text", side_effect=AssertionError("write forbidden")), \
             patch("urllib.request.urlopen", side_effect=AssertionError("network forbidden")), \
             patch("urllib.request.urlretrieve", side_effect=AssertionError("download forbidden")), \
             redirect_stdout(StringIO()):
            self.assertEqual(setup.main(["--dry-run", "--with-forgery", "--with-synthetic"]), 0)

    def test_model_dry_run_is_offline(self):
        with patch.object(models, "read_state", return_value={"synthetic": True, "forgery": True}), \
             patch.object(models, "prepare_group", side_effect=AssertionError("models forbidden")), \
             patch.object(models, "configure_environment", side_effect=AssertionError("environment mutation forbidden")), \
             redirect_stdout(StringIO()):
            self.assertEqual(models.main(["--dry-run"]), 0)

    def test_optional_synthetic_has_isolated_torch_environment(self):
        args = setup.arguments(["--with-synthetic"])
        commands = setup.install_commands(args, "cuda", "uv")
        synthetic = [command for command in commands if any(".venv-synthetic" in part for part in command)]
        self.assertTrue(any("torch==2.7.1" in command for command in synthetic))
        self.assertTrue(any("https://download.pytorch.org/whl/cpu" in command for command in synthetic))
        self.assertFalse(any("requirements/synthetic.txt" in part for command in commands if not any(".venv-synthetic" in part for part in command) for part in command))

    def test_low_disk_blocks_real_installs_before_any_package_command(self):
        args = setup.arguments([])
        with patch.object(setup, "free_gib", return_value=1):
            self.assertTrue(setup.disk_errors(args))
            args.allow_low_disk = True
            self.assertEqual(setup.disk_errors(args), [])

    def test_uninstalled_addons_are_not_downloaded_by_default(self):
        self.assertNotIn("synthetic", models.selected_groups("all", {}))
        self.assertNotIn("forgery", models.selected_groups("all", {}))

    def test_start_help_never_starts_or_checks_a_service(self):
        with patch("sys.argv", ["start.py", "--help"]), \
             patch.object(models, "ollama_binary", side_effect=AssertionError("process inspection forbidden")), \
             patch("urllib.request.urlopen", side_effect=AssertionError("network forbidden")), \
             redirect_stdout(StringIO()):
            self.assertEqual(start.main(), 0)

    def test_existing_ollama_is_never_owned_or_stopped(self):
        with patch.object(models, "ollama_ready", return_value=True), \
             patch("subprocess.Popen", side_effect=AssertionError("spawn forbidden")), \
             patch("subprocess.run", side_effect=AssertionError("terminate forbidden")):
            with models.ollama_session("ollama"):
                pass

    def test_core_and_optional_installs_preserve_torch_profile(self):
        args = setup.arguments(["--with-forgery", "--with-synthetic"])
        commands = setup.install_commands(args, "cuda-modern", "uv")
        for command in commands:
            if "-r" in command:
                self.assertIn("-c", command)


if __name__ == "__main__":
    unittest.main()
