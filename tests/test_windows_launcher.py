"""Real Windows process trees; no app servers, GPU models or downloads."""
import ctypes
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest


@unittest.skipUnless(os.name == "nt" and shutil.which("powershell.exe"), "Windows launcher test")
class WindowsLauncherTests(unittest.TestCase):
    def exercise(self, abrupt):
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        kernel.WaitForSingleObject.restype = ctypes.c_uint32
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        helper = Path(__file__).resolve().parents[1] / "scripts/windows_process_job.ps1"
        with tempfile.TemporaryDirectory(prefix="launcher lifecycle ") as directory:
            root = Path(directory)
            ready, finish = root / "ready.json", root / "finish"
            child = root / "child.py"
            child.write_text(
                "import subprocess,sys,os,json,time\n"
                "from pathlib import Path\n"
                "grandchild=subprocess.Popen([sys.executable,'-B','-c','import time;time.sleep(90)'], "
                "creationflags=subprocess.CREATE_NO_WINDOW)\n"
                "Path(sys.argv[1]).write_text(json.dumps([os.getpid(),grandchild.pid]))\n"
                "time.sleep(90)\n", encoding="utf8")
            launcher = root / "launcher.ps1"
            launcher.write_text(
                "param($Helper, $Python, $Child, $Ready, $Finish)\n"
                "$ErrorActionPreference='Stop'\n. $Helper\nEnable-LauncherProcessCleanup\n"
                "Start-Process -FilePath $Python -ArgumentList @('-B', ('\"'+$Child+'\"'), ('\"'+$Ready+'\"')) "
                "-WindowStyle Hidden | Out-Null\n"
                "while (-not (Test-Path -LiteralPath $Finish)) { Start-Sleep -Milliseconds 50 }\n",
                encoding="utf8")
            # This independent process must survive the launcher's shutdown.
            external = subprocess.Popen([sys.executable, "-B", "-c", "import time;time.sleep(90)"],
                                        creationflags=subprocess.CREATE_NO_WINDOW)
            owner = subprocess.Popen(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                                      str(launcher), str(helper), sys.executable, str(child), str(ready), str(finish)],
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     creationflags=subprocess.CREATE_NO_WINDOW)
            handles = []
            try:
                deadline = time.monotonic() + 15
                while not ready.exists() and owner.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.05)
                if not ready.exists():
                    owner.terminate()
                    output = owner.communicate(timeout=5)
                    self.fail(f"Launcher did not start its child: {output!r}")
                for pid in json.loads(ready.read_text()):
                    handle = kernel.OpenProcess(0x100000, False, pid)  # SYNCHRONIZE
                    self.assertTrue(handle, f"Child {pid} was not running")
                    handles.append(handle)
                if abrupt:
                    owner.terminate()  # No finally/atexit can run in the owner.
                else:
                    finish.touch()
                owner.communicate(timeout=10)
                for handle in handles:
                    self.assertEqual(kernel.WaitForSingleObject(handle, 5000), 0, "A descendant survived its launcher")
                self.assertIsNone(external.poll(), "An unrelated process was stopped")
            finally:
                if owner.poll() is None:
                    owner.terminate()
                owner.communicate(timeout=5)
                for handle in handles:
                    kernel.CloseHandle(handle)
                external.terminate()
                external.wait(timeout=5)

    def test_abrupt_launcher_exit_stops_children_and_grandchildren(self):
        self.exercise(abrupt=True)

    def test_normal_launcher_exit_stops_children_and_grandchildren(self):
        self.exercise(abrupt=False)
