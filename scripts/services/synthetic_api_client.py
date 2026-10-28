"""Synthetic service health/startup using plain configuration, independent of UI."""
from __future__ import annotations

import atexit
import contextlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path


_ownership_lock = threading.RLock()
_owned_processes: list[subprocess.Popen] = []
_closing = False
IS_WINDOWS = os.name == "nt"
FORCE_KILL = getattr(signal, "SIGKILL", 9)


def begin_service_session() -> None:
    """Allow startup for a newly bound API server; never discard owned processes."""
    global _closing
    with _ownership_lock:
        if _owned_processes:
            raise RuntimeError("A synthetic service session is already active in this process.")
        _closing = False


def _stop_process_tree(process: subprocess.Popen) -> None:
    """Terminate only a process/group created and registered by this backend."""
    if IS_WINDOWS:
        if process.poll() is None:
            try:
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                               capture_output=True, check=False, timeout=15)
            except (OSError, subprocess.SubprocessError):
                process.terminate()
    else:
        # Every owned POSIX process starts a new session; its PID is the group ID.
        # Signal the group even if its leader exited, so foreground children stop too.
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if IS_WINDOWS:
            process.kill()
        else:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, FORCE_KILL)
        process.wait(timeout=5)
    finally:
        if not IS_WINDOWS:
            # A leader can exit before a child; clear any remainder in our group.
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, FORCE_KILL)


def _release_owned(process: subprocess.Popen) -> None:
    with _ownership_lock:
        if process not in _owned_processes:
            return
        _owned_processes.remove(process)
    _stop_process_tree(process)


def stop_owned_services() -> None:
    """Stop our service trees and prevent a late worker from starting another."""
    global _closing
    with _ownership_lock:
        _closing = True
        processes = list(_owned_processes)
        _owned_processes.clear()
    for process in processes:
        try:
            _stop_process_tree(process)
        except (OSError, subprocess.SubprocessError) as exc:
            # Attempt every owned service even if one process has already vanished.
            print(f"Could not finish stopping owned synthetic process {process.pid}: {exc}", file=sys.stderr)


atexit.register(stop_owned_services)


@dataclass(frozen=True)
class SyntheticApiConfig:
    url: str
    auto_start: bool
    start_command: str
    timeout: float
    working_directory: Path


def is_healthy(url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/health", timeout=2) as response:
            return response.status == 200 and bool(json.load(response).get("ok", True))
    except Exception:
        return False


def ensure_available(config: SyntheticApiConfig) -> bool:
    """Return whether a process was launched; raise if the service stays unavailable."""
    with _ownership_lock:
        if _closing:
            raise RuntimeError("The analysis backend is shutting down; no synthetic service will be started.")
        if is_healthy(config.url):
            return False
    if not config.auto_start:
        raise RuntimeError(f"Synthetic API unavailable at {config.url}. Enable auto-start or start it manually.")
    if not config.start_command:
        raise ValueError("Synthetic API start command is empty.")
    candidate = Path(config.start_command).expanduser()
    if not candidate.is_absolute():
        candidate = config.working_directory / candidate
    shell = False
    environment = None
    if candidate.is_file():
        path = str(candidate.resolve())
        if candidate.resolve() == (config.working_directory / "install" / "start_synthetic.py").resolve():
            # Launch the isolated interpreter directly, so Windows has no wrapper
            # that could exit before taskkill can find its descendant process.
            from install.start_synthetic import service_command
            argv, environment = service_command()
        elif candidate.suffix.lower() in {".bat", ".cmd"}:
            argv = ["cmd", "/c", path]
        elif candidate.suffix.lower() == ".py":
            argv = [sys.executable, path]
        else:
            argv = [path]
    else:
        argv = config.start_command
        shell = True
    with _ownership_lock:
        if _closing:
            raise RuntimeError("The analysis backend is shutting down; no synthetic service will be started.")
        process = subprocess.Popen(
            argv, cwd=config.working_directory, shell=shell, env=environment,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            start_new_session=not IS_WINDOWS,
        )
        _owned_processes.append(process)
    try:
        deadline = time.monotonic() + config.timeout
        while time.monotonic() < deadline:
            with _ownership_lock:
                if _closing:
                    raise RuntimeError("The analysis backend shut down while the synthetic service was starting.")
            if is_healthy(config.url):
                with _ownership_lock:
                    if _closing:
                        raise RuntimeError("The analysis backend shut down while the synthetic service was starting.")
                return True
            if process.poll() is not None:
                raise RuntimeError(f"Synthetic API exited before becoming ready (exit code {process.returncode}).")
            time.sleep(0.4)
        raise RuntimeError(f"Synthetic API not ready after {config.timeout:g}s at {config.url}.")
    except BaseException:
        _release_owned(process)
        raise

