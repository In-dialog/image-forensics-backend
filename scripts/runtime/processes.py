"""Own short-lived process groups and reap them on failure or normal backend exit."""
from __future__ import annotations

import atexit
import contextlib
import os
import signal
import subprocess
import threading

_lock = threading.RLock()
_children = set()
_closing = False


def terminate_process_tree(process):
    if os.name == "nt":
        if process.poll() is None:
            try:
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                               capture_output=True, timeout=10,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            except (OSError, subprocess.SubprocessError):
                process.terminate()
    else:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            process.kill()
        else:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)
    finally:
        if os.name != "nt":
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)


@contextlib.contextmanager
def owned_process(command, **options):
    with _lock:
        if _closing:
            raise RuntimeError("The backend is shutting down; no inference worker can be started.")
        process = subprocess.Popen(command, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                                   start_new_session=os.name != "nt", **options)
        _children.add(process)
    try:
        yield process
    finally:
        try:
            if process.poll() is None or os.name != "nt":
                terminate_process_tree(process)
            with contextlib.suppress(OSError, ValueError, subprocess.SubprocessError):
                process.communicate(timeout=5)
        finally:
            with _lock:
                _children.discard(process)
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None:
                    stream.close()


def _shutdown_workers():
    global _closing
    with _lock:
        _closing = True
        children = list(_children)
        _children.clear()
    for process in children:
        try:
            terminate_process_tree(process)
        except (OSError, subprocess.SubprocessError) as exc:
            print(f"Could not stop inference worker {process.pid}: {exc}")


atexit.register(_shutdown_workers)
