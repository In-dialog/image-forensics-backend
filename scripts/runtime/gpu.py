"""Release inference resources once their owning function has returned.

Emptying CUDA's cache alone cannot free live tensors. In particular, an exception
traceback can keep an entire failed inference alive while the next model loads.
These boundaries discard completed frames before collecting/freeing the cache.
"""
from __future__ import annotations

import gc
import sys
import traceback
from functools import wraps


def clear_exception_frames(error: BaseException) -> None:
    """Keep traceback locations/messages, but release locals in unwound frames."""
    pending, seen = [error], set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        traceback.clear_frames(current.__traceback__)
        pending.extend(e for e in (current.__cause__, current.__context__) if e is not None)


def release_cuda_memory(*, report: bool = False, strict: bool = False) -> None:
    """Collect dead models/tensors and return unused allocations to the driver.

Never import torch, initialize an unused GPU, or reset a live CUDA context just
to clean up. A context reset would invalidate PyTorch's state for the next job.
"""
    gc.collect()
    torch = sys.modules.get("torch")
    cuda = getattr(torch, "cuda", None)
    if cuda is None or not getattr(cuda, "is_initialized", lambda: False)():
        return
    errors = []
    # Synchronize only devices used by this process. empty_cache releases the
    # allocator's unused blocks across devices, including completed workspaces.
    try:
        devices = [i for i in range(cuda.device_count()) if cuda.memory_reserved(i)]
    except Exception as exc:
        errors.append(str(exc))
        devices = []
    for device in devices:
        try:
            cuda.synchronize(device)
        except Exception as exc:
            errors.append(str(exc))
    for operation in (cuda.empty_cache, cuda.ipc_collect):
        try:
            operation()
        except Exception as exc:
            errors.append(str(exc))
    if report:
        try:
            allocated = sum(cuda.memory_allocated(i) for i in range(cuda.device_count()))
            reserved = sum(cuda.memory_reserved(i) for i in range(cuda.device_count()))
            print(f"VRAM cleanup: tensors {allocated / 2**20:.1f} MiB; cache {max(0, reserved - allocated) / 2**20:.1f} MiB")
            if allocated:
                errors.append(f"{allocated / 2**20:.1f} MiB of live tensors remain after the stage")
        except Exception as exc:
            errors.append(str(exc))
    if errors:
        message = "GPU cleanup: " + "; ".join(errors)
        if strict:
            raise RuntimeError(message)
        print(message)


def release_after_call(function):
    """For functions returning CPU data, or explicitly caller-owned models."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except BaseException as exc:
            clear_exception_frames(exc)
            raise
        finally:
            release_cuda_memory()
    return wrapped


def inference_call(function):
    """Disable unnecessary gradient storage and release a finished inference."""
    @release_after_call
    @wraps(function)
    def wrapped(*args, **kwargs):
        import torch
        with torch.no_grad():
            return function(*args, **kwargs)
    return wrapped
