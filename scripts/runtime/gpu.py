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
from types import FunctionType


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


def clear_model_caches(model) -> None:
    """Release memoized method outputs that model.to('cpu') does not move.

    Transformers can hide an lru_cache inside a compilation-aware wrapper.
    Those class-level caches retain both the module and CUDA result tensors.
    Inspect only the departing model's method wrappers, never the global heap.
    Call only when that model is no longer performing inference.
    """
    modules = getattr(model, "modules", None)
    owners = modules() if callable(modules) else (model,)
    seen = set()
    for owner in owners:
        for cls in type(owner).__mro__:
            for member in vars(cls).values():
                pending = [getattr(member, "__func__", member)]
                while pending:
                    function = pending.pop()
                    if id(function) in seen or not callable(function):
                        continue
                    seen.add(id(function))
                    clear = getattr(function, "cache_clear", None)
                    if callable(clear):
                        clear()
                    wrapped = getattr(function, "__wrapped__", None)
                    if wrapped is not None:
                        pending.append(wrapped)
                    if isinstance(function, FunctionType):
                        for cell in function.__closure__ or ():
                            try:
                                cached = cell.cell_contents
                            except ValueError:  # Empty closure cell.
                                continue
                            if callable(getattr(cached, "cache_clear", None)):
                                pending.append(cached)


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
    # Synchronize only devices used by this process before releasing workspaces.
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
    # cuBLAS workspaces remain allocated even after every model/tensor owner is
    # gone. empty_cache alone cannot release them. The guarded private hook is
    # needed on PyTorch versions without a public workspace-release API.
    operations = (
        getattr(torch, "clear_autocast_cache", None),
        getattr(getattr(torch, "_C", None), "_cuda_clearCublasWorkspaces", None),
        gc.collect,
        cuda.empty_cache,
        cuda.ipc_collect,
    )
    for operation in operations:
        if not callable(operation):
            continue
        try:
            operation()
        except Exception as exc:
            errors.append(str(exc))
    if report:
        try:
            allocated = sum(cuda.memory_allocated(i) for i in range(cuda.device_count()))
            reserved = sum(cuda.memory_reserved(i) for i in range(cuda.device_count()))
            # Allocator counters include library-owned buffers, so a nonzero
            # count alone does not prove a leaked model or invalidate a result.
            print(f"VRAM cleanup: allocated {allocated / 2**20:.1f} MiB; unused cache {max(0, reserved - allocated) / 2**20:.1f} MiB")
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
